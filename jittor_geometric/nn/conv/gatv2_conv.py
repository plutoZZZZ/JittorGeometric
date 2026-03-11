'''
Description: GATv2 graph convolutional operator from the "How Attentive are Graph Attention Networks?" paper
Author: lusz
Date: 2024-06-26 10:57:06
Modified for GATv2 with multi-head attention
'''
from typing import Optional, Tuple
from jittor_geometric.typing import Adj, OptVar

import jittor as jt
from jittor import Var
from jittor_geometric.nn.conv import MessagePassingNts
from jittor_geometric.utils import add_remaining_self_loops
from jittor_geometric.utils.num_nodes import maybe_num_nodes

from ..inits import glorot, zeros
from jittor_geometric.data import CSC, CSR
from jittor_geometric.ops import ScatterToEdge, EdgeSoftmax, aggregateWithWeight, ScatterToVertex


class GATv2Conv(MessagePassingNts):
    r"""The graph convolutional operator from the `"How Attentive are Graph Attention Networks?"
    <https://arxiv.org/abs/2105.14491>`_ paper (GATv2).

    GATv2 fixes the static attention issue of GAT by changing the order of operations:
    - GAT: e_ij = LeakyReLU(a^T [W h_i || W h_j])
    - GATv2: e_ij = a^T LeakyReLU([W h_i || W h_j])
    
    This allows the attention mechanism to be a universal approximator.
    
    Args:
        in_channels: Size of each input sample
        out_channels: Size of each output sample (per head)
        e_num: Number of edges in the graph
        heads: Number of multi-head attentions (default: 1)
        concat: If set to False, the multi-head attentions are averaged instead of concatenated (default: True)
        negative_slope: LeakyReLU angle of the negative slope (default: 0.2)
        dropout: Dropout probability of the normalized attention coefficients (default: 0.)
        improved: If set to True, the layer adds self-loops with increased weight (default: False)
        cached: If set to True, the layer caches the computation graph (default: False)
        add_self_loops: If set to True, add self-loops to the graph (default: True)
        normalize: If set to True, normalize the output (default: True)
        bias: If set to False, the layer will not learn an additive bias (default: True)
        share_weights: If set to True, share the weight matrix for src and dst nodes (default: False)
    """

    _cached_edge_index: Optional[Tuple[Var, Var]]
    _cached_csc: Optional[CSC]

    def __init__(self, in_channels: int, out_channels: int, e_num: int,
                 heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, share_weights: bool = False, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GATv2Conv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.share_weights = share_weights

        self._cached_edge_index = None
        self._cached_adj_t = None
        self._cached_csc = None

        # For multi-head attention: each head has its own weight and attention vector
        self.weight = jt.random((in_channels, heads * out_channels))
        
        # Attention vector for each head: (heads, 2 * out_channels)
        # This shape is better for einsum operations
        self.att = jt.random((heads, 2 * out_channels))
        
        if bias:
            if concat:
                self.bias = jt.zeros((heads * out_channels,))
            else:
                self.bias = jt.zeros((out_channels,))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        if self.bias is not None:
            zeros(self.bias)
        self._cached_adj_t = None
        self._cached_csc = None

    def execute(self, x: Var, csc: CSC) -> Var:
        """"""
        out = self.vertex_forward(x)
        out = self.propagate(x=out, csc=csc)
        
        if self.bias is not None:
            out = out + self.bias
            
        return out
    
    def propagate(self, x, csc):
        e_msg = self.scatter_to_edge(x, csc)
        out = self.edge_forward(e_msg, csc)
        out = self.scatter_to_vertex(out, csc)
        return out
    
    def vertex_forward(self, x: Var) -> Var:
        # x: [num_nodes, in_channels]
        # out: [num_nodes, heads * out_channels]
        out = x @ self.weight
        return out
    
    def scatter_to_edge(self, x, csc) -> Var:
        # x: [num_nodes, heads * out_channels]
        # Keep as 2D for ScatterToEdge compatibility
        out1 = ScatterToEdge(x, csc, "src")  # [num_edges, heads * out_channels]
        out2 = ScatterToEdge(x, csc, "dst")  # [num_edges, heads * out_channels]
        
        # Reshape to 3D for concatenation
        out1_3d = out1.view(-1, self.heads, self.out_channels)  # [num_edges, heads, out_channels]
        out2_3d = out2.view(-1, self.heads, self.out_channels)  # [num_edges, heads, out_channels]
        
        # Concatenate src and dst features
        # out: [num_edges, heads, 2 * out_channels]
        out_3d = jt.contrib.concat([out1_3d, out2_3d], dim=2)
        
        # Flatten back to 2D: [num_edges, heads * 2 * out_channels]
        out = out_3d.view(-1, self.heads * 2 * self.out_channels)
        return out
    
    def edge_forward(self, x, csc) -> Var:
        # ==============================================
        # THIS IS THE KEY DIFFERENCE BETWEEN GAT AND GATv2
        # ==============================================
        # GAT: e_ij = LeakyReLU(a^T [W h_i || W h_j])
        # GATv2: e_ij = a^T LeakyReLU([W h_i || W h_j])
        #
        # For multi-head attention, we process each head independently
        
        # x shape: [num_edges, heads * 2 * out_channels]
        num_edges = x.size(0)
        
        # Reshape to 3D: [num_edges, heads, 2 * out_channels]
        x_3d = x.view(num_edges, self.heads, 2 * self.out_channels)
        
        # 1. Apply non-linearity FIRST (GATv2 key difference)
        activated = jt.nn.leaky_relu(x_3d, scale=self.negative_slope)
        
        # 2. THEN compute attention scores with attention vector
        # self.att shape: [heads, 2 * out_channels]
        # e: [num_edges, heads]
        # Replace einsum with basic operations to avoid cupy dependency
        e = (activated * self.att.unsqueeze(0)).sum(dim=2)
        
        # Edge softmax for each head
        a = EdgeSoftmax(e, csc)  # [num_edges, heads]
        
        # Apply dropout to attention
        if self.dropout > 0:
            a = jt.nn.dropout(a, p=self.dropout)
        
        # Extract source features for message passing: [num_edges, heads, out_channels]
        x_src_3d = x_3d[:, :, :self.out_channels]
        
        # Apply attention weights: [num_edges, heads, out_channels] * [num_edges, heads, 1]
        e_msg_3d = x_src_3d * a.unsqueeze(2)
        
        # Flatten back to 2D: [num_edges, heads * out_channels]
        e_msg = e_msg_3d.view(-1, self.heads * self.out_channels)
        
        return e_msg
    
    def scatter_to_vertex(self, edge, csc) -> Var:
        # edge shape: [num_edges, heads * out_channels]
        # For message passing: edge i->j, aggregate message from i to j
        # So we use 'dst' to aggregate to destination nodes
        out = ScatterToVertex(edge, csc, 'dst')  # [num_nodes, heads * out_channels]
        
        # Combine heads
        if self.concat:
            # Keep as concatenated: [num_nodes, heads * out_channels]
            pass
        else:
            # Average heads: [num_nodes, heads, out_channels] -> [num_nodes, out_channels]
            num_nodes = out.size(0)
            out_3d = out.view(num_nodes, self.heads, self.out_channels)
            out = jt.mean(out_3d, dim=1)
            
        return out
    
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, 
                                             self.in_channels,
                                             self.out_channels, 
                                             self.heads)
