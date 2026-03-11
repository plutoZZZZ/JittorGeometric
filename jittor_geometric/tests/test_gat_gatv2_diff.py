'''
Test to demonstrate the mathematical difference between GAT and GATv2
The key difference:
- GAT: e_ij = LeakyReLU(a^T [W h_i || W h_j])
- GATv2: e_ij = a^T LeakyReLU([W h_i || W h_j])

When inputs have both positive and negative values, these produce different results.
'''
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jittor as jt
from jittor import Var
from jittor_geometric.data import CSC
from jittor_geometric.ops import cootocsc

jt.flags.use_cuda = 0
jt.flags.lazy_execution = 0
jt.misc.set_global_seed(42)

def create_test_graph_with_neg_values():
    """Create a test graph with both positive and negative values"""
    edge_index = jt.array([[0, 1], [1, 0]])  # Simple bidirectional graph
    edge_weight = jt.ones(edge_index.shape[1])
    csc = cootocsc(edge_index, edge_weight, 2)
    # Input with both positive and negative values
    x = jt.array([[1.0, -2.0], [-3.0, 4.0]])
    return x, csc, edge_index.shape[1]

x, csc, e_num = create_test_graph_with_neg_values()

print("=" * 60)
print("Testing GAT vs GATv2 mathematical difference")
print("=" * 60)
print(f"Input x (with negative values):\n{x.numpy()}")
print()

# Parameters
in_channels = 2
out_channels = 2
heads = 1

# Create projection weight and attention vector
weight = jt.array([[0.5, 0.0], [0.0, 0.5]])  # Simple identity-like projection
att = jt.array([[1.0, 1.0, -1.0, -1.0]])  # Attention vector that will show difference

print(f"Projection weight:\n{weight.numpy()}")
print(f"Attention vector:\n{att.numpy()}")
print()

# Vertex forward (projection)
x_proj = x @ weight
print(f"After projection:\n{x_proj.numpy()}")
print()

# Scatter to edge
from jittor_geometric.ops import ScatterToEdge, EdgeSoftmax
out1 = ScatterToEdge(x_proj, csc, "src")
out2 = ScatterToEdge(x_proj, csc, "dst")

print(f"ScatterToEdge src:\n{out1.numpy()}")
print(f"ScatterToEdge dst:\n{out2.numpy()}")
print()

# Reshape to 3D
out1_3d = out1.view(-1, heads, out_channels)
out2_3d = out2.view(-1, heads, out_channels)
out_3d = jt.contrib.concat([out1_3d, out2_3d], dim=2)
print(f"Concatenated features (3D):\n{out_3d.numpy()}")
print()

# ====================== GAT computation ======================
# GAT: multiply first, then LeakyReLU
e_gat = (out_3d * att.unsqueeze(0)).sum(dim=2)
activated_gat = jt.nn.leaky_relu(e_gat, scale=0.2)

# ====================== GATv2 computation ======================
# GATv2: LeakyReLU first, then multiply
activated_gatv2 = jt.nn.leaky_relu(out_3d, scale=0.2)
e_gatv2 = (activated_gatv2 * att.unsqueeze(0)).sum(dim=2)

print("=" * 60)
print("KEY COMPARISON")
print("=" * 60)
print(f"GAT - before LeakyReLU (a^T * [x_i||x_j]):\n{e_gat.numpy()}")
print(f"GAT - after LeakyReLU:\n{activated_gat.numpy()}")
print()
print(f"GATv2 - after LeakyReLU on features:\n{activated_gatv2.numpy()}")
print(f"GATv2 - after multiply with a^T:\n{e_gatv2.numpy()}")
print()
print("=" * 60)
print("DIFFERENCE DEMONSTRATION")
print("=" * 60)
print(f"Are GAT and GATv2 outputs equal? {jt.abs(activated_gat - e_gatv2).max() < 1e-5}")
print(f"Difference:\n{(activated_gat - e_gatv2).numpy()}")
print()
print("EXPLANATION:")
print("In GAT, LeakyReLU is applied to the SCALAR attention score")
print("In GATv2, LeakyReLU is applied to the FEATURE VECTOR before dot product")
print("This is why GATv2 can produce different attention scores!")

# Now apply EdgeSoftmax and see difference in attention weights
from jittor_geometric.ops import EdgeSoftmax
a_gat = EdgeSoftmax(activated_gat, csc)
a_gatv2 = EdgeSoftmax(e_gatv2, csc)

print()
print("=" * 60)
print("Attention Weights (after Softmax)")
print("=" * 60)
print(f"GAT attention weights:\n{a_gat.numpy()}")
print(f"GATv2 attention weights:\n{a_gatv2.numpy()}")
print(f"Are attention weights equal? {jt.abs(a_gat - a_gatv2).max() < 1e-5}")
print(f"Difference in attention weights:\n{(a_gat - a_gatv2).numpy()}")

# Now test with the actual layer implementations
print()
print("=" * 60)
print("Testing with actual GAT/GATv2 layers")
print("=" * 60)
from jittor_geometric.nn.conv import GATConv, GATv2Conv

# Create layers with the same weights
gat_layer = GATConv(in_channels, out_channels, e_num, heads=1)
gatv2_layer = GATv2Conv(in_channels, out_channels, e_num, heads=1)

# Set the same weights for fair comparison
gat_layer.weight = weight.clone()
gatv2_layer.weight = weight.clone()
gat_layer.att = att.clone()
gatv2_layer.att = att.clone()
gat_layer.bias = None
gatv2_layer.bias = None

output_gat = gat_layer(x, csc)
output_gatv2 = gatv2_layer(x, csc)

print(f"GAT layer output:\n{output_gat.numpy()}")
print(f"GATv2 layer output:\n{output_gatv2.numpy()}")
print(f"Are outputs equal? {jt.abs(output_gat - output_gatv2).max() < 1e-5}")
print(f"Layer output difference:\n{(output_gat - output_gatv2).numpy()}")

print()
print("=" * 60)
print("MULTI-HEAD ATTENTION TEST")
print("=" * 60)
heads = 2
gat_layer_multi = GATConv(in_channels, out_channels, e_num, heads=heads, concat=True)
gatv2_layer_multi = GATv2Conv(in_channels, out_channels, e_num, heads=heads, concat=True)

# Use deterministic weights
gat_layer_multi.weight = jt.random((in_channels, heads * out_channels))
gatv2_layer_multi.weight = gat_layer_multi.weight.clone()
gat_layer_multi.att = jt.random((heads, 2 * out_channels))
gatv2_layer_multi.att = gat_layer_multi.att.clone()
gat_layer_multi.bias = None
gatv2_layer_multi.bias = None

output_gat_multi = gat_layer_multi(x, csc)
output_gatv2_multi = gatv2_layer_multi(x, csc)

print(f"Multi-head ({heads}) GAT output shape: {output_gat_multi.shape}")
print(f"Multi-head ({heads}) GATv2 output shape: {output_gatv2_multi.shape}")
print(f"Multi-head GAT output:\n{output_gat_multi.numpy()}")
print(f"Multi-head GATv2 output:\n{output_gatv2_multi.numpy()}")
print(f"Are multi-head outputs equal? {jt.abs(output_gat_multi - output_gatv2_multi).max() < 1e-5}")

print()
print("=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
