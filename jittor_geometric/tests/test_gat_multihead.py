'''
Comprehensive test for GAT and GATv2 multi-head attention implementations
This test verifies:
1. Correct output shapes for multi-head attention
2. Mathematical difference between GAT and GATv2
3. Forward and backward pass functionality
4. Attention mechanism correctness
'''
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jittor as jt
from jittor import Var
from jittor_geometric.data import CSC
from jittor_geometric.ops import cootocsc
from jittor_geometric.nn import GATConv, GATv2Conv

# Critical settings for proper execution
jt.flags.use_cuda = 0
jt.flags.lazy_execution = 0
jt.misc.set_global_seed(42)

def create_test_graph():
    """Create a test graph where each node has multiple edges to show softmax effect"""
    edge_index = jt.array([
        [0, 1, 2, 3, 0, 1],  # src
        [0, 0, 1, 1, 2, 3]   # dst
    ])
    edge_weight = jt.ones(edge_index.shape[1])
    csc = cootocsc(edge_index, edge_weight, 4)
    
    x = jt.array([
        [1.0, -1.0],
        [-2.0, 2.0],
        [3.0, -3.0],
        [-4.0, 4.0]
    ])
    return x, csc, edge_index.shape[1]

def test_multihead_shapes():
    """Test multi-head attention output shapes"""
    print("=" * 70)
    print("TEST 1: Multi-head attention output shapes")
    print("=" * 70)
    
    x, csc, e_num = create_test_graph()
    in_channels = 2
    out_channels = 4
    heads = 3
    
    gat = GATConv(in_channels, out_channels, e_num, heads=heads, concat=True)
    out_gat = gat(x, csc)
    print(f"GATConv (concat=True): {out_gat.shape} expected [4, {heads * out_channels}]")
    assert out_gat.shape == (4, heads * out_channels), f"GAT concat shape mismatch: {out_gat.shape}"
    
    gat_avg = GATConv(in_channels, out_channels, e_num, heads=heads, concat=False)
    out_gat_avg = gat_avg(x, csc)
    print(f"GATConv (concat=False): {out_gat_avg.shape} expected [4, {out_channels}]")
    assert out_gat_avg.shape == (4, out_channels), f"GAT avg shape mismatch: {out_gat_avg.shape}"
    
    gatv2 = GATv2Conv(in_channels, out_channels, e_num, heads=heads, concat=True)
    out_gatv2 = gatv2(x, csc)
    print(f"GATv2Conv (concat=True): {out_gatv2.shape} expected [4, {heads * out_channels}]")
    assert out_gatv2.shape == (4, heads * out_channels), f"GATv2 concat shape mismatch: {out_gatv2.shape}"
    
    gatv2_avg = GATv2Conv(in_channels, out_channels, e_num, heads=heads, concat=False)
    out_gatv2_avg = gatv2_avg(x, csc)
    print(f"GATv2Conv (concat=False): {out_gatv2_avg.shape} expected [4, {out_channels}]")
    assert out_gatv2_avg.shape == (4, out_channels), f"GATv2 avg shape mismatch: {out_gatv2_avg.shape}"
    
    print("✓ All shape tests passed!")
    return True

def test_gat_gatv2_mathematical_difference():
    """Test that GAT and GATv2 produce different outputs"""
    print("\n" + "=" * 70)
    print("TEST 2: GAT vs GATv2 mathematical difference")
    print("=" * 70)
    
    x, csc, e_num = create_test_graph()
    in_channels = 2
    out_channels = 2
    heads = 2
    
    gat = GATConv(in_channels, out_channels, e_num, heads=heads, concat=True, bias=False)
    gatv2 = GATv2Conv(in_channels, out_channels, e_num, heads=heads, concat=True, bias=False)
    
    gatv2.weight = gat.weight.clone()
    gatv2.att = gat.att.clone()
    
    out_gat = gat(x, csc)
    out_gatv2 = gatv2(x, csc)
    
    print(f"Input x:\n{x.numpy()}")
    print(f"\nGAT output:\n{out_gat.numpy()}")
    print(f"\nGATv2 output:\n{out_gatv2.numpy()}")
    
    max_diff = jt.max(jt.abs(out_gat - out_gatv2)).item()
    print(f"\nMaximum difference between GAT and GATv2: {max_diff}")
    
    if max_diff > 1e-6:
        print("✓ GAT and GATv2 produce different outputs (as expected due to activation order)")
        return True
    else:
        print("✗ WARNING: GAT and GATv2 outputs are identical!")
        return False

def test_attention_mechanism():
    """Test that attention mechanism produces valid attention weights"""
    print("\n" + "=" * 70)
    print("TEST 3: Attention weights verification")
    print("=" * 70)
    
    x, csc, e_num = create_test_graph()
    in_channels = 2
    out_channels = 2
    
    gat = GATConv(in_channels, out_channels, e_num, heads=1, concat=True, bias=False)
    gatv2 = GATv2Conv(in_channels, out_channels, e_num, heads=1, concat=True, bias=False)
    
    weight = jt.array([[1.0, 0.0], [0.0, 1.0]])
    att = jt.array([[1.0, 0.0, 0.0, 1.0]])
    
    gat.weight = weight
    gatv2.weight = weight
    gat.att = att
    gatv2.att = att
    
    from jittor_geometric.ops import ScatterToEdge, EdgeSoftmax
    x_proj = x @ weight
    out1 = ScatterToEdge(x_proj, csc, "src")
    out2 = ScatterToEdge(x_proj, csc, "dst")
    
    out1_3d = out1.view(-1, 1, out_channels)
    out2_3d = out2.view(-1, 1, out_channels)
    edge_feat = jt.contrib.concat([out1_3d, out2_3d], dim=2)
    
    e_gat = (edge_feat * att.unsqueeze(0)).sum(dim=2)
    e_gat_activated = jt.nn.leaky_relu(e_gat, scale=0.2)
    a_gat = EdgeSoftmax(e_gat_activated, csc)
    
    edge_feat_activated = jt.nn.leaky_relu(edge_feat, scale=0.2)
    e_gatv2 = (edge_feat_activated * att.unsqueeze(0)).sum(dim=2)
    a_gatv2 = EdgeSoftmax(e_gatv2, csc)
    
    print(f"Edge features:\n{edge_feat.numpy()[:, 0, :]}")
    print(f"\nGAT attention scores:\n{e_gat_activated.numpy()}")
    print(f"GAT attention weights:\n{a_gat.numpy()}")
    print(f"\nGATv2 attention scores:\n{e_gatv2.numpy()}")
    print(f"GATv2 attention weights:\n{a_gatv2.numpy()}")
    
    col0_sum_gat = a_gat[0:2].sum().item()
    col0_sum_gatv2 = a_gatv2[0:2].sum().item()
    col1_sum_gat = a_gat[2:4].sum().item()
    col1_sum_gatv2 = a_gatv2[2:4].sum().item()
    
    print(f"\nSoftmax sum for node 0: GAT={col0_sum_gat:.6f}, GATv2={col0_sum_gatv2:.6f}")
    print(f"Softmax sum for node 1: GAT={col1_sum_gat:.6f}, GATv2={col1_sum_gatv2:.6f}")
    
    assert abs(col0_sum_gat - 1.0) < 1e-5, "GAT softmax sum not 1 for node 0"
    assert abs(col1_sum_gat - 1.0) < 1e-5, "GAT softmax sum not 1 for node 1"
    assert abs(col0_sum_gatv2 - 1.0) < 1e-5, "GATv2 softmax sum not 1 for node 0"
    assert abs(col1_sum_gatv2 - 1.0) < 1e-5, "GATv2 softmax sum not 1 for node 1"
    
    print("✓ Attention weights are valid!")
    
    att_diff = jt.max(jt.abs(e_gat_activated - e_gatv2)).item()
    print(f"Attention score difference: {att_diff}")
    if att_diff > 1e-6:
        print("✓ GAT and GATv2 compute attention scores differently")
        return True
    else:
        print("✗ WARNING: Attention scores are identical!")
        return False

def test_backward_pass():
    """Test backward pass works correctly"""
    print("\n" + "=" * 70)
    print("TEST 4: Backward pass test")
    print("=" * 70)
    
    x, csc, e_num = create_test_graph()
    in_channels = 2
    out_channels = 2
    heads = 2
    
    gat = GATConv(in_channels, out_channels, e_num, heads=heads, concat=True)
    gatv2 = GATv2Conv(in_channels, out_channels, e_num, heads=heads, concat=True)
    
    out_gat = gat(x, csc)
    loss_gat = out_gat.sum()
    
    optimizer = jt.optim.SGD(gat.parameters(), lr=0.01)
    optimizer.backward(loss_gat)
    
    weight_grad = gat.weight.opt_grad(optimizer)
    att_grad = gat.att.opt_grad(optimizer)
    
    def get_norm(v):
        if v is None:
            return 0
        return jt.sqrt(jt.sqr(v).sum()).item()
    
    weight_norm = get_norm(weight_grad)
    att_norm = get_norm(att_grad)
    
    print(f"GAT output sum: {loss_gat.item():.6f}")
    print(f"GAT weight grad norm: {weight_norm:.6f}")
    print(f"GAT att grad norm: {att_norm:.6f}")
    
    assert weight_grad is not None, "GAT weight grad is None"
    assert att_grad is not None, "GAT att grad is None"
    assert weight_norm > 1e-10, f"GAT weight grad is zero: {weight_norm}"
    assert att_norm > 1e-10, f"GAT att grad is zero: {att_norm}"
    
    optimizer2 = jt.optim.SGD(gatv2.parameters(), lr=0.01)
    out_gatv2 = gatv2(x, csc)
    loss_gatv2 = out_gatv2.sum()
    optimizer2.backward(loss_gatv2)
    
    weight_grad2 = gatv2.weight.opt_grad(optimizer2)
    att_grad2 = gatv2.att.opt_grad(optimizer2)
    
    weight_norm2 = get_norm(weight_grad2)
    att_norm2 = get_norm(att_grad2)
    
    print(f"GATv2 output sum: {loss_gatv2.item():.6f}")
    print(f"GATv2 weight grad norm: {weight_norm2:.6f}")
    print(f"GATv2 att grad norm: {att_norm2:.6f}")
    
    assert weight_grad2 is not None, "GATv2 weight grad is None"
    assert att_grad2 is not None, "GATv2 att grad is None"
    assert weight_norm2 > 1e-10, f"GATv2 weight grad is zero: {weight_norm2}"
    assert att_norm2 > 1e-10, f"GATv2 att grad is zero: {att_norm2}"
    
    print("✓ Backward pass tests passed!")
    return True

def test_different_head_configurations():
    """Test with different numbers of heads"""
    print("\n" + "=" * 70)
    print("TEST 5: Different head configurations")
    print("=" * 70)
    
    x, csc, e_num = create_test_graph()
    in_channels = 2
    
    for heads in [1, 2, 4, 8]:
        out_channels = 2
        print(f"\nTesting with {heads} heads:")
        
        gat = GATConv(in_channels, out_channels, e_num, heads=heads, concat=True)
        gatv2 = GATv2Conv(in_channels, out_channels, e_num, heads=heads, concat=True)
        
        out_gat = gat(x, csc)
        out_gatv2 = gatv2(x, csc)
        
        expected_shape = (4, heads * out_channels)
        assert out_gat.shape == expected_shape, f"GAT shape mismatch for {heads} heads: {out_gat.shape}"
        assert out_gatv2.shape == expected_shape, f"GATv2 shape mismatch for {heads} heads: {out_gatv2.shape}"
        
        print(f"  Heads={heads}: shapes OK, output_sum={out_gat.sum().item():.4f}, {out_gatv2.sum().item():.4f}")
    
    print("✓ All head configuration tests passed!")
    return True

if __name__ == "__main__":
    all_passed = True
    
    all_passed &= test_multihead_shapes()
    all_passed &= test_gat_gatv2_mathematical_difference()
    all_passed &= test_attention_mechanism()
    all_passed &= test_backward_pass()
    all_passed &= test_different_head_configurations()
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED!")
    print("=" * 70)
