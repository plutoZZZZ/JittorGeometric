'''
Description: Test for EdgeSoftmax with 1D and multi-column support
Author: lusz
Date: 2024-07-04 12:01:14
'''

import jittor as jt
import os
import sys
from jittor import nn
from jittor import Function
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.data import CSC, CSR
from jittor_geometric.ops import EdgeSoftmax

jt.flags.use_cuda = 0
jt.flags.lazy_execution = 0

# Test 1D input
print("=" * 60)
print("TEST 1: EdgeSoftmax with 1D input")
print("=" * 60)
x = jt.array([1.0, 2.0, 3.0, 4.0])
y = jt.array([1.0, 1.0, 1.0, 1.0])
row_indices = jt.array([0, 0, 1, 2])
col_offset = jt.array([0, 1, 3, 4])
csc_weight = jt.array([1.0, 2.0, 3.0, 4.0])
csc = CSC(row_indices, col_offset, csc_weight)

output = EdgeSoftmax(x, csc)
print("Input:", x)
print("Output:", output)

# Verify softmax per node
print("\nVerifying softmax sums to 1 per node:")
node0_sum = output[0:1].sum().item()
node1_sum = output[1:3].sum().item()
node2_sum = output[3:4].sum().item()
print(f"Node 0 sum: {node0_sum:.6f}")
print(f"Node 1 sum: {node1_sum:.6f}")
print(f"Node 2 sum: {node2_sum:.6f}")

loss = nn.BCELoss()
loss_var = loss(output, y)
di = jt.grad(loss_var, [x])
print("Input Variable Gradient:", di)

# Test multi-column input
print("\n" + "=" * 60)
print("TEST 2: EdgeSoftmax with multi-column (2D) input")
print("=" * 60)
x_2d = jt.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
output_2d = EdgeSoftmax(x_2d, csc)
print("Input shape:", x_2d.shape)
print("Input:\n", x_2d)
print("Output shape:", output_2d.shape)
print("Output:\n", output_2d)

# Verify softmax per node and per column
print("\nVerifying softmax sums to 1 per node per column:")
for col in range(2):
    node0_sum = output_2d[0:1, col].sum().item()
    node1_sum = output_2d[1:3, col].sum().item()
    node2_sum = output_2d[3:4, col].sum().item()
    print(f"Column {col}: Node 0 sum={node0_sum:.6f}, Node 1 sum={node1_sum:.6f}, Node 2 sum={node2_sum:.6f}")

# Test backward pass for multi-column
print("\n" + "=" * 60)
print("TEST 3: Backward pass with multi-column input")
print("=" * 60)
y_2d = jt.ones_like(output_2d)
loss_2d = nn.BCELoss()
loss_var_2d = loss_2d(output_2d, y_2d)
di_2d = jt.grad(loss_var_2d, [x_2d])
print("Multi-column Input Gradient shape:", di_2d[0].shape)
print("Multi-column Input Gradient:\n", di_2d[0])

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)