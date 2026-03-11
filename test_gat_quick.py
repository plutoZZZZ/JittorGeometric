#!/usr/bin/env python
import sys
sys.path.insert(0, '.')

import jittor as jt
jt.flags.use_cuda = 1
jt.flags.lazy_execution = 0

from examples import gat_example

print("=== 运行 GAT 示例 (快速测试 - 10轮) ===")
gat_example.best_val_acc = gat_example.test_acc = 0
for epoch in range(1, 11):
    gat_example.train()
    train_acc, val_acc, tmp_test_acc = gat_example.test()
    if val_acc > gat_example.best_val_acc:
        gat_example.best_val_acc = val_acc
        gat_example.test_acc = tmp_test_acc
    if epoch % 2 == 0:
        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {gat_example.best_val_acc:.4f}, Test: {gat_example.test_acc:.4f}')
print('GAT 示例测试完成!')
