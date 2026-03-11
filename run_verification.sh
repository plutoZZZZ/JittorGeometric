#!/bin/bash
source /home/lsz/miniconda3/etc/profile.d/conda.sh
conda activate jittor
cd /mnt/d/Code/qianshi0310/jittorgeometric/2/JittorGeometric

export PYTHONPATH=/mnt/d/Code/qianshi0310/jittorgeometric/2/JittorGeometric:$PYTHONPATH

echo "=== 环境检查 ==="
python -c "import jittor; print('Jittor版本:', jittor.__version__)"
python -c "import jittor_geometric; print('JittorGeometric导入成功')"

echo ""
echo "=== 测试文件列表 ==="
ls jittor_geometric/tests/

echo ""
echo "=== 运行 EdgeSoftmax 测试 ==="
python jittor_geometric/tests/test_edgesoftmax.py

echo ""
echo "=== 运行 GAT/GATv2 差异测试 ==="
python jittor_geometric/tests/test_gat_gatv2_diff.py

echo ""
echo "=== 运行多头注意力测试 ==="
python jittor_geometric/tests/test_gat_multihead.py

echo ""
echo "=== 运行 GAT 示例 (快速测试) ==="
python -c "
import jittor as jt
jt.flags.use_cuda = 1
jt.flags.lazy_execution = 0
import sys
sys.path.insert(0, '/mnt/d/Code/qianshi0310/jittorgeometric/2/JittorGeometric/examples')
import gat_example
gat_example.best_val_acc = gat_example.test_acc = 0
print('训练中...')
for epoch in range(1, 11):
    gat_example.train()
    if epoch % 5 == 0:
        train_acc, val_acc, tmp_test_acc = gat_example.test()
        if val_acc > gat_example.best_val_acc:
            gat_example.best_val_acc = val_acc
            gat_example.test_acc = tmp_test_acc
        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {gat_example.best_val_acc:.4f}, Test: {gat_example.test_acc:.4f}')
print('GAT 示例测试完成!')
"

echo ""
echo "=== 运行 GATv2 示例 (快速测试) ==="
python -c "
import jittor as jt
jt.flags.use_cuda = 1
jt.flags.lazy_execution = 0
import sys
sys.path.insert(0, '/mnt/d/Code/qianshi0310/jittorgeometric/2/JittorGeometric/examples')
import gatv2_example
gatv2_example.best_val_acc = gatv2_example.test_acc = 0
print('训练中...')
for epoch in range(1, 11):
    gatv2_example.train()
    if epoch % 5 == 0:
        train_acc, val_acc, tmp_test_acc = gatv2_example.test()
        if val_acc > gatv2_example.best_val_acc:
            gatv2_example.best_val_acc = val_acc
            gatv2_example.test_acc = tmp_test_acc
        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {gatv2_example.best_val_acc:.4f}, Test: {gatv2_example.test_acc:.4f}')
print('GATv2 示例测试完成!')
"

echo ""
echo "=== 运行诊断检查 ==="
python -c "
from jittor_geometric.nn.conv.gat_conv import GATConv
from jittor_geometric.nn.conv.gatv2_conv import GATv2Conv
print('GATConv 和 GATv2Conv 导入成功!')
print('GATConv 方法:', [m for m in dir(GATConv) if not m.startswith('_')])
print('GATv2Conv 方法:', [m for m in dir(GATv2Conv) if not m.startswith('_')])
"

echo ""
echo "=== 所有验证完成 ==="
