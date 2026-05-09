#!/usr/bin/env python3
from flask import Flask, render_template, jsonify, request, send_from_directory
import threading
import os
import json

app = Flask(__name__, template_folder='templates')

STATUS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_status.json')

MODEL_DATASETS = {
    'GCN': ['Cora', 'Citeseer', 'Pubmed', 'Reddit'],
    'GAT': ['Cora', 'Citeseer', 'Pubmed'],
    'GraphSAGE': ['Cora', 'Citeseer', 'Pubmed', 'Reddit'],
    'ChebNet2': ['Cora', 'Citeseer', 'Pubmed'],
    'SGC': ['Cora', 'Citeseer', 'Pubmed'],
    'APPNP': ['Cora', 'Citeseer', 'Pubmed'],
    'GPRGNN': ['Cora', 'Citeseer', 'Pubmed'],
    'BernNet': ['Cora', 'Citeseer', 'Pubmed'],
}

MODEL_PARAMS = {
    'GCN': [
        {'key': 'hidden_dim', 'label': '隐藏层维度', 'type': 'number', 'default': 256},
        {'key': 'dropout', 'label': 'Dropout率', 'type': 'number', 'default': 0.5, 'step': 0.1},
        {'key': 'lr', 'label': '学习率', 'type': 'number', 'default': 0.01, 'step': 0.001},
        {'key': 'weight_decay', 'label': '权重衰减', 'type': 'number', 'default': 5e-4, 'step': 1e-4},
    ],
    'GAT': [
        {'key': 'hidden_dim', 'label': '隐藏层维度', 'type': 'number', 'default': 128},
        {'key': 'dropout', 'label': 'Dropout率', 'type': 'number', 'default': 0.6, 'step': 0.1},
        {'key': 'lr', 'label': '学习率', 'type': 'number', 'default': 0.005, 'step': 0.001},
        {'key': 'weight_decay', 'label': '权重衰减', 'type': 'number', 'default': 1e-4, 'step': 1e-5},
    ],
    'GraphSAGE': [
        {'key': 'hidden_dim', 'label': '隐藏层维度', 'type': 'number', 'default': 256},
        {'key': 'dropout', 'label': 'Dropout率', 'type': 'number', 'default': 0.5, 'step': 0.1},
        {'key': 'lr', 'label': '学习率', 'type': 'number', 'default': 0.01, 'step': 0.001},
        {'key': 'weight_decay', 'label': '权重衰减', 'type': 'number', 'default': 5e-4, 'step': 1e-4},
    ],
    'ChebNet2': [
        {'key': 'hidden_dim', 'label': '隐藏层维度', 'type': 'number', 'default': 256},
        {'key': 'K', 'label': 'Chebyshev阶数K', 'type': 'number', 'default': 3},
        {'key': 'dropout', 'label': 'Dropout率', 'type': 'number', 'default': 0.5, 'step': 0.1},
        {'key': 'lr', 'label': '学习率', 'type': 'number', 'default': 0.01, 'step': 0.001},
    ],
    'SGC': [
        {'key': 'hidden_dim', 'label': '隐藏层维度', 'type': 'number', 'default': 64},
        {'key': 'K', 'label': '传播步数K', 'type': 'number', 'default': 2},
        {'key': 'dropout', 'label': 'Dropout率', 'type': 'number', 'default': 0.8, 'step': 0.1},
        {'key': 'lr', 'label': '学习率', 'type': 'number', 'default': 0.01, 'step': 0.001},
    ],
    'APPNP': [
        {'key': 'hidden_dim', 'label': '隐藏层维度', 'type': 'number', 'default': 64},
        {'key': 'K', 'label': 'PPR传播步数K', 'type': 'number', 'default': 10},
        {'key': 'alpha', 'label': 'PPR传送概率α', 'type': 'number', 'default': 0.1, 'step': 0.01},
        {'key': 'dropout', 'label': 'Dropout率', 'type': 'number', 'default': 0.5, 'step': 0.1},
    ],
    'GPRGNN': [
        {'key': 'hidden_dim', 'label': '隐藏层维度', 'type': 'number', 'default': 64},
        {'key': 'K', 'label': 'GPR传播阶数K', 'type': 'number', 'default': 10},
        {'key': 'alpha', 'label': 'PPR初始化参数α', 'type': 'number', 'default': 0.2, 'step': 0.01},
        {'key': 'dropout', 'label': 'Dropout率', 'type': 'number', 'default': 0.5, 'step': 0.1},
    ],
    'BernNet': [
        {'key': 'hidden_dim', 'label': '隐藏层维度', 'type': 'number', 'default': 64},
        {'key': 'K', 'label': 'Bernstein阶数K', 'type': 'number', 'default': 5},
        {'key': 'dropout', 'label': 'Dropout率', 'type': 'number', 'default': 0.5, 'step': 0.1},
        {'key': 'lr', 'label': '学习率', 'type': 'number', 'default': 0.01, 'step': 0.001},
    ],
}

MODEL_GRAPH_INFO = {
    'GCN': {
        'nodes': [
            {'id': 'input', 'label': 'Input\nx: Var[N, F]', 'category': 'io'},
            {'id': 'linear1', 'label': 'Linear1\nnn.Linear(F, H)', 'category': 'meta'},
            {'id': 'scatter1', 'label': 'ScatterToEdge1\nscatter_to_edge', 'category': 'custom'},
            {'id': 'aggregate1', 'label': 'Aggregate1\nAggregateWithWeight', 'category': 'custom'},
            {'id': 'relu', 'label': 'ReLU\nnn.relu', 'category': 'meta'},
            {'id': 'dropout1', 'label': 'Dropout\nnn.dropout', 'category': 'meta'},
            {'id': 'linear2', 'label': 'Linear2\nnn.Linear(H, C)', 'category': 'meta'},
            {'id': 'scatter2', 'label': 'ScatterToEdge2\nscatter_to_edge', 'category': 'custom'},
            {'id': 'aggregate2', 'label': 'Aggregate2\nAggregateWithWeight', 'category': 'custom'},
            {'id': 'logsoftmax', 'label': 'LogSoftmax\nnn.logsoftmax', 'category': 'meta'},
            {'id': 'output', 'label': 'Output\nVar[N, C]', 'category': 'io'},
        ],
        'edges': [
            ['input', 'linear1'], ['linear1', 'scatter1'], ['scatter1', 'aggregate1'],
            ['aggregate1', 'relu'], ['relu', 'dropout1'],
            ['dropout1', 'linear2'], ['linear2', 'scatter2'], ['scatter2', 'aggregate2'],
            ['aggregate2', 'logsoftmax'], ['logsoftmax', 'output'],
        ],
        'fusion_groups': [
            {
                'name': '元算子融合',
                'ops': ['x@W₁', '+b₁'],
                'type': 'meta_fusion',
            },
            {
                'name': '跨边界融合',
                'ops': ['Aggregate1', 'ReLU'],
                'type': 'cross_boundary',
            },
            {
                'name': '元算子融合',
                'ops': ['x@W₂', '+b₂'],
                'type': 'meta_fusion',
            },
            {
                'name': '跨边界融合',
                'ops': ['Aggregate2', 'LogSoftmax'],
                'type': 'cross_boundary',
            },
        ],
    },
    'GAT': {
        'nodes': [
            {'id': 'input', 'label': 'Input\nx: Var[N, F]', 'category': 'io'},
            {'id': 'linear1', 'label': 'Linear1\nx @ W₁', 'category': 'meta'},
            {'id': 'scatter_to_edge1', 'label': 'ScatterToEdge\nsrc & dst', 'category': 'custom'},
            {'id': 'edge_forward1', 'label': 'EdgeForward\nconcat + LeakyReLU', 'category': 'meta'},
            {'id': 'edge_softmax1', 'label': 'EdgeSoftmax\nsoftmax per node', 'category': 'custom'},
            {'id': 'scatter_to_vertex1', 'label': 'ScatterToVertex\nweighted sum', 'category': 'custom'},
            {'id': 'dropout1', 'label': 'Dropout\nnn.dropout', 'category': 'meta'},
            {'id': 'linear2', 'label': 'Linear2\nx @ W₂', 'category': 'meta'},
            {'id': 'scatter_to_edge2', 'label': 'ScatterToEdge\nsrc & dst', 'category': 'custom'},
            {'id': 'edge_forward2', 'label': 'EdgeForward\nconcat + LeakyReLU', 'category': 'meta'},
            {'id': 'edge_softmax2', 'label': 'EdgeSoftmax\nsoftmax per node', 'category': 'custom'},
            {'id': 'scatter_to_vertex2', 'label': 'ScatterToVertex\nweighted sum', 'category': 'custom'},
            {'id': 'output', 'label': 'Output\nVar[N, C]', 'category': 'io'},
        ],
        'edges': [
            ['input', 'linear1'], ['linear1', 'scatter_to_edge1'],
            ['scatter_to_edge1', 'edge_forward1'], ['edge_forward1', 'edge_softmax1'],
            ['edge_softmax1', 'scatter_to_vertex1'], ['scatter_to_vertex1', 'dropout1'],
            ['dropout1', 'linear2'], ['linear2', 'scatter_to_edge2'],
            ['scatter_to_edge2', 'edge_forward2'], ['edge_forward2', 'edge_softmax2'],
            ['edge_softmax2', 'scatter_to_vertex2'], ['scatter_to_vertex2', 'output'],
        ],
        'fusion_groups': [
            {
                'name': '跨边界融合',
                'ops': ['x@W₁+b₁', 'ScatterToEdge1'],
                'type': 'cross_boundary',
            },
            {
                'name': '跨边界融合',
                'ops': ['EdgeForward', 'EdgeSoftmax'],
                'type': 'cross_boundary',
            },
            {
                'name': '跨边界融合',
                'ops': ['EdgeSoftmax', 'ScatterToVertex'],
                'type': 'cross_boundary',
            },
            {
                'name': '跨边界融合',
                'ops': ['x@W₂+b₂', 'ScatterToEdge2'],
                'type': 'cross_boundary',
            },
        ],
    },
    'GraphSAGE': {
        'nodes': [
            {'id': 'input', 'label': 'Input\nx: Var[N, F]', 'category': 'io'},
            {'id': 'scatter1', 'label': 'ScatterToEdge1\nscatter_to_edge', 'category': 'custom'},
            {'id': 'aggregate1', 'label': 'Aggregate1\nAggregateWithWeight', 'category': 'custom'},
            {'id': 'linear1', 'label': 'Linear1\nnn.Linear(F, H)', 'category': 'meta'},
            {'id': 'relu', 'label': 'ReLU\nnn.relu', 'category': 'meta'},
            {'id': 'dropout1', 'label': 'Dropout\nnn.dropout', 'category': 'meta'},
            {'id': 'scatter2', 'label': 'ScatterToEdge2\nscatter_to_edge', 'category': 'custom'},
            {'id': 'aggregate2', 'label': 'Aggregate2\nAggregateWithWeight', 'category': 'custom'},
            {'id': 'linear2', 'label': 'Linear2\nnn.Linear(H, C)', 'category': 'meta'},
            {'id': 'output', 'label': 'Output\nVar[N, C]', 'category': 'io'},
        ],
        'edges': [
            ['input', 'scatter1'], ['scatter1', 'aggregate1'], ['aggregate1', 'linear1'],
            ['linear1', 'relu'], ['relu', 'dropout1'],
            ['dropout1', 'scatter2'], ['scatter2', 'aggregate2'], ['aggregate2', 'linear2'],
            ['linear2', 'output'],
        ],
        'fusion_groups': [
            {
                'name': '跨边界融合',
                'ops': ['Aggregate1', 'x@W₁+b₁'],
                'type': 'cross_boundary',
            },
            {
                'name': '元算子融合',
                'ops': ['x@W₁+b₁', 'ReLU'],
                'type': 'meta_fusion',
            },
            {
                'name': '跨边界融合',
                'ops': ['Aggregate2', 'x@W₂+b₂'],
                'type': 'cross_boundary',
            },
        ],
    },
    'ChebNet2': {
        'nodes': [
            {'id': 'input', 'label': 'Input\nx: Var[N, F]', 'category': 'io'},
            {'id': 'linear1', 'label': 'Linear1\nnn.Linear(F, H)', 'category': 'meta'},
            {'id': 'relu1', 'label': 'ReLU\nnn.relu', 'category': 'meta'},
            {'id': 'linear2', 'label': 'Linear2\nnn.Linear(H, C)', 'category': 'meta'},
            {'id': 'scatter_t0', 'label': 'ScatterToEdge(T₀)\nscatter_to_edge', 'category': 'custom'},
            {'id': 'aggregate_t0', 'label': 'Aggregate(T₀)\nAggregateWithWeight', 'category': 'custom'},
            {'id': 'scatter_t1', 'label': 'ScatterToEdge(T₁)\nscatter_to_edge', 'category': 'custom'},
            {'id': 'aggregate_t1', 'label': 'Aggregate(T₁)\nAggregateWithWeight', 'category': 'custom'},
            {'id': 'cheb_combine', 'label': 'ChebCombine\nΣ αₖTₖ(x)', 'category': 'meta'},
            {'id': 'output', 'label': 'Output\nVar[N, C]', 'category': 'io'},
        ],
        'edges': [
            ['input', 'linear1'], ['linear1', 'relu1'], ['relu1', 'linear2'],
            ['linear2', 'scatter_t0'], ['linear2', 'scatter_t1'],
            ['scatter_t0', 'aggregate_t0'], ['scatter_t1', 'aggregate_t1'],
            ['aggregate_t0', 'cheb_combine'], ['aggregate_t1', 'cheb_combine'],
            ['cheb_combine', 'output'],
        ],
        'fusion_groups': [
            {
                'name': '元算子融合',
                'ops': ['x@W₁+b₁', 'ReLU'],
                'type': 'meta_fusion',
            },
            {
                'name': '跨边界融合',
                'ops': ['Aggregate(T₁)', 'ChebCombine'],
                'type': 'cross_boundary',
            },
        ],
    },
    'SGC': {
        'nodes': [
            {'id': 'input', 'label': 'Input\nx: Var[N, F]', 'category': 'io'},
            {'id': 'scatter_k1_1', 'label': 'ScatterToEdge(K=1)\nscatter_to_edge', 'category': 'custom'},
            {'id': 'aggregate_k1_1', 'label': 'Aggregate(K=1)\nAggregateWithWeight', 'category': 'custom'},
            {'id': 'scatter_k2_1', 'label': 'ScatterToEdge(K=2)\nscatter_to_edge', 'category': 'custom'},
            {'id': 'aggregate_k2_1', 'label': 'Aggregate(K=2)\nAggregateWithWeight', 'category': 'custom'},
            {'id': 'linear1', 'label': 'Linear1\nnn.Linear(F, C)', 'category': 'meta'},
            {'id': 'output', 'label': 'Output\nVar[N, C]', 'category': 'io'},
        ],
        'edges': [
            ['input', 'scatter_k1_1'], ['scatter_k1_1', 'aggregate_k1_1'],
            ['aggregate_k1_1', 'scatter_k2_1'], ['scatter_k2_1', 'aggregate_k2_1'],
            ['aggregate_k2_1', 'linear1'], ['linear1', 'output'],
        ],
        'fusion_groups': [
            {
                'name': '跨边界融合',
                'ops': ['Aggregate(K=1)', 'Aggregate(K=2)'],
                'type': 'cross_boundary',
            },
            {
                'name': '跨边界融合',
                'ops': ['Aggregate(K=2)', 'x@W+b'],
                'type': 'cross_boundary',
            },
        ],
    },
    'APPNP': {
        'nodes': [
            {'id': 'input', 'label': 'Input\nx: Var[N, F]', 'category': 'io'},
            {'id': 'linear1', 'label': 'Linear1\nnn.Linear(F, H)', 'category': 'meta'},
            {'id': 'relu1', 'label': 'ReLU\nnn.relu', 'category': 'meta'},
            {'id': 'dropout1', 'label': 'Dropout\nnn.dropout', 'category': 'meta'},
            {'id': 'linear2', 'label': 'Linear2\nnn.Linear(H, C)', 'category': 'meta'},
            {'id': 'ppr_init', 'label': 'PPRInit\nh = Linear2(x)', 'category': 'meta'},
            {'id': 'scatter_k1', 'label': 'ScatterToEdge(K=1)\nscatter_to_edge', 'category': 'custom'},
            {'id': 'aggregate_k1', 'label': 'Aggregate(K=1)\nAggregateWithWeight', 'category': 'custom'},
            {'id': 'alpha_blend_k1', 'label': 'AlphaBlend(K=1)\n(1-α)·agg + α·h', 'category': 'meta'},
            {'id': 'scatter_k2', 'label': 'ScatterToEdge(K=2)\nscatter_to_edge', 'category': 'custom'},
            {'id': 'aggregate_k2', 'label': 'Aggregate(K=2)\nAggregateWithWeight', 'category': 'custom'},
            {'id': 'alpha_blend_k2', 'label': 'AlphaBlend(K=2)\n(1-α)·agg + α·h', 'category': 'meta'},
            {'id': 'logsoftmax', 'label': 'LogSoftmax\nnn.logsoftmax', 'category': 'meta'},
            {'id': 'output', 'label': 'Output\nVar[N, C]', 'category': 'io'},
        ],
        'edges': [
            ['input', 'linear1'], ['linear1', 'relu1'], ['relu1', 'dropout1'],
            ['dropout1', 'linear2'], ['linear2', 'ppr_init'],
            ['ppr_init', 'scatter_k1'], ['scatter_k1', 'aggregate_k1'],
            ['aggregate_k1', 'alpha_blend_k1'],
            ['alpha_blend_k1', 'scatter_k2'], ['scatter_k2', 'aggregate_k2'],
            ['aggregate_k2', 'alpha_blend_k2'],
            ['alpha_blend_k2', 'logsoftmax'], ['logsoftmax', 'output'],
        ],
        'fusion_groups': [
            {
                'name': '元算子融合',
                'ops': ['x@W₁+b₁', 'ReLU'],
                'type': 'meta_fusion',
            },
            {
                'name': '跨边界融合',
                'ops': ['Aggregate(K=1)', 'AlphaBlend(K=1)'],
                'type': 'cross_boundary',
            },
            {
                'name': '跨边界融合',
                'ops': ['Aggregate(K=2)', 'AlphaBlend(K=2)', 'LogSoftmax'],
                'type': 'cross_boundary',
            },
        ],
    },
    'GPRGNN': {
        'nodes': [
            {'id': 'input', 'label': 'Input\nx: Var[N, F]', 'category': 'io'},
            {'id': 'linear1', 'label': 'Linear1\nnn.Linear(F, H)', 'category': 'meta'},
            {'id': 'relu1', 'label': 'ReLU\nnn.relu', 'category': 'meta'},
            {'id': 'dropout1', 'label': 'Dropout\nnn.dropout', 'category': 'meta'},
            {'id': 'linear2', 'label': 'Linear2\nnn.Linear(H, C)', 'category': 'meta'},
            {'id': 'gpr_init', 'label': 'GPRInit\nout = α₀·Linear2(x)', 'category': 'meta'},
            {'id': 'scatter_k1', 'label': 'ScatterToEdge(K=1)\nscatter_to_edge', 'category': 'custom'},
            {'id': 'aggregate_k1', 'label': 'Aggregate(K=1)\nAggregateWithWeight', 'category': 'custom'},
            {'id': 'gpr_weighted_k1', 'label': 'GPRWeighted(K=1)\nout += α₁·agg', 'category': 'meta'},
            {'id': 'scatter_k2', 'label': 'ScatterToEdge(K=2)\nscatter_to_edge', 'category': 'custom'},
            {'id': 'aggregate_k2', 'label': 'Aggregate(K=2)\nAggregateWithWeight', 'category': 'custom'},
            {'id': 'gpr_weighted_k2', 'label': 'GPRWeighted(K=2)\nout += α₂·agg', 'category': 'meta'},
            {'id': 'logsoftmax', 'label': 'LogSoftmax\nnn.logsoftmax', 'category': 'meta'},
            {'id': 'output', 'label': 'Output\nVar[N, C]', 'category': 'io'},
        ],
        'edges': [
            ['input', 'linear1'], ['linear1', 'relu1'], ['relu1', 'dropout1'],
            ['dropout1', 'linear2'], ['linear2', 'gpr_init'],
            ['gpr_init', 'scatter_k1'], ['scatter_k1', 'aggregate_k1'],
            ['aggregate_k1', 'gpr_weighted_k1'],
            ['gpr_weighted_k1', 'scatter_k2'], ['scatter_k2', 'aggregate_k2'],
            ['aggregate_k2', 'gpr_weighted_k2'],
            ['gpr_weighted_k2', 'logsoftmax'], ['logsoftmax', 'output'],
        ],
        'fusion_groups': [
            {
                'name': '元算子融合',
                'ops': ['x@W₁+b₁', 'ReLU'],
                'type': 'meta_fusion',
            },
            {
                'name': '跨边界融合',
                'ops': ['Aggregate(K=1)', 'GPRWeighted(K=1)'],
                'type': 'cross_boundary',
            },
            {
                'name': '跨边界融合',
                'ops': ['Aggregate(K=2)', 'GPRWeighted(K=2)', 'LogSoftmax'],
                'type': 'cross_boundary',
            },
        ],
    },
    'BernNet': {
        'nodes': [
            {'id': 'input', 'label': 'Input\nx: Var[N, F]', 'category': 'io'},
            {'id': 'linear1', 'label': 'Linear1\nnn.Linear(F, H)', 'category': 'meta'},
            {'id': 'relu1', 'label': 'ReLU\nnn.relu', 'category': 'meta'},
            {'id': 'dropout1', 'label': 'Dropout\nnn.dropout', 'category': 'meta'},
            {'id': 'linear2', 'label': 'Linear2\nnn.Linear(H, C)', 'category': 'meta'},
            {'id': 'bern_init', 'label': 'BernInit\nout = α₀·Linear2(x)', 'category': 'meta'},
            {'id': 'scatter_lap', 'label': 'ScatterToEdge(L)\nscatter_to_edge', 'category': 'custom'},
            {'id': 'aggregate_lap', 'label': 'Aggregate(L)\nAggregateWithWeight', 'category': 'custom'},
            {'id': 'scatter_shift', 'label': 'ScatterToEdge(2I-L)\nscatter_to_edge', 'category': 'custom'},
            {'id': 'aggregate_shift', 'label': 'Aggregate(2I-L)\nAggregateWithWeight', 'category': 'custom'},
            {'id': 'bern_combine', 'label': 'BernCombine\nΣ C(K,i)/2^K · αᵢ · Tᵢ', 'category': 'meta'},
            {'id': 'logsoftmax', 'label': 'LogSoftmax\nnn.logsoftmax', 'category': 'meta'},
            {'id': 'output', 'label': 'Output\nVar[N, C]', 'category': 'io'},
        ],
        'edges': [
            ['input', 'linear1'], ['linear1', 'relu1'], ['relu1', 'dropout1'],
            ['dropout1', 'linear2'], ['linear2', 'bern_init'],
            ['bern_init', 'scatter_lap'], ['bern_init', 'scatter_shift'],
            ['scatter_lap', 'aggregate_lap'], ['scatter_shift', 'aggregate_shift'],
            ['aggregate_lap', 'bern_combine'], ['aggregate_shift', 'bern_combine'],
            ['bern_combine', 'logsoftmax'], ['logsoftmax', 'output'],
        ],
        'fusion_groups': [
            {
                'name': '元算子融合',
                'ops': ['x@W₁+b₁', 'ReLU'],
                'type': 'meta_fusion',
            },
            {
                'name': '跨边界融合',
                'ops': ['Aggregate(L)', 'Aggregate(2I-L)', 'BernCombine'],
                'type': 'cross_boundary',
            },
            {
                'name': '跨边界融合',
                'ops': ['BernCombine', 'LogSoftmax'],
                'type': 'cross_boundary',
            },
        ],
    },
}

def init_status_file():
    if not os.path.exists(STATUS_FILE):
        status = {
            'running': False,
            'epoch': 0,
            'total_epochs': 0,
            'loss': 0.0,
            'acc': 0.0,
            'error': None,
            'history': {'loss': [], 'acc': []},
            'finished': False
        }
        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f)

init_status_file()

def read_training_status():
    try:
        with open(STATUS_FILE, 'r') as f:
            status = json.load(f)
            default_status = {
                'running': False,
                'epoch': 0,
                'total_epochs': 0,
                'loss': 0.0,
                'acc': 0.0,
                'error': None,
                'history': {'loss': [], 'acc': []},
                'finished': False
            }
            for key, value in default_status.items():
                if key not in status:
                    status[key] = value
            if 'history' not in status:
                status['history'] = {'loss': [], 'acc': []}
            if 'loss' not in status['history']:
                status['history']['loss'] = []
            if 'acc' not in status['history']:
                status['history']['acc'] = []
            return status
    except Exception as e:
        return {
            'running': False,
            'epoch': 0,
            'total_epochs': 0,
            'loss': 0.0,
            'acc': 0.0,
            'error': None,
            'history': {'loss': [], 'acc': []},
            'finished': False
        }

def write_training_status(status):
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    assets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')
    return send_from_directory(assets_path, filename)

@app.route('/api/status')
def status():
    return jsonify(read_training_status())

@app.route('/api/models')
def get_models():
    return jsonify(list(MODEL_DATASETS.keys()))

@app.route('/api/datasets/<model_name>')
def get_datasets(model_name):
    datasets = MODEL_DATASETS.get(model_name, [])
    return jsonify(datasets)

@app.route('/api/params/<model_name>')
def get_params(model_name):
    params = MODEL_PARAMS.get(model_name, [])
    return jsonify(params)

@app.route('/api/graph-info/<model_name>')
def get_graph_info(model_name):
    info = MODEL_GRAPH_INFO.get(model_name)
    if info is None:
        return jsonify({'error': f'No graph info for model {model_name}'}), 404
    return jsonify(info)

@app.route('/api/train', methods=['POST'])
def train():
    data = request.get_json()
    model_name = data.get('model_name', 'GCN')
    dataset_name = data.get('dataset_name', 'Cora')
    epochs = data.get('epochs', 200)
    params = data.get('params', {})

    training_status = {
        'running': True,
        'epoch': 0,
        'total_epochs': epochs,
        'loss': 0.0,
        'acc': 0.0,
        'error': None,
        'history': {'loss': [], 'acc': []},
        'finished': False
    }
    write_training_status(training_status)

    def start_training():
        import subprocess
        import sys

        worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_worker.py')
        cmd = [
            sys.executable,
            worker_path,
            '--model', model_name,
            '--dataset', dataset_name,
            '--epochs', str(epochs),
        ]
        for k, v in params.items():
            cmd.extend([f'--{k}', str(v)])

        def output_reader(pipe):
            while True:
                line = pipe.readline()
                if not line:
                    break
                print(f"[TRAIN] {line.rstrip()}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        reader_thread = threading.Thread(target=output_reader, args=(process.stdout,))
        reader_thread.start()

        process.wait()
        reader_thread.join()
        if process.returncode != 0:
            final_status = read_training_status()
            final_status['error'] = f"Training failed with exit code {process.returncode}"
            write_training_status(final_status)

    thread = threading.Thread(target=start_training)
    thread.start()

    return jsonify({'status': 'Training started'})

@app.route('/api/stop-training', methods=['POST'])
def stop_training():
    status = read_training_status()
    status['error'] = 'Training stopped by user'
    status['running'] = False
    write_training_status(status)
    return jsonify({'success': True, 'message': 'Training stopped'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
