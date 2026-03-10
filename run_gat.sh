#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jittor
cd /mnt/d/Code/qianshi0310/jittorgeometric/merge0310/JittorGeometric
python examples/gat_example.py --dataset cora 2>&1 | tail -30
