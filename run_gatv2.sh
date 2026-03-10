#!/bin/bash
cd /mnt/d/Code/qianshi0310/jittorgeometric/merge0310/JittorGeometric
source /home/liuyuan/miniconda3/etc/profile.d/conda.sh
conda activate jittor
python examples/gatv2_example.py --dataset cora
