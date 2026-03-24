#!/usr/bin/env python3

import os
os.environ['JITTOR_CPU'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Import jittor after setting environment variables
import jittor as jt

# Test basic operations
print("Testing Jittor CPU mode...")

# Create a simple tensor
x = jt.randn(3, 3)
print(f"Tensor x: {x}")

# Perform basic operations
y = x + x
print(f"y = x + x: {y}")

print("Jittor CPU mode works!")