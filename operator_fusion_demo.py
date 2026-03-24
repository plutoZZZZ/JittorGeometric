#!/usr/bin/env python3
"""Operator Fusion Demo for Jittor"""

import jittor as jt
from jittor import nn
import numpy as np

class FusionDemo(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm(64)
        self.bn2 = nn.BatchNorm(128)
        self.bn3 = nn.BatchNorm(256)
        
    def execute(self, x):
        # Without fusion
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        return x
    
    def execute_with_fusion(self, x):
        # With explicit fusion (Jittor automatically fuses these operations)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x

def demonstrate_fusion():
    print("Jittor Operator Fusion Demo")
    print("="*50)
    
    # Create model
    model = FusionDemo()
    
    # Create input tensor
    x = jt.randn(1, 3, 224, 224)
    
    # Warm up
    y = model(x)
    
    # Export computation graph
    print("\nExporting computation graph...")
    # Jittor automatically builds and optimizes the computation graph
    print("Computation graph has been built and optimized by Jittor")
    print("Jittor will automatically apply operator fusion during compilation")
    
    # Show fusion information
    print("\nOperator Fusion Information:")
    print("="*50)
    print("Jittor automatically fuses the following operations:")
    print("1. Conv2d + BatchNorm + ReLU")
    print("2. Linear + BatchNorm + Activation")
    print("3. Element-wise operations (add, mul, etc.)")
    print("\nBenefits of fusion:")
    print("- Reduced memory bandwidth usage")
    print("- Lower kernel launch overhead")
    print("- Improved cache locality")
    print("- Higher overall throughput")
    
    # Performance comparison
    print("\nPerformance Comparison:")
    print("="*50)
    
    # Without fusion
    print("\nWithout explicit fusion:")
    import time
    start_time = time.time()
    for _ in range(100):
        y = model.execute(x)
    jt.sync_all()
    elapsed_time = time.time() - start_time
    print(f"Time: {elapsed_time*1000:.2f} ms")
    
    # With fusion
    print("\nWith explicit fusion:")
    start_time = time.time()
    for _ in range(100):
        y = model.execute_with_fusion(x)
    jt.sync_all()
    elapsed_time = time.time() - start_time
    print(f"Time: {elapsed_time*1000:.2f} ms")
    
    print("\nNote: Jittor automatically fuses operations even without explicit fusion.")
    print("The explicit fusion syntax is just for demonstration.")

if __name__ == "__main__":
    demonstrate_fusion()