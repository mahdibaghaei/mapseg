#!/usr/bin/env python3
'''
GPU Diagnostic Script for MAPSeg Multi-GPU Issues
Run this BEFORE training to verify GPU setup
'''

import torch
import torch.nn as nn
import os
import sys

print("="*70)
print("MAPSeg GPU DIAGNOSTIC")
print("="*70)

# Check CUDA
print(f"\n1. CUDA AVAILABILITY")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   CUDA version: {torch.version.cuda}")
print(f"   cuDNN version: {torch.backends.cudnn.version()}")

# Check GPU count
num_gpus = torch.cuda.device_count()
print(f"\n2. GPU COUNT: {num_gpus}")

if num_gpus == 0:
    print("   ❌ No GPUs detected!")
    sys.exit(1)

# Check each GPU
print(f"\n3. GPU DETAILS")
for i in range(num_gpus):
    print(f"   GPU {i}:")
    print(f"      Name: {torch.cuda.get_device_name(i)}")
    props = torch.cuda.get_device_properties(i)
    print(f"      Memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"      Compute Capability: {props.major}.{props.minor}")

    # Check current memory
    mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
    mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
    print(f"      Current Allocated: {mem_alloc:.2f} GB")
    print(f"      Current Reserved: {mem_reserved:.2f} GB")

# Test tensor allocation on both GPUs
print(f"\n4. TESTING TENSOR ALLOCATION")
try:
    for i in range(num_gpus):
        test_tensor = torch.randn(1000, 1000).cuda(i)
        print(f"   ✓ GPU {i}: Can allocate tensors")
        del test_tensor
    torch.cuda.empty_cache()
    print("   ✓ All GPUs can allocate tensors")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test DataParallel
print(f"\n5. TESTING DATAPARALLEL")
try:
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(1000, 1000)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel().cuda()
    print(f"   Model created on GPU 0")

    # Wrap with DataParallel
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"   Model wrapped with DataParallel")
        print(f"   Device IDs: {model.device_ids}")

    # Test forward pass with batch
    batch_size = 4 if num_gpus > 1 else 2
    x = torch.randn(batch_size, 1000).cuda()

    print(f"   Testing forward pass with batch_size={batch_size}")
    y = model(x)

    # Check GPU usage after forward
    print(f"   ✓ Forward pass successful")
    print(f"\n   GPU Memory After Forward Pass:")
    for i in range(num_gpus):
        mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
        print(f"      GPU {i}: {mem_alloc:.3f} GB")

    # Test backward pass
    loss = y.sum()
    loss.backward()
    print(f"   ✓ Backward pass successful")

    print(f"\n   GPU Memory After Backward Pass:")
    for i in range(num_gpus):
        mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
        print(f"      GPU {i}: {mem_alloc:.3f} GB")

    del model, x, y, loss
    torch.cuda.empty_cache()

    if num_gpus > 1:
        print(f"\n   ✓ DataParallel is working correctly!")

except Exception as e:
    print(f"   ❌ DataParallel test failed: {e}")
    import traceback
    traceback.print_exc()

# Check environment variables
print(f"\n6. ENVIRONMENT VARIABLES")
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")
pytorch_alloc = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')
print(f"   PYTORCH_CUDA_ALLOC_CONF: {pytorch_alloc}")

# Test with larger model (closer to actual use case)
print(f"\n7. TESTING WITH LARGER MODEL (Simulating MAPSeg)")
try:
    class LargerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
            self.conv4 = nn.Conv3d(64, 8, kernel_size=1)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            return self.conv4(x)

    model = LargerModel().cuda()

    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"   Model wrapped with DataParallel")

    # Test with 3D data (like medical images)
    batch_size = 2 if num_gpus > 1 else 1
    x = torch.randn(batch_size, 1, 64, 64, 64).cuda()

    print(f"   Testing 3D convolution with shape {x.shape}")
    y = model(x)

    print(f"\n   GPU Memory After 3D Forward:")
    for i in range(num_gpus):
        mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"      GPU {i}: {mem_alloc:.3f} GB allocated, {mem_reserved:.3f} GB reserved")

    loss = y.sum()
    loss.backward()

    print(f"\n   GPU Memory After 3D Backward:")
    for i in range(num_gpus):
        mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"      GPU {i}: {mem_alloc:.3f} GB allocated, {mem_reserved:.3f} GB reserved")

    del model, x, y, loss
    torch.cuda.empty_cache()

    print(f"\n   ✓ 3D model test passed!")

except Exception as e:
    print(f"   ❌ 3D model test failed: {e}")
    import traceback
    traceback.print_exc()

# Final recommendations
print(f"\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)

if num_gpus > 1:
    print(f"✓ {num_gpus} GPUs detected")
    print(f"\nRECOMMENDATIONS:")
    print(f"1. Use batch_size=1 per GPU (total={num_gpus}) to start")
    print(f"2. Reduce patch_size to [48, 48, 48] or [64, 64, 64]")
    print(f"3. Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    print(f"4. Monitor GPU usage with: watch -n 2 nvidia-smi")
elif num_gpus == 1:
    print(f"⚠ Only 1 GPU detected")
    print(f"\nRECOMMENDATIONS:")
    print(f"1. Use batch_size=1")
    print(f"2. Reduce patch_size to [48, 48, 48]")
    print(f"3. Consider gradient accumulation")
else:
    print(f"❌ No GPUs detected - cannot train")

print("="*70)
