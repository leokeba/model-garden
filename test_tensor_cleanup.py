#!/usr/bin/env python3
"""Test to see if we can fix the tensor memory leak."""

import gc
import psutil
import os
from PIL import Image
import torch
from torchvision import transforms


def get_memory_mb():
    """Get current process memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_tensor_leak_with_cleanup():
    """Test if explicit cleanup helps with tensor accumulation."""
    print("Test: Tensor conversion WITH aggressive cleanup")
    print(f"Baseline memory: {get_memory_mb():.1f} MB")
    
    img = Image.new("RGB", (512, 512))
    to_tensor = transforms.ToTensor()
    
    for i in range(100):
        # Convert to tensor
        tensor = to_tensor(img)
        
        # Simulate using it (like in training)
        _ = tensor.mean()
        
        # Explicitly delete
        del tensor
        
        # Aggressive GC every iteration
        if (i + 1) % 10 == 0:
            gc.collect()
            print(f"After {i + 1} conversions + cleanup: {get_memory_mb():.1f} MB")
    
    gc.collect()
    print(f"Final memory: {get_memory_mb():.1f} MB\n")


def test_detached_tensors():
    """Test if detaching tensors helps."""
    print("Test: Using detached tensors")
    print(f"Baseline memory: {get_memory_mb():.1f} MB")
    
    img = Image.new("RGB", (512, 512))
    to_tensor = transforms.ToTensor()
    
    for i in range(100):
        # Convert and immediately detach
        tensor = to_tensor(img).detach()
        _ = tensor.mean()
        del tensor
        
        if (i + 1) % 10 == 0:
            gc.collect()
            print(f"After {i + 1} detached conversions: {get_memory_mb():.1f} MB")
    
    gc.collect()
    print(f"Final memory: {get_memory_mb():.1f} MB\n")


def test_no_grad_context():
    """Test if using torch.no_grad() helps."""
    print("Test: Conversions within torch.no_grad()")
    print(f"Baseline memory: {get_memory_mb():.1f} MB")
    
    img = Image.new("RGB", (512, 512))
    to_tensor = transforms.ToTensor()
    
    with torch.no_grad():
        for i in range(100):
            tensor = to_tensor(img)
            _ = tensor.mean()
            del tensor
            
            if (i + 1) % 10 == 0:
                gc.collect()
                print(f"After {i + 1} no_grad conversions: {get_memory_mb():.1f} MB")
    
    gc.collect()
    print(f"Final memory: {get_memory_mb():.1f} MB\n")


def test_in_place_operations():
    """Test if the issue is related to how we handle the tensor."""
    print("Test: Store tensors in list vs immediate deletion")
    print(f"Baseline memory: {get_memory_mb():.1f} MB")
    
    img = Image.new("RGB", (512, 512))
    to_tensor = transforms.ToTensor()
    
    # Version 1: Store all tensors (BAD)
    tensors = []
    for i in range(50):
        tensor = to_tensor(img)
        tensors.append(tensor)
    print(f"Stored 50 tensors in list: {get_memory_mb():.1f} MB")
    
    tensors.clear()
    del tensors
    gc.collect()
    print(f"After clearing list: {get_memory_mb():.1f} MB")
    
    # Version 2: Immediate deletion (GOOD?)
    for i in range(50):
        tensor = to_tensor(img)
        del tensor
        if i == 49:
            gc.collect()
    print(f"Created 50 tensors with immediate deletion: {get_memory_mb():.1f} MB\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Tensor Memory Leak Fixes")
    print("=" * 60)
    print()
    
    test_tensor_leak_with_cleanup()
    test_detached_tensors()
    test_no_grad_context()
    test_in_place_operations()
    
    print("=" * 60)
    print("Tests complete!")
    print("=" * 60)
