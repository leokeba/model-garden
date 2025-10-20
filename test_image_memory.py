#!/usr/bin/env python3
"""Test to understand where the image memory leak is coming from."""

import gc
import psutil
import os
from PIL import Image
import torch


def get_memory_mb():
    """Get current process memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_pil_image_accumulation():
    """Test if PIL images accumulate when stored in a list."""
    print("Test 1: PIL Image accumulation")
    print(f"Baseline memory: {get_memory_mb():.1f} MB")
    
    # Create a list of PIL images
    images = []
    for i in range(100):
        # Create a 512x512 RGB image (similar to vision model inputs)
        img = Image.new("RGB", (512, 512))
        images.append(img)
    
    print(f"After creating 100 images: {get_memory_mb():.1f} MB")
    
    # Simulate accessing them multiple times (like a dataloader would)
    for epoch in range(3):
        for img in images:
            # Just access the image (simulates dataloader iteration)
            _ = img.size
        print(f"After epoch {epoch + 1}: {get_memory_mb():.1f} MB")
    
    # Clean up
    images.clear()
    gc.collect()
    print(f"After cleanup: {get_memory_mb():.1f} MB\n")


def test_image_tensor_conversion():
    """Test if converting PIL to tensor accumulates memory."""
    print("Test 2: PIL to Tensor conversion")
    print(f"Baseline memory: {get_memory_mb():.1f} MB")
    
    # Create a PIL image
    img = Image.new("RGB", (512, 512))
    print(f"After creating 1 image: {get_memory_mb():.1f} MB")
    
    # Simulate what a processor/collator might do
    from torchvision import transforms
    to_tensor = transforms.ToTensor()
    
    tensors = []
    for i in range(100):
        # Convert to tensor (this is what happens in collation)
        tensor = to_tensor(img)
        tensors.append(tensor)
        
        if (i + 1) % 20 == 0:
            print(f"After {i + 1} conversions: {get_memory_mb():.1f} MB")
    
    print(f"After 100 conversions: {get_memory_mb():.1f} MB")
    
    # Clean up
    tensors.clear()
    del tensor
    gc.collect()
    print(f"After cleanup: {get_memory_mb():.1f} MB\n")


def test_image_reloading():
    """Test if reloading the same image file causes accumulation."""
    print("Test 3: Image file reloading")
    
    # Create a temporary image file
    temp_img = Image.new("RGB", (512, 512))
    temp_img.save("/tmp/test_image.jpg")
    
    print(f"Baseline memory: {get_memory_mb():.1f} MB")
    
    images = []
    for i in range(100):
        # Load image from file (simulates loading from dataset)
        img = Image.open("/tmp/test_image.jpg")
        img.load()  # Force load pixels
        img = img.convert("RGB")
        images.append(img)
        
        if (i + 1) % 20 == 0:
            print(f"After loading {i + 1} times: {get_memory_mb():.1f} MB")
    
    # Clean up
    images.clear()
    gc.collect()
    os.remove("/tmp/test_image.jpg")
    print(f"After cleanup: {get_memory_mb():.1f} MB\n")


def test_dataloader_simulation():
    """Simulate what happens when a PyTorch DataLoader iterates over images."""
    print("Test 4: DataLoader simulation with PIL images")
    
    # Create dataset with PIL images
    dataset = []
    for i in range(50):
        img = Image.new("RGB", (512, 512))
        dataset.append({"image": img, "text": f"Example {i}"})
    
    print(f"Dataset created: {get_memory_mb():.1f} MB")
    
    # Simulate multiple epochs
    for epoch in range(5):
        for batch_idx in range(0, len(dataset), 2):  # batch_size=2
            batch = dataset[batch_idx:batch_idx+2]
            
            # Simulate collation - extract images
            batch_images = [item["image"] for item in batch]
            
            # This is the key: are we creating new references?
            # In real training, these would be converted to tensors
            _ = batch_images
        
        print(f"After epoch {epoch + 1}: {get_memory_mb():.1f} MB")
        gc.collect()
    
    # Clean up
    dataset.clear()
    gc.collect()
    print(f"After cleanup: {get_memory_mb():.1f} MB\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing PIL Image Memory Behavior")
    print("=" * 60)
    print()
    
    test_pil_image_accumulation()
    test_image_tensor_conversion()
    test_image_reloading()
    test_dataloader_simulation()
    
    print("=" * 60)
    print("Tests complete!")
    print("=" * 60)
