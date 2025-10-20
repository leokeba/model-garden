#!/usr/bin/env python3
"""Test if the Qwen2VL processor caches processed images."""

import gc
import psutil
import os
from PIL import Image
from transformers import Qwen2VLProcessor
import torch


def get_memory_mb():
    """Get current process memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


print("Loading Qwen2VL processor...")
processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
print(f"Processor loaded: {get_memory_mb():.1f} MB\n")

# Create a test image
img = Image.new("RGB", (512, 512))

# Test repeated processing
print("Processing the same image 100 times...")
baseline = get_memory_mb()
print(f"Baseline: {baseline:.1f} MB")

for i in range(100):
    # Simulate what happens in training
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Test"}
            ]
        }
    ]
    
    # Process (this is what the collator does)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(
        text=[text],
        images=[img],
        return_tensors="pt",
        padding=True
    )
    
    # Check if memory grows
    if (i + 1) % 10 == 0:
        current = get_memory_mb()
        growth = current - baseline
        print(f"After {i + 1} processings: {current:.1f} MB (+{growth:.1f} MB)")
        
        # Try cleanup
        del inputs
        del text
        gc.collect()

print(f"\nFinal memory: {get_memory_mb():.1f} MB")
print(f"Total growth: {get_memory_mb() - baseline:.1f} MB")

# Check if processor has any caches
print("\nChecking processor for caches...")
if hasattr(processor, '_cache'):
    print(f"Processor has _cache attribute: {len(processor._cache) if hasattr(processor._cache, '__len__') else 'unknown size'}")
if hasattr(processor, 'image_processor') and hasattr(processor.image_processor, '_cache'):
    print(f"Image processor has _cache attribute")
if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, '_cache'):
    print(f"Tokenizer has _cache attribute")

print("\nProcessor attributes:")
for attr in dir(processor):
    if 'cache' in attr.lower():
        print(f"  - {attr}")
