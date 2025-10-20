#!/usr/bin/env python3
"""Test what happens when we actually iterate through the HF dataset during training."""

import gc
import psutil
import os
from datasets import load_dataset
from PIL import Image
from model_garden.vision_training import VisionLanguageTrainer


def get_memory_mb():
    """Get current process memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


print("Loading dataset from HuggingFace...")
baseline = get_memory_mb()
print(f"Baseline memory: {baseline:.1f} MB")

# Load the actual dataset
ds = load_dataset('Barth371/cmr-all', split='train')
print(f"Dataset loaded: {get_memory_mb():.1f} MB (growth: {get_memory_mb() - baseline:.1f} MB)")
print(f"Dataset size: {len(ds)} examples\n")

# Create a trainer instance to use its format_dataset method
print("Creating trainer...")
trainer = VisionLanguageTrainer(
    base_model="Qwen/Qwen2.5-VL-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True
)
# Don't load the model - we just need the formatting logic
after_trainer = get_memory_mb()
print(f"Trainer created (no model loaded): {after_trainer:.1f} MB (growth: {after_trainer - baseline:.1f} MB)\n")

# Now format the dataset (this loads PIL images)
print("Formatting dataset (loading all images into PIL objects)...")
formatted = trainer.format_dataset(ds)
after_format = get_memory_mb()
print(f"Dataset formatted: {after_format:.1f} MB (growth: {after_format - baseline:.1f} MB)")
print(f"Formatted dataset length: {len(formatted)}\n")

# Simulate what happens during training - iterate through the dataset multiple times
print("Simulating training epochs...")
for epoch in range(3):
    for i, example in enumerate(formatted):
        # Just access the example (simulates dataloader iteration)
        _ = example
        
        if (i + 1) % 100 == 0:
            current = get_memory_mb()
            print(f"  Epoch {epoch + 1}, step {i + 1}: {current:.1f} MB (growth: {current - baseline:.1f} MB)")
    
    # GC after each epoch
    gc.collect()
    after_epoch = get_memory_mb()
    print(f"After epoch {epoch + 1} + GC: {after_epoch:.1f} MB (growth: {after_epoch - baseline:.1f} MB)\n")

print(f"\n{'='*60}")
print(f"Total memory growth: {get_memory_mb() - baseline:.1f} MB")
print(f"Expected growth (663 images * ~5MB): ~3315 MB")
print(f"{'='*60}")
