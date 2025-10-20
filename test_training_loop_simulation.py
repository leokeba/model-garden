"""Simulate the exact training loop to isolate the memory leak.

This test recreates the training loop step-by-step to find where memory accumulates.
"""

import gc
import os
import psutil
import torch
from pathlib import Path
from datasets import load_dataset

# Configure HF cache
HF_HOME = os.getenv('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))
os.environ['HF_HOME'] = HF_HOME
os.environ['TRANSFORMERS_CACHE'] = str(Path(HF_HOME) / 'hub')
os.environ['HF_DATASETS_CACHE'] = str(Path(HF_HOME) / 'datasets')

def get_memory_mb():
    """Get current process memory in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def format_mb(mb):
    """Format MB for display."""
    return f"{mb:.1f} MB"

print("=" * 80)
print("Training Loop Memory Leak Simulation")
print("=" * 80)

# Test 1: Load dataset and format it (like the trainer does)
print("\n1. Loading and formatting dataset...")
mem_start = get_memory_mb()
raw_dataset = load_dataset("Barth371/cmr-all", split="train")
mem_after_load = get_memory_mb()
print(f"   Memory after load: {format_mb(mem_after_load)} (delta: +{format_mb(mem_after_load - mem_start)})")

# Format dataset (this creates PIL images from base64)
print("   Formatting dataset (converting base64 to PIL images)...")
from model_garden.vision_training import VisionLanguageTrainer
trainer_instance = VisionLanguageTrainer(
    base_model="Qwen/Qwen2.5-VL-3B-Instruct",
    load_in_4bit=True
)
formatted_dataset = trainer_instance.format_dataset(raw_dataset)
mem_after_format = get_memory_mb()
print(f"   Memory after format: {format_mb(mem_after_format)} (delta: +{format_mb(mem_after_format - mem_after_load)})")

# Test 2: Load processor
print("\n2. Loading Qwen2VL processor...")
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
mem_after_processor = get_memory_mb()
print(f"   Memory after processor: {format_mb(mem_after_processor)} (delta: +{format_mb(mem_after_processor - mem_after_format)})")

# Test 3: Process images WITHOUT cleanup (like training loop)
print("\n3. Simulating training loop - Processing batches WITHOUT cleanup...")
print("   Processing 50 batches of 2 samples each (100 total samples)...")

mem_before_loop = get_memory_mb()
batch_size = 2
num_batches = 50

# Store processed batches to simulate what trainer might do
processed_batches = []

for i in range(num_batches):
    # Get batch from formatted dataset
    batch_samples = formatted_dataset[i * batch_size : (i + 1) * batch_size]
    
    # Process each sample (convert to tensors) - this is what UnslothVisionDataCollator does
    batch_data = []
    for sample in batch_samples:
        # Extract image and text from messages
        messages = sample["messages"]
        user_msg = messages[1]  # User message has image + text
        image = None
        text_parts = []
        for content in user_msg["content"]:
            if content["type"] == "image":
                image = content["image"]
            elif content["type"] == "text":
                text_parts.append(content["text"])
        text = " ".join(text_parts)
        
        # Process (this creates tensors)
        processed = processor(
            text=text,
            images=image,
            return_tensors="pt"
        )
        batch_data.append(processed)
    
    # Store the batch (simulating what trainer does)
    processed_batches.append(batch_data)
    
    if (i + 1) % 10 == 0:
        current_mem = get_memory_mb()
        print(f"   After batch {i+1}: {format_mb(current_mem)} (delta: +{format_mb(current_mem - mem_before_loop)})")

mem_after_loop_no_cleanup = get_memory_mb()
growth_no_cleanup = mem_after_loop_no_cleanup - mem_before_loop
print(f"\n   Final memory: {format_mb(mem_after_loop_no_cleanup)}")
print(f"   Total growth: +{format_mb(growth_no_cleanup)} for {num_batches} batches")
print(f"   Growth per batch: +{format_mb(growth_no_cleanup / num_batches)}")

# Test 4: Clear and run GC
print("\n4. Clearing processed batches and running GC...")
processed_batches.clear()
del processed_batches
gc.collect()
mem_after_cleanup = get_memory_mb()
freed = mem_after_loop_no_cleanup - mem_after_cleanup
print(f"   Memory after cleanup: {format_mb(mem_after_cleanup)}")
print(f"   Memory freed: {format_mb(freed)}")

# Test 5: Process batches WITH cleanup after each batch
print("\n5. Simulating training loop - Processing batches WITH cleanup...")
print("   Processing 50 batches of 2 samples each (100 total samples)...")

mem_before_loop2 = get_memory_mb()

for i in range(num_batches):
    # Get batch from formatted dataset
    batch_samples = formatted_dataset[i * batch_size : (i + 1) * batch_size]
    
    # Process each sample (convert to tensors)
    batch_data = []
    for sample in batch_samples:
        # Extract image and text from messages
        messages = sample["messages"]
        user_msg = messages[1]  # User message has image + text
        image = None
        text_parts = []
        for content in user_msg["content"]:
            if content["type"] == "image":
                image = content["image"]
            elif content["type"] == "text":
                text_parts.append(content["text"])
        text = " ".join(text_parts)
        
        # Process (this creates tensors)
        processed = processor(
            text=text,
            images=image,
            return_tensors="pt"
        )
        batch_data.append(processed)
    
    # CLEANUP: Delete batch data immediately after "processing"
    del batch_data
    gc.collect()
    
    if (i + 1) % 10 == 0:
        current_mem = get_memory_mb()
        print(f"   After batch {i+1}: {format_mb(current_mem)} (delta: +{format_mb(current_mem - mem_before_loop2)})")

mem_after_loop_with_cleanup = get_memory_mb()
growth_with_cleanup = mem_after_loop_with_cleanup - mem_before_loop2
print(f"\n   Final memory: {format_mb(mem_after_loop_with_cleanup)}")
print(f"   Total growth: +{format_mb(growth_with_cleanup)} for {num_batches} batches")
print(f"   Growth per batch: +{format_mb(growth_with_cleanup / num_batches)}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"WITHOUT cleanup: +{format_mb(growth_no_cleanup)} ({format_mb(growth_no_cleanup / num_batches)}/batch)")
print(f"WITH cleanup:    +{format_mb(growth_with_cleanup)} ({format_mb(growth_with_cleanup / num_batches)}/batch)")
print(f"Difference:      {format_mb(growth_no_cleanup - growth_with_cleanup)}")

if growth_with_cleanup > 100:
    print("\n⚠️  WARNING: Memory still growing even with cleanup!")
    print("   This suggests the leak is NOT from Python objects.")
    print("   Possible causes:")
    print("   - PyTorch internal memory pools")
    print("   - Processor caching in C++ layer")
    print("   - Image decoder memory not being released")
else:
    print("\n✓ Cleanup prevents the leak!")
    print("  The issue is that the trainer doesn't clean up batch data properly.")
