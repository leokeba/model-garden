#!/usr/bin/env python3
"""Deep memory profiling to find what's actually leaking."""

import tracemalloc
import gc
import sys
from datasets import load_dataset
from model_garden.vision_training import VisionLanguageTrainer


def take_snapshot(label):
    """Take a memory snapshot and return statistics."""
    gc.collect()
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"Top 10 lines allocating memory:")
    for stat in top_stats[:10]:
        print(f"  {stat}")
    
    return snapshot


# Start tracing
tracemalloc.start()
print("Memory tracing started")

# Load dataset
print("\n1. Loading dataset...")
snapshot1 = take_snapshot("After loading dataset")

ds = load_dataset('Barth371/cmr-all', split='train')
snapshot2 = take_snapshot("After dataset loaded")

# Create trainer
print("\n2. Creating trainer...")
trainer = VisionLanguageTrainer(
    base_model="Qwen/Qwen2.5-VL-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True
)
snapshot3 = take_snapshot("After trainer created")

# Format dataset
print("\n3. Formatting dataset...")
formatted = trainer.format_dataset(ds)
snapshot4 = take_snapshot("After formatting dataset")

# Now the critical test - simulate collation like in training
print("\n4. Simulating collation (this is where the leak likely happens)...")
from unsloth.trainer import UnslothVisionDataCollator

# Load model and processor
print("Loading model...")
trainer.load_model()
snapshot5 = take_snapshot("After model loaded")

# Create collator
collator = UnslothVisionDataCollator(trainer.model, trainer.processor)
snapshot6 = take_snapshot("After collator created")

# Simulate training - collate batches
print("\nCollating batches (this should reveal the leak)...")
batch_size = 2

for i in range(0, min(20, len(formatted)), batch_size):
    batch = formatted[i:i+batch_size]
    
    # This is what happens in training - collator processes the batch
    try:
        collated = collator(batch)
        
        # Delete immediately
        del collated
        
        if (i // batch_size + 1) % 5 == 0:
            gc.collect()
            snapshot = take_snapshot(f"After {i//batch_size + 1} batches")
    except Exception as e:
        print(f"Error collating batch {i}: {e}")
        break

print("\n" + "="*60)
print("Profiling complete")
print("="*60)

# Get final snapshot
final_snapshot = take_snapshot("Final state")

# Compare snapshots
print("\n" + "="*60)
print("Memory growth from start to end:")
print("="*60)

top_stats = final_snapshot.compare_to(snapshot1, 'lineno')
for stat in top_stats[:10]:
    print(f"  {stat}")
