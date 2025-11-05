"""Test to see if the collator's train_on_responses_only function is stateful or has bugs."""

import os
os.environ["HF_HOME"] = "/root/model-garden/storage/cache"
os.environ["TRANSFORMERS_CACHE"] = "/root/model-garden/storage/cache/hub"
os.environ["HF_DATASETS_CACHE"] = "/root/model-garden/storage/cache/datasets"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from PIL import Image
import torch

print("=" * 80)
print("DEEP DIVE: CHECKING IF train_on_responses_only FUNCTION IS BUGGY")
print("=" * 80)

# Load model
model, processor = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-2B-Instruct",
    load_in_4bit=True,
    max_seq_length=2048,
)

# Create sample
sample = {
    "messages": [
        {"role": "user", "content": [
            {"type": "image", "image": Image.new("RGB", (224, 224), color="red")},
            {"type": "text", "text": "What color?"}
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Red color."}
        ]}
    ]
}

# Create collator
collator = UnslothVisionDataCollator(
    model, 
    processor,
    train_on_responses_only=True,
    instruction_part="<|im_start|>user",
    response_part="<|im_start|>assistant",
    force_match=False
)

print("\n1. First, let's see what the collator produces BEFORE train_on_responses_only masking:")
print("   (We'll intercept the batch after parent collator but before masking)")

# Process batch
batch = collator([sample])

print(f"\n2. Batch after full collation (with train_on_responses_only masking):")
print(f"   input_ids shape: {batch['input_ids'].shape}")
print(f"   labels shape: {batch['labels'].shape}")

# Check the labels
labels = batch['labels'][0]
input_ids = batch['input_ids'][0]

# Count masked vs non-masked
total = len(labels)
non_masked_count = (labels != -100).sum().item()
masked_count = total - non_masked_count

print(f"\n3. Label masking statistics:")
print(f"   Total tokens: {total}")
print(f"   Masked tokens (==-100): {masked_count} ({100*masked_count/total:.1f}%)")
print(f"   Non-masked tokens (loss computed): {non_masked_count} ({100*non_masked_count/total:.1f}%)")

# Decode to see what's masked
print(f"\n4. Decoding the sequence to verify masking:")

# Find the assistant marker in input_ids
full_text = processor.decode(input_ids, skip_special_tokens=False)
print(f"   Full sequence: {repr(full_text[:200])}")

# Find non-masked region
non_masked_indices = (labels != -100).nonzero(as_tuple=True)[0]
if len(non_masked_indices) > 0:
    first_non_masked = non_masked_indices[0].item()
    last_non_masked = non_masked_indices[-1].item()
    
    non_masked_text = processor.decode(input_ids[first_non_masked:last_non_masked+1], skip_special_tokens=False)
    print(f"\n   Non-masked region (contributes to loss):")
    print(f"   {repr(non_masked_text)}")
    
    # Check if this is ONLY the assistant response
    if "<|im_start|>user" in non_masked_text or "What color" in non_masked_text:
        print("\n   ❌ BUG FOUND: User prompt is NOT masked!")
        print("   The train_on_responses_only masking is NOT working correctly!")
    elif "<|im_start|>assistant" in non_masked_text[:50]:
        print("\n   ⚠️  PARTIAL BUG: Assistant marker is included in loss")
        print("   This might be intentional, but could cause issues")
    else:
        print("\n   ✓ Looks correct: Only the response text is non-masked")

# Now let's check the actual train_on_responses_only FUNCTION that was created
print("\n5. Inspecting the train_on_responses_only function object:")
print(f"   Type: {type(collator.train_on_responses_only)}")
print(f"   Is callable: {callable(collator.train_on_responses_only)}")

if callable(collator.train_on_responses_only):
    # Try calling it directly to see what it does
    print("\n6. Testing the train_on_responses_only function directly:")
    
    # The function expects a batch dict with input_ids and labels
    test_batch = {
        "input_ids": batch["input_ids"],
        "labels": batch["input_ids"].clone()  # Start with all tokens unmasked
    }
    
    print(f"   Before function: labels[0] has {(test_batch['labels'][0] != -100).sum().item()} non-masked tokens")
    
    # Call the function
    result = collator.train_on_responses_only(test_batch)
    
    print(f"   After function: labels[0] has {(result['labels'][0] != -100).sum().item()} non-masked tokens")
    
    # Compare with what the collator produced
    collator_non_masked = non_masked_count
    function_non_masked = (result['labels'][0] != -100).sum().item()
    
    if collator_non_masked == function_non_masked:
        print(f"\n   ✓ Consistent: Both produce {non_masked_count} non-masked tokens")
    else:
        print(f"\n   ❌ INCONSISTENT:")
        print(f"      Collator produced: {collator_non_masked} non-masked tokens")
        print(f"      Function produced: {function_non_masked} non-masked tokens")
        print(f"      This is the bug!")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)

if non_masked_count < total * 0.15:  # Less than 15% non-masked
    print("✓ Masking appears to be working (>85% tokens masked)")
    print("  The issue is likely elsewhere in the training loop")
else:
    print("❌ Masking is NOT working properly (too few tokens masked)")
    print("  This explains why training loss is so low from the start!")

print("=" * 80)
