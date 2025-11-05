"""Debug script to investigate training vs validation loss computation."""

import os
os.environ["HF_HOME"] = "/root/model-garden/storage/cache"
os.environ["TRANSFORMERS_CACHE"] = "/root/model-garden/storage/cache/hub"
os.environ["HF_DATASETS_CACHE"] = "/root/model-garden/storage/cache/datasets"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from datasets import Dataset
from PIL import Image
import torch

print("=" * 80)
print("INVESTIGATING TRAIN VS EVAL LOSS COMPUTATION")
print("=" * 80)

# Load a small model
model, processor = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-2B-Instruct",
    load_in_4bit=True,
    max_seq_length=2048,
)

# Create sample data
sample_data = [
    {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": Image.new("RGB", (224, 224), color="red")},
                {"type": "text", "text": "What color is this?"}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "This is red."}
            ]}
        ]
    }
] * 4  # 4 samples

# Create data collator WITH prompt masking enabled
print("\n1. Creating data collator with train_on_responses_only=True")
collator = UnslothVisionDataCollator(
    model, 
    processor,
    train_on_responses_only=True,
    instruction_part="<|im_start|>user",
    response_part="<|im_start|>assistant",
    force_match=False
)

# Process one batch
print("\n2. Processing batch with collator...")
batch = collator(sample_data)

# Move batch to GPU
print("\n3. Moving batch to GPU...")
batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

print(f"\n4. Batch keys: {batch.keys()}")
print(f"   input_ids shape: {batch['input_ids'].shape}")
print(f"   labels shape: {batch['labels'].shape}")

# Check labels - count non-masked tokens
labels = batch['labels']
print(f"\n5. Labels statistics:")
for i in range(len(labels)):
    non_masked = (labels[i] != -100).sum().item()
    total = len(labels[i])
    masked_pct = 100 * (1 - non_masked / total)
    print(f"   Sample {i}: {non_masked}/{total} tokens for loss ({masked_pct:.1f}% masked)")

# Now simulate what happens during forward pass
print("\n6. Computing loss with model.forward()...")
model.eval()  # Set to eval mode
with torch.no_grad():
    outputs = model(**batch)
    loss = outputs.loss
    print(f"   Loss: {loss.item():.4f}")

print("\n7. Now let's check if there's a difference in train vs eval mode...")
model.train()  # Set to train mode
with torch.no_grad():
    outputs_train = model(**batch)
    loss_train = outputs_train.loss
    print(f"   Loss (train mode): {loss_train.item():.4f}")

print("\n" + "=" * 80)
print("CONCLUSION:")
if abs(loss.item() - loss_train.item()) < 0.001:
    print("✓ Loss is the SAME in train and eval mode with the SAME batch")
    print("  This means the issue is NOT in model.forward() behavior")
else:
    print("✗ Loss DIFFERS between train and eval mode!")
    print(f"  Difference: {abs(loss.item() - loss_train.item()):.4f}")

print("\n8. Let's manually check the labels to see if prompts are masked...")
# Decode the first sample to see what's masked
input_ids_0 = batch['input_ids'][0]
labels_0 = batch['labels'][0]

# Find where labels != -100 (these are the tokens that contribute to loss)
non_masked_indices = (labels_0 != -100).nonzero(as_tuple=True)[0]
if len(non_masked_indices) > 0:
    first_non_masked = non_masked_indices[0].item()
    last_non_masked = non_masked_indices[-1].item()
    
    # Decode the full input
    full_text = processor.decode(input_ids_0, skip_special_tokens=False)
    
    # Decode just the part that contributes to loss
    loss_text = processor.decode(input_ids_0[first_non_masked:last_non_masked+1], skip_special_tokens=False)
    
    print(f"\n   Full input text:\n   {repr(full_text)}")
    print(f"\n   Text that contributes to loss (non-masked):\n   {repr(loss_text)}")
    
    # Check if the prompt is in the non-masked part
    if "What color" in loss_text:
        print("\n   ⚠️  WARNING: Prompt 'What color' found in non-masked tokens!")
        print("   This means the prompt IS contributing to loss")
    else:
        print("\n   ✓ Prompt appears to be masked correctly")

print("\n" + "=" * 80)
