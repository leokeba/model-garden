"""Test if Unsloth model handles masked labels correctly in eval mode."""

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
print("TESTING IF MODEL HANDLES MASKED LABELS DIFFERENTLY IN TRAIN VS EVAL MODE")
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
            {"type": "text", "text": "Red."}
        ]}
    ]
}

# Create collator WITH prompt masking
collator = UnslothVisionDataCollator(
    model, 
    processor,
    train_on_responses_only=True,
    instruction_part="<|im_start|>user",
    response_part="<|im_start|>assistant",
    force_match=False
)

# Process batch
batch = collator([sample])
batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

print("\n1. Batch prepared with masked labels:")
labels = batch['labels'][0]
non_masked = (labels != -100).sum().item()
total = len(labels)
print(f"   Non-masked tokens: {non_masked}/{total} ({100*non_masked/total:.1f}%)")

# Test 1: Compute loss in EVAL mode (model.eval())
print("\n2. Computing loss in EVAL mode (model.eval())...")
model.eval()
with torch.no_grad():
    outputs_eval = model(**batch)
    loss_eval = outputs_eval.loss.item()
    print(f"   Loss: {loss_eval:.4f}")

# Test 2: Compute loss in TRAIN mode (model.train())
print("\n3. Computing loss in TRAIN mode (model.train())...")
model.train()
with torch.no_grad():  # Still no_grad to not track gradients
    outputs_train = model(**batch)
    loss_train = outputs_train.loss.item()
    print(f"   Loss: {loss_train:.4f}")

print("\n4. Comparison:")
print(f"   Eval loss:  {loss_eval:.4f}")
print(f"   Train loss: {loss_train:.4f}")
print(f"   Difference: {abs(loss_eval - loss_train):.4f}")

if abs(loss_eval - loss_train) < 0.001:
    print("\n   ✓ SAME - Model handles masked labels identically in both modes")
else:
    print("\n   ✗ DIFFERENT - Model behavior changes between train and eval!")

# Test 3: What if we DON'T mask the labels?
print("\n" + "=" * 80)
print("5. Testing WITHOUT label masking (compute loss on ALL tokens)...")
print("=" * 80)

# Create batch without masking
collator_no_mask = UnslothVisionDataCollator(
    model, 
    processor,
    train_on_responses_only=False,  # NO MASKING
)

batch_no_mask = collator_no_mask([sample])
batch_no_mask = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch_no_mask.items()}

labels_no_mask = batch_no_mask['labels'][0]
non_masked_no_mask = (labels_no_mask != -100).sum().item()
print(f"\n   Non-masked tokens: {non_masked_no_mask}/{len(labels_no_mask)} ({100*non_masked_no_mask/len(labels_no_mask):.1f}%)")

# Eval mode
model.eval()
with torch.no_grad():
    outputs_no_mask_eval = model(**batch_no_mask)
    loss_no_mask_eval = outputs_no_mask_eval.loss.item()
    print(f"   Loss (eval, no masking): {loss_no_mask_eval:.4f}")

# Train mode  
model.train()
with torch.no_grad():
    outputs_no_mask_train = model(**batch_no_mask)
    loss_no_mask_train = outputs_no_mask_train.loss.item()
    print(f"   Loss (train, no masking): {loss_no_mask_train:.4f}")

print(f"\n   Difference: {abs(loss_no_mask_eval - loss_no_mask_train):.4f}")

# Compare masked vs unmasked
print("\n" + "=" * 80)
print("6. Comparing MASKED vs UNMASKED losses:")
print("=" * 80)
print(f"\n   Masked (response only):   {loss_eval:.4f}")
print(f"   Unmasked (all tokens):    {loss_no_mask_eval:.4f}")
print(f"   Ratio (unmasked/masked):  {loss_no_mask_eval/loss_eval:.2f}x")

if loss_no_mask_eval > loss_eval * 1.5:
    print("\n   ✓ Unmasked loss is significantly higher - masking is working")
else:
    print("\n   ⚠️  Unmasked loss is similar to masked - masking might not be working!")

print("\n" + "=" * 80)
