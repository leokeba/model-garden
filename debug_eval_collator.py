"""Debug script to check if data collator processes train and eval datasets the same way."""

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
print("TESTING DATA COLLATOR ON TRAIN VS EVAL DATA")
print("=" * 80)

# Load model
model, processor = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-2B-Instruct",
    load_in_4bit=True,
    max_seq_length=2048,
)

# Create TWO DIFFERENT samples to simulate train vs eval split
train_sample = {
    "messages": [
        {"role": "user", "content": [
            {"type": "image", "image": Image.new("RGB", (224, 224), color="red")},
            {"type": "text", "text": "What color is this image?"}
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "The image is red."}
        ]}
    ]
}

eval_sample = {
    "messages": [
        {"role": "user", "content": [
            {"type": "image", "image": Image.new("RGB", (224, 224), color="blue")},
            {"type": "text", "text": "Describe this color please."}
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "This is a blue color."}
        ]}
    ]
}

# Create collator
print("\n1. Creating data collator with train_on_responses_only=True")
collator = UnslothVisionDataCollator(
    model, 
    processor,
    train_on_responses_only=True,
    instruction_part="<|im_start|>user",
    response_part="<|im_start|>assistant",
    force_match=False
)

# Process both samples
print("\n2. Processing TRAIN sample...")
train_batch = collator([train_sample])
train_batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in train_batch.items()}

print("\n3. Processing EVAL sample...")
eval_batch = collator([eval_sample])
eval_batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in eval_batch.items()}

# Analyze labels
def analyze_labels(batch, name):
    labels = batch['labels'][0]
    input_ids = batch['input_ids'][0]
    
    non_masked_count = (labels != -100).sum().item()
    total_count = len(labels)
    masked_pct = 100 * (1 - non_masked_count / total_count)
    
    print(f"\n{name} batch:")
    print(f"   Total tokens: {total_count}")
    print(f"   Tokens for loss: {non_masked_count} ({100 - masked_pct:.1f}%)")
    print(f"   Masked tokens: {total_count - non_masked_count} ({masked_pct:.1f}%)")
    
    # Decode non-masked part
    non_masked_indices = (labels != -100).nonzero(as_tuple=True)[0]
    if len(non_masked_indices) > 0:
        first_idx = non_masked_indices[0].item()
        last_idx = non_masked_indices[-1].item()
        loss_text = processor.decode(input_ids[first_idx:last_idx+1], skip_special_tokens=False)
        print(f"   Loss computed on: {repr(loss_text[:100])}")

analyze_labels(train_batch, "TRAIN")
analyze_labels(eval_batch, "EVAL")

# Now compute actual losses
print("\n4. Computing losses...")
model.eval()
with torch.no_grad():
    train_outputs = model(**train_batch)
    eval_outputs = model(**eval_batch)
    
    train_loss = train_outputs.loss.item()
    eval_loss = eval_outputs.loss.item()
    
    print(f"\n   Train sample loss: {train_loss:.4f}")
    print(f"   Eval sample loss:  {eval_loss:.4f}")
    
    if abs(train_loss - eval_loss) < 0.1:
        print("\n   ✓ Losses are similar - collator working consistently")
    else:
        print(f"\n   ⚠️  Loss difference: {abs(train_loss - eval_loss):.4f}")

print("\n" + "=" * 80)
print("CHECKING IF THE COLLATOR INSTANCE IS STATEFUL")
print("=" * 80)

# The key question: does calling the collator change its internal state?
print("\n5. Processing the same sample multiple times...")
losses = []
for i in range(3):
    batch = collator([train_sample])
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(**batch)
        loss = outputs.loss.item()
        losses.append(loss)
    
    non_masked = (batch['labels'][0] != -100).sum().item()
    print(f"   Iteration {i+1}: Loss={loss:.4f}, Non-masked tokens={non_masked}")

if all(abs(losses[0] - l) < 0.001 for l in losses):
    print("\n   ✓ Collator is consistent across calls")
else:
    print("\n   ✗ Collator produces different results!")
    print(f"   Losses: {losses}")

print("\n" + "=" * 80)
