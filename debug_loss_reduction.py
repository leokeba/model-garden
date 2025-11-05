"""Check if the model's internal loss computation uses the right reduction."""

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
import torch.nn.functional as F

print("=" * 80)
print("HYPOTHESIS: Model's internal loss uses wrong reduction/normalization")
print("=" * 80)

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

collator = UnslothVisionDataCollator(
    model, 
    processor,
    train_on_responses_only=True,
    instruction_part="<|im_start|>user",
    response_part="<|im_start|>assistant",
    force_match=False
)

batch = collator([sample])
batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

print("\n1. Batch statistics:")
labels = batch['labels']
input_ids = batch['input_ids']
seq_length = labels.size(1)
non_masked_tokens = (labels != -100).sum().item()
print(f"   Sequence length: {seq_length}")
print(f"   Non-masked tokens: {non_masked_tokens}")
print(f"   Percentage non-masked: {100*non_masked_tokens/seq_length:.1f}%")

print("\n2. Computing loss via model.forward():")
model.eval()
with torch.no_grad():
    outputs = model(**batch)
    model_loss = outputs.loss.item()
    logits = outputs.logits
    
print(f"   Model's loss: {model_loss:.4f}")

print("\n3. Manually computing loss different ways:")

# Shift for causal LM (predict next token)
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()

# Flatten
flat_logits = shift_logits.view(-1, shift_logits.size(-1))
flat_labels = shift_labels.view(-1)

# Method 1: reduction='mean' over ALL tokens (including -100)
loss_mean_all = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction='mean')
print(f"   Manual (reduction='mean', ignore -100): {loss_mean_all.item():.4f}")

# Method 2: reduction='sum' then divide by non-masked count
loss_sum = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction='sum')
valid_tokens = (flat_labels != -100).sum().item()
loss_sum_normalized = (loss_sum / valid_tokens).item()
print(f"   Manual (reduction='sum' / valid_tokens): {loss_sum_normalized:.4f}")

# Method 3: reduction='sum' then divide by TOTAL tokens (WRONG!)
loss_sum_wrong = (loss_sum / flat_labels.numel()).item()
print(f"   Manual (reduction='sum' / total_tokens): {loss_sum_wrong:.4f} [WRONG METHOD]")

# Method 4: reduction='none' then manual masking
loss_none = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction='none')
valid_mask = (flat_labels != -100)
loss_manual = loss_none[valid_mask].mean().item()
print(f"   Manual (reduction='none' then mean): {loss_manual:.4f}")

print("\n4. Comparison:")
print(f"   Model's loss:  {model_loss:.4f}")
print(f"   Method 1:      {loss_mean_all.item():.4f}  (diff: {abs(model_loss - loss_mean_all.item()):.6f})")
print(f"   Method 2:      {loss_sum_normalized:.4f}  (diff: {abs(model_loss - loss_sum_normalized):.6f})")
print(f"   Method 3:      {loss_sum_wrong:.4f}  (diff: {abs(model_loss - loss_sum_wrong):.6f}) [WRONG]")
print(f"   Method 4:      {loss_manual:.4f}  (diff: {abs(model_loss - loss_manual):.6f})")

# Find which matches
methods = [
    ("Method 1 (mean)", loss_mean_all.item()),
    ("Method 2 (sum/valid)", loss_sum_normalized),
    ("Method 3 (sum/total)", loss_sum_wrong),
    ("Method 4 (none+mean)", loss_manual),
]

closest = min(methods, key=lambda x: abs(model_loss - x[1]))
print(f"\n   ✓ Model uses: {closest[0]}")

if "sum/total" in closest[0]:
    print("\n   ❌ BUG CONFIRMED: Model divides by TOTAL tokens, not VALID tokens!")
    print("   This would cause lower loss when more tokens are masked.")
else:
    print("\n   ✓ Model correctly divides by valid (non-masked) tokens only")

print("\n" + "=" * 80)
