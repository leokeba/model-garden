"""Debug loss computation to understand why weighted loss is higher."""

import torch
import torch.nn.functional as F

# Simulate a simple scenario
batch_size = 2
seq_len = 10
vocab_size = 100

# Create fake logits and labels
logits = torch.randn(batch_size, seq_len, vocab_size)
labels = torch.randint(0, vocab_size, (batch_size, seq_len))

# Mask some tokens (simulate prompt masking)
labels[:, :3] = -100  # First 3 tokens are prompt

print("="*80)
print("Testing CrossEntropyLoss computation")
print("="*80)

# Method 1: Standard CrossEntropyLoss (reduction='mean')
loss_fct_mean = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
loss_mean = loss_fct_mean(logits.view(-1, vocab_size), labels.view(-1))
print(f"\n1. Standard loss (reduction='mean'): {loss_mean.item():.4f}")

# Method 2: CrossEntropyLoss with reduction='none', then manual mean
loss_fct_none = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
loss_none = loss_fct_none(logits.view(-1, vocab_size), labels.view(-1))

# CRITICAL: Check what reduction='none' returns for ignored tokens
print(f"\n2. Loss with reduction='none':")
print(f"   Shape: {loss_none.shape}")
print(f"   All values: {loss_none}")

# Count non-zero values
non_zero = (loss_none != 0.0).sum().item()
total = loss_none.numel()
print(f"   Non-zero losses: {non_zero}/{total}")

# Manual mean (only over valid tokens)
valid_mask = labels.view(-1) != -100
if valid_mask.any():
    manual_mean = loss_none[valid_mask].mean()
    print(f"   Manual mean (over valid tokens): {manual_mean.item():.4f}")
    print(f"   Matches standard? {abs(manual_mean.item() - loss_mean.item()) < 0.001}")

# Method 3: Weighted loss (our current implementation)
print(f"\n3. Weighted loss (our implementation):")

# Create weights: first 3 are prompt (will be ignored), next 4 are structural (0.1), last 3 are semantic (1.0)
weights = torch.ones(batch_size, seq_len)
weights[:, :3] = 0.0  # Prompt (doesn't matter, will be ignored)
weights[:, 3:7] = 0.1  # Structural
weights[:, 7:] = 1.0   # Semantic

weights_flat = weights.view(-1)

# Apply weights
weighted_loss = loss_none * weights_flat

print(f"   Weighted losses shape: {weighted_loss.shape}")
print(f"   Sum of weighted losses: {weighted_loss.sum().item():.4f}")
print(f"   Sum of weights (valid only): {weights_flat[valid_mask].sum().item():.4f}")
print(f"   Num valid tokens: {valid_mask.sum().item()}")

# Our current formula: sum(weighted_loss) / num_valid_tokens
our_loss = weighted_loss[valid_mask].sum() / valid_mask.sum()
print(f"   Our weighted loss: {our_loss.item():.4f}")

# Compare to standard
print(f"\n4. Comparison:")
print(f"   Standard loss: {loss_mean.item():.4f}")
print(f"   Our weighted loss: {our_loss.item():.4f}")
print(f"   Ratio (weighted/standard): {our_loss.item() / loss_mean.item():.2f}x")

# What SHOULD the weighted loss be?
# With 4 structural (weight=0.1) and 3 semantic (weight=1.0) out of 7 total valid tokens:
# Expected: (4 * avg_loss * 0.1 + 3 * avg_loss * 1.0) / 7
#         = avg_loss * (0.4 + 3.0) / 7
#         = avg_loss * 3.4 / 7
#         = avg_loss * 0.486
print(f"   Expected ratio: ~0.49x (should be LOWER, not higher!)")

print("\n" + "="*80)
print("Checking if CrossEntropyLoss with reduction='none' behaves correctly...")
print("="*80)

# Test: Does reduction='none' return 0 for ignored indices?
test_logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float)  # [2, 3]
test_labels = torch.tensor([0, -100], dtype=torch.long)  # [2]

loss_fct_test = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
test_loss = loss_fct_test(test_logits, test_labels)

print(f"\nTest logits shape: {test_logits.shape}")
print(f"Test labels: {test_labels}")
print(f"Test loss: {test_loss}")
print(f"Loss for ignored token (-100): {test_loss[1].item():.10f}")

if abs(test_loss[1].item()) < 1e-6:
    print("✅ Ignored tokens return 0.0 (as expected)")
else:
    print(f"❌ Ignored tokens return {test_loss[1].item():.4f} (NOT ZERO!)")
    print("   This could cause the inflated loss issue!")

print("\n" + "="*80)
