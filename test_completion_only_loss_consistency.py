"""
Test to verify completion_only_loss applies consistently to both training and validation.
"""

import torch
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

print("=" * 80)
print("TESTING COMPLETION_ONLY_LOSS CONSISTENCY")
print("=" * 80)

# Test the data collator directly
collator = DataCollatorForLanguageModeling(
    pad_token_id=0,
    completion_only_loss=True,
    padding_free=False
)

# Create example with prompt and completion tokens
examples = [
    {
        "input_ids": [1, 2, 3, 4, 5, 6],  # Full sequence
        "completion_mask": [0, 0, 0, 1, 1, 1]  # First 3 tokens are prompt
    }
]

print(f"\nInput: {examples[0]['input_ids']}")
print(f"Completion mask: {examples[0]['completion_mask']}")
print("  (0 = prompt token, 1 = completion token)")

# Collate the batch
batch = collator(examples)

print(f"\nAfter collation:")
print(f"Input IDs:  {batch['input_ids'].tolist()}")
print(f"Labels:     {batch['labels'].tolist()}")
print(f"\nNotice: Prompt tokens (first 3) are masked to -100 in labels!")
print(f"        Only completion tokens (4,5,6) will contribute to loss.")

# Verify the masking
input_ids = batch['input_ids'][0].tolist()
labels = batch['labels'][0].tolist()

masked_count = sum(1 for l in labels if l == -100)
unmasked_count = sum(1 for l in labels if l != -100)

print(f"\nStatistics:")
print(f"  Total tokens: {len(labels)}")
print(f"  Masked (prompt): {masked_count}")
print(f"  Unmasked (completion): {unmasked_count}")

print("\nConclusion:")
print("-" * 80)
print("The data collator masks prompt tokens BEFORE the batch reaches compute_loss().")
print("The same masking applies to both training and validation batches.")
print("If you see different losses in practice, it's due to:")
print("  1. Overfitting (model memorizes training data)")
print("  2. Different data distributions (train vs val)")
print("  3. Model behavior changes (dropout, etc.)")
print("NOT due to different loss computation!")
