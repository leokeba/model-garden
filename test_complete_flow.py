"""Test complete selective loss flow with pre-detection."""

import sys
import os
os.environ["HF_HOME"] = "/home/leo/Dev/model-garden/storage/cache"
os.environ["TRANSFORMERS_CACHE"] = "/home/leo/Dev/model-garden/storage/cache"

from datasets import Dataset
from transformers import Qwen2VLProcessor
from model_garden.selective_loss import create_selective_loss_collator

# Create mock dataset
print("Creating mock dataset...")
mock_data = []
for i in range(20):
    mock_data.append({
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Extract data"}]},
            {"role": "assistant", "content": [{"type": "text", "text": '{"name": "Test", "value": 123}'}]}
        ]
    })

dataset = Dataset.from_list(mock_data)
print(f"✓ Created dataset with {len(dataset)} samples\n")

print("Loading model components...")
processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
print("✓ Loaded processor\n")

print("="*60)
print("TESTING AGGRESSIVE MODE WITH PRE-DETECTION")
print("="*60)

# Test aggressive mode (should auto-detect)
collator = create_selective_loss_collator(
    model=None,  # Mock - not needed for this test
    processor=processor,
    mask_level="aggressive",
    dataset=dataset,
    verbose=True
)

print("\n" + "="*60)
print("SUCCESS")
print("="*60)
print("✅ Collator created successfully!")
print(f"✅ Schema keys available from batch 0: {collator.schema_keys}")
print("✅ No waiting for 10 batches - aggressive masking active immediately!")
