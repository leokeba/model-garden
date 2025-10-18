"""Test pre-training schema key detection."""

import sys
import os
os.environ["HF_HOME"] = "/home/leo/Dev/model-garden/storage/cache"
os.environ["TRANSFORMERS_CACHE"] = "/home/leo/Dev/model-garden/storage/cache"

# Test with mock dataset
from datasets import Dataset

# Create mock dataset with JSON responses
mock_data = []
for i in range(50):
    mock_data.append({
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Extract the data"}]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": '{"Marque": "Toyota", "Modele": "Corolla", "Year": 2020, "contents": [{"item": "info"}]}'}]
            }
        ]
    })

dataset = Dataset.from_list(mock_data)
print(f"Created mock dataset with {len(dataset)} samples")

# Import detection function
from model_garden.selective_loss import detect_schema_keys_from_dataset
from transformers import Qwen2VLProcessor

print("\nLoading processor...")
processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

print("\n" + "="*60)
print("TESTING PRE-TRAINING DETECTION")
print("="*60)

# Test detection
detected_keys = detect_schema_keys_from_dataset(
    dataset=dataset,
    processor=processor,
    num_samples=50,
    threshold=0.3,
    verbose=True
)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Detected keys: {detected_keys}")
print(f"Expected: {{'Marque', 'Modele', 'Year', 'contents', 'item'}}")

if detected_keys == {'Marque', 'Modele', 'Year', 'contents', 'item'}:
    print("\n✅ SUCCESS: All expected keys detected!")
else:
    print("\n❌ MISMATCH!")
    missing = {'Marque', 'Modele', 'Year', 'contents', 'item'} - detected_keys
    extra = detected_keys - {'Marque', 'Modele', 'Year', 'contents', 'item'}
    if missing:
        print(f"   Missing: {missing}")
    if extra:
        print(f"   Extra: {extra}")
