"""Test schema key detection with real training data flow."""

import sys
import os
os.environ["HF_HOME"] = "/home/leo/Dev/model-garden/storage/cache"
os.environ["TRANSFORMERS_CACHE"] = "/home/leo/Dev/model-garden/storage/cache"

from datasets import load_dataset
from transformers import Qwen2VLProcessor
import torch
import json

# Load dataset
print("Loading dataset...")
dataset = load_dataset("Barth371/cmr-all", split="train")
print(f"Loaded {len(dataset)} samples")

# Load tokenizer/processor
print("\nLoading processor...")
processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# Test detection logic on first few samples
print("\n" + "="*60)
print("TESTING SCHEMA KEY DETECTION")
print("="*60)

detected_keys_counter = {}

for idx in range(min(10, len(dataset))):
    sample = dataset[idx]
    print(f"\n=== Sample {idx} ===")
    
    # Extract assistant response from messages
    if "messages" in sample:
        messages = sample["messages"]
        assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)
        
        if assistant_msg:
            # Get text content from assistant message
            content = assistant_msg.get("content", [])
            if isinstance(content, list):
                text_content = next((c.get("text") for c in content if c.get("type") == "text"), None)
            else:
                text_content = content
            
            print(f"Assistant response: {text_content[:100]}...")
            
            # Try to parse as JSON
            try:
                json_data = json.loads(text_content)
                print(f"✅ Valid JSON")
                
                # Extract keys
                def extract_keys(obj, keys_found):
                    if isinstance(obj, dict):
                        for key in obj.keys():
                            keys_found.add(key)
                            extract_keys(obj[key], keys_found)
                    elif isinstance(obj, list):
                        for item in obj:
                            extract_keys(item, keys_found)
                
                keys_found = set()
                extract_keys(json_data, keys_found)
                print(f"Keys found: {len(keys_found)} - {list(keys_found)[:10]}")
                
                # Update counter
                for key in keys_found:
                    detected_keys_counter[key] = detected_keys_counter.get(key, 0) + 1
                    
            except Exception as e:
                print(f"❌ JSON parsing failed: {e}")
        else:
            print("❌ No assistant message found")
    else:
        print("❌ No messages field")

print("\n" + "="*60)
print("DETECTION RESULTS")
print("="*60)

# Apply threshold (30% of samples)
threshold = 10 * 0.3

detected_keys = {
    key for key, count in detected_keys_counter.items()
    if count >= threshold
}

print(f"\nTotal unique keys seen: {len(detected_keys_counter)}")
print(f"Threshold: {threshold} samples")
print(f"Keys meeting threshold: {len(detected_keys)}")

if detected_keys:
    print("\n🎯 Schema keys that would be masked:")
    sorted_keys = sorted(
        detected_keys_counter.items(),
        key=lambda x: x[1],
        reverse=True
    )[:20]
    for key, count in sorted_keys:
        if key in detected_keys:
            print(f"  ✓ {key} (appears in {count}/10 samples)")
else:
    print("\n⚠️  NO schema keys detected!")
    print("\nAll keys seen:")
    for key, count in list(detected_keys_counter.items())[:20]:
        print(f"  - {key}: {count}/10 samples")
