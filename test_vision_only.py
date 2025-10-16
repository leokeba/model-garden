#!/usr/bin/env python3
"""Test vision model WITHOUT structured outputs."""

import base64
import requests
import json

# Read and encode test image
with open("./data/test_images/blue_circle.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Prepare request WITHOUT structured output
payload = {
    "model": "Qwen/Qwen2.5-VL-3B-Instruct-ft",
    "messages": [
        {
            "role": "user",
            "content": f"data:image/jpeg;base64,{image_data}\n\nWhat shape and color is in this image?"
        }
    ],
    "max_tokens": 512,
    "temperature": 0.7
}

print("🧪 Sending request with vision ONLY (no structured outputs)...")
print(f"📊 Image size: {len(image_data)} chars")
print()

try:
    response = requests.post(
        "http://localhost:8888/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=60
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ SUCCESS - Vision model works!")
        result = response.json()
        print(json.dumps(result, indent=2))
    else:
        print(f"❌ ERROR - Status {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ EXCEPTION: {e}")
