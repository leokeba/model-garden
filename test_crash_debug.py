#!/usr/bin/env python3
"""Test script to reproduce the EngineCore crash with vision + structured outputs."""

import base64
import requests
import json

# Read and encode test image
with open("./data/test_images/blue_circle.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Prepare request with structured output
payload = {
    "model": "Qwen/Qwen2.5-VL-3B-Instruct-ft",
    "messages": [
        {
            "role": "user",
            "content": f"data:image/jpeg;base64,{image_data}\n\nWhat shape and color is in this image?"
        }
    ],
    "max_tokens": 2048,
    "temperature": 0.7,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "shape_detection",
            "schema": {
                "type": "object",
                "properties": {
                    "shape": {"type": "string"},
                    "color": {"type": "string"}
                },
                "required": ["shape", "color"]
            }
        }
    }
}

print("🧪 Sending request with vision + structured outputs...")
print(f"📊 Image size: {len(image_data)} chars")
print(f"📋 Response format: JSON schema")
print()

try:
    response = requests.post(
        "http://localhost:8888/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=60
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:500]}")
    
    if response.status_code == 200:
        print("✅ SUCCESS - No crash!")
        result = response.json()
        print(json.dumps(result, indent=2))
    else:
        print(f"❌ ERROR - Status {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ EXCEPTION: {e}")
