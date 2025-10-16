"""Simple API test without images first."""

import requests

API_BASE = "http://localhost:8000"

# Test 1: Simple text-only request
print("1. Testing text-only request...")
response = requests.post(
    f"{API_BASE}/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "max_tokens": 50
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"✅ Success: {result['choices'][0]['message']['content']}")
else:
    print(f"❌ Failed: {response.status_code} - {response.text[:500]}")

# Test 2: Text with structured output (no image)
print("\n2. Testing structured output (no image)...")
response2 = requests.post(
    f"{API_BASE}/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "messages": [
            {"role": "user", "content": "What is the capital of France? Answer in JSON format."}
        ],
        "max_tokens": 100,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "CityInfo",
                "schema": {
                    "type": "object",
                    "properties": {
                        "capital": {"type": "string"},
                        "country": {"type": "string"}
                    },
                    "required": ["capital"]
                }
            }
        }
    }
)

if response2.status_code == 200:
    result = response2.json()
    content = result['choices'][0]['message']['content']
    print(f"✅ Success: {content}")
else:
    print(f"❌ Failed: {response2.status_code} - {response2.text[:500]}")
