"""Test structured output via the API server."""

import requests
import json
import base64
from PIL import Image
import io

API_BASE = "http://localhost:8000"

def test_with_api():
    """Test structured output through the API."""
    print("Testing structured output via API server...")
    
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='blue')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_bytes = buf.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    print(f"Created test image, base64 length: {len(img_base64)}")
    
    # Test 1: Without structured output
    print("\n1. Testing WITHOUT structured output...")
    response1 = requests.post(
        f"{API_BASE}/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What color is this image?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
    )
    
    if response1.status_code == 200:
        result = response1.json()
        print(f"✅ Success (no struct): {result['choices'][0]['message']['content'][:200]}")
    else:
        print(f"❌ Failed: {response1.status_code} - {response1.text[:500]}")
        return
    
    # Test 2: WITH structured output
    print("\n2. Testing WITH structured output...")
    
    schema = {
        "type": "object",
        "properties": {
            "color": {"type": "string"},
            "description": {"type": "string"}
        },
        "required": ["color"]
    }
    
    response2 = requests.post(
        f"{API_BASE}/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What color is this image? Answer in JSON format."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 150,
            "temperature": 0.7,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ColorInfo",
                    "schema": schema
                }
            }
        }
    )
    
    if response2.status_code == 200:
        result = response2.json()
        content = result['choices'][0]['message']['content']
        print(f"✅ Success (with struct): {content}")
        
        # Validate JSON
        try:
            parsed = json.loads(content)
            print(f"✅ Valid JSON: {parsed}")
        except Exception as e:
            print(f"❌ Invalid JSON: {e}")
    else:
        print(f"❌ Failed: {response2.status_code}")
        print(f"Error: {response2.text[:1000]}")

if __name__ == "__main__":
    try:
        # Check API status
        health = requests.get(f"{API_BASE}/api/v1/system/status", timeout=5)
        if health.status_code != 200:
            print("❌ API not available")
            exit(1)
        
        print("✅ API is available\n")
        test_with_api()
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Start server with: model-garden serve")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
