"""Test that structured outputs work with vLLM defaults (no custom penalties)."""

import requests
import json
import base64
from pathlib import Path

API_BASE = "http://localhost:8000"

def test_structured_with_defaults():
    """Test structured output with a real image."""
    print("Testing structured output with vLLM defaults...")
    
    # Load a test image
    image_path = Path("/root/model-garden/data/test_images/red_square.jpg")
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    print(f"Loaded test image: {image_path.name}")
    print(f"Base64 length: {len(img_base64)}")
    
    # Define a simple schema
    schema = {
        "type": "object",
        "properties": {
            "shape": {"type": "string"},
            "color": {"type": "string"},
            "description": {"type": "string"}
        },
        "required": ["shape", "color"]
    }
    
    # Send request with structured output
    print("\nSending request with structured output...")
    response = requests.post(
        f"{API_BASE}/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe what you see in this image. What shape and color is it?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 200,
            "temperature": 0.7,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ShapeDescription",
                    "schema": schema
                }
            }
        },
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        content = result['choices'][0]['message']['content']
        print(f"\n✅ Success!")
        print(f"Response: {content}")
        
        # Try to parse as JSON
        try:
            parsed = json.loads(content)
            print(f"\n✅ Valid JSON:")
            print(json.dumps(parsed, indent=2))
            return True
        except json.JSONDecodeError as e:
            print(f"\n❌ Invalid JSON: {e}")
            print(f"Content preview: {content[:500]}")
            return False
    else:
        print(f"\n❌ Request failed: {response.status_code}")
        print(f"Error: {response.text[:500]}")
        return False

if __name__ == "__main__":
    success = test_structured_with_defaults()
    exit(0 if success else 1)
