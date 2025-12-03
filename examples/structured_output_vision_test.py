#!/usr/bin/env python3
"""
Test structured outputs with vision-language models.

This script tests that structured JSON outputs work correctly with image inputs.

Prerequisites:
1. Start Model Garden API with a vision model:
   model-garden serve --model Qwen/Qwen2.5-VL-7B-Instruct

2. Have a test image available
"""

import base64
import json
from pathlib import Path

import requests
from pydantic import BaseModel

API_BASE = "http://localhost:8000"


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class ImageAnalysis(BaseModel):
    """Structured analysis of an image."""
    description: str
    main_objects: list[str]
    colors: list[str]
    scene_type: str
    estimated_time_of_day: str | None = None


class ProductInfo(BaseModel):
    """Product information extracted from an image."""
    product_name: str
    category: str
    visible_features: list[str]
    condition: str
    estimated_value_range: str | None = None


def test_image_with_generic_json():
    """Test 1: Generic JSON object with image input."""
    print("\n" + "="*60)
    print("Test 1: Generic JSON Object with Image")
    print("="*60)

    # Use a placeholder image URL for testing
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    response = requests.post(
        f"{API_BASE}/v1/chat/completions",
        json={
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image and provide a JSON description"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "response_format": {
                "type": "json_object"
            }
        }
    )

    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print("\n✅ Success!")
        print(f"Response: {content}")

        # Validate it's valid JSON
        try:
            parsed = json.loads(content)
            print("\n✅ Valid JSON structure:")
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError as e:
            print(f"\n❌ Invalid JSON: {e}")
    else:
        print(f"\n❌ Request failed: {response.status_code}")
        print(response.text)


def test_image_with_specific_schema():
    """Test 2: Specific JSON schema with image input."""
    print("\n" + "="*60)
    print("Test 2: Specific Schema with Image")
    print("="*60)

    schema = ImageAnalysis.model_json_schema()
    print(f"\nUsing schema: {json.dumps(schema, indent=2)}")

    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    response = requests.post(
        f"{API_BASE}/v1/chat/completions",
        json={
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image following the provided schema"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ImageAnalysis",
                    "schema": schema
                }
            }
        }
    )

    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print("\n✅ Success!")

        try:
            analysis = ImageAnalysis.model_validate_json(content)
            print("\n✅ Valid and matches schema:")
            print(f"  Description: {analysis.description}")
            print(f"  Main Objects: {', '.join(analysis.main_objects)}")
            print(f"  Colors: {', '.join(analysis.colors)}")
            print(f"  Scene Type: {analysis.scene_type}")
            if analysis.estimated_time_of_day:
                print(f"  Time of Day: {analysis.estimated_time_of_day}")
        except Exception as e:
            print(f"\n❌ Validation failed: {e}")
            print(f"Raw response: {content}")
    else:
        print(f"\n❌ Request failed: {response.status_code}")
        print(response.text)


def test_product_extraction_from_image():
    """Test 3: Product information extraction from image."""
    print("\n" + "="*60)
    print("Test 3: Product Extraction from Image")
    print("="*60)

    schema = ProductInfo.model_json_schema()

    # Example product image URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Apple_logo_black.svg/200px-Apple_logo_black.svg.png"

    response = requests.post(
        f"{API_BASE}/v1/chat/completions",
        json={
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract product information from this image"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ProductInfo",
                    "schema": schema
                }
            }
        }
    )

    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print("\n✅ Success!")

        try:
            product = ProductInfo.model_validate_json(content)
            print("\n✅ Valid product information:")
            print(f"  📦 Product: {product.product_name}")
            print(f"  📁 Category: {product.category}")
            print(f"  ✨ Features: {', '.join(product.visible_features)}")
            print(f"  📊 Condition: {product.condition}")
            if product.estimated_value_range:
                print(f"  💰 Value Range: {product.estimated_value_range}")
        except Exception as e:
            print(f"\n❌ Validation failed: {e}")
            print(f"Raw response: {content}")
    else:
        print(f"\n❌ Request failed: {response.status_code}")
        print(response.text)


def test_local_image_file():
    """Test 4: Using a local image file with structured output."""
    print("\n" + "="*60)
    print("Test 4: Local Image File with Structured Output")
    print("="*60)

    # Check if test images directory exists
    test_images_dir = Path("data/test_images")
    if not test_images_dir.exists():
        print(f"\n⚠️  Test images directory not found: {test_images_dir}")
        print("Skipping this test.")
        return

    # Find first image in test_images directory
    image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    if not image_files:
        print(f"\n⚠️  No image files found in {test_images_dir}")
        print("Skipping this test.")
        return

    image_path = image_files[0]
    print(f"\nUsing image: {image_path}")

    # Encode image to base64
    image_base64 = encode_image_to_base64(str(image_path))
    image_data_url = f"data:image/jpeg;base64,{image_base64}"

    schema = ImageAnalysis.model_json_schema()

    response = requests.post(
        f"{API_BASE}/v1/chat/completions",
        json={
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image in detail"},
                        {"type": "image_url", "image_url": {"url": image_data_url}}
                    ]
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ImageAnalysis",
                    "schema": schema
                }
            }
        }
    )

    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print("\n✅ Success!")

        try:
            analysis = ImageAnalysis.model_validate_json(content)
            print("\n✅ Analysis of local image:")
            print(f"  {analysis.description}")
            print(f"  Objects: {', '.join(analysis.main_objects)}")
            print(f"  Colors: {', '.join(analysis.colors)}")
        except Exception as e:
            print(f"\n❌ Validation failed: {e}")
            print(f"Raw response: {content}")
    else:
        print(f"\n❌ Request failed: {response.status_code}")
        print(response.text)


def main():
    print("🌱 Testing Structured Outputs with Vision Models")
    print("=" * 60)
    print("Make sure Model Garden API is running with a vision model!")
    print("Example: model-garden serve --model Qwen/Qwen2.5-VL-7B-Instruct")
    print()

    # Check if API is available
    try:
        response = requests.get(f"{API_BASE}/api/v1/system/status", timeout=5)
        if response.status_code == 200:
            print("✅ API is available")
        else:
            print("❌ API returned error status")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print(f"\nMake sure the API is running on {API_BASE}")
        return

    try:
        # Run all tests
        test_image_with_generic_json()
        test_image_with_specific_schema()
        test_product_extraction_from_image()
        test_local_image_file()

        print("\n" + "="*60)
        print("✅ All vision + structured output tests completed!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
