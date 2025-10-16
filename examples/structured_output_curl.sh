#!/bin/bash
# Simple curl examples for Model Garden Structured Outputs
# Make sure the API is running: model-garden serve --model <your-model>

API_URL="http://localhost:8000/v1/chat/completions"

echo "🌱 Model Garden Structured Outputs - Curl Examples"
echo "=================================================="

# Example 1: Generic JSON Object
echo ""
echo "Example 1: Generic JSON Object"
echo "------------------------------"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [
      {"role": "user", "content": "List 3 colors with their hex codes"}
    ],
    "response_format": {
      "type": "json_object"
    }
  }' | jq -r '.choices[0].message.content' | jq .

# Example 2: Specific JSON Schema
echo ""
echo "Example 2: Specific JSON Schema - Landmark"
echo "-------------------------------------------"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [
      {"role": "user", "content": "Tell me about the Eiffel Tower"}
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "LandmarkInfo",
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "location": {"type": "string"},
            "height_meters": {"type": "integer"},
            "year_built": {"type": "integer"},
            "description": {"type": "string"}
          },
          "required": ["name", "location", "height_meters", "year_built", "description"]
        }
      }
    }
  }' | jq -r '.choices[0].message.content' | jq .

# Example 3: Product Information
echo ""
echo "Example 3: Product Information Extraction"
echo "------------------------------------------"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [
      {"role": "user", "content": "Extract product info: The iPhone 15 Pro costs $999, has features like A17 chip, titanium frame, and USB-C port. Currently in stock."}
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "Product",
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "price": {"type": "number"},
            "features": {"type": "array", "items": {"type": "string"}},
            "in_stock": {"type": "boolean"}
          },
          "required": ["name", "price", "features", "in_stock"]
        }
      }
    }
  }' | jq -r '.choices[0].message.content' | jq .

# Example 4: Function Calling
echo ""
echo "Example 4: Function Calling"
echo "---------------------------"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [
      {"role": "user", "content": "Send an email to john@example.com with subject Meeting and body Lets meet tomorrow at 3pm"}
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "FunctionCall",
        "schema": {
          "type": "object",
          "properties": {
            "function_name": {"type": "string"},
            "arguments": {"type": "object"}
          },
          "required": ["function_name", "arguments"]
        }
      }
    }
  }' | jq -r '.choices[0].message.content' | jq .

echo ""
echo "=================================================="
echo "✅ All examples completed!"
echo "=================================================="
