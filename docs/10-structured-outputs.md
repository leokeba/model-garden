# Structured Outputs with Model Garden

Model Garden now supports **structured outputs** with an OpenAI-compatible API! This allows you to generate JSON that strictly follows a schema, perfect for extracting structured data, function calling, and building reliable LLM applications.

## Overview

Structured outputs leverage vLLM's guided generation capabilities to ensure that model responses conform to specified JSON schemas. The feature uses the **xgrammar** backend by default for efficient constrained decoding.

### Key Features

- ✅ **OpenAI-Compatible API**: Drop-in replacement for OpenAI's structured outputs
- ✅ **Multiple Format Types**: Support for generic JSON objects and specific JSON schemas
- ✅ **Pydantic Integration**: Generate schemas from Pydantic models
- ✅ **Complex Schemas**: Nested objects, arrays, and `$ref` support
- ✅ **Type Safety**: Guaranteed valid JSON matching your schema
- ✅ **Vision Model Support**: Works with image inputs for multimodal structured extraction

## Quick Start

### 1. Basic JSON Object

Request any JSON object without a specific schema:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "messages": [
            {"role": "user", "content": "List 3 colors with their hex codes"}
        ],
        "response_format": {
            "type": "json_object"
        }
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
# Output: {"colors": [{"name": "Red", "hex": "#FF0000"}, ...]}
```

### 2. Specific JSON Schema

Define a schema for precise control:

```python
from pydantic import BaseModel

class LandmarkInfo(BaseModel):
    name: str
    location: str
    height_meters: int
    year_built: int
    description: str

schema = LandmarkInfo.model_json_schema()

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "messages": [
            {"role": "user", "content": "Tell me about the Eiffel Tower"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "LandmarkInfo",
                "schema": schema
            }
        }
    }
)

result = response.json()
# Guaranteed to match LandmarkInfo schema
landmark = LandmarkInfo.model_validate_json(
    result["choices"][0]["message"]["content"]
)
print(f"{landmark.name} is {landmark.height_meters}m tall")
```

### 3. Complex Nested Schema

Handle nested objects and arrays:

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class Person(BaseModel):
    name: str
    age: int
    email: str
    address: Address
    hobbies: list[str]

schema = Person.model_json_schema()

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "messages": [
            {"role": "user", "content": "Generate a random person profile"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "Person",
                "schema": schema
            }
        }
    }
)
```

## API Reference

### Request Format

Add `response_format` to your chat completion request:

```python
{
    "model": "your-model-name",
    "messages": [...],
    "response_format": {
        "type": "json_object" | "json_schema",
        "json_schema": {  # Only for type="json_schema"
            "name": "SchemaName",
            "schema": {...}  # JSON Schema object
        }
    }
}
```

### Response Format Types

| Type | Description | Use Case |
|------|-------------|----------|
| `text` | Default behavior, no structured output | Regular chat |
| `json_object` | Generic JSON object, any valid JSON | Flexible structured data |
| `json_schema` | Specific JSON schema validation | Type-safe, strict schemas |

## Examples

### Example 1: Extract Product Information

```python
from pydantic import BaseModel
from typing import List

class Product(BaseModel):
    name: str
    price: float
    features: List[str]
    rating: float
    in_stock: bool

schema = Product.model_json_schema()

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": "Extract product info: The SuperWidget 3000 costs $29.99, has features like WiFi and Bluetooth, rated 4.5 stars, currently in stock"
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "Product",
                "schema": schema
            }
        }
    }
)

product = Product.model_validate_json(
    response.json()["choices"][0]["message"]["content"]
)
```

### Example 2: Function Calling

```python
class FunctionCall(BaseModel):
    function_name: str
    arguments: dict

schema = FunctionCall.model_json_schema()

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": "I need to send an email to john@example.com with subject 'Meeting' and body 'Let's meet tomorrow'"
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "FunctionCall",
                "schema": schema
            }
        }
    }
)

function_call = FunctionCall.model_validate_json(
    response.json()["choices"][0]["message"]["content"]
)
print(f"Call: {function_call.function_name}")
print(f"Args: {function_call.arguments}")
```

### Example 3: Batch Data Extraction

```python
class Item(BaseModel):
    title: str
    category: str
    price: float

class ItemList(BaseModel):
    items: List[Item]

schema = ItemList.model_json_schema()

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": "Extract items: Book - Literature - $15.99, Laptop - Electronics - $899, Shirt - Clothing - $29"
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "ItemList",
                "schema": schema
            }
        }
    }
)

items = ItemList.model_validate_json(
    response.json()["choices"][0]["message"]["content"]
)
for item in items.items:
    print(f"{item.title}: ${item.price}")
```

### Example 4: Vision Model with Structured Output

Extract structured information from images using vision-language models:

```python
from pydantic import BaseModel
from typing import List

class ImageAnalysis(BaseModel):
    description: str
    main_objects: List[str]
    colors: List[str]
    scene_type: str

schema = ImageAnalysis.model_json_schema()

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
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

analysis = ImageAnalysis.model_validate_json(
    response.json()["choices"][0]["message"]["content"]
)
print(f"Description: {analysis.description}")
print(f"Objects: {', '.join(analysis.main_objects)}")
```

## Using with curl

### Generic JSON Object

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [
      {"role": "user", "content": "List 3 programming languages with their year of creation"}
    ],
    "response_format": {
      "type": "json_object"
    }
  }'
```

### Specific JSON Schema

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
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
  }'
```

## Best Practices

### 1. Choose the Right Format Type

- Use `json_object` for flexible, exploratory queries
- Use `json_schema` for production applications requiring type safety
- Use regular `text` format for conversational responses

### 2. Schema Design Tips

- Keep schemas simple and focused
- Use descriptive field names
- Mark required fields appropriately
- Add descriptions for better model understanding

### 3. Error Handling

Always validate the JSON response:

```python
import json
from pydantic import ValidationError

try:
    response_data = response.json()
    content = response_data["choices"][0]["message"]["content"]
    
    # Parse JSON
    parsed = json.loads(content)
    
    # Validate against schema
    validated = YourModel.model_validate(parsed)
    
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")
except ValidationError as e:
    print(f"Schema validation failed: {e}")
```

### 4. Performance Considerations

- Structured outputs add minimal overhead (typically <5%)
- Use streaming for long responses (coming soon)
- Cache schemas for repeated queries

## Compatibility with OpenAI

Model Garden's structured output API is designed to be compatible with OpenAI's API:

```python
# Works with both OpenAI and Model Garden
from openai import OpenAI

# For OpenAI
client = OpenAI(api_key="sk-...")

# For Model Garden
client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

# Same code works for both!
response = client.chat.completions.create(
    model="gpt-4",  # or your local model
    messages=[{"role": "user", "content": "..."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "MySchema",
            "schema": {...}
        }
    }
)
```

## Technical Details

### Backend Selection

Model Garden uses vLLM's structured output implementation with these backends:

1. **xgrammar** (default) - Fastest, most efficient
2. **guidance** - Alternative backend
3. **outlines** - Another option
4. **lm-format-enforcer** - Legacy support

The backend is selected automatically by vLLM based on availability.

### Schema Conversion

Model Garden converts OpenAI's `response_format` to vLLM's `StructuredOutputsParams`:

- `type: "json_object"` → Generic JSON schema with `additionalProperties: true`
- `type: "json_schema"` → Extracts schema from `json_schema.schema`
- Handles nested `"schema"` keys for compatibility

### Version Compatibility

- Requires vLLM >= 0.11.0 for full support
- Falls back gracefully for older versions with a warning
- Tested with Qwen, Llama, and other popular models

### Vision Model Support

Structured outputs work seamlessly with vision-language models:

- **Supported Models**: Qwen2.5-VL, LLaVA, and other multimodal models
- **Image Formats**: URLs, file paths, and base64-encoded data
- **Use Cases**: 
  - Product extraction from images
  - Document parsing with structured data
  - Image classification with detailed attributes
  - Scene understanding with object detection

Example workflow:
```python
# 1. Define schema for image analysis
class ProductInfo(BaseModel):
    name: str
    category: str
    features: List[str]
    price_estimate: Optional[str]

# 2. Send image + structured output request
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract product info"},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "ProductInfo", "schema": schema}
    }
)

# 3. Get validated structured data
product = ProductInfo.model_validate_json(result)
```

See `examples/structured_output_vision_test.py` for complete examples.

## Troubleshooting

### Common Issues

**Issue**: "StructuredOutputsParams not available"
```
Solution: Upgrade vLLM to >= 0.11.0
$ pip install --upgrade vllm
```

**Issue**: Model generates invalid JSON
```
Solution: 
1. Add system message: "You are a helpful assistant that responds in JSON"
2. Use more explicit prompts
3. Try a different model (Qwen2.5 works well)
```

**Issue**: Schema validation fails
```
Solution:
1. Check your schema is valid JSON Schema
2. Ensure all required fields are marked
3. Verify field types match expected data
```

### Debug Mode

Enable logging to see structured output parameters:

```python
import logging
logging.basicConfig(level=logging.INFO)

# You'll see: "✅ Added structured output parameters: ['json']"
```

## Testing

Run the test suite to verify everything works:

```bash
# Text-only models
model-garden serve --model Qwen/Qwen2.5-3B-Instruct

# In another terminal, run tests
python examples/structured_output_test.py
```

Expected output:
```
✅ Test 1: Generic JSON Object Format - PASSED
✅ Test 2: Specific JSON Schema Format - PASSED  
✅ Test 3: Complex Nested Schema - PASSED
✅ Test 4: Without Structured Output (Control) - PASSED
```

### Testing with Vision Models

```bash
# Start with a vision model
model-garden serve --model Qwen/Qwen2.5-VL-7B-Instruct

# Run vision-specific tests
python examples/structured_output_vision_test.py
```

This will test:
- Generic JSON extraction from images
- Specific schema validation with image inputs
- Product information extraction from images
- Local image file support with structured outputs

## Future Enhancements

- [ ] Streaming support for structured outputs
- [ ] More backend options (regex, CFG)
- [ ] Schema auto-generation from examples
- [ ] Multi-turn structured conversations
- [ ] Schema validation caching

## References

- [vLLM Structured Outputs Documentation](https://docs.vllm.ai/en/latest/features/structured_outputs.html)
- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs)
- [JSON Schema Specification](https://json-schema.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

## Contributing

Found a bug or have a feature request? Please open an issue on GitHub!

Want to contribute? Check out [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
