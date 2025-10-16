# Structured Outputs - Quick Reference

## Basic Usage

### Generic JSON Object
```python
response_format = {"type": "json_object"}
```

### Specific JSON Schema
```python
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "SchemaName",
        "schema": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }
}
```

## Response Format Types

| Type | Description | Example Use Case |
|------|-------------|------------------|
| `text` | Default, no structured output | Regular chat |
| `json_object` | Generic JSON, any structure | Flexible data extraction |
| `json_schema` | Strict schema validation | Type-safe applications |

## Quick Examples

### With Pydantic
```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

schema = Person.model_json_schema()

response = client.chat.completions.create(
    model="...",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "Person",
            "schema": schema
        }
    }
)
```

### With curl
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "...",
    "messages": [...],
    "response_format": {"type": "json_object"}
  }'
```

### With OpenAI Client
```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="...",
    messages=[...],
    response_format={"type": "json_object"}
)
```

## Common Patterns

### Product Extraction
```python
class Product(BaseModel):
    name: str
    price: float
    features: List[str]
```

### Function Calling
```python
class FunctionCall(BaseModel):
    function_name: str
    arguments: dict
```

### Data Collection
```python
class ItemList(BaseModel):
    items: List[Item]
```

## Tips

✅ Use `json_schema` for production applications  
✅ Use `json_object` for exploration  
✅ Validate responses with Pydantic  
✅ Add clear field descriptions  
✅ Mark required fields explicitly  

❌ Don't use overly complex schemas  
❌ Don't forget error handling  
❌ Don't skip validation  

## Files

- **Documentation**: `docs/10-structured-outputs.md`
- **Test Suite**: `examples/structured_output_test.py`
- **OpenAI Client**: `examples/structured_output_openai_client.py`
- **Curl Examples**: `examples/structured_output_curl.sh`

## Requirements

- vLLM >= 0.11.0
- OpenAI client (optional): `pip install openai`
- Model Garden API running

## More Info

Run examples:
```bash
# Test suite
python examples/structured_output_test.py

# OpenAI client examples
python examples/structured_output_openai_client.py

# Curl examples
bash examples/structured_output_curl.sh
```

Full documentation: [docs/10-structured-outputs.md](../docs/10-structured-outputs.md)
