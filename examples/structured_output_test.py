"""Test structured output feature with Model Garden API."""

import requests
import json
from pydantic import BaseModel

# API endpoint
API_BASE = "http://localhost:8000"

def test_json_object_format():
    """Test with generic JSON object format."""
    print("\n" + "="*60)
    print("Test 1: Generic JSON Object Format")
    print("="*60)
    
    response = requests.post(
        f"{API_BASE}/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Generate a JSON object with information about the Eiffel Tower. Include name, location, height, and year_built."
                }
            ],
            "max_tokens": 200,
            "temperature": 0.7,
            "response_format": {
                "type": "json_object"
            }
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print(f"\n✅ Success!")
        print(f"Response: {content}")
        
        # Try to parse as JSON
        try:
            parsed = json.loads(content)
            print(f"\n✅ Valid JSON structure:")
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError as e:
            print(f"\n❌ Failed to parse as JSON: {e}")
    else:
        print(f"\n❌ Request failed: {response.status_code}")
        print(f"Error: {response.text}")


def test_json_schema_format():
    """Test with specific JSON schema."""
    print("\n" + "="*60)
    print("Test 2: Specific JSON Schema Format")
    print("="*60)
    
    # Define a Pydantic model for the expected response
    class LandmarkInfo(BaseModel):
        name: str
        location: str
        height_meters: int
        year_built: int
        description: str
    
    # Get JSON schema from Pydantic model
    schema = LandmarkInfo.model_json_schema()
    
    print(f"\nUsing schema:")
    print(json.dumps(schema, indent=2))
    
    response = requests.post(
        f"{API_BASE}/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Provide information about the Eiffel Tower in the specified JSON format."
                }
            ],
            "max_tokens": 300,
            "temperature": 0.7,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "LandmarkInfo",
                    "schema": schema,
                    "strict": True
                }
            }
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print(f"\n✅ Success!")
        print(f"Response: {content}")
        
        # Try to parse and validate
        try:
            parsed = json.loads(content)
            validated = LandmarkInfo(**parsed)
            print(f"\n✅ Valid and matches schema:")
            print(f"  Name: {validated.name}")
            print(f"  Location: {validated.location}")
            print(f"  Height: {validated.height_meters}m")
            print(f"  Built: {validated.year_built}")
            print(f"  Description: {validated.description}")
        except json.JSONDecodeError as e:
            print(f"\n❌ Failed to parse as JSON: {e}")
        except Exception as e:
            print(f"\n❌ Failed to validate against schema: {e}")
    else:
        print(f"\n❌ Request failed: {response.status_code}")
        print(f"Error: {response.text}")


def test_complex_schema():
    """Test with a more complex nested schema."""
    print("\n" + "="*60)
    print("Test 3: Complex Nested Schema")
    print("="*60)
    
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
    
    print(f"\nUsing complex schema:")
    print(json.dumps(schema, indent=2))
    
    response = requests.post(
        f"{API_BASE}/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Create a fictional person profile with name, age, email, address (with street, city, country), and a list of 3 hobbies."
                }
            ],
            "max_tokens": 400,
            "temperature": 0.8,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Person",
                    "schema": schema
                }
            }
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print(f"\n✅ Success!")
        print(f"Response: {content}")
        
        try:
            parsed = json.loads(content)
            validated = Person(**parsed)
            print(f"\n✅ Valid and matches complex schema:")
            print(f"  Name: {validated.name}")
            print(f"  Age: {validated.age}")
            print(f"  Email: {validated.email}")
            print(f"  Address: {validated.address.street}, {validated.address.city}, {validated.address.country}")
            print(f"  Hobbies: {', '.join(validated.hobbies)}")
        except json.JSONDecodeError as e:
            print(f"\n❌ Failed to parse as JSON: {e}")
        except Exception as e:
            print(f"\n❌ Failed to validate against schema: {e}")
    else:
        print(f"\n❌ Request failed: {response.status_code}")
        print(f"Error: {response.text}")


def test_without_structured_output():
    """Test without structured output for comparison."""
    print("\n" + "="*60)
    print("Test 4: Without Structured Output (Control)")
    print("="*60)
    
    response = requests.post(
        f"{API_BASE}/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Tell me about the Eiffel Tower."
                }
            ],
            "max_tokens": 200,
            "temperature": 0.7
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print(f"\n✅ Success!")
        print(f"Response: {content}")
    else:
        print(f"\n❌ Request failed: {response.status_code}")
        print(f"Error: {response.text}")


if __name__ == "__main__":
    print("🧪 Testing Structured Output Feature")
    print("=" * 60)
    print("Make sure Model Garden API is running with a model loaded!")
    print("Run: model-garden serve --model <your-model>")
    print()
    
    try:
        # Check if API is available
        health = requests.get(f"{API_BASE}/api/v1/system/status", timeout=5)
        if health.status_code != 200:
            print("❌ API is not responding. Please start the server first.")
            exit(1)
        
        print("✅ API is available")
        
        # Run tests
        test_without_structured_output()
        test_json_object_format()
        test_json_schema_format()
        test_complex_schema()
        
        print("\n" + "="*60)
        print("✅ All tests completed!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running at http://localhost:8000")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
