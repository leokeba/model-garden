#!/usr/bin/env python3
"""
Example: Using Model Garden's Structured Outputs with OpenAI Python Client

This example demonstrates how to use structured outputs with the official
OpenAI Python client library, which provides a more convenient interface
than raw HTTP requests.

Prerequisites:
1. Install OpenAI client: pip install openai
2. Start Model Garden API: model-garden serve --model Qwen/Qwen2.5-3B-Instruct
"""

from openai import OpenAI
from pydantic import BaseModel
from typing import List
import json

# Initialize OpenAI client pointing to Model Garden
client = OpenAI(
    api_key="not-needed",  # Model Garden doesn't require authentication
    base_url="http://localhost:8000/v1"
)

# Define schemas using Pydantic models


class Product(BaseModel):
    """Product information extracted from text"""
    name: str
    category: str
    price: float
    features: List[str]
    in_stock: bool


class LandmarkInfo(BaseModel):
    """Information about a famous landmark"""
    name: str
    location: str
    height_meters: int
    year_built: int
    description: str


class FunctionCall(BaseModel):
    """Structured function calling"""
    function_name: str
    arguments: dict


def example_1_generic_json():
    """Example 1: Generic JSON object format"""
    print("\n" + "="*60)
    print("Example 1: Generic JSON Object")
    print("="*60)
    
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct",
        messages=[
            {
                "role": "user",
                "content": "List 3 programming languages with their year of creation and primary paradigm"
            }
        ],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(completion.choices[0].message.content)
    print(json.dumps(result, indent=2))


def example_2_specific_schema():
    """Example 2: Specific JSON schema with Pydantic"""
    print("\n" + "="*60)
    print("Example 2: Specific Schema - Landmark Info")
    print("="*60)
    
    schema = LandmarkInfo.model_json_schema()
    
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct",
        messages=[
            {
                "role": "user",
                "content": "Tell me about the Golden Gate Bridge"
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "LandmarkInfo",
                "schema": schema
            }
        }
    )
    
    # Validate response against schema
    landmark = LandmarkInfo.model_validate_json(
        completion.choices[0].message.content
    )
    
    print(f"Name: {landmark.name}")
    print(f"Location: {landmark.location}")
    print(f"Height: {landmark.height_meters}m")
    print(f"Built: {landmark.year_built}")
    print(f"Description: {landmark.description}")


def example_3_product_extraction():
    """Example 3: Extract structured product information"""
    print("\n" + "="*60)
    print("Example 3: Product Information Extraction")
    print("="*60)
    
    schema = Product.model_json_schema()
    
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct",
        messages=[
            {
                "role": "user",
                "content": """
                Extract product info from this text:
                
                The MacBook Pro 16-inch is a high-performance laptop in the 
                Electronics category, priced at $2499. It features an M3 Max chip, 
                36GB RAM, 1TB SSD storage, and a Liquid Retina XDR display. 
                Currently in stock.
                """
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "Product",
                "schema": schema
            }
        }
    )
    
    product = Product.model_validate_json(
        completion.choices[0].message.content
    )
    
    print(f"📦 Product: {product.name}")
    print(f"📁 Category: {product.category}")
    print(f"💰 Price: ${product.price}")
    print(f"✨ Features:")
    for feature in product.features:
        print(f"   - {feature}")
    print(f"📦 In Stock: {'Yes' if product.in_stock else 'No'}")


def example_4_function_calling():
    """Example 4: Function calling with structured outputs"""
    print("\n" + "="*60)
    print("Example 4: Function Calling")
    print("="*60)
    
    schema = FunctionCall.model_json_schema()
    
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct",
        messages=[
            {
                "role": "user",
                "content": """
                I need to send an email to sarah@example.com with the subject
                "Project Update" and the body "The project is on track for Q1 delivery."
                
                Extract the function call with arguments.
                """
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "FunctionCall",
                "schema": schema
            }
        }
    )
    
    function_call = FunctionCall.model_validate_json(
        completion.choices[0].message.content
    )
    
    print(f"🔧 Function: {function_call.function_name}")
    print(f"📋 Arguments:")
    for key, value in function_call.arguments.items():
        print(f"   {key}: {value}")


def example_5_batch_extraction():
    """Example 5: Extract multiple items with array schema"""
    print("\n" + "="*60)
    print("Example 5: Batch Data Extraction")
    print("="*60)
    
    class Item(BaseModel):
        title: str
        category: str
        price: float
    
    class ItemList(BaseModel):
        items: List[Item]
    
    schema = ItemList.model_json_schema()
    
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct",
        messages=[
            {
                "role": "user",
                "content": """
                Extract all items from this receipt:
                
                - The Hobbit (Book) - $14.99
                - Wireless Mouse (Electronics) - $29.99
                - Coffee Mug (Kitchen) - $12.99
                - Python Textbook (Book) - $59.99
                """
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ItemList",
                "schema": schema
            }
        }
    )
    
    items = ItemList.model_validate_json(
        completion.choices[0].message.content
    )
    
    print(f"📦 Found {len(items.items)} items:")
    total = 0
    for item in items.items:
        print(f"   {item.title} ({item.category}) - ${item.price}")
        total += item.price
    print(f"💰 Total: ${total:.2f}")


def main():
    print("🌱 Model Garden Structured Outputs - OpenAI Client Examples")
    print("=" * 60)
    print("Using OpenAI Python client with Model Garden backend")
    
    try:
        # Run all examples
        example_1_generic_json()
        example_2_specific_schema()
        example_3_product_extraction()
        example_4_function_calling()
        example_5_batch_extraction()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. Model Garden API is running: model-garden serve --model <model>")
        print("2. OpenAI client is installed: pip install openai")
        print("3. A model is loaded in Model Garden")


if __name__ == "__main__":
    main()
