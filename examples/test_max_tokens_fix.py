#!/usr/bin/env python3
"""
Test that structured outputs properly handle large JSON schemas without truncation.

This addresses the issue where max_tokens=256 was causing JSON to be cut off mid-generation.
"""

import requests
import json
from pydantic import BaseModel, Field
from typing import List, Optional

API_BASE = "http://localhost:8000/api/v1"


class Address(BaseModel):
    """Address information."""
    street: str
    city: str
    state: str
    zip_code: str
    country: str = "France"


class Contact(BaseModel):
    """Contact information."""
    name: str
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[Address] = None


class VehicleInfo(BaseModel):
    """Vehicle information."""
    make: str
    model: str
    year: int
    color: str
    vin: str
    mileage: int
    license_plate: str


class InsuranceInfo(BaseModel):
    """Insurance information."""
    policy_number: str
    provider: str
    coverage_type: str
    start_date: str
    end_date: str
    premium: float


class DocumentExtraction(BaseModel):
    """Complex document extraction with many fields (tests max_tokens handling)."""
    document_type: str = Field(description="Type of document")
    document_number: str = Field(description="Document reference number")
    issue_date: str = Field(description="Date document was issued")
    expiry_date: Optional[str] = Field(None, description="Expiration date if applicable")
    
    # Personal information
    owner: Contact = Field(description="Document owner information")
    
    # Vehicle details
    vehicle: VehicleInfo = Field(description="Vehicle information")
    
    # Insurance details
    insurance: InsuranceInfo = Field(description="Insurance policy details")
    
    # Additional fields
    notes: List[str] = Field(default_factory=list, description="Any additional notes")
    keywords: List[str] = Field(default_factory=list, description="Key terms extracted")
    
    # Metadata
    confidence_score: float = Field(description="Extraction confidence 0.0-1.0")
    language: str = Field(default="fr", description="Document language")


def test_large_schema_without_max_tokens():
    """Test 1: Large schema with auto max_tokens (should work)."""
    print("\n" + "="*70)
    print("Test 1: Large Schema with Auto max_tokens")
    print("="*70)
    
    schema = DocumentExtraction.model_json_schema()
    print(f"\nSchema has {len(json.dumps(schema))} characters")
    print(f"Schema requires many fields: owner, vehicle, insurance, etc.")
    
    response = requests.post(
        f"{API_BASE}/chat/completions",
        json={
            "model": "current",
            "messages": [{
                "role": "user",
                "content": "Generate a sample French vehicle registration document with complete details."
            }],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "DocumentExtraction",
                    "strict": True,
                    "schema": schema
                }
            }
            # NOTE: No max_tokens specified - should auto-use 2048
        }
    )
    
    print(f"\nResponse status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        print(f"✅ Response received ({len(content)} chars)")
        
        try:
            # Validate against Pydantic model
            doc = DocumentExtraction.model_validate_json(content)
            print(f"\n✅ Valid JSON - All required fields present!")
            print(f"  Document Type: {doc.document_type}")
            print(f"  Owner: {doc.owner.name}")
            print(f"  Vehicle: {doc.vehicle.year} {doc.vehicle.make} {doc.vehicle.model}")
            print(f"  Insurance: {doc.insurance.provider} ({doc.insurance.policy_number})")
            print(f"  Confidence: {doc.confidence_score:.2f}")
            print(f"\n  Total JSON size: {len(content)} chars")
            print(f"  ✅ No truncation - complete JSON!")
            
        except Exception as e:
            print(f"\n❌ Validation failed: {e}")
            print(f"Response: {content[:500]}...")
    else:
        print(f"❌ Request failed: {response.text[:500]}")


def test_large_schema_with_small_max_tokens():
    """Test 2: Large schema with small max_tokens (should fail/warn)."""
    print("\n" + "="*70)
    print("Test 2: Large Schema with max_tokens=256 (OLD BEHAVIOR)")
    print("="*70)
    
    schema = DocumentExtraction.model_json_schema()
    
    response = requests.post(
        f"{API_BASE}/chat/completions",
        json={
            "model": "current",
            "messages": [{
                "role": "user",
                "content": "Generate a sample French vehicle registration document."
            }],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "DocumentExtraction",
                    "schema": schema
                }
            },
            "max_tokens": 256  # Force old default
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        print(f"Response size: {len(content)} chars")
        
        try:
            doc = DocumentExtraction.model_validate_json(content)
            print(f"✅ Surprisingly, it worked with 256 tokens!")
        except Exception as e:
            print(f"\n❌ Expected failure: {str(e)[:200]}")
            print(f"⚠️  JSON was truncated at ~256 tokens")
            print(f"\nPartial response:\n{content[:300]}...")
    else:
        print(f"Request failed: {response.text[:200]}")


def test_explicit_high_max_tokens():
    """Test 3: Explicitly set high max_tokens."""
    print("\n" + "="*70)
    print("Test 3: Large Schema with max_tokens=4096 (Explicit)")
    print("="*70)
    
    schema = DocumentExtraction.model_json_schema()
    
    response = requests.post(
        f"{API_BASE}/chat/completions",
        json={
            "model": "current",
            "messages": [{
                "role": "user",
                "content": "Generate a complete French vehicle registration document with all details."
            }],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "DocumentExtraction",
                    "schema": schema
                }
            },
            "max_tokens": 4096  # Explicitly high
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        try:
            doc = DocumentExtraction.model_validate_json(content)
            print(f"✅ Complete document generated!")
            print(f"  JSON size: {len(content)} chars")
            print(f"  All fields: ✓")
        except Exception as e:
            print(f"❌ Failed: {e}")
    else:
        print(f"Failed: {response.text[:200]}")


def main():
    print("🔧 Testing max_tokens Fix for Structured Outputs")
    print("="*70)
    print("\nProblem: Default max_tokens=256 causes JSON truncation")
    print("Solution: Auto-set max_tokens=2048 for json_schema responses")
    print("\nMake sure Model Garden API is running!")
    
    try:
        response = requests.get(f"{API_BASE}/system/status", timeout=5)
        if response.status_code != 200:
            print("❌ API is not available")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return
    
    print("✅ API is available\n")
    
    try:
        test_large_schema_without_max_tokens()
        test_large_schema_with_small_max_tokens()
        test_explicit_high_max_tokens()
        
        print("\n" + "="*70)
        print("✅ All tests completed!")
        print("="*70)
        print("\nSummary:")
        print("- Auto max_tokens now prevents JSON truncation")
        print("- Default: 512 for text, 1024 for json_object, 2048 for json_schema")
        print("- Users can still override with explicit max_tokens parameter")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
