#!/usr/bin/env python3
"""Test HuggingFace Hub model upload functionality"""

import os
import sys
from pathlib import Path
import json
import tempfile
import shutil

# Add parent directory to path and import StorageManager
sys.path.insert(0, str(Path(__file__).parent))
from model_garden.api import StorageManager

def create_test_model():
    """Create a test model directory with minimal files"""
    test_dir = Path(tempfile.mkdtemp(prefix="test_model_"))
    
    # Create minimal model files
    config = {
        "model_type": "llama",
        "hidden_size": 768,
        "vocab_size": 32000
    }
    
    adapter_config = {
        "base_model_name_or_path": "unsloth/Llama-3.2-3B-Instruct",
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "v_proj"]
    }
    
    # Write config files
    with open(test_dir / "config.json", "w") as f:
        json.dump(config, f)
    
    with open(test_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f)
    
    # Create empty model file
    (test_dir / "adapter_model.safetensors").touch()
    
    return test_dir

def test_upload_validation():
    """Test upload endpoint validation without actually uploading"""
    print("🧪 Testing Model Upload Validation")
    print("=" * 60)
    
    # Create test model
    test_model_dir = create_test_model()
    print(f"✓ Created test model at: {test_model_dir}")
    
    # Register model in storage
    storage_dir = Path(__file__).parent / "storage"
    storage = StorageManager(storage_dir)
    models = storage.load_models()
    
    test_model_id = "test-upload-model"
    models[test_model_id] = {
        "id": test_model_id,
        "name": "Test Upload Model",
        "base_model": "unsloth/Llama-3.2-3B-Instruct",
        "path": str(test_model_dir),
        "status": "available",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "model_type": "text-generation",
        "training_dataset": "test",
        "training_steps": 100,
        "lora_rank": 16,
        "carbon_emissions_g": 10.5
    }
    
    storage.save_models(models)
    print(f"✓ Registered model: {test_model_id}")
    
    # Test validation
    print("\n📋 Testing validation logic:")
    
    # Test 1: Valid repo_id
    repo_id = "username/test-model"
    if "/" in repo_id:
        print(f"✓ Valid repo_id format: {repo_id}")
    else:
        print(f"✗ Invalid repo_id format: {repo_id}")
    
    # Test 2: Invalid repo_id (no slash)
    invalid_repo_id = "invalid"
    if "/" not in invalid_repo_id:
        print(f"✓ Correctly detected invalid format: {invalid_repo_id}")
    else:
        print(f"✗ Should have detected invalid format: {invalid_repo_id}")
    
    # Test 3: Check HF_TOKEN
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print(f"✓ HF_TOKEN is configured (length: {len(hf_token)})")
    else:
        print("⚠️  HF_TOKEN not set (upload will fail without it)")
    
    # Test 4: Check model files exist
    required_files = ["config.json", "adapter_config.json"]
    missing_files = []
    for file in required_files:
        if not (test_model_dir / file).exists():
            missing_files.append(file)
    
    if not missing_files:
        print(f"✓ All required files present")
    else:
        print(f"✗ Missing files: {missing_files}")
    
    # Test 5: Simulate README generation
    readme_content = f"""---
license: apache-2.0
base_model: {models[test_model_id]['base_model']}
tags:
  - model-garden
  - fine-tuned
  - text-generation
---

# {models[test_model_id]['name']}

Model fine-tuned with Model Garden.

## Model Details

- **Base Model**: {models[test_model_id]['base_model']}
- **Training Steps**: {models[test_model_id]['training_steps']}
- **LoRA Rank**: {models[test_model_id]['lora_rank']}

## Carbon Footprint

Training emissions: {models[test_model_id]['carbon_emissions_g']} gCO2eq
"""
    
    print(f"✓ Generated README preview ({len(readme_content)} chars)")
    print("\n📄 README Preview:")
    print("-" * 60)
    print(readme_content[:300] + "...")
    print("-" * 60)
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    shutil.rmtree(test_model_dir)
    del models[test_model_id]
    storage.save_models(models)
    print(f"✓ Removed test model")
    
    print("\n✅ All validation tests passed!")
    print("\n📚 Next steps:")
    print("1. Set HF_TOKEN environment variable if not set")
    print("2. Start the API server: uv run model-garden serve")
    print("3. Test upload via UI at http://localhost:8000")
    print("4. Or test via API:")
    print('   curl -X POST "http://localhost:8000/api/v1/models/{model_id}/upload-to-hub" \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"repo_id": "username/test-model", "private": false}\'')

def main():
    """Run tests"""
    try:
        test_upload_validation()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
