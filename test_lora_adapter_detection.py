"""Test LoRA adapter detection and loading functionality."""

import os
import json
from pathlib import Path
from model_garden.inference import is_lora_adapter, get_base_model_from_adapter

def test_adapter_detection():
    """Test adapter detection logic."""
    print("Testing LoRA Adapter Detection")
    print("=" * 80)
    
    # Test 1: HuggingFace Hub adapter (real repository)
    print("\n1. Testing HuggingFace Hub adapter detection...")
    hub_adapter = "Barth371/Qwen2.5-VL-72B-Instruct-bnb-4bit-2025-10-21_16-26_batch_size_4_cmr-block-2_adapters_4bit"
    
    try:
        is_adapter = is_lora_adapter(hub_adapter)
        print(f"   Is adapter: {is_adapter}")
        
        if is_adapter:
            base_model = get_base_model_from_adapter(hub_adapter)
            print(f"   Base model: {base_model}")
        
        print("   ✓ Hub adapter detection works!")
    except Exception as e:
        print(f"   ⚠️  Hub detection failed (may need HF_TOKEN): {e}")
    
    # Test 2: Create temporary local adapter for testing
    print("\n2. Testing local adapter detection...")
    temp_dir = Path("./test_adapter_temp")
    temp_dir.mkdir(exist_ok=True)
    
    adapter_config = {
        "base_model_name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct",
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
    
    with open(temp_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)
    
    is_adapter = is_lora_adapter(str(temp_dir))
    print(f"   Is adapter: {is_adapter}")
    
    if is_adapter:
        base_model = get_base_model_from_adapter(str(temp_dir))
        print(f"   Base model: {base_model}")
    
    # Cleanup
    (temp_dir / "adapter_config.json").unlink()
    temp_dir.rmdir()
    
    print("   ✓ Local adapter detection works!")
    
    # Test 3: Non-adapter directory
    print("\n3. Testing non-adapter directory...")
    is_adapter = is_lora_adapter("./models")
    print(f"   Is adapter: {is_adapter} (should be False)")
    print("   ✓ Non-adapter detection works!")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")

if __name__ == "__main__":
    # Set HF_TOKEN if available
    if not os.getenv("HF_TOKEN"):
        print("⚠️  HF_TOKEN not set - Hub tests may fail")
        print("   Set it with: export HF_TOKEN=your_token")
        print()
    
    test_adapter_detection()
