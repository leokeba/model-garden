#!/usr/bin/env python3
"""Test script to find working configuration for Qwen2.5-VL-72B-Instruct."""

import os
import sys
import subprocess
import gc

# Set HF_HOME BEFORE importing any HF libraries
os.environ['HF_HOME'] = '/scratch/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/huggingface/hub'
os.environ['HF_DATASETS_CACHE'] = '/scratch/huggingface/datasets'

def cleanup():
    """Clean up GPU memory."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def test_unsloth_loading(model_name: str, **kwargs):
    """Test loading with FastVisionModel."""
    print(f"\n{'='*80}")
    print(f"Test: FastVisionModel.from_pretrained")
    print(f"Model: {model_name}")
    print(f"Args: {kwargs}")
    print(f"{'='*80}")
    
    try:
        from unsloth import FastVisionModel
        import torch
        
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            **kwargs
        )
        
        print(f"✅ SUCCESS!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Device: {model.device if hasattr(model, 'device') else 'N/A'}")
        
        del model, tokenizer
        cleanup()
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)[:300]}")
        cleanup()
        return False

def test_transformers_direct(model_name: str, **kwargs):
    """Test loading directly with transformers."""
    print(f"\n{'='*80}")
    print(f"Test: Qwen2_5_VLForConditionalGeneration.from_pretrained")
    print(f"Model: {model_name}")
    print(f"Args: {kwargs}")
    print(f"{'='*80}")
    
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import torch
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            **kwargs
        )
        processor = AutoProcessor.from_pretrained(model_name)
        
        print(f"✅ SUCCESS!")
        print(f"   Model type: {type(model).__name__}")
        
        del model, processor
        cleanup()
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)[:300]}")
        cleanup()
        return False

def test_version_combo(transformers_ver: str, unsloth_ver: str):
    """Install and test a specific version combination."""
    print(f"\n{'#'*80}")
    print(f"# Installing versions: transformers={transformers_ver}, unsloth={unsloth_ver}")
    print(f"{'#'*80}")
    
    try:
        result = subprocess.run(
            ["uv", "pip", "install", "--quiet", 
             f"transformers=={transformers_ver}", 
             f"unsloth=={unsloth_ver}"],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            print(f"❌ Installation failed: {result.stderr[:200]}")
            return False
        
        # Force module reload
        for mod in list(sys.modules.keys()):
            if any(x in mod for x in ['unsloth', 'transformers', 'qwen']):
                del sys.modules[mod]
        
        print("✅ Versions installed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Installation error: {str(e)[:200]}")
        return False

def main():
    print("="*80)
    print("Qwen2.5-VL-72B-Instruct Loading Test Suite")
    print("="*80)
    
    # Get current versions
    try:
        import transformers
        import unsloth
        print(f"\nStarting versions:")
        print(f"  transformers: {transformers.__version__}")
        print(f"  unsloth: {unsloth.__version__}")
    except Exception as e:
        print(f"Could not get versions: {e}")
    
    print("\n" + "="*80)
    print("PHASE 1: Test unsloth pre-quantized 72B models")
    print("="*80)
    
    # Test unsloth's pre-quantized versions
    unsloth_72b_models = [
        "unsloth/Qwen2.5-VL-72B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-VL-72B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-VL-72B-Instruct",
    ]
    
    for model in unsloth_72b_models:
        if test_unsloth_loading(model, load_in_4bit=True):
            print(f"\n🎉 FOUND WORKING MODEL: {model}")
            return 0
    
    print("\n" + "="*80)
    print("PHASE 2: Test original Qwen model with quantization")
    print("="*80)
    
    # Test original model with different quantization approaches
    original_model = "Qwen/Qwen2.5-VL-72B-Instruct"
    
    test_configs = [
        {"load_in_4bit": True},
        {"load_in_8bit": True},
        {"device_map": "auto", "load_in_4bit": True},
        {"device_map": "auto", "load_in_8bit": True},
    ]
    
    for config in test_configs:
        if test_unsloth_loading(original_model, **config):
            print(f"\n🎉 FOUND WORKING CONFIG: {config}")
            return 0
    
    print("\n" + "="*80)
    print("PHASE 3: Test with transformers directly")
    print("="*80)
    
    # Test with transformers directly (no unsloth)
    direct_configs = [
        {"device_map": "auto", "torch_dtype": "auto"},
        {"device_map": "auto", "load_in_4bit": True},
        {"device_map": "auto", "load_in_8bit": True},
    ]
    
    for config in direct_configs:
        if test_transformers_direct(original_model, **config):
            print(f"\n🎉 FOUND WORKING CONFIG (transformers): {config}")
            print("Note: This works without unsloth. You may need to adapt your code.")
            return 0
    
    print("\n" + "="*80)
    print("PHASE 4: Test different version combinations")
    print("="*80)
    
    # Test different version combinations
    version_combos = [
        # Latest combinations
        ("4.46.0", "2025.10.8"),
        ("4.45.2", "2025.10.8"),
        ("4.45.1", "2025.10.8"),
        ("4.45.0", "2025.10.8"),
        ("4.44.2", "2025.10.8"),
        ("4.44.1", "2025.10.8"),
        ("4.44.0", "2025.10.8"),
        
        # Try with slightly older unsloth
        ("4.45.0", "2024.12"),
        ("4.44.0", "2024.12"),
        ("4.45.0", "2024.11"),
        ("4.44.0", "2024.11"),
        
        # Older stable combinations
        ("4.43.4", "2024.11"),
        ("4.43.3", "2024.10"),
        ("4.43.2", "2024.10"),
        ("4.43.1", "2024.10"),
        ("4.43.0", "2024.10"),
        ("4.42.4", "2024.10"),
        ("4.42.3", "2024.10"),
        ("4.42.0", "2024.9"),
        ("4.41.2", "2024.9"),
        ("4.41.0", "2024.9"),
    ]
    
    for tf_ver, unsloth_ver in version_combos:
        print(f"\nTesting combination: transformers={tf_ver}, unsloth={unsloth_ver}")
        
        if not test_version_combo(tf_ver, unsloth_ver):
            continue
        
        # Test with this version combination
        for model in unsloth_72b_models[:2]:  # Test first 2 models only
            if test_unsloth_loading(model, load_in_4bit=True):
                print(f"\n{'='*80}")
                print("🎉 FOUND WORKING COMBINATION!")
                print(f"{'='*80}")
                print(f"\nWorking configuration:")
                print(f"  transformers: {tf_ver}")
                print(f"  unsloth: {unsloth_ver}")
                print(f"  model: {model}")
                print(f"\nUpdate your pyproject.toml:")
                print(f'  "transformers=={tf_ver}",')
                print(f'  "unsloth=={unsloth_ver}",')
                return 0
    
    print("\n" + "="*80)
    print("❌ NO WORKING CONFIGURATION FOUND")
    print("="*80)
    print("\nPossible issues:")
    print("1. The 72B model may require special handling or more GPU memory")
    print("2. There might be compatibility issues with current library versions")
    print("3. The model might not be available or accessible")
    print("\nTry:")
    print("- Check if you have enough GPU memory (72B needs significant VRAM)")
    print("- Verify HuggingFace token access if model is gated")
    print("- Check unsloth GitHub issues for known problems with this model")
    
    return 1

if __name__ == "__main__":
    sys.exit(main())
