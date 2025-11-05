#!/usr/bin/env python3
"""Comprehensive test script for FastVisionModel loading with various configurations."""

import os
import sys
import subprocess
from typing import Tuple, Optional

# Set HF_HOME BEFORE importing any HF libraries
os.environ['HF_HOME'] = '/scratch/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/huggingface/hub'
os.environ['HF_DATASETS_CACHE'] = '/scratch/huggingface/datasets'

def test_model_loading(model_name: str, load_in_4bit: bool = True) -> Tuple[bool, Optional[str]]:
    """Test loading a specific model."""
    try:
        from unsloth import FastVisionModel
        import torch
        
        print(f"\n{'='*80}")
        print(f"Testing: {model_name}")
        print(f"load_in_4bit: {load_in_4bit}")
        print(f"{'='*80}")
        
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=load_in_4bit,
        )
        
        print(f"✅ SUCCESS!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Tokenizer type: {type(tokenizer).__name__}")
        
        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        return True, None
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ FAILED: {error_msg[:200]}")
        return False, error_msg


def test_version_combination(transformers_ver: str, unsloth_ver: str) -> bool:
    """Test a specific combination of transformers and unsloth versions."""
    print(f"\n{'#'*80}")
    print(f"# Testing version combination:")
    print(f"#   transformers: {transformers_ver}")
    print(f"#   unsloth: {unsloth_ver}")
    print(f"{'#'*80}")
    
    try:
        # Install specific versions
        subprocess.run(
            ["uv", "pip", "install", f"transformers=={transformers_ver}", f"unsloth=={unsloth_ver}"],
            check=True,
            capture_output=True
        )
        
        # Force reload of modules
        if 'unsloth' in sys.modules:
            del sys.modules['unsloth']
        if 'transformers' in sys.modules:
            del sys.modules['transformers']
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to install versions: {e}")
        return False


def main():
    print("="*80)
    print("FastVisionModel Loading Test Suite")
    print("="*80)
    
    # Get current versions
    try:
        import transformers
        import unsloth
        print(f"\nCurrent versions:")
        print(f"  transformers: {transformers.__version__}")
        print(f"  unsloth: {unsloth.__version__}")
    except Exception as e:
        print(f"Could not get current versions: {e}")
    
    # Models to test (in order of preference)
    models_to_test = [
        # Unsloth pre-quantized models (recommended)
        "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
        "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
        
        # Unsloth unsloth-bnb-4bit versions (newer format)
        "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit",
        
        # Original models with 4-bit loading
        "unsloth/Qwen2.5-VL-7B-Instruct",
        "unsloth/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        
        # Qwen3 models (newest)
        "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit",
        "unsloth/Qwen3-VL-4B-Instruct-bnb-4bit",
        "unsloth/Qwen3-VL-2B-Instruct-bnb-4bit",
    ]
    
    print(f"\n{'='*80}")
    print("Phase 1: Testing models with current versions")
    print(f"{'='*80}")
    
    successful_models = []
    failed_models = []
    
    for model_name in models_to_test:
        success, error = test_model_loading(model_name)
        if success:
            successful_models.append(model_name)
            print(f"\n🎉 Found working model: {model_name}")
            break  # Stop at first success
        else:
            failed_models.append((model_name, error))
    
    if successful_models:
        print(f"\n{'='*80}")
        print("✅ SUCCESS!")
        print(f"{'='*80}")
        print(f"\nWorking model: {successful_models[0]}")
        print(f"\nYou can use this model in your training scripts.")
        return 0
    
    print(f"\n{'='*80}")
    print("Phase 2: Testing different version combinations")
    print(f"{'='*80}")
    
    # Version combinations to test (recent versions that might work)
    version_combos = [
        # Latest stable combinations
        ("4.45.0", "2025.10.8"),
        ("4.44.0", "2025.10.8"),
        ("4.43.0", "2025.10.8"),
        ("4.42.0", "2025.10.8"),
        
        # Try with older unsloth
        ("4.45.0", "2024.11"),
        ("4.44.0", "2024.11"),
        ("4.43.0", "2024.11"),
        
        # Even older combinations that were stable
        ("4.41.0", "2024.10"),
        ("4.40.0", "2024.10"),
    ]
    
    for tf_ver, unsloth_ver in version_combos:
        if test_version_combination(tf_ver, unsloth_ver):
            # Try loading a small model
            for model_name in ["unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit", 
                              "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit"]:
                success, error = test_model_loading(model_name)
                if success:
                    print(f"\n{'='*80}")
                    print("✅ FOUND WORKING COMBINATION!")
                    print(f"{'='*80}")
                    print(f"\nWorking configuration:")
                    print(f"  transformers: {tf_ver}")
                    print(f"  unsloth: {unsloth_ver}")
                    print(f"  model: {model_name}")
                    print(f"\nUpdate your pyproject.toml with these versions:")
                    print(f'  "transformers=={tf_ver}",')
                    print(f'  "unsloth=={unsloth_ver}",')
                    return 0
    
    print(f"\n{'='*80}")
    print("❌ No working combination found")
    print(f"{'='*80}")
    print("\nFailed models:")
    for model, error in failed_models[:5]:
        print(f"  - {model}")
        print(f"    Error: {error[:100]}")
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
