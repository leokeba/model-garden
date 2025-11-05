#!/usr/bin/env python3
"""Test specific version combinations for Qwen2.5-VL-72B-Instruct with unsloth."""

import os
import sys
import subprocess
import gc

# Set HF_HOME BEFORE importing
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

def install_versions(tf_ver, unsloth_ver, bitsandbytes_ver=None):
    """Install specific versions."""
    packages = [f"transformers=={tf_ver}", f"unsloth=={unsloth_ver}"]
    if bitsandbytes_ver:
        packages.append(f"bitsandbytes=={bitsandbytes_ver}")
    
    print(f"\nInstalling: {', '.join(packages)}")
    
    try:
        result = subprocess.run(
            ["uv", "pip", "install", "--quiet"] + packages,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            print(f"❌ Installation failed")
            return False
        
        # Force module reload
        for mod in list(sys.modules.keys()):
            if any(x in mod for x in ['unsloth', 'transformers', 'qwen', 'bitsandbytes']):
                try:
                    del sys.modules[mod]
                except:
                    pass
        
        print("✅ Installed")
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)[:100]}")
        return False

def test_loading(model_name, **kwargs):
    """Test loading with FastVisionModel."""
    try:
        # Import fresh
        from unsloth import FastVisionModel
        import torch
        
        print(f"Testing: {model_name} with {kwargs}")
        
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            **kwargs
        )
        
        print(f"✅ SUCCESS! Model loaded successfully")
        del model, tokenizer
        cleanup()
        return True
    except Exception as e:
        error = str(e)[:200]
        print(f"❌ FAILED: {error}")
        cleanup()
        return False

def main():
    print("="*80)
    print("Testing specific version combinations for Qwen2.5-VL-72B")
    print("="*80)
    
    # Version combinations to test based on when Qwen2.5-VL was released
    # Qwen2.5-VL was released around September-October 2024
    test_combos = [
        # Very recent - November/December 2024
        ("4.47.0", "2024.12", "0.45.0"),
        ("4.46.3", "2024.12", "0.45.0"),
        ("4.46.2", "2024.12", "0.44.1"),
        ("4.46.1", "2024.12", "0.44.1"),
        ("4.46.0", "2024.12", "0.44.0"),
        
        # November 2024
        ("4.45.2", "2024.11", "0.44.1"),
        ("4.45.1", "2024.11", "0.44.0"),
        ("4.45.0", "2024.11", "0.44.0"),
        ("4.44.2", "2024.11", "0.43.3"),
        ("4.44.1", "2024.11", "0.43.3"),
        ("4.44.0", "2024.11", "0.43.2"),
        
        # October 2024 (around Qwen2.5-VL release)
        ("4.45.0", "2024.10", "0.43.3"),
        ("4.44.0", "2024.10", "0.43.2"),
        ("4.43.4", "2024.10", "0.43.1"),
        ("4.43.3", "2024.10", "0.43.1"),
        ("4.43.2", "2024.10", "0.43.0"),
        ("4.43.1", "2024.10", "0.43.0"),
        ("4.43.0", "2024.10", "0.43.0"),
        
        # September 2024
        ("4.42.4", "2024.9", "0.43.0"),
        ("4.42.3", "2024.9", "0.43.0"),
        ("4.42.0", "2024.9", "0.42.0"),
        ("4.41.2", "2024.9", "0.42.0"),
        ("4.41.1", "2024.9", "0.42.0"),
    ]
    
    models_to_test = [
        "unsloth/Qwen2.5-VL-72B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-VL-72B-Instruct-unsloth-bnb-4bit",
    ]
    
    for tf_ver, unsloth_ver, bnb_ver in test_combos:
        print(f"\n{'='*80}")
        print(f"Combo: transformers={tf_ver}, unsloth={unsloth_ver}, bitsandbytes={bnb_ver}")
        print(f"{'='*80}")
        
        if not install_versions(tf_ver, unsloth_ver, bnb_ver):
            continue
        
        for model in models_to_test:
            if test_loading(model, load_in_4bit=True):
                print(f"\n{'#'*80}")
                print(f"🎉 FOUND WORKING COMBINATION!")
                print(f"{'#'*80}")
                print(f"\nWorking versions:")
                print(f"  transformers=={tf_ver}")
                print(f"  unsloth=={unsloth_ver}")
                print(f"  bitsandbytes=={bnb_ver}")
                print(f"  model: {model}")
                print(f"\nUpdate pyproject.toml:")
                print(f'  "transformers=={tf_ver}",')
                print(f'  "unsloth=={unsloth_ver}",')
                print(f'  "bitsandbytes=={bnb_ver}",')
                return 0
    
    print(f"\n{'='*80}")
    print("❌ No working combination found with unsloth")
    print(f"{'='*80}")
    print("\nThe 72B model loads fine with transformers directly.")
    print("The issue appears to be with unsloth's quantized versions.")
    print("\nOptions:")
    print("1. Use transformers directly (no unsloth)")
    print("2. Report this issue to unsloth GitHub")
    print("3. Use a smaller model (7B or 3B) which work fine with unsloth")
    return 1

if __name__ == "__main__":
    sys.exit(main())
