#!/usr/bin/env python3
"""Test Qwen2.5-VL-72B with different unsloth versions using uv for dependency management."""

import os
import sys
import subprocess
import gc
import time

# Set HF_HOME BEFORE importing
os.environ['HF_HOME'] = '/scratch/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/huggingface/hub'
os.environ['HF_DATASETS_CACHE'] = '/scratch/huggingface/datasets'

def cleanup():
    """Clean up GPU memory."""
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass

def install_versions_with_uv(tf_ver, unsloth_ver, bnb_ver, install_zoo=False):
    """Install specific versions using uv."""
    print(f"\n{'='*80}")
    print(f"Installing with uv:")
    print(f"  transformers=={tf_ver}")
    print(f"  unsloth=={unsloth_ver}")
    print(f"  bitsandbytes=={bnb_ver}")
    if install_zoo:
        print(f"  unsloth-zoo (latest)")
    print(f"{'='*80}")
    
    try:
        # Install packages
        packages = [
            f"transformers=={tf_ver}",
            f"unsloth=={unsloth_ver}",
            f"bitsandbytes=={bnb_ver}",
        ]
        
        if install_zoo:
            packages.append("unsloth-zoo")
        
        result = subprocess.run(
            ["uv", "pip", "install"] + packages,
            capture_output=True,
            text=True,
            timeout=180
        )
        
        if result.returncode != 0:
            print(f"❌ Installation failed:")
            print(result.stderr[:500])
            return False
        
        print("✅ Packages installed successfully")
        
        # Force reload modules
        modules_to_clear = []
        for mod in list(sys.modules.keys()):
            if any(x in mod.lower() for x in ['unsloth', 'transformers', 'qwen', 'bitsandbytes']):
                modules_to_clear.append(mod)
        
        for mod in modules_to_clear:
            try:
                del sys.modules[mod]
            except:
                pass
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"❌ Installation timeout")
        return False
    except Exception as e:
        print(f"❌ Installation error: {str(e)[:200]}")
        return False

def test_loading(model_name, **kwargs):
    """Test loading with FastVisionModel."""
    print(f"\nTesting: {model_name}")
    print(f"Options: {kwargs}")
    
    try:
        # Import fresh
        from unsloth import FastVisionModel
        import torch
        
        print("Loading model...")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            **kwargs
        )
        
        print(f"✅ SUCCESS! Model loaded")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Device: {model.device if hasattr(model, 'device') else 'N/A'}")
        
        # Check memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        del model, tokenizer
        cleanup()
        return True
        
    except Exception as e:
        error = str(e)
        print(f"❌ FAILED: {error[:300]}")
        cleanup()
        return False

def main():
    print("="*80)
    print("Qwen2.5-VL-72B-Instruct Loading Test with UV")
    print("="*80)
    
    # Test current version first
    print("\n" + "="*80)
    print("PHASE 1: Test with current installed versions")
    print("="*80)
    
    try:
        import transformers
        import unsloth
        print(f"\nCurrent versions:")
        print(f"  transformers: {transformers.__version__}")
        print(f"  unsloth: {unsloth.__version__}")
        
        models_to_test = [
            "unsloth/Qwen2.5-VL-72B-Instruct-bnb-4bit",
            "unsloth/Qwen2.5-VL-72B-Instruct-unsloth-bnb-4bit",
        ]
        
        for model in models_to_test:
            if test_loading(model, load_in_4bit=True):
                print(f"\n🎉 SUCCESS with current versions!")
                print(f"Model: {model}")
                return 0
    except Exception as e:
        print(f"Current version test failed: {e}")
    
    print("\n" + "="*80)
    print("PHASE 2: Test specific version combinations")
    print("="*80)
    
    # Based on unsloth releases and Qwen2.5-VL availability
    # Reference: https://github.com/unslothai/unsloth/releases
    version_combos = [
        # Latest stable combinations (2025.10.x - current)
        ("4.46.2", "2025.10.8", "0.45.0", False),
        ("4.46.1", "2025.10.8", "0.44.1", False),
        ("4.46.0", "2025.10.7", "0.44.1", False),
        ("4.45.2", "2025.10.6", "0.44.1", False),
        ("4.45.1", "2025.10.5", "0.44.0", False),
        ("4.45.0", "2025.10.4", "0.44.0", False),
        
        # Try with unsloth-zoo for older versions
        ("4.44.2", "2024.10", "0.43.3", True),
        ("4.44.1", "2024.10", "0.43.3", True),
        ("4.44.0", "2024.10", "0.43.2", True),
        ("4.43.4", "2024.10", "0.43.1", True),
        ("4.43.3", "2024.10", "0.43.1", True),
        ("4.43.2", "2024.10", "0.43.0", True),
        
        # Even older - when Qwen2.5-VL support was added
        ("4.45.0", "2024.9.post4", "0.43.0", True),
        ("4.45.0", "2024.9.post3", "0.43.0", True),
        ("4.44.0", "2024.9.post2", "0.43.0", True),
        ("4.43.0", "2024.9.post1", "0.42.0", True),
        ("4.43.0", "2024.9", "0.42.0", True),
    ]
    
    models_to_test = [
        "unsloth/Qwen2.5-VL-72B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-VL-72B-Instruct-unsloth-bnb-4bit",
        "Qwen/Qwen2.VL-72B-Instruct",  # Original model without unsloth prefix
    ]
    
    for tf_ver, unsloth_ver, bnb_ver, need_zoo in version_combos:
        if not install_versions_with_uv(tf_ver, unsloth_ver, bnb_ver, need_zoo):
            continue
        
        # Give system a moment to stabilize
        time.sleep(2)
        
        for model in models_to_test:
            if test_loading(model, load_in_4bit=True):
                print(f"\n{'#'*80}")
                print(f"🎉 FOUND WORKING COMBINATION!")
                print(f"{'#'*80}")
                print(f"\nWorking configuration:")
                print(f"  transformers=={tf_ver}")
                print(f"  unsloth=={unsloth_ver}")
                print(f"  bitsandbytes=={bnb_ver}")
                if need_zoo:
                    print(f"  unsloth-zoo (required)")
                print(f"  model: {model}")
                print(f"\nTo install:")
                if need_zoo:
                    print(f'  uv pip install transformers=={tf_ver} unsloth=={unsloth_ver} bitsandbytes=={bnb_ver} unsloth-zoo')
                else:
                    print(f'  uv pip install transformers=={tf_ver} unsloth=={unsloth_ver} bitsandbytes=={bnb_ver}')
                print(f"\nUpdate pyproject.toml:")
                print(f'  "transformers=={tf_ver}",')
                print(f'  "unsloth=={unsloth_ver}",')
                print(f'  "bitsandbytes=={bnb_ver}",')
                if need_zoo:
                    print(f'  "unsloth-zoo",  # Required for this version')
                return 0
        
        # Clear for next iteration
        cleanup()
    
    print("\n" + "="*80)
    print("❌ NO WORKING COMBINATION FOUND")
    print("="*80)
    print("\nDiagnosis:")
    print("The issue appears to be:")
    print("1. Latest unsloth (2025.10.x) has issues with 72B quantized models")
    print("2. Older unsloth versions need unsloth-zoo which may not be compatible")
    print("3. The quantized checkpoint format may be incompatible")
    print("\nRecommendations:")
    print("1. Open an issue on unsloth GitHub about 72B quantized model loading")
    print("2. Check if there's a specific unsloth version for Qwen2.5-VL-72B")
    print("3. Consider using the non-quantized model with tensor parallelism")
    print("4. Use the 7B or 3B models which work fine with current unsloth")
    
    return 1

if __name__ == "__main__":
    sys.exit(main())
