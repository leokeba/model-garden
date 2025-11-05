#!/usr/bin/env python3
"""Quick verification that Qwen2.5-VL-72B-Instruct-bnb-4bit loads successfully."""

import os
import sys

# Set HF_HOME BEFORE importing
os.environ['HF_HOME'] = '/scratch/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/huggingface/hub'
os.environ['HF_DATASETS_CACHE'] = '/scratch/huggingface/datasets'

print("="*80)
print("Verifying Qwen2.5-VL-72B-Instruct-bnb-4bit Configuration")
print("="*80)

# Check versions
print("\n📦 Checking package versions...")
try:
    import transformers
    import unsloth
    import bitsandbytes
    
    print(f"✅ transformers: {transformers.__version__} (expected: 4.56.2)")
    print(f"✅ unsloth: {unsloth.__version__} (expected: 2025.10.8)")
    print(f"✅ bitsandbytes: {bitsandbytes.__version__} (expected: 0.48.1)")
    
    assert transformers.__version__ == "4.56.2", "transformers version mismatch!"
    assert unsloth.__version__ == "2025.10.8", "unsloth version mismatch!"
    assert bitsandbytes.__version__ == "0.48.1", "bitsandbytes version mismatch!"
    
except Exception as e:
    print(f"❌ Version check failed: {e}")
    sys.exit(1)

# Try loading the model
print("\n🚀 Testing model loading...")
try:
    from unsloth import FastVisionModel
    import torch
    
    model_name = "unsloth/Qwen2.5-VL-72B-Instruct-bnb-4bit"
    print(f"Model: {model_name}")
    print("Loading... (this may take a minute)")
    
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=True,
    )
    
    print("✅ Model loaded successfully!")
    
    # Check memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"\n💾 GPU Memory:")
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved: {reserved:.2f} GB")
    
    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print("✅ SUCCESS! Configuration verified")
    print("="*80)
    print("\nThe 72B model is ready to use for training and inference!")
    print("Model registry has been updated with: unsloth/Qwen2.5-VL-72B-Instruct-bnb-4bit")
    
except Exception as e:
    print(f"\n❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
