#!/usr/bin/env python3
"""Test dtype detection for vision models.

This script verifies that we correctly detect model parameter dtypes.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure HF cache before imports
from dotenv import load_dotenv
load_dotenv()

HF_HOME = os.getenv('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))
os.environ['HF_HOME'] = HF_HOME
os.environ['TRANSFORMERS_CACHE'] = str(Path(HF_HOME) / 'hub')
os.environ['HF_DATASETS_CACHE'] = str(Path(HF_HOME) / 'datasets')

import torch
from unsloth import FastLanguageModel

def test_dtype_detection():
    """Test dtype detection methods on Qwen2.5-VL-3B."""
    print("Loading Qwen2.5-VL-3B-Instruct...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-VL-3B-Instruct",
        max_seq_length=2048,
        dtype=None,  # Auto-detect (should load in native precision)
        load_in_4bit=False,
        load_in_8bit=False,
    )
    
    print("\n=== Model Dtype Detection Methods ===")
    
    # Method 1: model.dtype (unreliable)
    if hasattr(model, 'dtype'):
        print(f"1. model.dtype: {model.dtype}")
    else:
        print("1. model.dtype: NOT AVAILABLE")
    
    # Method 2: model.model.dtype (for wrapped models)
    if hasattr(model, 'model') and hasattr(model.model, 'dtype'):
        print(f"2. model.model.dtype: {model.model.dtype}")
    else:
        print("2. model.model.dtype: NOT AVAILABLE")
    
    # Method 3: config.torch_dtype
    if hasattr(model, 'config') and hasattr(model.config, 'torch_dtype'):
        print(f"3. model.config.torch_dtype: {model.config.torch_dtype}")
    else:
        print("3. model.config.torch_dtype: NOT AVAILABLE")
    
    # Method 4: Actual parameter dtype (MOST RELIABLE)
    try:
        first_param = next(model.parameters())
        print(f"4. next(model.parameters()).dtype: {first_param.dtype} ✓ MOST RELIABLE")
    except StopIteration:
        print("4. next(model.parameters()).dtype: NO PARAMETERS")
    
    # Check a few more parameters to verify consistency
    print("\n=== Parameter Dtypes (first 5) ===")
    for i, (name, param) in enumerate(model.named_parameters()):
        if i >= 5:
            break
        print(f"  {name}: {param.dtype}")
    
    print("\n=== Conclusion ===")
    param_dtype = next(model.parameters()).dtype
    if param_dtype == torch.bfloat16:
        print("✓ Model parameters are in bfloat16")
        print("✓ Training should use bf16=True, fp16=False")
    elif param_dtype == torch.float16:
        print("✓ Model parameters are in float16")
        print("✓ Training should use fp16=True, bf16=False")
    elif param_dtype == torch.float32:
        print("⚠️  Model parameters are in float32")
        print("⚠️  Training should use fp16=False, bf16=False")
    else:
        print(f"? Model parameters are in {param_dtype}")

if __name__ == "__main__":
    test_dtype_detection()
