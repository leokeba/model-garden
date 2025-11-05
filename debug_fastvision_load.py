#!/usr/bin/env python3
"""Debug script to test FastVisionModel loading for Qwen2-VL-72B-Instruct-bnb-4bit."""

import os
import sys

# Set HF_HOME to /scratch/huggingface BEFORE importing any HF libraries
os.environ['HF_HOME'] = '/scratch/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/huggingface/hub'
os.environ['HF_DATASETS_CACHE'] = '/scratch/huggingface/datasets'

print(f"HF_HOME: {os.environ['HF_HOME']}")
print(f"TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")
print()

# Import unsloth FIRST
from unsloth import FastVisionModel
import torch

print("=" * 80)
print("Testing FastVisionModel with Unsloth pre-quantized model")
print("=" * 80)
print()

model_name = "Qwen/Qwen2.VL-72B-Instruct"
print(f"Model: {model_name}")
print(f"Strategy: Load Unsloth's pre-quantized version directly")
print()

try:
    print("Loading model with FastVisionModel...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=True,
    )
    
    print()
    print("✅ SUCCESS! Model loaded")
    print(f"Model type: {type(model)}")
    print(f"Tokenizer type: {type(tokenizer)}")
    
except Exception as e:
    print()
    print("❌ FAILED!")
    print(f"Error: {e}")
    
    import traceback
    traceback.print_exc()
    sys.exit(1)
