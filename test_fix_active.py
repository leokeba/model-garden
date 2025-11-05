#!/usr/bin/env python3
"""
Test if FixedSFTTrainer is actually being used during training
"""
import os
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'

import sys
sys.path.insert(0, '/root/model-garden')

# Test 1: Check if FixedSFTTrainer has our override
from model_garden.vision_training import FixedSFTTrainer
import inspect

print("="*80)
print("Test 1: Checking FixedSFTTrainer.prediction_step source")
print("="*80)

source = inspect.getsource(FixedSFTTrainer.prediction_step)
if "num_items_in_batch" in source and "KEY FIX" in source:
    print("✓ FixedSFTTrainer.prediction_step contains our fix!")
    print("\nKey lines found:")
    for line in source.split('\n'):
        if 'num_items_in_batch' in line or 'KEY FIX' in line:
            print(f"  {line.strip()}")
else:
    print("✗ FixedSFTTrainer.prediction_step DOES NOT contain our fix!")
    print("\nSource preview:")
    print(source[:500])

# Test 2: Check method resolution order
print("\n" + "="*80)
print("Test 2: Method Resolution Order")
print("="*80)
print(f"FixedSFTTrainer.__mro__ = {FixedSFTTrainer.__mro__}")

# Test 3: Verify it's different from parent
from trl.trainer.sft_trainer import SFTTrainer
parent_prediction_step = SFTTrainer.prediction_step
child_prediction_step = FixedSFTTrainer.prediction_step

print("\n" + "="*80)
print("Test 3: Override verification")
print("="*80)
print(f"Parent method: {parent_prediction_step}")
print(f"Child method: {child_prediction_step}")
print(f"Are they different? {parent_prediction_step is not child_prediction_step}")

if parent_prediction_step is not child_prediction_step:
    print("✓ Method is properly overridden!")
else:
    print("✗ Method is NOT overridden - using parent implementation!")
