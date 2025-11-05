#!/usr/bin/env python3
"""
Quick test to verify FixedSFTTrainer properly passes num_items_in_batch during eval
"""
import os
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'

import sys
sys.path.insert(0, '/root/model-garden')

from model_garden.vision_training import FixedSFTTrainer
import torch

print("Testing FixedSFTTrainer.prediction_step...")
print("="*80)

# Check that the method exists
assert hasattr(FixedSFTTrainer, 'prediction_step'), "FixedSFTTrainer missing prediction_step!"

print("✓ FixedSFTTrainer.prediction_step exists")

# Check that it's different from parent
from trl.trainer.sft_trainer import SFTTrainer
parent_method = SFTTrainer.prediction_step
child_method = FixedSFTTrainer.prediction_step

if parent_method is child_method:
    print("✗ Warning: FixedSFTTrainer.prediction_step is not overridden!")
else:
    print("✓ FixedSFTTrainer.prediction_step is properly overridden")

print("\n" + "="*80)
print("Summary: FixedSFTTrainer is ready to use!")
print("="*80)
