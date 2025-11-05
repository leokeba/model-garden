#!/usr/bin/env python3
"""
Test the fixed SFTTrainer to verify eval loss is now correctly normalized
"""
import os
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'

# Start training with evaluation
print("Testing fixed SFTTrainer with evaluation...")
print("="*80)

import subprocess
result = subprocess.run([
    "uv", "run", "model-garden", "train-vision",
    "--base-model", "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
    "--dataset", "/root/model-garden/data/vision_test_dataset.jsonl",
    "--output-dir", "/root/model-garden/test_fixed_trainer",
    "--max-steps", "10",
    "--eval-steps", "5",
    "--save-steps", "100",  # Don't save checkpoints
    "--per-device-train-batch-size", "1",
    "--per-device-eval-batch-size", "1",
    "--learning-rate", "2e-4",
    "--lora-r", "8",
    "--logging-steps", "1",
], capture_output=False, text=True)

print("\n" + "="*80)
if result.returncode == 0:
    print("✓ Training completed successfully!")
    print("\nCheck the training logs above:")
    print("  - Training loss should be similar to eval loss (not 3-4x different)")
    print("  - Both should converge similarly over time")
else:
    print("✗ Training failed!")
