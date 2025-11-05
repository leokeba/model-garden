#!/usr/bin/env python3
"""Load a model using InferenceService with an optional max_model_len.

Usage:
  python scripts/test_load_good_model.py <model_path_or_hf_id> [max_model_len]

Exits non-zero on failure.
"""
import sys
import asyncio
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python scripts/test_load_good_model.py <model_path_or_hf_id> [max_model_len]")
    sys.exit(2)

MODEL_PATH = sys.argv[1]
MAX_MODEL_LEN = int(sys.argv[2]) if len(sys.argv) > 2 else None
print(f"Attempting to load model: {MODEL_PATH} with max_model_len={MAX_MODEL_LEN}")

from model_garden.inference import InferenceService

async def main():
    svc = InferenceService(model_path=MODEL_PATH, gpu_memory_utilization=0.2, max_model_len=MAX_MODEL_LEN)
    try:
        await svc.load_model()
        print("Model loaded successfully")
        await svc.unload_model()
        print("Model unloaded successfully")
    except Exception as e:
        print(f"Load failed: {e}")
        raise

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception:
        sys.exit(1)
