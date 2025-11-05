#!/usr/bin/env python3
"""Test loader that instantiates InferenceService and attempts to load a model.

Usage:
  PYTHONUNBUFFERED=1 python3 scripts/test_load_cleaned.py /path/to/model

Exits with non-zero on failure.
"""
import sys
import asyncio

from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python scripts/test_load_cleaned.py <model_path>")
    sys.exit(2)

MODEL_PATH = sys.argv[1]
print(f"Attempting to load model: {MODEL_PATH}")

from model_garden.inference import InferenceService

async def main():
    svc = InferenceService(model_path=MODEL_PATH, gpu_memory_utilization=0.2)
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
