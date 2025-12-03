#!/usr/bin/env python3
"""Run InferenceService for a model path with configurable gpu_memory_utilization.

Usage:
  python scripts/run_load_with_util.py <model_path> <gpu_memory_utilization> [max_model_len]
"""
import asyncio
import sys

if len(sys.argv) < 3:
    print("Usage: python scripts/run_load_with_util.py <model_path> <gpu_memory_utilization> [max_model_len]")
    sys.exit(2)

MODEL_PATH = sys.argv[1]
GPU_UTIL = float(sys.argv[2])
MAX_MODEL_LEN = int(sys.argv[3]) if len(sys.argv) > 3 else None

print(f"Loading model: {MODEL_PATH} with gpu_memory_utilization={GPU_UTIL} max_model_len={MAX_MODEL_LEN}")

from model_garden.inference import InferenceService


async def main():
    svc = InferenceService(model_path=MODEL_PATH, gpu_memory_utilization=GPU_UTIL, max_model_len=MAX_MODEL_LEN)
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
