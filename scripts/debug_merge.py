#!/usr/bin/env python3
"""Debug runner to attempt merging a vision LoRA adapter in-process.

Sets MODEL_GARDEN_DEBUG_RUN_MERGE_IN_MAIN=1 then calls InferenceService.load_model()
so the merge runs in the main process and prints full tracebacks/logs.

Usage: PYTHONUNBUFFERED=1 python3 scripts/debug_merge.py
"""
import asyncio
import os
import sys
import traceback

# Run in debug in-process merge mode
os.environ['MODEL_GARDEN_DEBUG_RUN_MERGE_IN_MAIN'] = '1'
# Optional: set HF_HOME to avoid filling root
os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))

print(f"MODEL_GARDEN_DEBUG_RUN_MERGE_IN_MAIN={os.environ.get('MODEL_GARDEN_DEBUG_RUN_MERGE_IN_MAIN')}")
print(f"HF_HOME={os.environ.get('HF_HOME')}")

# Adapter to test — change if you want to test a different adapter
ADAPTER = os.environ.get('DEBUG_TEST_ADAPTER', 'terra-cognita-ai/qwen-72b-cmr-blocks')

print(f"Attempting to load adapter: {ADAPTER}")

# Add repo root to path (should already be importable when run from repo)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from model_garden.inference import InferenceService
except Exception as e:
    print("Failed to import InferenceService:", e)
    traceback.print_exc()
    sys.exit(2)

svc = InferenceService(model_path=ADAPTER, gpu_memory_utilization=0.0)

async def main():
    try:
        await svc.load_model()
        print("LOAD_SUCCESS: model loaded/merged without raising an exception")
    except Exception as e:
        print("LOAD_FAILED:", e)
        traceback.print_exc()
        # Exit non-zero so callers can detect failure
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())
