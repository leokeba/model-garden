#!/usr/bin/env python3
"""
Load a merged model directory and attempt to re-save it using Unsloth's
`save_pretrained_merged(..., save_method="merged_16bit")` helper.

Usage:
  HF_HOME=/scratch/hf_cache python scripts/re_save_merged.py /scratch/hf_cache/temp_merges/model-garden-merged-1761907383 /scratch/hf_cache/temp_merges/model-garden-merged-1761907383-16bit

This script forces loading on CPU to avoid GPU OOM and prints full tracebacks.
"""
import os
import sys
import traceback

from transformers import AutoModelForVision2Seq


def main():
    if len(sys.argv) < 3:
        print("Usage: re_save_merged.py <merged_dir> <out_dir>")
        sys.exit(2)

    merged_dir = sys.argv[1]
    out_dir = sys.argv[2]

    print(f"Loading merged model from: {merged_dir}")
    try:
        # Load on CPU to avoid allocating GPU memory
        model = AutoModelForVision2Seq.from_pretrained(
            merged_dir,
            device_map="cpu",
            trust_remote_code=True,
        )
        print("Model loaded on CPU. Attempting Unsloth save_pretrained_merged(..., save_method=\"merged_16bit\")")
        # Some versions of Unsloth patch the model with save_pretrained_merged.
        # Try calling it if present, otherwise raise.
        save_fn = getattr(model, "save_pretrained_merged", None)
        if save_fn is None:
            raise RuntimeError("Model instance has no attribute save_pretrained_merged (is Unsloth installed?)")

        os.makedirs(out_dir, exist_ok=True)
        save_fn(out_dir, save_method="merged_16bit")
        print(f"Unsloth save_pretrained_merged succeeded -> {out_dir}")

    except Exception:
        print("Exception during Unsloth re-save:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
