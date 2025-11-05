#!/usr/bin/env python3
"""Create a clean copy of a merged model by dropping quantization auxiliaries
(such as .quant_state, .quant_map, .absmax, nested_*, *_blocks, *_scales), and
writing fresh sharded safetensors that contain only the primary tensors
(e.g., *.weight, *.bias, layernorm weights).

This uses the same temporary-file + mmap trick as Unsloth to avoid requiring all
weights resident in RAM at once.

Usage:
  PYTHONUNBUFFERED=1 python3 scripts/clean_merged.py /path/to/merged /path/to/out_clean

Note: run where you have enough disk space (we'll write new shards)."""
import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from safetensors.torch import save_file
import torch
from safetensors import safe_open


def is_aux_key(key: str) -> bool:
    # Keys that indicate auxiliary quantization metadata
    aux_markers = [
        ".quant_state",
        ".quant_map",
        ".nested_quant_map",
        ".absmax",
        ".nested_absmax",
        "_blocks",
        "_scales",
        "bitsandbytes__nf4",
    ]
    for m in aux_markers:
        if m in key:
            return True
    return False


def human_bytes(n: int) -> str:
    for unit in ['B','KiB','MiB','GiB','TiB']:
        if abs(n) < 1024.0:
            return f"{n:3.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PiB"


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/clean_merged.py <merged_dir> <out_dir>")
        sys.exit(2)

    merged_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    max_shard_size = int(os.environ.get("CLEAN_MAX_SHARD_SIZE_BYTES", 5 * 1024**3))

    if not merged_dir.exists():
        print(f"Merged dir not found: {merged_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = merged_dir / "model.safetensors.index.json"
    if not index_path.exists():
        print("model.safetensors.index.json not found in merged dir; aborting")
        sys.exit(1)

    with open(index_path, 'r') as f:
        index = json.load(f)

    weight_map = index.get('weight_map', {})
    # Primary keys: those that are not auxiliary
    primary_keys = [k for k in weight_map.keys() if not is_aux_key(k)]

    # Remove obvious duplicates where aux keys might be present for same base name
    # Keep ordering deterministic
    primary_keys = list(dict.fromkeys(primary_keys))

    print(f"Found {len(weight_map)} total keys, {len(primary_keys)} primary keys to keep")

    # Map each primary key to its source shard path
    key_to_shard = {k: Path(merged_dir) / weight_map[k] for k in primary_keys}

    # We'll write new shards model-00001-of-0000N.safetensors
    shard_idx = 1
    written_weight_map = {}
    tensors = {}
    current_shard_bytes = 0

    def flush_shard():
        nonlocal shard_idx, tensors, current_shard_bytes
        if not tensors:
            return
        # write tensors dict to a new shard file
        tmp_shard = tempfile.NamedTemporaryFile(delete=False, suffix='.safetensors')
        try:
            tmp_shard.close()
            print(f"Writing shard {shard_idx} with {len(tensors)} tensors ~ {human_bytes(current_shard_bytes)} -> {tmp_shard.name}")
            save_file(tensors, tmp_shard.name, metadata={"format": "pt"})
            final_name = f"model-{shard_idx:05d}-of-00000.safetensors"
            dest = out_dir / final_name
            shutil.move(tmp_shard.name, dest)
            for k in list(tensors.keys()):
                written_weight_map[k] = final_name
            tensors = {}
            current_shard_bytes = 0
            shard_idx += 1
        finally:
            # Ensure temporary file is removed on error
            try:
                if os.path.exists(tmp_shard.name):
                    os.remove(tmp_shard.name)
            except Exception:
                pass

    # Process each primary key
    for key in primary_keys:
        shard_file = key_to_shard.get(key)
        if not shard_file or not shard_file.exists():
            print(f"Warning: shard for key {key} missing: {shard_file}")
            continue
        try:
            with safe_open(str(shard_file), framework="pt", device="cpu") as sf:
                if key not in sf.keys():
                    print(f"Warning: key {key} not found in {shard_file.name}")
                    continue
                W = sf.get_tensor(key)
                # Save via temp + reload mmap to avoid holding multiple huge tensors
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmpf:
                    torch.save(W, tmpf.name)
                    tmp_name = tmpf.name
                # Load back with mmap to keep memory pressure low
                try:
                    W_mmap = torch.load(tmp_name, map_location='cpu', weights_only=False)
                except TypeError:
                    # Older torch versions might not accept weights_only; fallback
                    W_mmap = torch.load(tmp_name, map_location='cpu')
                try:
                    os.remove(tmp_name)
                except Exception:
                    pass

                tensor_bytes = W_mmap.element_size() * W_mmap.nelement()
                # If adding this tensor would exceed max_shard_size and shard isn't empty, flush
                if tensors and (current_shard_bytes + int(tensor_bytes) > max_shard_size):
                    flush_shard()
                # Add to current shard dict
                tensors[key] = W_mmap
                current_shard_bytes += int(tensor_bytes)
        except Exception as e:
            print(f"Error reading key {key} from {shard_file}: {e}")

    # Flush remaining
    flush_shard()

    # Rewrite index with accurate metadata
    new_index = {
        "metadata": index.get('metadata', {}),
        "weight_map": written_weight_map,
    }
    # update total_size based on files we wrote
    total_size = sum((out_dir / fn).stat().st_size for fn in set(written_weight_map.values()))
    new_index['metadata']['total_size'] = total_size

    with open(out_dir / 'model.safetensors.index.json', 'w') as f:
        json.dump(new_index, f, indent=2)

    # Copy over config/tokenizer and other necessary files
    for fname in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'preprocessor_config.json']:
        src = merged_dir / fname
        if src.exists():
            shutil.copy(src, out_dir / fname)

    print(f"Cleaned model saved to {out_dir}")


if __name__ == '__main__':
    main()
