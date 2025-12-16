# Axolotl Backend (CLI)

Axolotl is an optional training backend that Model Garden can drive via the Axolotl CLI. It is useful when you want Axolotl's training stack (DeepSpeed/FSDP options, multipacking, etc.) while keeping Model Garden's CLI/API, job management, and carbon tracking.

## Installation

```bash
# Install optional extra
uv pip install 'model-garden[axolotl]'
# Or directly
uv pip install axolotl
```

## Usage

- Text fine-tuning:
  ```bash
  uv run model-garden train \
    --backend axolotl \
    --base-model <model> \
    --dataset ./data/sample.jsonl \
    --output-dir ./models/my-axolotl-model
  ```

- Vision-language fine-tuning (OpenChat-style JSONL is auto-built for Axolotl):
  ```bash
  uv run model-garden train-vision \
    --backend axolotl \
    --base-model Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset ./data/vision_dataset.jsonl \
    --output-dir ./models/my-axolotl-vlm
  ```

## Dataset handling

- Local files: pass `--dataset` as usual; we hand the path to Axolotl.
- Hugging Face Hub datasets: pass `--from-hub`; we download then materialize a temporary JSONL for Axolotl.
- Text format: `{"instruction": "...", "input": "...", "output": "..."}` is converted to Axolotl `alpaca` format.
- Vision format: `{"text": "...", "image": "/path/to/img", "response": "..."}` becomes OpenChat messages with `messages` and `images` fields. A `system_message` is prepended when provided.

## Current limitations

- Eval datasets are not yet wired into the Axolotl config; run eval separately after training.
- Selective loss masking is not mapped for Axolotl.
- Axolotl writes checkpoints into `output_dir`; `save_model` is a no-op.
- Ensure the `axolotl` CLI is on PATH; the backend shells out with `python -m axolotl.cli.train`.
- Web UI: once `axolotl` is installed, the backend appears in the backend list; restart the service if you added the dependency while it was running.

## References

- Official docs: https://docs.axolotl.ai/
- GitHub: https://github.com/axolotl-ai-cloud/axolotl
