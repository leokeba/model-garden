#!/bin/bash
# Start Model Garden server with vision model autoload

echo "🚀 Starting Model Garden with Qwen2.5-VL-3B-Instruct..."
echo ""

# Disable torch.compile() to prevent memory-hungry worker processes
export TORCH_COMPILE_DISABLE=1
# Also disable torch._dynamo (compile backend)
export TORCHDYNAMO_DISABLE=1

MODEL_GARDEN_AUTOLOAD_MODEL="Qwen/Qwen2.5-VL-3B-Instruct" uv run model-garden serve
