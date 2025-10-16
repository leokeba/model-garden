#!/bin/bash
# Start Model Garden server with vision model autoload

echo "🚀 Starting Model Garden with Qwen2.5-VL-3B-Instruct..."
echo ""

MODEL_GARDEN_AUTOLOAD_MODEL="Qwen/Qwen2.5-VL-3B-Instruct" uv run model-garden serve
