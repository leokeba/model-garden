#!/bin/bash
# Cleanup GPU memory by killing all Model Garden and vLLM processes

echo "🧹 Cleaning up GPU memory..."
echo ""

# Kill all model-garden processes
echo "1. Stopping Model Garden processes..."
pkill -9 -f "model-garden serve" 2>/dev/null
pkill -9 -f "model-garden serve-model" 2>/dev/null
sleep 1

# Kill all vLLM engine core processes
echo "2. Stopping vLLM engine cores..."
pkill -9 -f "VLLM::EngineCore" 2>/dev/null
pkill -9 -f "vllm" 2>/dev/null
sleep 1

# Kill any remaining GPU processes
echo "3. Killing remaining GPU processes..."
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
if [ -n "$GPU_PIDS" ]; then
    echo "$GPU_PIDS" | xargs -r kill -9 2>/dev/null
    sleep 2
fi

# Show final GPU status
echo ""
echo "✅ Cleanup complete. Current GPU status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{printf "   GPU Memory: %d MB / %d MB (%.1f%% free)\n", $1, $2, ($2-$1)/$2*100}'

# Check if GPU is fully cleared
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$USED" -lt 100 ]; then
    echo "   🎉 GPU is clear!"
else
    echo "   ⚠️  Warning: $USED MB still in use"
fi
