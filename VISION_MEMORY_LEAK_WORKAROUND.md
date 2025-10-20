# Vision Training Memory Leak Workaround

## Problem
Vision-language model fine-tuning with Unsloth has a severe memory leak (~1-2 GB/minute) that will cause OOM crashes on systems with <64GB RAM.

## Root Cause
The leak appears to be in Unsloth's `UnslothVisionDataCollator`. We cannot fix this directly as it's in the Unsloth library.

## Workaround Implemented

### 1. Subprocess Isolation
Training now runs in a separate subprocess that is killed after completion. This prevents the leak from affecting the main API server.

**Location**: `model_garden/api.py` line ~1538

### 2. Memory Limit with Graceful Stop
Training now monitors its own memory usage and stops gracefully before OOM.

**Location**: `model_garden/vision_training.py` - `MemoryCleanupCallback.on_optimizer_step()`

**Default limit**: 28 GB (out of 32 GB total)

**Behavior**:
- Checks memory after each optimizer step
- If memory exceeds limit, stops training gracefully
- Saves progress (checkpoints already saved every N steps)
- Shows clear error message to user

### 3. Configuration
You can adjust the memory limit when creating a training job:

```python
# In model_garden/vision_training.py, modify the callback initialization:
cleanup_callback = MemoryCleanupCallback(
    cleanup_every_n_steps=10,
    processor=self.processor,
    max_memory_gb=28.0  # Adjust this value
)
```

## Recommendations for Users

### For Small Datasets (<200 examples)
- Should complete training before hitting memory limit
- No action needed

### For Medium Datasets (200-500 examples)
- May hit memory limit partway through
- Use checkpointing: `save_steps=50` or similar
- Training will stop gracefully, resume from last checkpoint

### For Large Datasets (>500 examples)
- WILL hit memory limit
- Options:
  1. **Split dataset**: Train in multiple smaller batches
  2. **Use text-only models**: No memory leak with text training
  3. **Use system with more RAM**: 64GB+ recommended
  4. **Wait for Unsloth fix**: Report issue to Unsloth project

## Monitoring

Use the included monitoring script to track memory during training:

```bash
# Get the training subprocess PID (it will be a child of the main server)
pgrep -f "python.*--run-training-job" 

# Monitor it
uv run python monitor_training_memory.py <PID>
```

## Reporting to Unsloth

If you want to help get this fixed upstream:

1. File an issue at: https://github.com/unslothai/unsloth/issues
2. Title: "Memory leak in UnslothVisionDataCollator during vision-language training"
3. Include:
   - Memory growth rate (~1-2 GB/minute)
   - Model: Qwen2.5-VL-3B-Instruct
   - Dataset size and format
   - Reference this analysis document

## Files Modified

- `model_garden/api.py` - Subprocess execution
- `model_garden/vision_training.py` - Memory monitoring callback  
- `model_garden/memory_management.py` - Enhanced cleanup (doesn't fix leak but helps)
- `MEMORY_LEAK_ANALYSIS.md` - Detailed analysis
- `VISION_MEMORY_LEAK_WORKAROUND.md` - This file
