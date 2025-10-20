# Memory Leak - True Root Cause Investigation

## What We Know

### Leak Characteristics
- **Rate**: ~600 MB/minute (10 MB/second)
- **Location**: Main training process USS (Unique Set Size)
- **Pattern**: Continuous growth, not just spikes
- **Scope**: System RAM, not GPU VRAM

### What's NOT the Cause
1. ✅ **torch.compile() workers** - They're stable at 8 GB total
2. ✅ **Selective loss collator** - Not even being used in this training run
3. ✅ **PIL image loading** - Images loaded once at startup
4. ✅ **Gradient accumulation** - Cleared every 10 steps
5. ✅ **Python objects in metrics** - Too slow (~few KB/min, not 600 MB/min)

### What MIGHT Be the Cause

#### Most Likely: Processor/Tokenizer Caching

**Evidence:**
- 663 images in dataset
- Processed vision tensors are 10-20 MB each
- If processor caches ALL processed images: 663 × 15 MB = **~10 GB**
- If building cache incrementally at 10 MB/image: matches our leak rate!

**Hypothesis:**
The Qwen2VL processor or image processor has an internal cache that's growing:
- Each batch processes 2 images
- Processed pixel_values tensors are cached
- Cache never evicts old entries
- Over 5 minutes: ~60 batches × 2 images × 10 MB = 1.2 GB ✓ (matches observed)

#### Other Possibilities:
1. **Unsloth's UnslothVisionDataCollator** - May be keeping references
2. **Transformers Trainer** - Known to accumulate state
3. **PyTorch autograd graph** - Despite clearing gradients
4. **HuggingFace datasets library** - Arrow table growing

## The Fix Attempt

### Added Aggressive Cache Clearing

Modified `MemoryCleanupCallback` to:
1. **Clear tokenizer caches** every 10 steps
2. **Clear image processor caches** every 10 steps  
3. **Zero gradients** with `set_to_none=True`
4. **Run garbage collection**

```python
# Clear processor caches
if hasattr(self.processor, 'tokenizer'):
    if hasattr(self.processor.tokenizer, '_cache'):
        self.processor.tokenizer._cache.clear()
        
if hasattr(self.processor, 'image_processor'):
    if hasattr(self.processor.image_processor, '_cache'):
        self.processor.image_processor._cache.clear()
```

## Testing Plan

1. **Restart service** with new cache-clearing code
2. **Start training** via UI
3. **Monitor with precise tracker**:
   ```bash
   uv run python track_leak_precisely.py --find-training 10 3
   ```
4. **Expected result if fix works**:
   - Memory should cycle (6GB → 6.2GB → 6GB)
   - No continuous growth
   - Leak rate < 50 MB/minute (acceptable)

5. **If still leaking**:
   - Need to profile what's actually in memory
   - May need to patch Unsloth or use different collator
   - Could be unfixable bug requiring upstream fix

## Next Steps

### If Fix Works ✅
- Document the solution
- Consider making cache clearing frequency configurable
- Report findings to Unsloth team

### If Fix Doesn't Work ❌
- Use detailed profiler to see WHAT objects are accumulating:
  ```bash
  uv run python profile_memory_detailed.py --pid <PID>
  ```
- May need to:
  - Disable image caching entirely (reload images each batch - slower but stable)
  - Use custom data collator without caching
  - Limit training to shorter runs with periodic restarts
  - Report bug to HuggingFace/Unsloth

## Files Modified
- `model_garden/vision_training.py` - Added processor cache clearing to MemoryCleanupCallback

## Monitoring Commands

```bash
# Find and track precisely
uv run python track_leak_precisely.py $(pgrep -f "model-garden serve") 10 3

# Detailed profiling  
uv run python profile_memory_detailed.py --pid $(pgrep -f "model-garden serve")

# Watch for leak
watch -n 5 'ps aux | grep model-garden | grep -v grep | awk "{print \$6/1024 \" MB\"}"'
```
