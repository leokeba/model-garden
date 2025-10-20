# Vision Training Memory Leak Analysis

## Summary
Vision-language model training (Qwen2.5-VL-3B with Unsloth) exhibits catastrophic memory leak of **1.2-2.0 GB/minute** in system RAM (not GPU VRAM).

## Observations

### Memory Growth Pattern
- Training with 663 examples, batch_size=2, gradient_accumulation=4
- Memory grows ~200 MB every few training steps
- Growth is irregular but persistent
- Total growth: ~6-8 GB in first 3 minutes
- **System will OOM in 15-20 minutes** with 32GB RAM

### What We've Ruled Out

1. **NOT the dataset loading** - Would see one big jump at start, then plateau
2. **NOT gradient accumulation** - Leak doesn't align with grad_accum steps (every 4)
3. **NOT log_history** - Tried clearing it every 10 steps, no effect
4. **NOT processor/tokenizer caches** - Tried clearing them, no effect  
5. **NOT garbage collection** - Tried aggressive GC (3 passes), no effect
6. **NOT optimizer state** - Added cleanup in on_optimizer_step, no effect
7. **NOT PyTorch compile workers** - Leak is in main/training process, not workers
8. **NOT metrics accumulation** - Lists too small (~249 entries max)
9. **NOT selective loss collator** - Leak happens without it

### What It IS

The leak appears to be in the **actual training loop**, growing **every training step**. Likely causes:

1. **UnslothVisionDataCollator internal caching** - Most likely culprit
   - Unsloth's collator may be accumulating processed images/tensors
   - We cannot modify Unsloth's internals
   
2. **PIL Images in dataset** - Contributing factor
   - 663 PIL Images loaded into memory (~3-6 GB baseline)
   - But doesn't explain continuous growth during training
   
3. **HuggingFace Trainer internals** - Possible
   - Something in SFTTrainer accumulating state
   - Despite cleanup callbacks

## Attempted Fixes

### 1. MemoryCleanupCallback (Failed)
```python
- Clear state.log_history every 10 steps
- Clear processor caches
- model.zero_grad(set_to_none=True)
- gc.collect() + torch.cuda.empty_cache()
```
**Result**: No effect on leak rate

### 2. on_optimizer_step cleanup (Failed)
```python
- GC after every optimizer.step()
- Clear gradients immediately
```
**Result**: No effect on leak rate

### 3. Subprocess isolation (Partial)
```python
- Run training in separate subprocess
- Process exits and frees ALL memory when done
```
**Result**: Isolates leak from main server, but **training still OOMs**

## Current Status

- **Main server**: Now isolated via subprocess, won't leak
- **Training subprocess**: Still leaks 1-2 GB/min, will OOM

## Recommended Solutions

### Short-term (Workaround)
1. **Add memory monitoring** - Kill training gracefully before OOM
2. **Reduce dataset size** - Use smaller batches of data
3. **Use checkpoint-resume** - Train in smaller chunks, restart process between chunks

### Long-term (Proper Fix)
1. **Report to Unsloth** - This appears to be an Unsloth UnslothVisionDataCollator bug
2. **Alternative collator** - Implement custom collator that doesn't leak
3. **Lazy image loading** - Don't keep PIL Images in memory, load on-demand

## Files Modified

- `model_garden/vision_training.py` - Added MemoryCleanupCallback with on_optimizer_step
- `model_garden/memory_management.py` - Added callback cleanup, multiple GC passes
- `model_garden/api.py` - Changed to subprocess execution model
- `monitor_training_memory.py` - Created monitoring tool

## Test Results

```
Duration: 2.9 minutes
Start memory: 1254 MB
End memory: 6625 MB  
Total growth: +5371 MB
Rate: +1873 MB/minute
```

Training would OOM in ~17 minutes with 32GB RAM.
