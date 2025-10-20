# Vision Training Fixes - October 20, 2025

## Issues Fixed

### 1. Model Precision Mismatch Error

**Issue**: TypeError when starting vision model training:
```
TypeError: Unsloth: Model is in bfloat16 precision but you want to use float16 precision. 
Set fp16 to `False` and bf16 to `True`
```

**Root Cause**: The model's dtype detection was not properly handling wrapped models (PeftModel, etc.). The code was only checking `self.model.dtype`, which doesn't always reflect the actual model precision for wrapped models.

**Fix Applied** (in `model_garden/vision_training.py` around line 773):
- **CRITICAL**: Check actual parameter dtypes instead of model attributes
- The model's `.dtype` attribute is unreliable (returns float32 even when params are bfloat16)
- For 16-bit models, use `next(model.parameters()).dtype` to get actual precision
- For quantized models (4-bit/8-bit), always use bfloat16 for training
- Added debug logging to show detected dtype and training precision
- Correctly sets `fp16=False` and `bf16=True` when model is in bfloat16

**Code Changes**:
```python
# Before (BROKEN - model.dtype is unreliable)
model_dtype = None
if self.model is not None and hasattr(self.model, 'dtype'):
    model_dtype = self.model.dtype
is_bfloat16_model = model_dtype == torch.bfloat16

# After (FIXED - check actual parameter dtypes)
model_dtype = None

# For 16-bit (non-quantized) models, we need to detect the actual dtype
# Many modern models (like Qwen2.5-VL) use bfloat16 by default
if not self.load_in_4bit and not self.load_in_8bit:
    if self.model is not None:
        # Check the actual parameter dtypes (most reliable method)
        try:
            # Get first parameter and check its dtype
            first_param = next(self.model.parameters())
            model_dtype = first_param.dtype
        except (StopIteration, AttributeError):
            # Fallback to other methods if no parameters
            # ... (fallback logic)
else:
    # For quantized models, always use bfloat16 for training
    model_dtype = torch.bfloat16

is_bfloat16_model = model_dtype == torch.bfloat16

# Log detected dtype for debugging
console.print(f"[cyan]🔍 Detected model dtype: {model_dtype}[/cyan]")
console.print(f"[cyan]📊 Training precision: {'bf16' if is_bfloat16_model else 'fp16'}[/cyan]")
```

### 2. Memory Monitor Callback Crash

**Issue**: ReferenceError during training:
```
ReferenceError: weakly-referenced object no longer exists
```

**Root Cause**: The `MemoryMonitorCallback.on_step_end()` method was iterating over `gc.get_objects()` to count tensors. During garbage collection, some objects become weak references that are already collected, causing a crash when accessed via `isinstance()`.

**Fix Applied** (in `model_garden/vision_training.py` around line 726):
- Wrapped tensor counting in individual try-except blocks to handle weak references
- Added outer try-except to catch any other memory monitoring errors
- Prevents training from crashing if memory monitoring fails
- Logs warning but continues training

**Code Changes**:
```python
# Before
tensors = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)]

# After
try:
    # Count tensor objects for debugging
    # Note: Wrap in try-except to handle weak references that may have been collected
    tensors = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor):
                tensors.append(obj)
        except ReferenceError:
            # Object was weakly referenced and has been collected
            continue
    
    cpu_tensors = [t for t in tensors if t.device.type == 'cpu']
    cuda_tensors = [t for t in tensors if t.device.type == 'cuda']
    
    # ... rest of monitoring code ...
except Exception as e:
    # If memory monitoring fails, log but don't crash training
    console.print(f"[yellow]⚠️  Memory monitoring error at step {state.global_step}: {e}[/yellow]")
```

## Testing

After applying both fixes:
1. Service restarted successfully
2. No immediate errors in logs
3. Ready to accept training jobs

## Impact

These fixes ensure:
- ✅ Vision model training can start with proper precision detection
- ✅ Training won't crash due to memory monitoring errors
- ✅ Better debugging visibility with dtype logging
- ✅ More robust handling of edge cases during garbage collection

## Next Steps

1. Monitor the next training job to ensure both fixes work correctly
2. Consider whether memory monitoring callback should be optional
3. Document these patterns for future callback implementations
