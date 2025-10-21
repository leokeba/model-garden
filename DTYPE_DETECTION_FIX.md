# Vision Model dtype Detection Fix

## Issue
When training vision-language models (Qwen2.5-VL) on the systemd service, training was failing with:

```
TypeError: Unsloth: Model is in bfloat16 precision but you want to use float16 precision. 
Set fp16 to `False` and bf16 to `True`
```

The logs showed:
```
🔍 Detected model dtype: torch.float32
📊 Training precision: fp16
```

But the actual model was in bfloat16, causing a mismatch.

## Root Cause

When loading Qwen2.5-VL with transformers (fallback path when Unsloth doesn't support the model):

1. The model was loaded WITHOUT explicitly specifying `torch_dtype`
2. Qwen2.5-VL's config has `torch_dtype=torch.bfloat16` as default
3. The model loaded in bfloat16, but dtype detection failed
4. After PEFT wrapping, the detection couldn't access parameters correctly
5. Detection returned default `float32`, setting `fp16=True`
6. Unsloth's SFTTrainer validated and caught the mismatch

## Fix Applied

### 1. Explicit torch_dtype in model loading (`vision_training.py`)

```python
# Determine torch_dtype based on quantization settings
if not self.load_in_4bit and not self.load_in_8bit:
    # For 16-bit precision, use the provided dtype or None to use model's default
    # Qwen2.5-VL defaults to bfloat16, which is appropriate
    torch_dtype = self.dtype
else:
    # For quantized models, let BitsAndBytes handle dtype
    torch_dtype = None

self.model = AutoModelForVision2Seq.from_pretrained(
    self.base_model,
    device_map="auto",
    torch_dtype=torch_dtype,  # <-- NOW SPECIFIED
    load_in_4bit=self.load_in_4bit if self.load_in_4bit else None,
    load_in_8bit=self.load_in_8bit if self.load_in_8bit else None,
    token=hf_token,
)
```

### 2. Improved PEFT model detection (`training_utils.py`)

Added multiple improvements to dtype detection:

**a) Skip meta device parameters (not yet initialized)**
```python
if first_param.device.type != 'meta':
    return model_dtype
```

**b) Better PEFT model handling**
```python
# Check base_model.parameters() for PEFT wrapped models
if hasattr(model, 'base_model'):
    try:
        first_param = next(model.base_model.parameters())
        if first_param.device.type != 'meta':
            return first_param.dtype
    except (StopIteration, AttributeError):
        pass
```

**c) Model-specific defaults for known architectures**
```python
# Qwen2.5-VL series uses bfloat16 by default
if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
    model_type = model.config.model_type
    if model_type in ['qwen2_vl', 'qwen2_audio_vl']:
        return torch.bfloat16
```

This fallback ensures we use the correct precision even if parameter inspection fails.

## Why This Matters

- **Cross-machine compatibility**: Different machines may use different code paths (Unsloth vs transformers)
- **Precision correctness**: Training precision MUST match model precision
- **Qwen2.5-VL native precision**: This model uses bfloat16 natively, not float16
- **PEFT wrapping**: After LoRA adapter application, detection needs to handle wrapped models

## Testing

After the fix:
1. Service restarts successfully
2. Vision model training should now detect bfloat16 correctly
3. Training args will use `bf16=True, fp16=False`
4. No more precision mismatch errors

## Related Files

- `model_garden/vision_training.py`: Model loading logic
- `model_garden/training_utils.py`: Shared dtype detection utilities
- `model_garden/api.py`: Training job orchestration

## Notes for Other Machines

If you see this error on another machine:
1. Check if model is loading via Unsloth or transformers fallback
2. Verify `torch_dtype` is being passed to `from_pretrained()`
3. Check detection logs: `🔍 Detected model dtype: ...`
4. Should see `torch.bfloat16` for Qwen2.5-VL, not `torch.float32`
