# Fix Applied: Vision Training Hub Dataset Support

## Issue
When using the `::` syntax with vision models (e.g., `Barth371/cmr-block-2::train.jsonl`), the training failed with:

```
HFValidationError: Repo id must use alphanumeric chars or '-', '_', '.', 
'--' and '..' are forbidden, '-' and '.' cannot start or end the name, 
max length is 96: 'Barth371/cmr-block-2::train.jsonl'.
```

## Root Cause
The `::` separator support was only implemented in `model_garden/training.py` (for text models), but **not** in `model_garden/vision_training.py` (for vision models).

When you selected a vision model, the system used the `VisionTrainer` class which had the old implementation without `::` parsing.

## Fix Applied
Updated `model_garden/vision_training.py` method `load_dataset_from_hub()` to:

1. ✅ Parse `::` separator to split repo name and file name
2. ✅ Use `data_files` parameter when specific file is provided
3. ✅ Fall back to standard split loading when no `::` present
4. ✅ Match the same behavior as text training

## Code Changes

**File:** `model_garden/vision_training.py`

**Before:**
```python
def load_dataset_from_hub(self, dataset_name: str, split: str = "train", **kwargs):
    dataset = load_dataset(dataset_name, split=split, token=hf_token, **kwargs)
```

**After:**
```python
def load_dataset_from_hub(self, dataset_name: str, split: str = "train", **kwargs):
    if "::" in dataset_name:
        repo_name, file_name = dataset_name.split("::", 1)
        dataset = load_dataset(repo_name, data_files=file_name, split="train", token=hf_token, **kwargs)
    else:
        dataset = load_dataset(dataset_name, split=split, token=hf_token, **kwargs)
```

## Testing

Now you can use the same syntax for both text and vision models:

### Vision Model Training:
```
Model Type: Vision-Language (VLM)
Base Model: Qwen/Qwen2.5-VL-3B-Instruct

Training Dataset: Barth371/cmr-block-2::train.jsonl
✅ Load from HuggingFace Hub

Validation Dataset: Barth371/cmr-block-2::validation.jsonl
✅ Load validation dataset from HuggingFace Hub
```

### Text Model Training:
```
Model Type: Text-Only (LLM)
Base Model: unsloth/mistral-7b-bnb-4bit

Training Dataset: username/repo::train.jsonl
✅ Load from HuggingFace Hub

Validation Dataset: username/repo::validation.jsonl
✅ Load validation dataset from HuggingFace Hub
```

## Status

✅ **Fixed and ready to use!**

Both text and vision training now support:
- Specific files with `::` separator
- Standard split names
- Private repositories (with `HF_TOKEN`)
- Validation datasets

## Try Again

Your training with `Barth371/cmr-block-2::train.jsonl` should now work! 🚀

Just restart your training job from the web UI.
