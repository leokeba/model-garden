# HuggingFace Hub Dataset Support

## Overview

Model Garden now supports loading datasets directly from HuggingFace Hub with automatic base64 image decoding for vision-language models.

## Features

✅ **Direct Hub Integration**: Load datasets from HuggingFace Hub without downloading files  
✅ **Base64 Image Support**: Automatically decode base64-encoded JPEG/PNG images  
✅ **OpenAI Messages Format**: Native support for OpenAI-style multimodal messages  
✅ **Flexible Dataset Sources**: Works with both local files and Hub datasets  
✅ **Web UI Support**: Checkbox to toggle between local files and Hub datasets  

## Quick Start

### CLI Usage

```bash
# Train with HuggingFace Hub dataset
uv run model-garden train-vision \
  --base-model Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset Barth371/train_pop_valet_no_wrong_doc \
  --from-hub \
  --output-dir ./models/form-extraction-model \
  --max-steps 100 \
  --batch-size 1
```

### Web UI Usage

1. Navigate to **Training Jobs** → **New Job**
2. Select **Vision-Language (VLM)** model type
3. Choose a vision model (e.g., Qwen2.5-VL-3B-Instruct)
4. Check **"Load from HuggingFace Hub"**
5. Enter dataset identifier: `Barth371/train_pop_valet_no_wrong_doc`
6. Configure hyperparameters and start training

## Dataset Formats

### HuggingFace Hub Format (OpenAI Messages)

```json
## Dataset Formats

### Recommended Format (Simple JSONL)

For best compatibility with Unsloth's UnslothVisionDataCollator, use this simple format:

```json
{"text": "What is in this image?", "image": "/path/to/image.jpg", "response": "A cat"}
{"text": "Describe the scene", "image": "data:image/jpeg;base64,...", "response": "A sunset"}
```

**Supported image formats:**
- File paths (absolute or relative)
- Base64-encoded images with data URI (`data:image/jpeg;base64,...`)
- URLs (if supported by your setup)

### OpenAI Messages Format

While the system can load OpenAI messages format from HuggingFace Hub, it's **automatically converted to the simple format** for compatibility:

```json
{
  "messages": [
    {
      "role": "system",
      "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "data:image/jpeg;base64,..."},
        {"type": "text", "text": "What is in this image?"}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "A cat sitting on a table"}]
    }
  ]
}
```

**Note**: The system extracts the first image and text from user message and the assistant's response, converting them to the simple format. Complex multi-turn conversations or multiple images per message may not be fully supported.
```

### Local File Format (JSONL)

```json
{"text": "What is in this image?", "image": "/path/to/image.jpg", "response": "A cat"}
{"text": "Describe the scene", "image": "data:image/jpeg;base64,...", "response": "A sunset"}
```

**Supported image formats:**
- File paths (absolute or relative)
- Base64-encoded images with data URI
- URLs (if supported by your setup)

## Example Datasets

### Barth371/train_pop_valet_no_wrong_doc
- **Task**: Form content extraction from smartphone pictures/scans
- **Size**: 492 examples, ~147MB
- **Format**: OpenAI messages with base64 JPEG images
- **Use case**: Document understanding, OCR, form extraction

## Implementation Details

### VisionLanguageTrainer Updates

1. **`load_dataset()` method**: Unified interface for local files and Hub datasets
2. **`load_dataset_from_hub()` method**: Direct HuggingFace Hub loading
3. **`_decode_base64_image()` method**: Converts base64 strings to PIL Images
4. **`_load_image()` method**: Smart image loading from multiple sources
5. **`format_dataset()` enhancement**: Automatic format detection and conversion

### Image Loading Logic

```python
# Automatically handles:
# 1. Base64 strings (with or without data URI prefix)
# 2. File paths (absolute or relative)
# 3. PIL Image objects
# 4. Fallback to blank image if loading fails

image = trainer._load_image(image_data)  # Returns PIL.Image.Image
```

### API Integration

The FastAPI backend now accepts `from_hub` flag in `TrainingJobRequest`:

```python
class TrainingJobRequest(BaseModel):
    name: str
    base_model: str
    dataset_path: str  # File path or Hub identifier
    from_hub: bool = False  # Toggle Hub loading
    is_vision: bool = False
    # ... other fields
```

## Testing

Test the HuggingFace Hub integration:

```bash
python3 test_hub_dataset.py
```

This script:
1. Loads 5 examples from a Hub dataset
2. Formats the dataset (converts base64 to PIL Images)
3. Verifies image conversion and data structure
4. Displays example content

## Performance

- **Base64 decoding**: ~10-50ms per image (depending on size)
- **Memory usage**: Images decoded on-the-fly during training
- **Network**: Only downloads dataset metadata + data (not separate image files)

## Benefits

1. **No manual downloads**: Images embedded in dataset, no separate file management
2. **Easy sharing**: Share datasets with collaborators via HuggingFace Hub
3. **Version control**: Hub handles dataset versioning
4. **Streaming support**: Can stream large datasets without loading all into memory
5. **Reproducibility**: Dataset identifier ensures exact version used

## Troubleshooting

### Issue: Base64 decode errors
**Solution**: Verify image data has correct format (data URI or raw base64)

### Issue: Out of memory
**Solution**: Use smaller batch sizes or enable dataset streaming

### Issue: Slow loading
**Solution**: Consider caching dataset locally or using faster network connection

## Future Enhancements

- [ ] Dataset streaming for very large datasets
- [ ] Automatic image preprocessing (resize, normalize)
- [ ] Support for more image formats (WebP, AVIF)
- [ ] Dataset caching for faster repeated access
- [ ] Dataset validation and preview in web UI

## Related Documentation

- [Vision-Language Training Guide](docs/08-vision-language-training.md)
- [Vision Support Overview](VISION_SUPPORT.md)
- [API Specification](docs/03-api-specification.md)
