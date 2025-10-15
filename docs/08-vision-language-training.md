# Vision-Language Model Training Guide

## Overview
Model Garden now supports fine-tuning vision-language models like **Qwen2.5-VL** that can understand and reason about images and text together.

## Supported Models

### Qwen2.5-VL Series
- `Qwen/Qwen2.5-VL-3B-Instruct` - 3B parameter vision-language model
- `Qwen/Qwen2.5-VL-7B-Instruct` - 7B parameter vision-language model (recommended)
- `Qwen/Qwen2.5-VL-72B-Instruct` - 72B parameter vision-language model (requires multiple GPUs)

## Quick Start

### 1. Create a Sample Dataset

```bash
uv run model-garden create-vision-dataset \
  --output ./data/vision_sample.jsonl \
  --num-examples 10
```

This creates a template dataset. **Important**: You must replace the placeholder image paths with actual images!

### 2. Dataset Format

Your vision-language dataset should be in JSONL format with these fields:

```json
{"text": "What is shown in this image?", "image": "/path/to/image1.jpg", "response": "A cat sitting on a windowsill"}
{"text": "Describe the scene", "image": "/path/to/image2.jpg", "response": "A sunset over mountains with orange and pink clouds"}
{"text": "What color is the car?", "image": "/path/to/image3.jpg", "response": "The car is red"}
```

**Required fields:**
- `text` - The question or instruction about the image
- `image` - Path to the image file (JPEG, PNG, etc.)
- `response` - The expected answer or description

**Image paths can be:**
- Absolute paths: `/home/user/images/photo.jpg`
- Relative paths: `./data/images/photo.jpg`
- URLs: `https://example.com/image.jpg` (if your setup supports it)

### 3. Fine-tune the Model

```bash
uv run model-garden train-vision \
  --base-model Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset ./data/vision_dataset.jsonl \
  --output-dir ./models/my-vision-model \
  --epochs 3 \
  --batch-size 1 \
  --learning-rate 2e-5
```

## Advanced Usage

### Custom Hyperparameters

```bash
uv run model-garden train-vision \
  --base-model Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset ./data/my_vision_data.jsonl \
  --output-dir ./models/custom-vision \
  --epochs 5 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --gradient-accumulation-steps 16 \
  --max-seq-length 2048 \
  --lora-r 32 \
  --lora-alpha 32 \
  --logging-steps 5 \
  --save-steps 50
```

### Custom Dataset Fields

If your dataset uses different field names:

```bash
uv run model-garden train-vision \
  --base-model Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset ./data/custom_format.jsonl \
  --output-dir ./models/my-model \
  --text-field "question" \
  --image-field "img_path"
```

## Memory Requirements

Vision-language models require more VRAM than text-only models:

| Model | VRAM (4-bit) | VRAM (Full) | Recommended GPU |
|-------|--------------|-------------|-----------------|
| Qwen2.5-VL-3B | ~6GB | ~12GB | RTX 3090, A100 |
| Qwen2.5-VL-7B | ~10GB | ~28GB | A100, H100 |
| Qwen2.5-VL-72B | ~45GB | ~144GB | Multiple A100s/H100s |

**Tips for reducing memory:**
- Use batch size of 1
- Increase gradient accumulation steps (8-16)
- Use 4-bit quantization (enabled by default)
- Reduce max sequence length if possible

## Dataset Preparation Tips

### 1. Image Quality
- Use consistent image sizes (will be resized automatically)
- Supported formats: JPEG, PNG, WebP
- Recommended: 224x224 to 512x512 pixels
- Keep images under 5MB each

### 2. Data Diversity
- Include various image types relevant to your task
- Mix different scenes, objects, and contexts
- Balance question types (description, counting, reasoning, etc.)

### 3. Response Quality
- Keep responses concise and accurate
- Match the style you want the model to learn
- Include both simple and complex examples

### 4. Example Datasets

**Image Captioning:**
```json
{"text": "Describe this image", "image": "img1.jpg", "response": "A golden retriever playing in a park"}
{"text": "What is happening here?", "image": "img2.jpg", "response": "Children playing soccer on a field"}
```

**Visual Question Answering:**
```json
{"text": "How many people are in the image?", "image": "img1.jpg", "response": "There are 3 people"}
{"text": "What is the person wearing?", "image": "img2.jpg", "response": "A blue shirt and jeans"}
```

**Scene Understanding:**
```json
{"text": "Describe the setting", "image": "img1.jpg", "response": "An office with modern furniture and large windows"}
{"text": "What objects do you see?", "image": "img2.jpg", "response": "A laptop, coffee mug, and notepad on a desk"}
```

## Limitations & Known Issues

### Current Limitations
1. **Experimental Feature** - Vision-language training is in early development
2. **Basic Implementation** - Full multimodal training pipeline coming soon
3. **Model Support** - Currently optimized for Qwen2.5-VL series
4. **Image Processing** - Uses basic image handling (advanced preprocessing coming)

### Known Issues
- Unsloth may not fully support vision-language models yet (falls back to transformers)
- Merged model saving not yet implemented (LoRA adapters only)
- Custom vision processors need manual integration
- Batch processing of images requires optimization

### Workarounds
- Use LoRA fine-tuning (recommended and faster)
- Keep batch size at 1 for stability
- Monitor GPU memory closely
- Test with small datasets first

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
--batch-size 1

# Increase gradient accumulation
--gradient-accumulation-steps 16

# Reduce sequence length
--max-seq-length 1024
```

### Image Loading Errors
- Verify all image paths exist
- Check image file formats
- Ensure images aren't corrupted
- Use absolute paths for reliability

### Slow Training
- Vision models train slower than text-only
- Expect 2-3x longer training time
- Use gradient checkpointing (enabled by default)
- Consider using smaller images

## Next Steps

After fine-tuning your vision-language model:

1. **Test the Model**
   ```bash
   # Coming soon: vision-language inference
   uv run model-garden generate-vision ./models/my-vision-model \
     --image ./test_image.jpg \
     --prompt "What is in this image?"
   ```

2. **Export for Production**
   - LoRA adapters can be merged with base model
   - Export to GGUF for llama.cpp (when supported)
   - Deploy with vLLM (vision support coming)

3. **Evaluate Performance**
   - Test on validation images
   - Check response quality
   - Measure inference speed

## API Integration

Vision-language models can be trained via API (coming soon):

```python
import requests

response = requests.post('http://localhost:8000/api/v1/training/jobs', json={
    "name": "vision-model-1",
    "base_model": "Qwen/Qwen2.5-VL-3B-Instruct",
    "dataset_path": "./data/vision_dataset.jsonl",
    "output_dir": "./models/vision-model-1",
    "model_type": "vision-language",  # New field
    "hyperparameters": {
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "batch_size": 1,
        "gradient_accumulation_steps": 8
    }
})
```

## Resources

### Documentation
- [Qwen2.5-VL Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Vision-Language Models Overview](https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder)
- [LoRA for Vision Models](https://arxiv.org/abs/2106.09685)

### Example Datasets
- [COCO Captions](https://cocodataset.org/)
- [Visual Question Answering (VQA)](https://visualqa.org/)
- [TextVQA](https://textvqa.org/)

### Community
- GitHub Issues: Report bugs or request features
- Discussions: Share your vision-language projects
- Discord: Get help from the community

---

**Status**: Experimental  
**Version**: 0.2.0  
**Last Updated**: October 2025

## Contributing

Help improve vision-language support:
- Test with different models
- Share dataset preprocessing scripts
- Report compatibility issues
- Contribute training optimizations

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.
