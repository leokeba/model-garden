# Vision-Language Model Support

## Overview

Model Garden now supports fine-tuning vision-language models (VLMs) like Qwen2.5-VL that can understand both images and text!

## Features

### Supported Models
- **Qwen/Qwen2.5-VL-3B-Instruct** (3 billion parameters)
- **Qwen/Qwen2.5-VL-7B-Instruct** (7 billion parameters)
- **Qwen/Qwen2.5-VL-72B-Instruct** (72 billion parameters)
- **Unsloth 4-bit quantized versions** for lower memory usage

### Web UI Integration

The training form now includes:
- **Model Type Selector**: Choose between Text-only (LLM) or Vision-Language (VLM)
- **Vision Model Options**: Dropdown with Qwen2.5-VL models
- **Auto-Configuration**: Vision models automatically get optimized hyperparameters:
  - Batch Size: 1 (vision models need more memory per sample)
  - Gradient Accumulation: 8 (compensates for smaller batch size)
  - Learning Rate: 2e-5 (lower than text models)
- **Dataset Format Helper**: Shows example format for vision datasets
- **Visual Indicators**: Clear UI cues for vision-specific requirements

### Backend Support

- **VisionLanguageTrainer Class**: Specialized trainer for multimodal models
- **Unsloth Integration**: Uses `UnslothVisionDataCollator` for efficient training
- **Automatic Model Detection**: API automatically routes to correct trainer based on `is_vision` flag
- **OpenAI Message Format**: Datasets use standard message format with PIL Image objects

## Dataset Format

Vision-language datasets should be in JSONL format:

```jsonl
{"text": "What is in this image?", "image": "/path/to/image.jpg", "response": "A cat sitting on a table"}
{"text": "Describe the color", "image": "/path/to/image2.jpg", "response": "The image shows a blue sky"}
```

## CLI Commands

### Train a Vision Model
```bash
model-garden train-vision \
  --base-model Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset ./data/my_vision_dataset.jsonl \
  --output-dir ./models/my-vision-model \
  --epochs 3 \
  --batch-size 1 \
  --max-steps 100
```

### Create Sample Vision Dataset
```bash
model-garden create-vision-dataset \
  --output ./data/sample_vision.jsonl \
  --num-examples 10
```

## API Usage

### Create Vision Training Job

```bash
curl -X POST http://localhost:8000/api/v1/training/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-vision-model",
    "base_model": "Qwen/Qwen2.5-VL-3B-Instruct",
    "dataset_path": "./data/vision_dataset.jsonl",
    "output_dir": "./models/my-vision-model",
    "is_vision": true,
    "model_type": "vision",
    "hyperparameters": {
      "learning_rate": 0.00002,
      "num_epochs": 3,
      "batch_size": 1,
      "gradient_accumulation_steps": 8,
      "max_steps": -1
    },
    "lora_config": {
      "r": 16,
      "lora_alpha": 16
    }
  }'
```

## Technical Details

### Architecture
- **Base**: Qwen2.5-VL models with Vision Transformer (ViT) + Language Model
- **Training Method**: LoRA (Low-Rank Adaptation) for efficient fine-tuning
- **Quantization**: 4-bit quantization supported via Unsloth
- **Memory Usage**: ~12-16GB VRAM for 3B model with 4-bit quantization

### Training Configuration
| Parameter | Text Models | Vision Models | Reason |
|-----------|------------|---------------|---------|
| Batch Size | 2-4 | 1 | Vision models need more memory per sample |
| Gradient Accumulation | 4 | 8+ | Compensates for smaller batch size |
| Learning Rate | 2e-4 | 2e-5 | Vision models are more sensitive |
| LoRA Rank | 16 | 16 | Standard rank works for both |

### Dataset Preprocessing
1. Images are loaded as PIL Image objects
2. Text and images are formatted as OpenAI-style messages
3. UnslothVisionDataCollator handles batch processing
4. Automatic image preprocessing (resize, normalize)

## Use Cases

### Image Understanding
- **OCR**: Extract text from images
- **Visual Question Answering**: Answer questions about images
- **Image Captioning**: Generate descriptions of images
- **Object Detection**: Identify objects in images

### Document Analysis
- **Invoice Processing**: Extract data from invoices
- **Form Understanding**: Parse forms and documents
- **Chart Analysis**: Understand graphs and charts
- **Table Extraction**: Extract tabular data from images

### Creative Applications
- **Image-to-Story**: Generate stories from images
- **Style Analysis**: Describe artistic styles
- **Product Description**: Generate product descriptions from photos
- **Scene Understanding**: Detailed scene descriptions

## Performance

### Test Results (Qwen2.5-VL-3B on H100)
- **Training Speed**: ~12 seconds/step (batch_size=1, grad_accum=8)
- **Memory Usage**: ~35GB VRAM with 4-bit quantization
- **Dataset**: 10 example images (224x224)
- **Training Time**: ~36 seconds for 3 steps

### Optimization Tips
1. Use 4-bit quantized models for lower memory
2. Increase gradient accumulation instead of batch size
3. Use smaller images (224x224) for faster training
4. Enable gradient checkpointing for memory efficiency

## Limitations

- **Image Size**: Optimal at 224x224, larger images use more memory
- **Batch Size**: Limited to 1-2 for most GPUs
- **Training Time**: Slower than text-only models
- **Dataset Size**: Requires good quality image-text pairs

## Documentation

- **Full Guide**: See `docs/08-vision-language-training.md`
- **API Docs**: See `docs/03-api-specification.md`
- **Examples**: See `data/vision_test_dataset.jsonl`

## Testing

A successful test run was completed with:
- Model: Qwen/Qwen2.5-VL-3B-Instruct
- Dataset: 10 test images (shapes and colors)
- Training: 3 steps completed successfully
- Output: Model saved to `./models/test-vision-model`

## Next Steps

1. Test the web UI by navigating to http://localhost:8000/training/new
2. Select "Vision-Language (VLM)" model type
3. Choose a Qwen2.5-VL model
4. Provide your vision dataset path
5. Start training and monitor progress

---

**Note**: Vision-language training is experimental. For production use, thorough testing with your specific dataset is recommended.
