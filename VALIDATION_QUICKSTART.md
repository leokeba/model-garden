# Training with Validation Datasets - Quick Start

This guide shows you how to use the new train/validation dataset features with real-time loss curves.

## Overview

Model Garden now supports:
- ✅ **Separate validation datasets** for better model evaluation
- ✅ **Real-time loss curves** with beautiful Chart.js visualization
- ✅ **HuggingFace Hub integration** for both train and validation datasets
- ✅ **Automatic best model selection** based on validation loss
- ✅ **Live metrics streaming** via WebSocket

## Quick Examples

### 1. Text Model with Local Datasets

```bash
# Create training job with validation
curl -X POST http://localhost:8000/api/v1/training/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-model",
    "base_model": "unsloth/tinyllama-bnb-4bit",
    "dataset_path": "./data/train.jsonl",
    "validation_dataset_path": "./data/val.jsonl",
    "output_dir": "./models/my-model",
    "hyperparameters": {
      "num_epochs": 3,
      "batch_size": 2,
      "learning_rate": 0.0002,
      "eval_steps": 50
    }
  }'
```

### 2. Vision Model with HuggingFace Hub

```bash
# Train Qwen2.5-VL with Hub datasets
curl -X POST http://localhost:8000/api/v1/training/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "vision-model",
    "base_model": "Qwen/Qwen2.5-VL-3B-Instruct",
    "dataset_path": "username/train-dataset",
    "validation_dataset_path": "username/val-dataset",
    "from_hub": true,
    "validation_from_hub": true,
    "is_vision": true,
    "output_dir": "./models/vision-model",
    "hyperparameters": {
      "num_epochs": 3,
      "batch_size": 1,
      "learning_rate": 0.00002,
      "eval_steps": 100
    }
  }'
```

### 3. Using the Web UI

1. Navigate to **Training** → **New Training Job**
2. Fill in model name and base model
3. Set **Dataset Path** (with optional HuggingFace Hub checkbox)
4. Set **Validation Dataset Path** (optional but recommended)
5. Configure **Evaluation Steps** in hyperparameters
6. Click **Start Training**
7. View real-time loss curves on the job detail page!

## Testing It Out

Run the included test script to see it in action:

```bash
# Install dependencies
pip install httpx websockets

# Run test (creates sample datasets and monitors training)
python test_validation_training.py
```

This will:
1. Create sample train/val datasets
2. Start a training job with validation
3. Monitor training via WebSocket
4. Show real-time training and validation metrics
5. Print a summary at the end

## Dataset Format

### Text Datasets
```jsonl
{"instruction": "What is 2+2?", "input": "", "output": "4"}
{"instruction": "Translate to French", "input": "Hello", "output": "Bonjour"}
```

### Vision Datasets (Local)
```jsonl
{"text": "What is in this image?", "image": "/path/to/image.jpg", "response": "A cat"}
```

### Vision Datasets (HuggingFace Hub)
```jsonl
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "data:image/jpeg;base64,..."},
        {"type": "text", "text": "Describe this"}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "A beautiful sunset"}]
    }
  ]
}
```

## Viewing Loss Curves

Once training starts:
1. Go to the training job detail page
2. You'll see a live loss curve chart updating in real-time
3. Training loss (blue line) and validation loss (green line)
4. Metrics table showing recent values
5. Live logs at the bottom

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `validation_dataset_path` | Path to validation dataset | `None` |
| `validation_from_hub` | Load validation from HuggingFace Hub | `false` |
| `eval_steps` | Evaluate every N steps | Same as `save_steps` |
| `load_best_model_at_end` | Save best model based on val loss | `True` (auto when val dataset provided) |

## Benefits

✨ **Early Detection**: Spot overfitting immediately with validation loss
✨ **Better Models**: Automatically saves the best checkpoint
✨ **Visual Insights**: Beautiful charts make trends obvious
✨ **Real-time Monitoring**: No need to SSH and check logs
✨ **Production Ready**: All metrics stored and accessible via API

## Full Documentation

See [TRAINING_METRICS.md](TRAINING_METRICS.md) for complete details including:
- Advanced configuration options
- WebSocket event specifications
- Performance considerations
- Troubleshooting guide
- API reference

## Next Steps

- Try training with your own datasets
- Experiment with different `eval_steps` values
- Compare models using validation metrics
- Export metrics for external analysis

Happy training! 🚀
