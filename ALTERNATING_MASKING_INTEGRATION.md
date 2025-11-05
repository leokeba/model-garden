# Alternating Masking Strategy - Integration Complete ✅

## Summary

Successfully implemented the alternating masking strategy across all layers of Model Garden:
- ✅ Core selective loss module (`selective_loss.py`)
- ✅ Vision training logic (`vision_training.py`)
- ✅ API backend (`api.py`)
- ✅ CLI interface (`cli.py`)
- ✅ Frontend UI (`frontend/`)

## Changes Made

### 1. Core Module (`model_garden/selective_loss.py`)
- Added `masking_strategy` parameter with "epoch_based" and "alternating" options
- Added `mask_every_n_steps` (cycle length) and `mask_for_n_steps` (masking ON duration)
- Implemented alternating logic using modulo arithmetic
- Enhanced logging to show masking state transitions
- Backward compatible - defaults to epoch_based strategy

### 2. Training Logic (`model_garden/vision_training.py`)
- Added new parameters to `train()` method signature
- Updated collator initialization to pass new parameters
- Enhanced console output to show strategy and cycle information

### 3. API Backend (`model_garden/api.py`)
- Updated `TrainingJobRequest` Pydantic model with new fields:
  - `selective_loss_masking_strategy: str = "epoch_based"`
  - `selective_loss_mask_every_n_steps: int = 100`
  - `selective_loss_mask_for_n_steps: int = 50`
- Updated `TrainingJobInfo` model for job storage
- Updated job creation, loading, and rerun logic
- All endpoints now properly serialize/deserialize new parameters

### 4. CLI Interface (`model_garden/cli.py`)
- Added `--selective-loss-masking-strategy` option (epoch_based | alternating)
- Added `--selective-loss-mask-every-n-steps` option (default: 100)
- Added `--selective-loss-mask-for-n-steps` option (default: 50)
- Updated help text with clear descriptions
- Parameters properly flow to vision training function

### 5. Frontend UI (`frontend/src/routes/training/new/+page.svelte`)
- Added masking strategy selector dropdown
- Conditional UI based on selected strategy:
  - **Epoch-based**: Shows epoch slider (existing control)
  - **Alternating**: Shows two new sliders for cycle configuration
- Real-time preview of masking pattern (e.g., "50% masking / 50% structure")
- Color-coded explanations (blue for strategy, purple for alternating pattern)
- Updated TypeScript types (`client.ts`)

## Usage Examples

### CLI Usage

```bash
# Epoch-based strategy (original behavior)
uv run model-garden train-vision \
  --base-model Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset ./data/vision.jsonl \
  --output-dir ./models/my-model \
  --selective-loss \
  --selective-loss-masking-strategy epoch_based \
  --selective-loss-masking-start-epoch 0.5

# Alternating strategy (new feature)
uv run model-garden train-vision \
  --base-model Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset ./data/vision.jsonl \
  --output-dir ./models/my-model \
  --selective-loss \
  --selective-loss-masking-strategy alternating \
  --selective-loss-mask-every-n-steps 100 \
  --selective-loss-mask-for-n-steps 50 \
  --selective-loss-verbose
```

### API Usage

```python
import requests

# Create training job with alternating masking
response = requests.post("http://localhost:8000/api/training", json={
    "name": "My Vision Model",
    "base_model": "Qwen/Qwen2.5-VL-3B-Instruct",
    "dataset_path": "./data/vision.jsonl",
    "output_dir": "./models/my-model",
    "is_vision": True,
    "selective_loss": True,
    "selective_loss_level": "aggressive",
    "selective_loss_masking_strategy": "alternating",
    "selective_loss_mask_every_n_steps": 100,
    "selective_loss_mask_for_n_steps": 50,
    "selective_loss_verbose": True,
    "hyperparameters": {
        "num_epochs": 3,
        "per_device_train_batch_size": 2,
        "learning_rate": 2e-5
    }
})
```

### Frontend Usage

1. Navigate to **Training → New Training Job**
2. Select **Vision Model** tab
3. Scroll to **🎯 Selective Loss (Structured Outputs)** section
4. Enable **Selective Loss Masking**
5. Select **Masking Strategy**: "Alternating (Cycle ON/OFF)"
6. Configure:
   - **Cycle Length**: 100 steps (how often to alternate)
   - **Masking ON per cycle**: 50 steps (50% masking, 50% structure)
7. Click **Start Training**

## Verification Tests

### ✅ Core Module Test
```bash
uv run python test_alternating_masking.py
# Output: All tests PASSED! ✓
```

### ✅ API Validation Test
```bash
uv run python -c "from model_garden.api import TrainingJobRequest; \
req = TrainingJobRequest(name='test', base_model='test', dataset_path='test', output_dir='test', \
selective_loss_masking_strategy='alternating', selective_loss_mask_every_n_steps=100, \
selective_loss_mask_for_n_steps=50); \
print(f'✓ Strategy: {req.selective_loss_masking_strategy}')"
# Output: ✓ Strategy: alternating
```

### ✅ CLI Help Test
```bash
uv run model-garden train-vision --help | grep "masking-strategy"
# Output shows new --selective-loss-masking-strategy option
```

### ✅ Frontend Compilation Test
```bash
cd frontend && npm run check
# Output: svelte-check found 0 errors and 0 warnings
```

## Expected Training Output

When using alternating strategy with verbose mode, you'll see:

```
🎯 Using selective loss masking (level: aggressive)
   Strategy: alternating
   🔄 Alternating: ON for 50/100 steps per cycle

🔄 Step 0: Masking ON for next 50 steps
🔄 Step 50: Masking OFF for next 50 steps
🔄 Step 100: Masking ON for next 50 steps
...
```

## Default Behavior

- **Strategy**: `epoch_based` (backward compatible)
- **Start Epoch**: `0.0` (immediate masking)
- **Cycle Length**: `100` steps
- **Masking ON**: `50` steps (50% of cycle)

## Recommendations

### For Most Users
```
Strategy: alternating
Cycle Length: 100 steps
Masking ON: 50 steps (50/50 split)
```

### For Structure-Heavy Tasks
```
Strategy: alternating
Cycle Length: 100 steps
Masking ON: 40 steps (40% masking, 60% structure)
```

### For Semantics-Heavy Tasks
```
Strategy: alternating
Cycle Length: 100 steps
Masking ON: 70 steps (70% masking, 30% structure)
```

### For Initial Experiments
```
Strategy: epoch_based
Start Epoch: 0.5 (learn structure first)
```

## Documentation

- **User Guide**: `ALTERNATING_MASKING_STRATEGY.md`
- **Implementation Summary**: This file
- **Test Suite**: `test_alternating_masking.py`

## Next Steps

To use the alternating masking strategy:

1. **Via CLI**: Add `--selective-loss-masking-strategy alternating` to your training command
2. **Via API**: Set `"selective_loss_masking_strategy": "alternating"` in your request
3. **Via Frontend**: Select "Alternating (Cycle ON/OFF)" in the masking strategy dropdown

The feature is fully integrated and ready to use! 🚀
