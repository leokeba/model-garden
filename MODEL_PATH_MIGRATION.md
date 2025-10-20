# Model Path Simplification

## Overview

Fixed the redundant `models/models/` directory structure by simplifying model paths throughout the system.

## Changes Made

### 1. Backend API (`model_garden/api.py`)

- Added new `resolve_model_path()` function that handles multiple path formats:
  - Simple model names: `"my-model"` → `models/my-model`
  - Legacy paths with prefix: `"./models/my-model"` → `models/my-model`
  - Absolute paths: `/path/to/model` → `/path/to/model` (unchanged)
  - HuggingFace IDs: `"org/model"` → `org/model` (unchanged)

- Updated `create_training_job()` to use `resolve_model_path()` for output directories

- Updated `load_inference_model()` to use `resolve_model_path()` for model loading

### 2. Frontend (`frontend/src/routes/training/new/+page.svelte`)

- Changed output directory generation from `./models/${name}` to just `${name}`
- Updated placeholder from `./models/my-model` to `my-model`
- Added help text showing the full path: `models/{model-name}`

### 3. Documentation

- Updated `README.md` with simpler path examples and explanation
- Updated `QUICKSTART.md` to use simplified paths
- Added note about backward compatibility (both formats work)

### 4. Migration

- Created `migrate_models.py` script to move existing models from `models/models/` to `models/`
- Script updates training job metadata in `storage/training_jobs.json`
- Successfully migrated 11 models and updated 33 job records

## User Impact

### Before
```bash
# Training
--output-dir ./models/my-model

# Loading (API)
"model_path": "models/models/my-model"  # Had to specify twice!

# Directory structure
models/
  models/          # Redundant nesting
    my-model/
    other-model/
```

### After
```bash
# Training (recommended)
--output-dir my-model

# Training (still works for backward compatibility)
--output-dir ./models/my-model

# Loading (API)
"model_path": "my-model"  # Much simpler!

# Directory structure
models/
  my-model/
  other-model/
```

## Backward Compatibility

The system automatically handles both path formats:
- New format: `my-model` (recommended)
- Legacy format: `./models/my-model` (still supported)

Existing scripts and API calls will continue to work without modification.

## Migration Steps

If you have existing models in the old `models/models/` structure:

```bash
# Run the migration script
uv run python migrate_models.py

# Restart the service
sudo systemctl restart model-garden.service
```

The migration script will:
1. Show all models to be migrated
2. Ask for confirmation
3. Move models from `models/models/` to `models/`
4. Update training job metadata paths
5. Remove the empty nested directory

## Testing

Verified functionality:
- ✅ Training creates models in `models/{name}` (not `models/models/{name}`)
- ✅ Model loading accepts simple names: `"my-model"`
- ✅ Model loading still works with legacy paths: `"./models/my-model"`
- ✅ Frontend shows simplified paths and generates correct output_dir
- ✅ Existing models migrated successfully
- ✅ Training job metadata updated correctly
