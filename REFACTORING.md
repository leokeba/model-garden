# Model Garden Refactoring Progress

## Overview

This document tracks the refactoring of `model_garden/` from a flat module structure to a well-organized package hierarchy.

## Completed (December 2025)

### Phase 1: API Package (Previous Session)
- ✅ Created `api/` package from 4000+ line `api.py` monolith
- ✅ Organized into `models/`, `routes/`, and core modules
- ✅ Old `api.py` kept for reference

### Phase 2: Training Package
- ✅ Created `training/` package with:
  - `trainer.py` (was `training.py`) - ModelTrainer, create_text_trainer
  - `vision_trainer.py` (was `vision_training.py`) - VisionLanguageTrainer, create_vision_trainer
  - `selective_loss.py` - SelectiveLossVisionCollator
  - `early_stopping.py` - EarlyStoppingCallback
  - `weighted_loss.py` - WeightedLossTrainer
  - `utils.py` (was `training_utils.py`) - detect_model_dtype, MemoryMonitorCallback
  - `subprocess_runner.py` - run_training_in_subprocess

### Phase 3: Inference Package
- ✅ Created `inference/` package with:
  - `service.py` (was `inference.py`) - InferenceService class (~1150 lines)
  - `utils.py` - Helper functions (GPU memory, model detection, quantization)

### Phase 4: Queue Package
- ✅ Created `queue/` package with:
  - `job_queue.py` - JobQueue, JobStatus, JobType
  - `worker.py` (was `queue_worker.py`) - QueueWorker

### Phase 5: Utils Package
- ✅ Created `utils/` package with:
  - `memory.py` (was `memory_management.py`) - cleanup_training_resources
  - `dataset_validator.py` - DatasetValidator, DatasetStats

### Phase 6: Experiments Package
- ✅ Created `experiments/` package with:
  - `hyperparameter_explorer.py` - HyperparameterExplorer
  - `visualizer.py` (was `exploration_visualizer.py`) - ExplorationVisualizer

### Phase 7: Import Updates
- ✅ Updated imports in all `api/` route files
- ✅ Updated imports in `cli.py`
- ✅ Updated imports in `backends/` (unsloth_backend.py, transformers_backend.py)
- ✅ Updated internal imports within training package
- ✅ Updated internal imports within queue package
- ✅ Created comprehensive `__init__.py` files for all packages

### Phase 8: CLI Package
- ✅ Created `cli/` package from 1495 line `cli.py` monolith
- ✅ Organized into logical command groups:
  - `train.py` - train, train-vision commands
  - `inference.py` - serve-model, inference-generate, inference-chat commands
  - `dataset.py` - create-dataset, create-vision-dataset commands
  - `server.py` - serve command (FastAPI server)
  - `utils.py` - generate, list-backends commands
- ✅ Old `cli.py` converted to re-export shim for backward compatibility
- ✅ All 10 CLI commands verified working

### Phase 9: Backends Deduplication (Latest)
- ✅ Created `transformers_base.py` with `TransformersTrainerMixin` shared class
- ✅ Extracted ~300 lines of duplicate code into mixin:
  - `_get_hf_token()`, `_get_torch_dtype()`, `_get_quantization_config()`
  - `_configure_lora()` - shared LoRA setup
  - `_load_dataset_from_local_file()`, `_load_dataset_from_hf_hub()`
  - `_setup_carbon_tracking()`, `_stop_carbon_tracking()`
  - `_create_training_args()` - shared TrainingArguments creation
  - `_get_callbacks()`, `_save_model_internal()`
- ✅ Refactored `TransformersTextTrainer` to use mixin (reduced ~200 lines)
- ✅ Refactored `TransformersVisionTrainer` to use mixin (reduced ~200 lines)
- ✅ Decision: Keep `backends/` as top-level package (used by training, cli, api)
- ✅ All backends verified working with `list-backends` command

## Current Package Structure

```
model_garden/
├── __init__.py           # Lazy-loading for main classes
├── cli.py               # Re-export shim (backward compat)
├── model_registry.py    # Model registry (standalone)
├── api.py               # OLD - kept for reference
│
├── cli/                 # CLI commands
│   ├── __init__.py      # Main entry point
│   ├── train.py         # train, train-vision
│   ├── inference.py     # serve-model, inference-generate, inference-chat
│   ├── dataset.py       # create-dataset, create-vision-dataset
│   ├── server.py        # serve (FastAPI)
│   └── utils.py         # generate, list-backends
│
├── api/                 # FastAPI application
│   ├── __init__.py
│   ├── app.py
│   ├── storage.py
│   ├── tasks.py
│   ├── websocket.py
│   ├── models/
│   └── routes/
│
├── training/            # Training components
│   ├── __init__.py
│   ├── trainer.py
│   ├── vision_trainer.py
│   ├── selective_loss.py
│   ├── early_stopping.py
│   ├── weighted_loss.py
│   ├── utils.py
│   └── subprocess_runner.py
│
├── inference/           # Inference service
│   ├── __init__.py
│   ├── service.py
│   └── utils.py
│
├── queue/               # Job queue management
│   ├── __init__.py
│   ├── job_queue.py
│   └── worker.py
│
├── utils/               # General utilities
│   ├── __init__.py
│   ├── memory.py
│   └── dataset_validator.py
│
├── experiments/         # Hyperparameter exploration
│   ├── __init__.py
│   ├── hyperparameter_explorer.py
│   └── visualizer.py
│
├── backends/            # Training backends
│   ├── __init__.py      # Backend registry and exports
│   ├── base.py          # Abstract base classes (TextTrainer, VisionTrainer, TrainingBackend)
│   ├── registry.py      # get_backend, list_backends, register_backend
│   ├── transformers_base.py  # TransformersTrainerMixin (shared logic)
│   ├── transformers_backend.py  # TransformersTextTrainer, TransformersVisionTrainer
│   └── unsloth_backend.py  # UnslothBackend (delegates to training/)
│
└── carbon/              # Carbon tracking (unchanged)
    └── ...
```

## Import Migration Guide

| Old Import | New Import |
|------------|------------|
| `from model_garden.job_queue import ...` | `from model_garden.queue import ...` |
| `from model_garden.queue_worker import ...` | `from model_garden.queue import ...` |
| `from model_garden.vision_training import ...` | `from model_garden.training import ...` |
| `from model_garden.training_utils import ...` | `from model_garden.training.utils import ...` |
| `from model_garden.selective_loss import ...` | `from model_garden.training.selective_loss import ...` |
| `from model_garden.early_stopping import ...` | `from model_garden.training import ...` |
| `from model_garden.memory_management import ...` | `from model_garden.utils import ...` |
| `from model_garden.dataset_validator import ...` | `from model_garden.utils import ...` |
| `from model_garden.hyperparameter_explorer import ...` | `from model_garden.experiments import ...` |
| `from model_garden.exploration_visualizer import ...` | `from model_garden.experiments import ...` |
| `from model_garden.inference import ...` | `from model_garden.inference import ...` (unchanged) |
| `from model_garden.cli import main` | `from model_garden.cli import main` (unchanged, re-exports) |

## Pending Tasks

- [ ] Remove old flat module files after validation period
- [ ] Update documentation examples to use new import paths
- [ ] Update example scripts in `examples/` directory
- [ ] Consider moving `model_registry.py` to a `registry/` package

## Old Files to Remove (After Validation)

These files at `model_garden/` root are the old versions, now replaced by packages:
- `api.py` → replaced by `api/` package
- `training.py` → replaced by `training/trainer.py`
- `vision_training.py` → replaced by `training/vision_trainer.py`
- `training_utils.py` → replaced by `training/utils.py`
- `inference.py` → replaced by `inference/service.py`
- `job_queue.py` → replaced by `queue/job_queue.py`
- `queue_worker.py` → replaced by `queue/worker.py`
- `memory_management.py` → replaced by `utils/memory.py`
- `dataset_validator.py` → replaced by `utils/dataset_validator.py`
- `hyperparameter_explorer.py` → replaced by `experiments/hyperparameter_explorer.py`
- `exploration_visualizer.py` → replaced by `experiments/visualizer.py`
- `selective_loss.py` → replaced by `training/selective_loss.py`
- `early_stopping.py` → replaced by `training/early_stopping.py`
- `weighted_loss_trainer.py` → replaced by `training/weighted_loss.py`
