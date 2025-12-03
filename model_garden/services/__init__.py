"""Service layer for Model Garden.

This module provides backend-agnostic service classes that encapsulate
business logic for training, inference, and dataset operations.

Both the CLI and API use these services as the single source of truth,
eliminating code duplication and ensuring consistent behavior.

Architecture:
    ┌─────────────────┐     ┌─────────────────┐
    │   CLI           │     │   FastAPI API   │
    │ (cli/*.py)      │     │ (api/routes/)   │
    └────────┬────────┘     └────────┬────────┘
             │                       │
             ▼                       ▼
    ┌────────────────────────────────────────┐
    │          Service Layer                 │
    │  (TrainingService, InferenceService,   │
    │   DatasetService)                      │
    └────────────────────┬───────────────────┘
                         │
                         ▼
    ┌────────────────────────────────────────┐
    │       Training/Inference Backends      │
    │  (training/backends/, inference/)      │
    └────────────────────────────────────────┘

Usage:
    >>> from model_garden.services import TrainingService, TrainingRequest
    >>> service = TrainingService()
    >>> request = TrainingRequest(
    ...     name="my-model",
    ...     base_model="unsloth/tinyllama-bnb-4bit",
    ...     dataset_path="./data/train.jsonl",
    ...     output_dir="./models/my-model"
    ... )
    >>> result = service.train(request)
"""

from model_garden.services.dataset_service import (
    DatasetInfo,
    DatasetService,
    DatasetValidationResult,
)
from model_garden.services.inference_service import (
    InferenceRequest,
    InferenceService,
    ModelLoadRequest,
)
from model_garden.services.training_service import (
    TrainingRequest,
    TrainingService,
)

__all__ = [
    # Training
    "TrainingService",
    "TrainingRequest",
    # Inference
    "InferenceService",
    "InferenceRequest",
    "ModelLoadRequest",
    # Dataset
    "DatasetService",
    "DatasetInfo",
    "DatasetValidationResult",
]
