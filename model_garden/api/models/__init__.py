"""Pydantic models for the Model Garden API.

This package contains all request/response schemas used by the API.
"""

from model_garden.api.models.common import (
    APIResponse,
    PaginatedResponse,
)
from model_garden.api.models.datasets import (
    DatasetValidationRequest,
    DatasetValidationResponse,
)
from model_garden.api.models.inference import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    CompletionRequest,
)
from model_garden.api.models.models import (
    ModelInfo,
    ModelRenameRequest,
)
from model_garden.api.models.training import (
    TrainingJobInfo,
    TrainingJobRequest,
)

__all__ = [
    # Common
    "APIResponse",
    "PaginatedResponse",
    # Training
    "TrainingJobRequest",
    "TrainingJobInfo",
    # Models
    "ModelInfo",
    "ModelRenameRequest",
    # Inference
    "ChatCompletionRequest",
    "ChatCompletionMessage",
    "CompletionRequest",
    # Datasets
    "DatasetValidationRequest",
    "DatasetValidationResponse",
]
