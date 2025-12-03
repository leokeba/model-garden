# API package - FastAPI application and related components
"""
Modular API package for Model Garden.

This package provides a well-organized FastAPI application with:
- models/: Pydantic request/response models
- routes/: API route handlers organized by domain
- websocket.py: WebSocket connection management
- storage.py: Persistent storage management
- tasks.py: Background task functions
- app.py: FastAPI application factory

Usage:
    from model_garden.api import app
    # or
    from model_garden.api import create_app
    app = create_app()
"""

from .app import app, create_app

# Re-export commonly used models
from .models import (
    APIResponse,
    ModelInfo,
    ModelRenameRequest,
    PaginatedResponse,
    TrainingJobInfo,
    TrainingJobRequest,
)

# Re-export inference models and utilities
from .routes.inference import (
    ChatCompletionRequest,
    ChatMessage,
    InferenceRequest,
    ResponseFormat,
    convert_response_format_to_structured_outputs,
)
from .storage import StorageManager, get_storage_manager
from .tasks import (
    cancellation_events,
    create_progress_callback,
    early_stop_requests,
    run_model_loading,
    run_training_job,
)
from .websocket import ConnectionManager, get_connection_manager

# Backward compatibility: provide training_jobs as a module-level dict
# that reads from storage. Note that this dict is loaded once at import time.
# For real-time data, use get_storage_manager().load_training_jobs() instead.
_storage = get_storage_manager()
training_jobs = _storage.load_training_jobs()


__all__ = [
    # Application
    "app",
    "create_app",
    # Storage
    "StorageManager",
    "get_storage_manager",
    # WebSocket
    "ConnectionManager",
    "get_connection_manager",
    # Background tasks
    "run_training_job",
    "run_model_loading",
    "create_progress_callback",
    "cancellation_events",
    "early_stop_requests",
    # Models
    "APIResponse",
    "PaginatedResponse",
    "TrainingJobRequest",
    "TrainingJobInfo",
    "ModelInfo",
    "ModelRenameRequest",
    # Inference models
    "ChatCompletionRequest",
    "ChatMessage",
    "InferenceRequest",
    "ResponseFormat",
    "convert_response_format_to_structured_outputs",
    # Backward compatibility
    "training_jobs",
]
