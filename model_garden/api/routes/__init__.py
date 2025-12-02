# Routes package - API route handlers organized by domain
"""
API routes organized by functionality:
- models: Model management (list, info, rename, delete, upload to Hub)
- training: Training job management and queue
- inference: Model loading, chat completions, text generation
- datasets: Dataset management and validation
- carbon: Carbon emissions tracking and BoAmps reports
- system: System status and GPU management
"""

from .carbon import router as carbon_router
from .datasets import router as datasets_router
from .inference import router as inference_router
from .models import router as models_router
from .system import router as system_router
from .training import router as training_router

__all__ = [
    "models_router",
    "training_router",
    "inference_router",
    "datasets_router",
    "carbon_router",
    "system_router",
]
