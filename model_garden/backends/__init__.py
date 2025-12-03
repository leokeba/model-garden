"""Backwards compatibility shim for model_garden.backends.

This module is deprecated. Please import from model_garden.training.backends instead.

All backend functionality has been moved to model_garden.training.backends to
reflect that backends are training-specific components.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "model_garden.backends is deprecated. Please use model_garden.training.backends instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new location for backwards compatibility
from model_garden.training.backends import (
    TextTrainer,
    TrainingBackend,
    VisionTrainer,
    get_backend,
    list_backends,
    register_backend,
)
from model_garden.training.backends.registry import (
    get_registered_backend_names,
    is_backend_available,
)

__all__ = [
    "TrainingBackend",
    "TextTrainer",
    "VisionTrainer",
    "get_backend",
    "list_backends",
    "register_backend",
    "is_backend_available",
    "get_registered_backend_names",
]
