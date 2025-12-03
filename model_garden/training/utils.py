"""Shared utilities for training configuration.

DEPRECATED: This module is maintained for backward compatibility.
New code should import from model_garden.training.mixins instead.

This module re-exports the following from mixins.py:
- detect_model_dtype
- get_training_precision_config
- MemoryMonitorCallback
- cleanup_memory
"""

import warnings

# Re-export from mixins.py for backward compatibility
from model_garden.training.mixins import (
    MemoryMonitorCallback,
    cleanup_memory,
    detect_model_dtype,
    get_training_precision_config,
)

# Emit deprecation warning on import
warnings.warn(
    "model_garden.training.utils is deprecated. Import from model_garden.training.mixins instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "detect_model_dtype",
    "get_training_precision_config",
    "MemoryMonitorCallback",
    "cleanup_memory",
]
