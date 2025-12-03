# Utils package
"""
Utility functions and classes for Model Garden:
- console.py: Shared Rich console instance
- hf_cache.py: HuggingFace cache configuration
- image.py: Image loading and processing utilities
- memory.py: Memory management utilities
- dataset_validator.py: Dataset validation and statistics
"""

# Console
from .console import console

# Dataset validation
from .dataset_validator import DatasetStats, DatasetValidator

# HuggingFace cache configuration
from .hf_cache import (
    configure_all,
    configure_hf_cache,
    configure_pytorch_memory,
    configure_unsloth_settings,
    get_hf_token,
)

# Image utilities
from .image import decode_base64_image, load_image, load_image_safe

# Memory management
from .memory import (
    cleanup_training_resources,
    clear_trainer_internals,
    get_process_memory_mb,
    report_memory_usage,
)

__all__ = [
    # Console
    "console",
    # HF Cache
    "configure_hf_cache",
    "configure_pytorch_memory",
    "configure_unsloth_settings",
    "configure_all",
    "get_hf_token",
    # Image
    "decode_base64_image",
    "load_image",
    "load_image_safe",
    # Memory
    "clear_trainer_internals",
    "cleanup_training_resources",
    "get_process_memory_mb",
    "report_memory_usage",
    # Dataset
    "DatasetStats",
    "DatasetValidator",
]
