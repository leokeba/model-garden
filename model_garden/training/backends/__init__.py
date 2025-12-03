"""Training backends for Model Garden.

This module provides an abstraction layer for different training backends,
allowing Model Garden to support multiple training frameworks (Unsloth, Transformers, etc.).
"""

from model_garden.training.backends.base import TextTrainer, TrainingBackend, VisionTrainer
from model_garden.training.backends.registry import (
    get_backend,
    get_default_backend,
    list_backends,
    register_backend,
)


# Auto-register available backends
def _register_backends():
    """Register all available backends.

    Backends are registered with try/except to gracefully handle missing dependencies.
    - Unsloth backend: Requires 'unsloth' package (optional dependency)
    - Transformers backend: Uses standard HuggingFace + PEFT (always available)
    """
    # Register Unsloth backend (optional - requires unsloth package)
    try:
        from model_garden.utils.optional_deps import is_unsloth_installed

        if is_unsloth_installed():
            from model_garden.training.backends.unsloth_backend import UnslothBackend

            register_backend("unsloth", UnslothBackend)
        # If unsloth not installed, silently skip - transformers backend will be default
    except Exception as e:
        # Log unexpected errors but don't fail
        import sys

        print(f"Warning: Error checking Unsloth availability: {e}", file=sys.stderr)

    # Register Transformers backend (standard HF - always available)
    try:
        from model_garden.training.backends.transformers_backend import TransformersBackend

        register_backend("transformers", TransformersBackend)
    except ImportError as e:
        import sys

        print(f"Warning: Failed to register Transformers backend: {e}", file=sys.stderr)
    except Exception as e:
        import sys

        print(f"Error registering Transformers backend: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()

    # Future backends can be registered here
    # e.g., DeepSpeedBackend, etc.


_register_backends()

__all__ = [
    "TrainingBackend",
    "TextTrainer",
    "VisionTrainer",
    "get_backend",
    "get_default_backend",
    "list_backends",
    "register_backend",
]
