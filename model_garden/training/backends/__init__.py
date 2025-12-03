"""Training backends for Model Garden.

This module provides an abstraction layer for different training backends,
allowing Model Garden to support multiple training frameworks (Unsloth, Transformers, etc.).
"""

from model_garden.training.backends.base import TextTrainer, TrainingBackend, VisionTrainer
from model_garden.training.backends.registry import get_backend, list_backends, register_backend


# Auto-register available backends
def _register_backends():
    """Register all available backends."""
    # Register Unsloth backend (always available)
    try:
        from model_garden.training.backends.unsloth_backend import UnslothBackend

        register_backend("unsloth", UnslothBackend)
    except ImportError as e:
        # Log the error but don't fail - backend might not be available
        import sys

        print(f"Warning: Failed to register Unsloth backend: {e}", file=sys.stderr)
    except Exception as e:
        # Log unexpected errors
        import sys

        print(f"Error registering Unsloth backend: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()

    # Register Transformers backend (standard HF)
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
    "list_backends",
    "register_backend",
]
