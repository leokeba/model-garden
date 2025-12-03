"""Backend registry for managing training backends.

This module provides a registry system for dynamically discovering and
instantiating training backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from model_garden.training.backends.base import TrainingBackend

# Type definitions for better IDE support
# Using TypeAlias for Python 3.11 compatibility (type syntax requires Python 3.12)
from typing import TypeAlias

BackendClass: TypeAlias = type["TrainingBackend"]


class BackendInfo(TypedDict):
    """Type definition for backend information returned by list_backends()."""

    name: str
    description: str
    supports_text: bool
    supports_vision: bool


# Global registry of available backends
_BACKENDS: dict[str, BackendClass] = {}

# Default backend name
DEFAULT_BACKEND: str = "unsloth"


def register_backend(name: str, backend_class: BackendClass) -> None:
    """Register a training backend.

    Args:
        name: Name to register the backend under (e.g., 'unsloth', 'transformers')
        backend_class: The backend class to register (must inherit from TrainingBackend)

    Raises:
        ValueError: If backend_class doesn't inherit from TrainingBackend
        TypeError: If backend_class is not a class
    """
    from model_garden.training.backends.base import TrainingBackend

    if not isinstance(backend_class, type):
        raise TypeError(f"Expected a class, got {type(backend_class).__name__}")

    if not issubclass(backend_class, TrainingBackend):
        raise ValueError(
            f"Backend class {backend_class.__name__} must inherit from TrainingBackend"
        )

    _BACKENDS[name.lower()] = backend_class


def get_backend(name: str = DEFAULT_BACKEND) -> TrainingBackend:
    """Get a training backend instance by name.

    Args:
        name: Name of the backend to instantiate (default: 'unsloth')

    Returns:
        An instance of the requested backend

    Raises:
        ValueError: If the backend is not registered
    """
    name_lower = name.lower()

    if name_lower not in _BACKENDS:
        available = ", ".join(sorted(_BACKENDS.keys()))
        raise ValueError(f"Backend '{name}' not found. Available backends: {available}")

    backend_class = _BACKENDS[name_lower]
    return backend_class()


def list_backends() -> list[BackendInfo]:
    """List all registered backends with their information.

    Returns:
        List of BackendInfo dicts containing backend name, description, and capabilities
    """
    backends: list[BackendInfo] = []

    for name in sorted(_BACKENDS.keys()):
        backend_class = _BACKENDS[name]
        # Instantiate to get properties (backends should be lightweight)
        backend = backend_class()
        backends.append(
            BackendInfo(
                name=name,
                description=backend.description,
                supports_text=backend.supports_text_training(),
                supports_vision=backend.supports_vision_training(),
            )
        )

    return backends


def is_backend_available(name: str) -> bool:
    """Check if a backend is available.

    Args:
        name: Name of the backend to check

    Returns:
        True if the backend is registered, False otherwise
    """
    return name.lower() in _BACKENDS


def get_registered_backend_names() -> list[str]:
    """Get list of all registered backend names.

    Returns:
        Sorted list of backend names
    """
    return sorted(_BACKENDS.keys())
