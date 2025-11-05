"""Backend registry for managing training backends.

This module provides a registry system for dynamically discovering and
instantiating training backends.
"""

from typing import Dict, List, Optional, Type

from model_garden.backends.base import TrainingBackend

# Global registry of available backends
_BACKENDS: Dict[str, Type[TrainingBackend]] = {}


def register_backend(name: str, backend_class: Type[TrainingBackend]) -> None:
    """Register a training backend.

    Args:
        name: Name to register the backend under (e.g., 'unsloth', 'transformers')
        backend_class: The backend class to register (must inherit from TrainingBackend)

    Raises:
        ValueError: If backend_class doesn't inherit from TrainingBackend
    """
    if not issubclass(backend_class, TrainingBackend):
        raise ValueError(
            f"Backend class {backend_class.__name__} must inherit from TrainingBackend"
        )

    _BACKENDS[name.lower()] = backend_class


def get_backend(name: str = "unsloth") -> TrainingBackend:
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
        available = ", ".join(_BACKENDS.keys())
        raise ValueError(
            f"Backend '{name}' not found. Available backends: {available}"
        )

    backend_class = _BACKENDS[name_lower]
    return backend_class()


def list_backends() -> List[Dict[str, str]]:
    """List all registered backends with their information.

    Returns:
        List of dicts containing backend name, description, and capabilities
    """
    backends = []

    for name, backend_class in _BACKENDS.items():
        # Instantiate to get properties (backends should be lightweight)
        backend = backend_class()
        backends.append({
            "name": name,
            "description": backend.description,
            "supports_text": backend.supports_text_training(),
            "supports_vision": backend.supports_vision_training(),
        })

    return backends


def is_backend_available(name: str) -> bool:
    """Check if a backend is available.

    Args:
        name: Name of the backend to check

    Returns:
        True if the backend is registered, False otherwise
    """
    return name.lower() in _BACKENDS
