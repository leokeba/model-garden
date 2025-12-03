"""Optional dependency checking utilities.

This module provides a single source of truth for checking optional dependencies
like Unsloth. All availability checks should use these functions.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def is_unsloth_installed() -> bool:
    """Check if Unsloth is installed and importable.

    Returns:
        True if Unsloth is available, False otherwise.

    Note:
        Result is cached after first call for performance.
        Catches both ImportError (not installed) and other exceptions
        that may occur during import (e.g., version incompatibilities
        with dependencies like datasets).
    """
    try:
        import unsloth  # noqa: F401

        return True
    except (ImportError, NotImplementedError, Exception):
        # ImportError: Unsloth not installed
        # NotImplementedError: Unsloth has version conflicts (e.g., datasets 4.4.0)
        # Exception: Other import-time errors
        return False


def require_unsloth(feature_name: str = "This feature") -> None:
    """Raise ImportError if Unsloth is not installed.

    Args:
        feature_name: Name of the feature requiring Unsloth (for error message)

    Raises:
        ImportError: If Unsloth is not installed
    """
    if not is_unsloth_installed():
        raise ImportError(
            f"{feature_name} requires Unsloth. "
            "Install it with: pip install 'model-garden[unsloth]' "
            "or: pip install unsloth"
        )


def get_unsloth_import_error() -> str:
    """Get a helpful error message for missing Unsloth installation.

    Returns:
        Error message string with installation instructions
    """
    return (
        "Unsloth is not installed. The Unsloth backend provides 2x faster training "
        "and 60% memory savings.\n\n"
        "To install Unsloth:\n"
        "  pip install 'model-garden[unsloth]'\n"
        "  # or\n"
        "  pip install unsloth\n\n"
        "Alternatively, use the 'transformers' backend which works without Unsloth:\n"
        "  model-garden train --backend transformers ..."
    )
