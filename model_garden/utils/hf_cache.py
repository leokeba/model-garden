"""HuggingFace cache configuration utilities.

This module provides centralized configuration for HuggingFace-related
environment variables. Import and call configure_hf_cache() at the top
of any module that uses HuggingFace libraries BEFORE importing them.

This avoids duplicating the same environment setup code across multiple files.
"""

import os
from pathlib import Path


def configure_hf_cache() -> dict[str, str]:
    """Configure HuggingFace cache environment variables.

    Sets up HF_HOME, TRANSFORMERS_CACHE, HF_DATASETS_CACHE, and HUGGINGFACE_HUB_CACHE
    based on the HF_HOME environment variable (or defaults to ~/.cache/huggingface).

    This function should be called BEFORE importing any HuggingFace libraries
    (transformers, datasets, huggingface_hub) to ensure the cache paths are
    set correctly.

    Returns:
        Dictionary containing the configured cache paths:
        - hf_home: Base HuggingFace home directory
        - transformers_cache: Transformers model cache
        - datasets_cache: Datasets cache
        - hub_cache: HuggingFace Hub cache

    Example:
        >>> from model_garden.utils.hf_cache import configure_hf_cache
        >>> configure_hf_cache()  # Call this first!
        >>> from transformers import AutoModel  # Now import HF libraries
    """
    # Load from .env file if available
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # dotenv is optional

    # Get or set HF_HOME
    hf_home = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

    # Set all HuggingFace-related environment variables
    os.environ["HF_HOME"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = str(Path(hf_home) / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(Path(hf_home) / "datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(hf_home) / "hub")

    return {
        "hf_home": hf_home,
        "transformers_cache": os.environ["TRANSFORMERS_CACHE"],
        "datasets_cache": os.environ["HF_DATASETS_CACHE"],
        "hub_cache": os.environ["HUGGINGFACE_HUB_CACHE"],
    }


def get_hf_token() -> str | None:
    """Get the HuggingFace token from environment.

    Checks HF_TOKEN environment variable.

    Returns:
        The HuggingFace token or None if not set.
    """
    return os.getenv("HF_TOKEN")


def configure_pytorch_memory() -> None:
    """Configure PyTorch CUDA memory settings for optimal performance.

    Sets PYTORCH_CUDA_ALLOC_CONF and PYTORCH_ALLOC_CONF to reduce
    memory fragmentation and enable expandable segments.

    This should be called before importing PyTorch/torch.
    """
    # Configure CUDA memory allocator for better performance
    # max_split_size_mb limits memory fragmentation
    # expandable_segments allows dynamic growth
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def configure_unsloth_settings() -> None:
    """Configure Unsloth-specific settings.

    Disables statistics collection to avoid thread safety issues
    (stats collection uses signal.alarm which must be called from main thread).
    """
    os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"


def configure_all() -> dict[str, str]:
    """Configure all environment variables for Model Garden.

    Convenience function that calls all configuration functions:
    - configure_hf_cache()
    - configure_pytorch_memory()
    - configure_unsloth_settings()

    Returns:
        Dictionary from configure_hf_cache() with cache paths.

    Example:
        >>> from model_garden.utils.hf_cache import configure_all
        >>> configure_all()
        >>> # Now safe to import any ML libraries
    """
    cache_paths = configure_hf_cache()
    configure_pytorch_memory()
    configure_unsloth_settings()
    return cache_paths
