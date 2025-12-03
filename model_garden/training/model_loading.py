"""Model loading utilities with retry support.

This module provides shared model loading functionality with retry logic
and exponential backoff for handling network failures. Can be used by
both text and vision trainers.
"""

import time
from typing import Any, Protocol, TypeVar

from model_garden.training.constants import (
    DEFAULT_RETRY_ATTEMPTS,
    RETRY_BASE_DELAY_SECONDS,
    RETRY_EXPONENTIAL_BACKOFF,
    RETRY_MAX_DELAY_SECONDS,
)
from model_garden.utils.console import console


T = TypeVar("T")


class ModelLoader(Protocol):
    """Protocol for model loading functions."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class ModelLoadingError(Exception):
    """Exception raised when model loading fails after all retries."""

    def __init__(
        self,
        model_name: str,
        attempts: int,
        last_error: Exception | None = None,
    ):
        self.model_name = model_name
        self.attempts = attempts
        self.last_error = last_error
        message = f"Failed to load model '{model_name}' after {attempts} attempts"
        if last_error:
            message += f": {last_error}"
        super().__init__(message)


def with_retry(
    func: ModelLoader,
    model_name: str,
    max_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    base_delay: float = RETRY_BASE_DELAY_SECONDS,
    backoff_factor: float = RETRY_EXPONENTIAL_BACKOFF,
    max_delay: float = RETRY_MAX_DELAY_SECONDS,
    verbose: bool = True,
) -> Any:
    """Execute a function with retry logic and exponential backoff.

    This is useful for model loading operations that may fail due to
    network issues when downloading from HuggingFace Hub.

    Args:
        func: The function to execute (should be a callable)
        model_name: Name of the model (for error messages)
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        backoff_factor: Multiplier for delay after each retry
        max_delay: Maximum delay between retries (seconds)
        verbose: Whether to print retry messages

    Returns:
        The result of the function call

    Raises:
        ModelLoadingError: If all retry attempts fail

    Example:
        >>> result = with_retry(
        ...     lambda: load_model("org/model"),
        ...     model_name="org/model",
        ...     max_attempts=3
        ... )
    """
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                delay = min(
                    base_delay * (backoff_factor ** attempt),
                    max_delay,
                )
                if verbose:
                    console.print(
                        f"[yellow]⚠️  Model loading attempt {attempt + 1}/{max_attempts} "
                        f"failed: {e}[/yellow]"
                    )
                    console.print(f"[yellow]   Retrying in {delay:.1f}s...[/yellow]")
                time.sleep(delay)

    raise ModelLoadingError(model_name, max_attempts, last_error)


class ModelLoaderWithFallback:
    """Load models with automatic fallback to alternative loaders.

    This class supports loading models with a primary loader (e.g., Unsloth)
    and automatically falling back to alternative loaders (e.g., Transformers)
    if the primary fails.

    Example:
        >>> loader = ModelLoaderWithFallback(
        ...     model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        ...     primary_loader=load_with_unsloth,
        ...     fallback_loaders=[load_with_transformers],
        ... )
        >>> model, tokenizer = loader.load()
    """

    def __init__(
        self,
        model_name: str,
        primary_loader: ModelLoader,
        fallback_loaders: list[ModelLoader] | None = None,
        verbose: bool = True,
    ):
        """Initialize the loader.

        Args:
            model_name: Name/path of the model to load
            primary_loader: Primary loading function to try first
            fallback_loaders: List of fallback loaders to try if primary fails
            verbose: Whether to print loading progress
        """
        self.model_name = model_name
        self.primary_loader = primary_loader
        self.fallback_loaders = fallback_loaders or []
        self.verbose = verbose

    def load(self, **kwargs: Any) -> Any:
        """Load the model using primary or fallback loaders.

        Args:
            **kwargs: Additional arguments passed to loaders

        Returns:
            The loaded model (format depends on loader)

        Raises:
            ModelLoadingError: If all loaders fail
        """
        # Try primary loader first
        try:
            result = with_retry(
                lambda: self.primary_loader(**kwargs),
                model_name=self.model_name,
                verbose=self.verbose,
            )
            if self.verbose:
                console.print("[green]✓ Model loaded with primary loader[/green]")
            return result
        except ModelLoadingError as e:
            if not self.fallback_loaders:
                raise e
            if self.verbose:
                console.print(
                    f"[yellow]⚠️  Primary loader failed: {e.last_error}[/yellow]"
                )

        # Try fallback loaders
        last_error: Exception | None = None
        for i, loader in enumerate(self.fallback_loaders, 1):
            try:
                result = with_retry(
                    lambda: loader(**kwargs),
                    model_name=self.model_name,
                    verbose=self.verbose,
                )
                if self.verbose:
                    console.print(f"[green]✓ Model loaded with fallback loader {i}[/green]")
                return result
            except ModelLoadingError as e:
                last_error = e.last_error
                if self.verbose:
                    console.print(
                        f"[yellow]⚠️  Fallback loader {i} failed: {e.last_error}[/yellow]"
                    )

        raise ModelLoadingError(
            self.model_name,
            DEFAULT_RETRY_ATTEMPTS * (1 + len(self.fallback_loaders)),
            last_error,
        )


def get_hf_token_or_none() -> str | None:
    """Get HuggingFace token from environment, returning None if not set.

    This is a convenience wrapper that returns None instead of raising
    an error if no token is found.
    """
    try:
        from model_garden.utils.hf_cache import get_hf_token
        return get_hf_token()
    except Exception:
        return None
