"""Lazy loading dataset for vision-language models.

This module provides the LazyVisionDataset class which loads images on-demand
during training instead of loading all images into memory at once.
"""

from collections.abc import Callable
from typing import Any

from PIL import Image


class LazyVisionDataset:
    """A dataset wrapper that loads images on-demand to prevent memory exhaustion.

    Instead of loading all PIL Images into RAM at format time (which can easily
    exhaust memory with 1000+ images), this class stores only the image references
    (file paths or base64 strings) and loads them lazily when accessed.

    This is particularly important for:
    - Large vision datasets (1000+ images)
    - High-resolution images
    - Multi-epoch training where data is iterated multiple times

    The tradeoff is slightly more I/O during training, but dramatically reduced
    memory usage. For SSD storage, the I/O overhead is minimal.

    Attributes:
        examples: List of example metadata (text, image reference, response, etc.)
        system_message: System message to use for all examples
        image_loader: Function to load images from references
        _cache: Optional LRU cache for recently accessed images

    Example:
        >>> def load_image(ref):
        ...     return Image.open(ref) if isinstance(ref, str) else ref
        ...
        >>> dataset = LazyVisionDataset(examples, system_message, load_image)
        >>> len(dataset)  # Fast - just returns count
        1000
        >>> dataset[0]  # Loads image on-demand
        {"messages": [...]}  # With PIL Image loaded
    """

    def __init__(
        self,
        examples: list[dict],
        system_message: str,
        image_loader: Callable[[Any], Image.Image],
        cache_size: int = 0,
    ):
        """Initialize the lazy dataset.

        Args:
            examples: List of example dicts with text, image reference, and response.
                     Image references can be file paths, base64 strings, or URLs.
            system_message: System message to prepend to each example.
            image_loader: Function to load PIL Image from image reference.
            cache_size: Number of images to cache in memory (0 = no caching).
                       Caching can help if the same images are accessed repeatedly,
                       but uses memory proportional to cache_size * avg_image_size.
        """
        self.examples = examples
        self.system_message = system_message
        self.image_loader = image_loader
        self._cache: dict[int, Image.Image] | None = None
        self._cache_size = cache_size
        if cache_size > 0:
            self._cache = {}
            self._cache_order: list[int] = []

    def __len__(self) -> int:
        """Return the number of examples (fast, no I/O)."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        """Load and return a single example with its image.

        This method is called by the DataLoader during training. The image
        is loaded on-demand here, keeping memory usage proportional to
        batch_size rather than dataset_size.

        Args:
            idx: Index of the example to load

        Returns:
            Formatted message dict with PIL Image loaded
        """
        example = self.examples[idx]

        # Check cache first
        pil_image = None
        if self._cache is not None and idx in self._cache:
            pil_image = self._cache[idx]
        else:
            # Load image on-demand
            image_ref = example.get("image")
            pil_image = self.image_loader(image_ref)

            # Update cache if enabled
            if self._cache is not None:
                # Simple LRU eviction
                if len(self._cache) >= self._cache_size:
                    oldest = self._cache_order.pop(0)
                    del self._cache[oldest]
                self._cache[idx] = pil_image
                self._cache_order.append(idx)

        # Get text and response
        text = example.get("text", "")
        response = example.get("response", "")
        effective_system = example.get("system", self.system_message)

        # Format as OpenAI messages
        return {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": effective_system}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}],
                },
            ],
        }

    def clear_cache(self) -> None:
        """Clear the image cache to free memory."""
        if self._cache is not None:
            self._cache.clear()
            self._cache_order.clear()


class LazyVisionDatasetWithMultipleImages(LazyVisionDataset):
    """Extension of LazyVisionDataset that supports multiple images per example.

    Some vision tasks require multiple images in a single prompt (e.g., comparing
    images, before/after analysis). This class extends LazyVisionDataset to
    support an "images" key containing a list of image references.

    Example:
        >>> examples = [
        ...     {
        ...         "text": "Compare these two images",
        ...         "images": ["image1.jpg", "image2.jpg"],
        ...         "response": "The first image shows..."
        ...     }
        ... ]
        >>> dataset = LazyVisionDatasetWithMultipleImages(examples, system_msg, loader)
    """

    def __getitem__(self, idx: int) -> dict:
        """Load and return a single example with multiple images.

        Args:
            idx: Index of the example to load

        Returns:
            Formatted message dict with PIL Images loaded
        """
        example = self.examples[idx]

        # Get text and response
        text = example.get("text", "")
        response = example.get("response", "")
        effective_system = example.get("system", self.system_message)

        # Build user content with multiple images
        user_content: list[dict[str, Any]] = []

        # Handle single image (backwards compatible)
        if "image" in example:
            image_ref = example["image"]
            pil_image = self._load_with_cache(idx, 0, image_ref)
            user_content.append({"type": "image", "image": pil_image})

        # Handle multiple images
        if "images" in example:
            for img_idx, image_ref in enumerate(example["images"]):
                pil_image = self._load_with_cache(idx, img_idx, image_ref)
                user_content.append({"type": "image", "image": pil_image})

        # Add text content
        user_content.append({"type": "text", "text": text})

        # Format as OpenAI messages
        return {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": effective_system}],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}],
                },
            ],
        }

    def _load_with_cache(self, example_idx: int, image_idx: int, image_ref: Any) -> Image.Image:
        """Load image with optional caching.

        Args:
            example_idx: Index of the example
            image_idx: Index of the image within the example
            image_ref: Reference to the image (path, base64, etc.)

        Returns:
            Loaded PIL Image
        """
        # Create composite cache key
        cache_key = example_idx * 1000 + image_idx  # Assumes < 1000 images per example

        if self._cache is not None and cache_key in self._cache:
            return self._cache[cache_key]

        pil_image = self.image_loader(image_ref)

        if self._cache is not None:
            if len(self._cache) >= self._cache_size:
                oldest = self._cache_order.pop(0)
                del self._cache[oldest]
            self._cache[cache_key] = pil_image
            self._cache_order.append(cache_key)

        return pil_image
