"""Image loading and processing utilities.

This module provides centralized image loading functions for use across
Model Garden, including training and inference services.
"""

import base64
import io
import os
from typing import Any

from PIL import Image


def decode_base64_image(image_str: str) -> Image.Image:
    """Decode a base64-encoded image string to PIL Image.

    Args:
        image_str: Base64-encoded image string (with or without data URI prefix)

    Returns:
        PIL Image object

    Raises:
        ValueError: If the base64 string cannot be decoded

    Example:
        >>> img = decode_base64_image("data:image/png;base64,iVBORw0KGgo...")
        >>> img.size
        (100, 100)
    """
    try:
        # Remove data URI prefix if present (e.g., "data:image/jpeg;base64,")
        if image_str.startswith("data:"):
            image_str = image_str.split(",", 1)[1]

        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_str)

        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {e}") from e


def load_image(
    image_data: Any,
    fallback_size: tuple[int, int] = (224, 224),
    convert_to_rgb: bool = True,
) -> Image.Image:
    """Load image from various sources (file path, base64, PIL Image, etc.).

    This function handles multiple input formats and returns a consistent
    PIL Image object.

    Args:
        image_data: Image data in one of the following formats:
            - PIL.Image.Image: Returned as-is (optionally converted to RGB)
            - str (file path): Loaded from disk
            - str (base64): Decoded from base64 string
            - str (data URI): Decoded from data URI format
        fallback_size: Size for blank fallback image if loading fails
        convert_to_rgb: Whether to convert image to RGB format

    Returns:
        PIL Image object (in RGB format if convert_to_rgb is True)

    Note:
        Images are loaded fully into memory (no lazy loading) to ensure
        consistent behavior across different use cases.

    Example:
        >>> img = load_image("/path/to/image.jpg")
        >>> img.mode
        'RGB'
        >>> img = load_image("data:image/png;base64,...")
        >>> img.size
        (100, 100)
    """
    # Already a PIL Image
    if isinstance(image_data, Image.Image):
        if convert_to_rgb and image_data.mode != "RGB":
            return image_data.convert("RGB")
        return image_data

    # Check for PIL.Image subclasses (PngImageFile, JpegImageFile, etc.)
    if image_data is not None and hasattr(image_data, "mode") and hasattr(image_data, "convert"):
        # It's an image-like object
        if convert_to_rgb and image_data.mode != "RGB":
            return image_data.convert("RGB")
        return image_data

    # String input - could be file path or base64
    if isinstance(image_data, str):
        # Check if it looks like base64 data
        is_data_uri = image_str.startswith("data:image") if (image_str := image_data) else False
        is_long_string = len(image_data) > 100 and not os.path.exists(image_data)

        if is_data_uri or is_long_string:
            # Looks like base64
            try:
                img = decode_base64_image(image_data)
                if convert_to_rgb and img.mode != "RGB":
                    img = img.convert("RGB")
                return img
            except ValueError:
                # Fall through to fallback
                pass
        elif os.path.exists(image_data):
            # Load image from file
            img = Image.open(image_data)
            # Convert to RGB to ensure consistent format
            if convert_to_rgb and img.mode != "RGB":
                img = img.convert("RGB")
            # Force load to ensure pixels are in memory (avoid lazy loading)
            img.load()
            return img

    # Fallback: create blank image
    return Image.new("RGB", fallback_size)


def load_image_safe(
    image_data: Any,
    fallback_size: tuple[int, int] = (224, 224),
    convert_to_rgb: bool = True,
    warn_on_fallback: bool = True,
) -> Image.Image:
    """Load image with warning on fallback (for use in training/inference).

    Same as load_image() but prints a warning when using the fallback image.
    This is useful for training and inference where you want to know if
    image loading failed.

    Args:
        image_data: Image data (see load_image for supported formats)
        fallback_size: Size for blank fallback image if loading fails
        convert_to_rgb: Whether to convert image to RGB format
        warn_on_fallback: Whether to print a warning when using fallback

    Returns:
        PIL Image object
    """
    # Try the standard load first
    result = load_image(image_data, fallback_size, convert_to_rgb)

    # Check if we got a fallback image (blank image of fallback_size)
    if warn_on_fallback and result.size == fallback_size:
        # Check if original was not already a valid image
        original_is_image = isinstance(image_data, Image.Image) or (
            hasattr(image_data, "mode") and hasattr(image_data, "convert")
        )
        if not original_is_image:
            from model_garden.utils.console import console

            console.print(
                f"[yellow]⚠️  Unknown image format (type: {type(image_data).__name__}), "
                f"using blank image[/yellow]"
            )

    return result
