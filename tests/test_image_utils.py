"""Tests for image loading utilities."""

import base64
import io
from pathlib import Path

import pytest
from PIL import Image

from model_garden.utils.image import decode_base64_image, load_image


class TestDecodeBase64Image:
    """Tests for decode_base64_image function."""

    def test_decode_simple_base64(self):
        """Test decoding a simple base64 image."""
        # Create a small test image
        img = Image.new("RGB", (10, 10), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode()

        # Decode it
        result = decode_base64_image(base64_str)

        assert isinstance(result, Image.Image)
        assert result.size == (10, 10)

    def test_decode_data_uri_format(self):
        """Test decoding a data URI formatted base64 image."""
        # Create a small test image
        img = Image.new("RGB", (10, 10), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode()
        data_uri = f"data:image/png;base64,{base64_str}"

        # Decode it
        result = decode_base64_image(data_uri)

        assert isinstance(result, Image.Image)
        assert result.size == (10, 10)

    def test_decode_invalid_base64_raises_error(self):
        """Test that invalid base64 raises ValueError."""
        with pytest.raises(ValueError, match="Failed to decode"):
            decode_base64_image("not-valid-base64!!!")


class TestLoadImage:
    """Tests for load_image function."""

    def test_load_pil_image(self):
        """Test loading a PIL Image directly."""
        img = Image.new("RGBA", (20, 20), color="green")
        result, success = load_image(img)

        assert success is True
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"  # Should be converted to RGB
        assert result.size == (20, 20)

    def test_load_rgb_pil_image(self):
        """Test loading an RGB PIL Image (no conversion needed)."""
        img = Image.new("RGB", (20, 20), color="yellow")
        result, success = load_image(img)

        assert success is True
        assert result is img  # Should return same image
        assert result.mode == "RGB"

    def test_load_base64_string(self):
        """Test loading from base64 string."""
        # Create a larger test image (load_image only tries base64 for strings > 200 chars)
        img = Image.new("RGB", (100, 100), color="purple")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode()

        result, success = load_image(base64_str)

        assert success is True
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (100, 100)

    def test_load_data_uri_string(self):
        """Test loading from data URI string."""
        img = Image.new("RGB", (12, 12), color="orange")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        base64_str = base64.b64encode(buffer.getvalue()).decode()
        data_uri = f"data:image/jpeg;base64,{base64_str}"

        result, success = load_image(data_uri)

        assert success is True
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (12, 12)

    def test_load_file_path(self, tmp_path: Path):
        """Test loading from file path."""
        # Create and save a test image
        img = Image.new("RGB", (25, 25), color="cyan")
        img_path = tmp_path / "test_image.png"
        img.save(img_path)

        result, success = load_image(str(img_path))

        assert success is True
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (25, 25)

    def test_load_file_path_converts_to_rgb(self, tmp_path: Path):
        """Test that loading from file converts to RGB."""
        # Create and save an RGBA image
        img = Image.new("RGBA", (25, 25), color=(255, 0, 0, 128))
        img_path = tmp_path / "test_rgba.png"
        img.save(img_path)

        result, success = load_image(str(img_path))

        assert success is True
        assert result.mode == "RGB"

    def test_fallback_for_unknown_type(self):
        """Test that unknown types get a fallback blank image."""
        result, success = load_image(12345)  # Invalid type

        assert success is False
        assert isinstance(result, Image.Image)
        assert result.size == (224, 224)  # Default fallback size

    def test_custom_fallback_size(self):
        """Test custom fallback size."""
        result, success = load_image(None, fallback_size=(100, 100))

        assert success is False
        assert result.size == (100, 100)

    def test_skip_rgb_conversion(self):
        """Test skipping RGB conversion."""
        img = Image.new("L", (20, 20), color=128)  # Grayscale
        result, success = load_image(img, convert_to_rgb=False)

        assert success is True
        assert result.mode == "L"  # Should stay grayscale

    def test_nonexistent_file_returns_fallback(self):
        """Test that a nonexistent file path returns fallback."""
        result, success = load_image("/nonexistent/path/to/image.jpg")

        assert success is False
        assert isinstance(result, Image.Image)
        assert result.size == (224, 224)
