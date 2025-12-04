"""Tests for training/lazy_dataset.py - Lazy loading dataset utilities.

These tests verify the LazyVisionDataset class which loads images on-demand
during training to prevent memory exhaustion.
"""

from unittest.mock import MagicMock

import pytest

from model_garden.training.lazy_dataset import (
    LazyVisionDataset,
    LazyVisionDatasetWithMultipleImages,
)


class TestLazyVisionDataset:
    """Tests for LazyVisionDataset class."""

    @pytest.fixture
    def mock_image_loader(self):
        """Create a mock image loader function."""

        def loader(ref):
            mock_image = MagicMock()
            mock_image.ref = ref
            return mock_image

        return loader

    @pytest.fixture
    def sample_examples(self):
        """Create sample examples for testing."""
        return [
            {"text": "What is this?", "image": "image1.jpg", "response": "A cat"},
            {"text": "Describe this.", "image": "image2.jpg", "response": "A dog"},
            {"text": "What color?", "image": "image3.jpg", "response": "Blue"},
        ]

    def test_init(self, sample_examples, mock_image_loader):
        """Test LazyVisionDataset initialization."""
        dataset = LazyVisionDataset(
            examples=sample_examples,
            system_message="Be helpful.",
            image_loader=mock_image_loader,
        )

        assert len(dataset.examples) == 3
        assert dataset.system_message == "Be helpful."
        assert dataset._cache is None  # No cache by default

    def test_init_with_cache(self, sample_examples, mock_image_loader):
        """Test initialization with caching enabled."""
        dataset = LazyVisionDataset(
            examples=sample_examples,
            system_message="Be helpful.",
            image_loader=mock_image_loader,
            cache_size=10,
        )

        assert dataset._cache is not None
        assert dataset._cache_size == 10

    def test_len(self, sample_examples, mock_image_loader):
        """Test __len__ method."""
        dataset = LazyVisionDataset(
            examples=sample_examples,
            system_message="Be helpful.",
            image_loader=mock_image_loader,
        )

        assert len(dataset) == 3

    def test_getitem_returns_formatted_message(self, sample_examples, mock_image_loader):
        """Test __getitem__ returns properly formatted message."""
        dataset = LazyVisionDataset(
            examples=sample_examples,
            system_message="Be helpful.",
            image_loader=mock_image_loader,
        )

        result = dataset[0]

        assert "messages" in result
        messages = result["messages"]
        assert len(messages) == 3

        # System message
        assert messages[0]["role"] == "system"
        assert messages[0]["content"][0]["text"] == "Be helpful."

        # User message
        assert messages[1]["role"] == "user"
        assert len(messages[1]["content"]) == 2
        assert messages[1]["content"][0]["type"] == "image"
        assert messages[1]["content"][1]["type"] == "text"
        assert messages[1]["content"][1]["text"] == "What is this?"

        # Assistant message
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"][0]["text"] == "A cat"

    def test_getitem_loads_image(self, sample_examples):
        """Test that __getitem__ calls image loader."""
        loader_calls = []

        def tracking_loader(ref):
            loader_calls.append(ref)
            return MagicMock()

        dataset = LazyVisionDataset(
            examples=sample_examples,
            system_message="Test",
            image_loader=tracking_loader,
        )

        dataset[0]
        assert "image1.jpg" in loader_calls

        dataset[1]
        assert "image2.jpg" in loader_calls

    def test_getitem_with_example_system_override(self, mock_image_loader):
        """Test that per-example system message overrides default."""
        examples = [
            {
                "text": "Question",
                "image": "img.jpg",
                "response": "Answer",
                "system": "Custom system message",
            }
        ]

        dataset = LazyVisionDataset(
            examples=examples,
            system_message="Default system",
            image_loader=mock_image_loader,
        )

        result = dataset[0]
        system_text = result["messages"][0]["content"][0]["text"]
        assert system_text == "Custom system message"

    def test_caching_behavior(self, sample_examples):
        """Test that caching prevents repeated image loads."""
        load_count = {"count": 0}

        def counting_loader(ref):
            load_count["count"] += 1
            return MagicMock()

        dataset = LazyVisionDataset(
            examples=sample_examples,
            system_message="Test",
            image_loader=counting_loader,
            cache_size=10,
        )

        # First access - should load
        dataset[0]
        assert load_count["count"] == 1

        # Second access - should use cache
        dataset[0]
        assert load_count["count"] == 1  # Still 1

        # Access different index - should load
        dataset[1]
        assert load_count["count"] == 2

    def test_cache_eviction(self):
        """Test LRU cache eviction."""
        examples = [
            {"text": f"Q{i}", "image": f"img{i}.jpg", "response": f"A{i}"} for i in range(5)
        ]

        load_count = {"count": 0}

        def counting_loader(ref):
            load_count["count"] += 1
            return MagicMock()

        dataset = LazyVisionDataset(
            examples=examples,
            system_message="Test",
            image_loader=counting_loader,
            cache_size=2,  # Small cache
        )

        # Load indices 0, 1, 2 (exceeds cache)
        dataset[0]  # Load, cache [0]
        dataset[1]  # Load, cache [0, 1]
        dataset[2]  # Load, evict 0, cache [1, 2]

        assert load_count["count"] == 3

        # Access index 0 again - should reload (was evicted)
        dataset[0]
        assert load_count["count"] == 4

        # Access index 2 again - should use cache
        dataset[2]
        assert load_count["count"] == 4  # Still 4

    def test_clear_cache(self, sample_examples, mock_image_loader):
        """Test clear_cache method."""
        dataset = LazyVisionDataset(
            examples=sample_examples,
            system_message="Test",
            image_loader=mock_image_loader,
            cache_size=10,
        )

        # Populate cache
        dataset[0]
        dataset[1]

        assert len(dataset._cache) == 2

        # Clear cache
        dataset.clear_cache()

        assert len(dataset._cache) == 0
        assert len(dataset._cache_order) == 0

    def test_clear_cache_no_cache(self, sample_examples, mock_image_loader):
        """Test clear_cache when caching is disabled."""
        dataset = LazyVisionDataset(
            examples=sample_examples,
            system_message="Test",
            image_loader=mock_image_loader,
            cache_size=0,  # No cache
        )

        # Should not raise
        dataset.clear_cache()


class TestLazyVisionDatasetWithMultipleImages:
    """Tests for LazyVisionDatasetWithMultipleImages class."""

    @pytest.fixture
    def mock_image_loader(self):
        """Create a mock image loader function."""

        def loader(ref):
            mock_image = MagicMock()
            mock_image.ref = ref
            return mock_image

        return loader

    def test_single_image_backward_compatible(self, mock_image_loader):
        """Test backward compatibility with single image field."""
        examples = [{"text": "Question", "image": "single.jpg", "response": "Answer"}]

        dataset = LazyVisionDatasetWithMultipleImages(
            examples=examples,
            system_message="Test",
            image_loader=mock_image_loader,
        )

        result = dataset[0]
        user_content = result["messages"][1]["content"]

        # Should have one image and one text
        image_items = [item for item in user_content if item.get("type") == "image"]
        text_items = [item for item in user_content if item.get("type") == "text"]

        assert len(image_items) == 1
        assert len(text_items) == 1

    def test_multiple_images(self, mock_image_loader):
        """Test handling of multiple images."""
        examples = [
            {
                "text": "Compare these images",
                "images": ["img1.jpg", "img2.jpg", "img3.jpg"],
                "response": "Comparison result",
            }
        ]

        dataset = LazyVisionDatasetWithMultipleImages(
            examples=examples,
            system_message="Test",
            image_loader=mock_image_loader,
        )

        result = dataset[0]
        user_content = result["messages"][1]["content"]

        # Should have three images and one text
        image_items = [item for item in user_content if item.get("type") == "image"]
        text_items = [item for item in user_content if item.get("type") == "text"]

        assert len(image_items) == 3
        assert len(text_items) == 1
        assert text_items[0]["text"] == "Compare these images"

    def test_mixed_single_and_multiple_images(self, mock_image_loader):
        """Test handling of both 'image' and 'images' fields."""
        examples = [
            {
                "text": "Question",
                "image": "main.jpg",
                "images": ["extra1.jpg", "extra2.jpg"],
                "response": "Answer",
            }
        ]

        dataset = LazyVisionDatasetWithMultipleImages(
            examples=examples,
            system_message="Test",
            image_loader=mock_image_loader,
        )

        result = dataset[0]
        user_content = result["messages"][1]["content"]

        image_items = [item for item in user_content if item.get("type") == "image"]

        # Should have all images (1 from 'image' + 2 from 'images')
        assert len(image_items) == 3

    def test_caching_with_multiple_images(self):
        """Test caching behavior with multiple images."""
        examples = [
            {
                "text": "Q1",
                "images": ["img1.jpg", "img2.jpg"],
                "response": "A1",
            }
        ]

        load_count = {"count": 0}

        def counting_loader(ref):
            load_count["count"] += 1
            return MagicMock()

        dataset = LazyVisionDatasetWithMultipleImages(
            examples=examples,
            system_message="Test",
            image_loader=counting_loader,
            cache_size=10,
        )

        # First access - should load both images
        dataset[0]
        assert load_count["count"] == 2

        # Second access - should use cache
        dataset[0]
        assert load_count["count"] == 2  # Still 2

    def test_per_example_system_message(self, mock_image_loader):
        """Test per-example system message override."""
        examples = [
            {
                "text": "Question",
                "images": ["img.jpg"],
                "response": "Answer",
                "system": "Custom system",
            }
        ]

        dataset = LazyVisionDatasetWithMultipleImages(
            examples=examples,
            system_message="Default system",
            image_loader=mock_image_loader,
        )

        result = dataset[0]
        system_text = result["messages"][0]["content"][0]["text"]
        assert system_text == "Custom system"

    def test_load_with_cache_method(self, mock_image_loader):
        """Test _load_with_cache internal method."""
        examples = [{"text": "Q", "images": ["a.jpg", "b.jpg"], "response": "A"}]

        dataset = LazyVisionDatasetWithMultipleImages(
            examples=examples,
            system_message="Test",
            image_loader=mock_image_loader,
            cache_size=10,
        )

        # Direct call to internal method
        img1 = dataset._load_with_cache(0, 0, "test1.jpg")
        img2 = dataset._load_with_cache(0, 1, "test2.jpg")

        # Both should be loaded
        assert img1 is not None
        assert img2 is not None

        # Cache should have both
        assert 0 in dataset._cache  # 0 * 1000 + 0
        assert 1 in dataset._cache  # 0 * 1000 + 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_examples(self):
        """Test with empty examples list."""
        dataset = LazyVisionDataset(
            examples=[],
            system_message="Test",
            image_loader=lambda x: MagicMock(),
        )

        assert len(dataset) == 0

    def test_missing_text_field(self):
        """Test handling of missing text field."""
        examples = [{"image": "img.jpg", "response": "Answer"}]

        dataset = LazyVisionDataset(
            examples=examples,
            system_message="Test",
            image_loader=lambda x: MagicMock(),
        )

        result = dataset[0]
        user_text = result["messages"][1]["content"][1]["text"]
        assert user_text == ""  # Should default to empty string

    def test_missing_response_field(self):
        """Test handling of missing response field."""
        examples = [{"text": "Question", "image": "img.jpg"}]

        dataset = LazyVisionDataset(
            examples=examples,
            system_message="Test",
            image_loader=lambda x: MagicMock(),
        )

        result = dataset[0]
        response_text = result["messages"][2]["content"][0]["text"]
        assert response_text == ""  # Should default to empty string

    def test_image_loader_exception(self):
        """Test that image loader exceptions propagate."""

        def failing_loader(ref):
            raise ValueError("Failed to load image")

        examples = [{"text": "Q", "image": "bad.jpg", "response": "A"}]

        dataset = LazyVisionDataset(
            examples=examples,
            system_message="Test",
            image_loader=failing_loader,
        )

        with pytest.raises(ValueError, match="Failed to load image"):
            dataset[0]

    def test_none_image_reference(self):
        """Test handling of None image reference."""
        loader_calls = []

        def tracking_loader(ref):
            loader_calls.append(ref)
            return MagicMock()

        examples = [{"text": "Q", "image": None, "response": "A"}]

        dataset = LazyVisionDataset(
            examples=examples,
            system_message="Test",
            image_loader=tracking_loader,
        )

        result = dataset[0]

        # Loader should be called with None
        assert None in loader_calls
