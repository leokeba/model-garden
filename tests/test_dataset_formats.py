"""Tests for training/dataset_formats.py - Dataset format conversion utilities.

These tests verify dataset format detection and conversion between different
formats (simple, messages, VQA).
"""

import pytest

from model_garden.training.dataset_formats import DatasetFormatConverter


class TestDetectFormat:
    """Tests for DatasetFormatConverter.detect_format method."""

    def test_detect_simple_format_with_text_and_response(self):
        """Test detection of simple format with text and response."""
        example = {"text": "What is this?", "image": "img.jpg", "response": "A cat"}
        assert DatasetFormatConverter.detect_format(example) == "simple"

    def test_detect_simple_format_with_output(self):
        """Test detection of simple format with output field (alternative to response)."""
        example = {"text": "Hello", "output": "World"}
        assert DatasetFormatConverter.detect_format(example) == "simple"

    def test_detect_simple_format_image_only(self):
        """Test detection of simple format with image and response only."""
        example = {"image": "img.jpg", "response": "A picture"}
        assert DatasetFormatConverter.detect_format(example) == "simple"

    def test_detect_messages_format(self):
        """Test detection of OpenAI messages format."""
        example = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        assert DatasetFormatConverter.detect_format(example) == "messages"

    def test_detect_messages_format_empty_list(self):
        """Test that empty messages list is not detected as messages format."""
        example = {"messages": []}
        # Empty messages list should not be detected as valid messages format
        assert DatasetFormatConverter.detect_format(example) != "messages"

    def test_detect_vqa_format(self):
        """Test detection of VQA format."""
        example = {"question": "What color is the sky?", "answer": "Blue", "image": "sky.jpg"}
        assert DatasetFormatConverter.detect_format(example) == "vqa"

    def test_detect_vqa_format_with_answers_list(self):
        """Test detection of VQA format with answers as list."""
        example = {"question": "What is shown?", "answers": ["cat", "kitten"], "image": "cat.jpg"}
        assert DatasetFormatConverter.detect_format(example) == "vqa"

    def test_detect_unknown_format_empty_dict(self):
        """Test that empty dict returns unknown."""
        assert DatasetFormatConverter.detect_format({}) == "unknown"

    def test_detect_unknown_format_non_dict(self):
        """Test that non-dict input returns unknown."""
        assert DatasetFormatConverter.detect_format("not a dict") == "unknown"
        assert DatasetFormatConverter.detect_format([1, 2, 3]) == "unknown"
        assert DatasetFormatConverter.detect_format(None) == "unknown"

    def test_detect_unknown_format_missing_required_fields(self):
        """Test that missing required fields returns unknown."""
        # Missing response/output
        example = {"text": "Hello", "image": "img.jpg"}
        assert DatasetFormatConverter.detect_format(example) == "unknown"


class TestDetectVqaFormat:
    """Tests for DatasetFormatConverter.detect_vqa_format method."""

    def test_vqa_format_with_answer(self):
        """Test VQA detection with single answer."""
        example = {"question": "What?", "answer": "Something", "image": "img.jpg"}
        assert DatasetFormatConverter.detect_vqa_format(example) is True

    def test_vqa_format_with_answers(self):
        """Test VQA detection with answers list."""
        example = {"question": "What?", "answers": ["a", "b"], "image": "img.jpg"}
        assert DatasetFormatConverter.detect_vqa_format(example) is True

    def test_vqa_format_missing_image(self):
        """Test VQA detection fails without image."""
        example = {"question": "What?", "answer": "Something"}
        assert DatasetFormatConverter.detect_vqa_format(example) is False

    def test_vqa_format_missing_question(self):
        """Test VQA detection fails without question."""
        example = {"answer": "Something", "image": "img.jpg"}
        assert DatasetFormatConverter.detect_vqa_format(example) is False

    def test_vqa_format_non_dict(self):
        """Test VQA detection returns False for non-dict."""
        assert DatasetFormatConverter.detect_vqa_format("not a dict") is False
        assert DatasetFormatConverter.detect_vqa_format(None) is False


class TestConvertVqaToSimple:
    """Tests for DatasetFormatConverter.convert_vqa_to_simple method."""

    def test_convert_basic_vqa(self):
        """Test basic VQA conversion."""
        example = {
            "question": "What is in the image?",
            "answer": "A dog",
            "image": "dog.jpg",
        }
        result = DatasetFormatConverter.convert_vqa_to_simple(example)

        assert result["text"] == "What is in the image?"
        assert result["image"] == "dog.jpg"
        assert result["response"] == "A dog"

    def test_convert_vqa_with_answers_list(self):
        """Test VQA conversion with answers as list of strings."""
        example = {
            "question": "What color?",
            "answers": ["red", "crimson", "scarlet"],
            "image": "color.jpg",
        }
        result = DatasetFormatConverter.convert_vqa_to_simple(example)

        assert result["text"] == "What color?"
        assert result["response"] == "red"  # Takes first answer

    def test_convert_vqa_with_answers_list_of_dicts(self):
        """Test VQA conversion with answers as list of dicts."""
        example = {
            "question": "What is shown?",
            "answers": [{"answer": "cat"}, {"answer": "kitten"}],
            "image": "cat.jpg",
        }
        result = DatasetFormatConverter.convert_vqa_to_simple(example)

        assert result["response"] == "cat"

    def test_convert_vqa_with_answers_string(self):
        """Test VQA conversion with answers as string (edge case)."""
        example = {
            "question": "What?",
            "answers": "single answer",
            "image": "img.jpg",
        }
        result = DatasetFormatConverter.convert_vqa_to_simple(example)

        assert result["response"] == "single answer"

    def test_convert_scienceqa_format(self):
        """Test ScienceQA format with choices and index."""
        example = {
            "question": "What planet is this?",
            "choices": ["Mars", "Venus", "Earth", "Jupiter"],
            "answer": 2,  # Index into choices
            "solution": "The blue planet is Earth.",
            "image": "planet.jpg",
        }
        result = DatasetFormatConverter.convert_vqa_to_simple(example)

        assert result["text"] == "What planet is this?"
        assert "Earth" in result["response"]
        assert "blue planet" in result["response"]

    def test_convert_scienceqa_without_solution(self):
        """Test ScienceQA format without solution."""
        example = {
            "question": "Which is bigger?",
            "choices": ["Small", "Medium", "Large"],
            "answer": 2,
            "image": "sizes.jpg",
        }
        result = DatasetFormatConverter.convert_vqa_to_simple(example)

        assert result["response"] == "Large"

    def test_convert_vqa_missing_fields(self):
        """Test VQA conversion with missing optional fields."""
        example = {"image": "test.jpg"}  # Minimal
        result = DatasetFormatConverter.convert_vqa_to_simple(example)

        assert result["text"] == ""
        assert result["image"] == "test.jpg"
        assert result["response"] == ""

    def test_convert_vqa_raises_for_non_dict(self):
        """Test that non-dict input raises ValueError."""
        with pytest.raises(ValueError, match="Expected VQA example to be a dict"):
            DatasetFormatConverter.convert_vqa_to_simple("not a dict")


class TestConvertMessagesToSimple:
    """Tests for DatasetFormatConverter.convert_messages_to_simple method."""

    def test_convert_standard_messages(self):
        """Test conversion of standard OpenAI messages format."""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are helpful."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "test.jpg"},
                    {"type": "text", "text": "What is this?"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "This is a test image."}],
            },
        ]
        result = DatasetFormatConverter.convert_messages_to_simple(messages)

        assert result["system"] == "You are helpful."
        assert result["text"] == "What is this?"
        assert result["image"] == "test.jpg"
        assert result["response"] == "This is a test image."

    def test_convert_messages_with_image_url_format(self):
        """Test conversion with image_url format."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                    {"type": "text", "text": "Describe this."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "An image."}],
            },
        ]
        result = DatasetFormatConverter.convert_messages_to_simple(messages)

        assert result["image"] == "https://example.com/img.jpg"
        assert result["text"] == "Describe this."
        assert result["response"] == "An image."

    def test_convert_messages_simple_string_content(self):
        """Test conversion with simplified string content format."""
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = DatasetFormatConverter.convert_messages_to_simple(messages)

        assert result["system"] == "Be helpful."
        assert result["text"] == "Hello!"
        assert result["response"] == "Hi there!"

    def test_convert_messages_without_system(self):
        """Test conversion without system message."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Question"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Answer"}],
            },
        ]
        result = DatasetFormatConverter.convert_messages_to_simple(messages)

        assert result["system"] == ""
        assert result["text"] == "Question"
        assert result["response"] == "Answer"

    def test_convert_messages_empty_list(self):
        """Test conversion of empty messages list."""
        result = DatasetFormatConverter.convert_messages_to_simple([])

        assert result["text"] == ""
        assert result["image"] is None
        assert result["response"] == ""
        assert result["system"] == ""

    def test_convert_messages_raises_for_non_list(self):
        """Test that non-list input raises ValueError."""
        with pytest.raises(ValueError, match="Expected 'messages' to be a list"):
            DatasetFormatConverter.convert_messages_to_simple("not a list")

    def test_convert_messages_skips_invalid_entries(self):
        """Test that invalid message entries are skipped gracefully."""
        messages = [
            "not a dict",  # Invalid - should be skipped
            {"role": "user", "content": [{"type": "text", "text": "Valid text"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Valid response"}]},
        ]
        result = DatasetFormatConverter.convert_messages_to_simple(messages)

        # Should still extract from valid messages
        assert result["text"] == "Valid text"
        assert result["response"] == "Valid response"

    def test_convert_messages_handles_invalid_content(self):
        """Test handling of invalid content types."""
        messages = [
            {"role": "user", "content": 12345},  # Invalid content type
            {"role": "assistant", "content": [{"type": "text", "text": "Response"}]},
        ]
        result = DatasetFormatConverter.convert_messages_to_simple(messages)

        assert result["response"] == "Response"


class TestToOpenaiMessages:
    """Tests for DatasetFormatConverter.to_openai_messages method."""

    def test_basic_conversion(self):
        """Test basic conversion to OpenAI messages format."""
        result = DatasetFormatConverter.to_openai_messages(
            text="What is this?",
            image="test.jpg",
            response="A test image.",
            system_message="Be helpful.",
        )

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
        assert messages[1]["content"][0]["image"] == "test.jpg"
        assert messages[1]["content"][1]["type"] == "text"
        assert messages[1]["content"][1]["text"] == "What is this?"

        # Assistant message
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"][0]["text"] == "A test image."

    def test_default_system_message(self):
        """Test that default system message is used."""
        result = DatasetFormatConverter.to_openai_messages(
            text="Hello",
            image=None,
            response="Hi",
        )

        system_text = result["messages"][0]["content"][0]["text"]
        assert "helpful assistant" in system_text.lower()

    def test_with_pil_image(self):
        """Test conversion with PIL Image object."""
        from unittest.mock import MagicMock

        mock_image = MagicMock()
        mock_image.__class__.__name__ = "Image"

        result = DatasetFormatConverter.to_openai_messages(
            text="Describe",
            image=mock_image,
            response="Description",
        )

        # Image should be passed through unchanged
        assert result["messages"][1]["content"][0]["image"] is mock_image


class TestRoundTrip:
    """Tests for round-trip conversion between formats."""

    def test_simple_to_messages_to_simple(self):
        """Test round-trip: simple -> messages -> simple."""
        original = {
            "text": "What is this?",
            "image": "test.jpg",
            "response": "A test.",
        }

        # Convert to messages
        messages_format = DatasetFormatConverter.to_openai_messages(
            text=original["text"],
            image=original["image"],
            response=original["response"],
        )

        # Convert back to simple
        result = DatasetFormatConverter.convert_messages_to_simple(messages_format["messages"])

        assert result["text"] == original["text"]
        assert result["image"] == original["image"]
        assert result["response"] == original["response"]
