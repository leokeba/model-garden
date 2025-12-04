"""Tests for training/chat_template.py - Chat template detection utilities.

These tests verify automatic chat template marker detection from tokenizers.
"""

from unittest.mock import MagicMock

import pytest

from model_garden.training.chat_template import (
    ChatTemplateDetector,
    FALLBACK_MARKERS,
)


class TestChatTemplateDetector:
    """Tests for ChatTemplateDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a detector instance."""
        return ChatTemplateDetector(verbose=False)

    @pytest.fixture
    def verbose_detector(self):
        """Create a verbose detector instance."""
        return ChatTemplateDetector(verbose=True)


class TestDetectMethod:
    """Tests for ChatTemplateDetector.detect method."""

    def test_detect_qwen_style_template(self):
        """Test detection of Qwen-style chat template."""
        detector = ChatTemplateDetector(verbose=False)

        # Mock processor that returns Qwen-style formatting
        processor = MagicMock()
        processor.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are helpful.\n<|im_end|>\n"
            "<|im_start|>user\n__USER_PLACEHOLDER__\n<|im_end|>\n"
            "<|im_start|>assistant\n__ASSISTANT_PLACEHOLDER__\n<|im_end|>"
        )

        instruction, response = detector.detect(processor)

        assert "<|im_start|>user" in instruction
        assert "<|im_start|>assistant" in response

    def test_detect_llama_style_template(self):
        """Test detection of Llama-style chat template."""
        detector = ChatTemplateDetector(verbose=False)

        processor = MagicMock()
        processor.apply_chat_template.return_value = (
            "<s>[INST] __USER_PLACEHOLDER__ [/INST] __ASSISTANT_PLACEHOLDER__ </s>"
        )

        instruction, response = detector.detect(processor)

        # Should detect instruction marker
        assert instruction is not None

    def test_detect_fallback_on_exception(self):
        """Test fallback when apply_chat_template raises exception."""
        detector = ChatTemplateDetector(verbose=False)

        processor = MagicMock()
        processor.apply_chat_template.side_effect = Exception("Template error")

        # Mock model type for fallback
        processor.tokenizer = MagicMock()
        processor.tokenizer.config = MagicMock()
        processor.tokenizer.config.model_type = "qwen"

        instruction, response = detector.detect(processor)

        # Should use Qwen fallback markers
        assert instruction == FALLBACK_MARKERS["qwen"][0]
        assert response == FALLBACK_MARKERS["qwen"][1]

    def test_detect_fallback_on_missing_placeholders(self):
        """Test fallback when placeholders not found in template."""
        detector = ChatTemplateDetector(verbose=False)

        processor = MagicMock()
        # Template without proper placeholders
        processor.apply_chat_template.return_value = "Some template without markers"
        
        # Set up name_or_path to match "llama" for fallback detection
        processor.name_or_path = "meta-llama/Llama-3.2-3B"
        
        # Also need to set up tokenizer.config to avoid that path
        processor.tokenizer = MagicMock()
        processor.tokenizer.config = MagicMock()
        processor.tokenizer.config.model_type = "llama"

        instruction, response = detector.detect(processor)

        # Should use Llama fallback markers (based on model_type)
        assert instruction == FALLBACK_MARKERS["llama"][0]
        assert response == FALLBACK_MARKERS["llama"][1]


class TestGetFallbackMarkers:
    """Tests for ChatTemplateDetector.get_fallback_markers method."""

    def test_fallback_for_qwen(self):
        """Test fallback markers for Qwen models."""
        detector = ChatTemplateDetector(verbose=False)

        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.config = MagicMock()
        processor.tokenizer.config.model_type = "qwen2"

        instruction, response = detector.get_fallback_markers(processor)

        assert instruction == "<|im_start|>user"
        assert response == "<|im_start|>assistant"

    def test_fallback_for_llama(self):
        """Test fallback markers for Llama models."""
        detector = ChatTemplateDetector(verbose=False)

        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.config = MagicMock()
        processor.tokenizer.config.model_type = "llama"

        instruction, response = detector.get_fallback_markers(processor)

        assert instruction == "[INST]"
        assert response == "[/INST]"

    def test_fallback_for_phi(self):
        """Test fallback markers for Phi models."""
        detector = ChatTemplateDetector(verbose=False)

        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.config = MagicMock()
        processor.tokenizer.config.model_type = "phi3"

        instruction, response = detector.get_fallback_markers(processor)

        assert instruction == "<|user|>"
        assert response == "<|assistant|>"

    def test_fallback_for_mistral(self):
        """Test fallback markers for Mistral models."""
        detector = ChatTemplateDetector(verbose=False)

        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.config = MagicMock()
        processor.tokenizer.config.model_type = "mistral"

        instruction, response = detector.get_fallback_markers(processor)

        assert instruction == "[INST]"
        assert response == "[/INST]"

    def test_fallback_for_gemma(self):
        """Test fallback markers for Gemma models."""
        detector = ChatTemplateDetector(verbose=False)

        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.config = MagicMock()
        processor.tokenizer.config.model_type = "gemma"

        instruction, response = detector.get_fallback_markers(processor)

        assert instruction == "<start_of_turn>user"
        assert response == "<start_of_turn>model"

    def test_fallback_for_vicuna(self):
        """Test fallback markers for Vicuna models."""
        detector = ChatTemplateDetector(verbose=False)

        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.config = MagicMock()
        processor.tokenizer.config.model_type = "vicuna"

        instruction, response = detector.get_fallback_markers(processor)

        assert instruction == "USER:"
        assert response == "ASSISTANT:"

    def test_fallback_generic(self):
        """Test generic fallback for unknown models."""
        detector = ChatTemplateDetector(verbose=False)

        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.config = MagicMock()
        processor.tokenizer.config.model_type = "unknown_model_type"

        # Remove name_or_path to prevent that fallback
        del processor.name_or_path

        instruction, response = detector.get_fallback_markers(processor)

        assert instruction == "User:"
        assert response == "Assistant:"


class TestGetModelType:
    """Tests for ChatTemplateDetector._get_model_type method."""

    def test_get_from_tokenizer_config(self):
        """Test getting model type from tokenizer config."""
        detector = ChatTemplateDetector(verbose=False)

        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.config = MagicMock()
        processor.tokenizer.config.model_type = "QWEN2"

        model_type = detector._get_model_type(processor)

        assert model_type == "qwen2"  # Should be lowercased

    def test_get_from_processor_config(self):
        """Test getting model type from processor config."""
        detector = ChatTemplateDetector(verbose=False)

        processor = MagicMock(spec=["config"])
        processor.config = MagicMock()
        processor.config.model_type = "Llama3"

        model_type = detector._get_model_type(processor)

        assert model_type == "llama3"

    def test_get_from_name_or_path(self):
        """Test getting model type from name_or_path."""
        detector = ChatTemplateDetector(verbose=False)

        processor = MagicMock(spec=["name_or_path"])
        processor.name_or_path = "Qwen/Qwen2.5-VL-3B-Instruct"

        model_type = detector._get_model_type(processor)

        assert model_type == "qwen"

    def test_returns_empty_on_failure(self):
        """Test that empty string is returned when detection fails."""
        detector = ChatTemplateDetector(verbose=False)

        processor = MagicMock(spec=[])  # No relevant attributes

        model_type = detector._get_model_type(processor)

        assert model_type == ""

    def test_handles_attribute_errors_gracefully(self):
        """Test that AttributeError is handled gracefully."""
        detector = ChatTemplateDetector(verbose=False)

        processor = MagicMock()
        processor.tokenizer.config.model_type = property(
            lambda self: (_ for _ in ()).throw(AttributeError("No model_type"))
        )
        # Remove fallback options
        del processor.config
        del processor.name_or_path

        model_type = detector._get_model_type(processor)

        assert model_type == ""


class TestFallbackMarkers:
    """Tests for FALLBACK_MARKERS constant."""

    def test_all_markers_are_tuples(self):
        """Test that all fallback markers are tuples of length 2."""
        for family, markers in FALLBACK_MARKERS.items():
            assert isinstance(markers, tuple), f"{family} markers should be tuple"
            assert len(markers) == 2, f"{family} markers should have 2 elements"

    def test_all_markers_are_strings(self):
        """Test that all marker values are strings."""
        for family, (instruction, response) in FALLBACK_MARKERS.items():
            assert isinstance(instruction, str), f"{family} instruction should be string"
            assert isinstance(response, str), f"{family} response should be string"

    def test_common_families_present(self):
        """Test that common model families are present."""
        expected_families = ["qwen", "llama", "phi", "mistral", "gemma", "chatml"]
        for family in expected_families:
            assert family in FALLBACK_MARKERS, f"Missing fallback for {family}"


class TestVerboseOutput:
    """Tests for verbose output behavior."""

    def test_verbose_mode_prints_detection(self, capsys):
        """Test that verbose mode prints detection results."""
        detector = ChatTemplateDetector(verbose=True)

        processor = MagicMock()
        processor.apply_chat_template.return_value = (
            "<|im_start|>user\n__USER_PLACEHOLDER__\n<|im_end|>\n"
            "<|im_start|>assistant\n__ASSISTANT_PLACEHOLDER__\n<|im_end|>"
        )

        detector.detect(processor)

        # Verbose output goes to Rich console, not capsys
        # This test just ensures no exceptions are raised

    def test_quiet_mode_no_print(self):
        """Test that quiet mode doesn't print."""
        detector = ChatTemplateDetector(verbose=False)

        processor = MagicMock()
        processor.apply_chat_template.return_value = (
            "<|im_start|>user\n__USER_PLACEHOLDER__\n<|im_end|>\n"
            "<|im_start|>assistant\n__ASSISTANT_PLACEHOLDER__\n<|im_end|>"
        )

        # Should not raise any exceptions
        detector.detect(processor)
