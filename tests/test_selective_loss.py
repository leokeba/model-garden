"""Tests for selective loss computation module."""

from unittest.mock import MagicMock

import pytest
import torch

from model_garden.training.selective_loss import (
    SelectiveLossCollator,
    SelectiveLossMixin,
    detect_schema_keys_from_dataset,
)


class TestSelectiveLossMixin:
    """Tests for the SelectiveLossMixin class."""

    def test_structural_chars_defined(self):
        """Test that structural characters are properly defined."""
        expected_chars = {"{", "}", "[", "]", ":", ",", '"', " ", "\n", "\t", "\r"}
        assert SelectiveLossMixin.STRUCTURAL_CHARS == expected_chars

    def test_json_keywords_defined(self):
        """Test that JSON keywords are properly defined."""
        assert "null" in SelectiveLossMixin.JSON_KEYWORDS

    def test_json_type_keywords_defined(self):
        """Test that JSON type keywords are properly defined."""
        expected = {"object", "array", "string", "number", "integer", "boolean", "null"}
        assert SelectiveLossMixin.JSON_TYPE_KEYWORDS == expected

    def test_schema_keywords_defined(self):
        """Test that schema keywords are properly defined."""
        # Check some key schema keywords exist
        assert "type" in SelectiveLossMixin.SCHEMA_KEYWORDS
        assert "properties" in SelectiveLossMixin.SCHEMA_KEYWORDS
        assert "required" in SelectiveLossMixin.SCHEMA_KEYWORDS
        assert "$schema" in SelectiveLossMixin.SCHEMA_KEYWORDS


class MockTokenizer:
    """Mock tokenizer for testing."""

    def decode(self, token_ids, skip_special_tokens=False):
        """Decode token IDs - maps IDs to predefined strings."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # Simple mapping for testing
        token_map = {
            1: "{",
            2: "}",
            3: "[",
            4: "]",
            5: ":",
            6: ",",
            7: '"',
            8: " ",
            9: "\n",
            10: "null",
            11: "hello",
            12: "world",
            13: "type",
            14: "string",
            15: "properties",
            100: "<pad>",
        }

        result = []
        for tid in token_ids:
            if isinstance(tid, torch.Tensor):
                tid = tid.item()
            result.append(token_map.get(tid, f"<unk:{tid}>"))

        return "".join(result)


class MockProcessor:
    """Mock processor containing a tokenizer."""

    def __init__(self):
        self.tokenizer = MockTokenizer()


class TestSelectiveLossCollator:
    """Tests for the SelectiveLossCollator class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        mock_base_collator = MagicMock()
        mock_processor = MockProcessor()

        collator = SelectiveLossCollator(
            base_collator=mock_base_collator,
            processor=mock_processor,
        )

        assert collator.mask_structural is True
        assert collator.mask_keys is False
        assert collator.schema_keys == set()
        assert collator.mask_keywords is False
        assert collator.masking_strategy == "epoch_based"
        assert collator.masking_start_epoch == 0.0
        assert collator.verbose is False

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        mock_base_collator = MagicMock()
        mock_processor = MockProcessor()

        collator = SelectiveLossCollator(
            base_collator=mock_base_collator,
            processor=mock_processor,
            mask_structural_tokens=False,
            mask_schema_keys=True,
            schema_keys=["name", "value"],
            mask_json_keywords=True,
            masking_strategy="alternating",
            mask_every_n_steps=50,
            mask_for_n_steps=25,
            verbose=True,
        )

        assert collator.mask_structural is False
        assert collator.mask_keys is True
        assert collator.schema_keys == {"name", "value"}
        assert collator.mask_keywords is True
        assert collator.masking_strategy == "alternating"
        assert collator.mask_every_n_steps == 50
        assert collator.mask_for_n_steps == 25
        assert collator.verbose is True

    def test_init_invalid_strategy(self):
        """Test that invalid masking strategy raises error."""
        mock_base_collator = MagicMock()
        mock_processor = MockProcessor()

        with pytest.raises(ValueError, match="Invalid masking_strategy"):
            SelectiveLossCollator(
                base_collator=mock_base_collator,
                processor=mock_processor,
                masking_strategy="invalid_strategy",
            )

    def test_init_invalid_structural_weight(self):
        """Test that invalid structural weight raises error."""
        mock_base_collator = MagicMock()
        mock_processor = MockProcessor()

        with pytest.raises(ValueError, match="structural_weight must be between"):
            SelectiveLossCollator(
                base_collator=mock_base_collator,
                processor=mock_processor,
                masking_strategy="weighted",
                structural_weight=1.5,  # Invalid: > 1.0
            )

    def test_get_tokenizer_from_processor(self):
        """Test getting tokenizer from processor."""
        mock_base_collator = MagicMock()
        mock_processor = MockProcessor()

        collator = SelectiveLossCollator(
            base_collator=mock_base_collator,
            processor=mock_processor,
        )

        tokenizer = collator._get_tokenizer()
        assert tokenizer is mock_processor.tokenizer

    def test_get_tokenizer_processor_is_tokenizer(self):
        """Test when processor itself is the tokenizer."""
        mock_base_collator = MagicMock()
        mock_tokenizer = MockTokenizer()

        collator = SelectiveLossCollator(
            base_collator=mock_base_collator,
            processor=mock_tokenizer,  # Pass tokenizer directly
        )

        tokenizer = collator._get_tokenizer()
        assert tokenizer is mock_tokenizer

    def test_call_delegates_to_base_collator(self):
        """Test that __call__ delegates to base collator first."""
        mock_base_collator = MagicMock()
        mock_base_collator.return_value = {
            "input_ids": torch.tensor([[1, 11, 12, 2]]),
            "labels": torch.tensor([[1, 11, 12, 2]]),
        }
        mock_processor = MockProcessor()

        collator = SelectiveLossCollator(
            base_collator=mock_base_collator,
            processor=mock_processor,
            mask_structural_tokens=False,  # Disable masking for this test
        )

        features = [{"input_ids": [1, 2, 3]}]
        result = collator(features)

        mock_base_collator.assert_called_once_with(features)
        assert "input_ids" in result
        assert "labels" in result

    def test_get_masking_stats(self):
        """Test getting masking statistics."""
        mock_base_collator = MagicMock()
        mock_processor = MockProcessor()

        collator = SelectiveLossCollator(
            base_collator=mock_base_collator,
            processor=mock_processor,
        )

        # Initial stats should be zero
        stats = collator.get_masking_stats()
        assert stats["total_tokens"] == 0
        assert stats["masked_tokens"] == 0
        assert stats["mask_percentage"] == 0.0
        assert stats["batch_count"] == 0

    def test_set_trainer(self):
        """Test setting trainer reference."""
        mock_base_collator = MagicMock()
        mock_processor = MockProcessor()

        collator = SelectiveLossCollator(
            base_collator=mock_base_collator,
            processor=mock_processor,
        )

        mock_trainer = MagicMock()
        collator.set_trainer(mock_trainer)

        # Trainer should be accessible
        retrieved_trainer = collator._get_trainer()
        assert retrieved_trainer is mock_trainer

    def test_set_trainer_none(self):
        """Test setting trainer to None."""
        mock_base_collator = MagicMock()
        mock_processor = MockProcessor()

        collator = SelectiveLossCollator(
            base_collator=mock_base_collator,
            processor=mock_processor,
        )

        collator.set_trainer(None)
        assert collator._get_trainer() is None

    def test_should_enable_masking_epoch_based(self):
        """Test masking enable logic for epoch-based strategy."""
        mock_base_collator = MagicMock()
        mock_processor = MockProcessor()

        collator = SelectiveLossCollator(
            base_collator=mock_base_collator,
            processor=mock_processor,
            masking_strategy="epoch_based",
            masking_start_epoch=1.0,
        )

        # Mock trainer with epoch state
        mock_trainer = MagicMock()
        mock_trainer.state.epoch = 0.5
        collator.set_trainer(mock_trainer)

        # Should not enable before start epoch
        assert collator._should_enable_masking() is False

        # Should enable after start epoch
        mock_trainer.state.epoch = 1.5
        assert collator._should_enable_masking() is True

    def test_should_enable_masking_alternating(self):
        """Test masking enable logic for alternating strategy."""
        mock_base_collator = MagicMock()
        mock_processor = MockProcessor()

        collator = SelectiveLossCollator(
            base_collator=mock_base_collator,
            processor=mock_processor,
            masking_strategy="alternating",
            mask_every_n_steps=100,
            mask_for_n_steps=50,
        )

        # Step 0 - within masking window (0-49)
        collator.current_step = 0
        assert collator._should_enable_masking() is True

        # Step 25 - still within masking window
        collator.current_step = 25
        assert collator._should_enable_masking() is True

        # Step 50 - outside masking window (50-99)
        collator.current_step = 50
        assert collator._should_enable_masking() is False

        # Step 75 - still outside
        collator.current_step = 75
        assert collator._should_enable_masking() is False

        # Step 100 - new cycle starts, within window again
        collator.current_step = 100
        assert collator._should_enable_masking() is True

    def test_is_structural_token_pure_whitespace(self):
        """Test that pure whitespace is detected as structural."""
        mock_base_collator = MagicMock()
        mock_processor = MockProcessor()

        collator = SelectiveLossCollator(
            base_collator=mock_base_collator,
            processor=mock_processor,
        )

        assert collator._is_structural_token("   ", "", check_schema_keys=False) is True
        assert collator._is_structural_token("\n\t", "", check_schema_keys=False) is True

    def test_is_structural_token_json_brackets(self):
        """Test that JSON brackets are detected as structural."""
        mock_base_collator = MagicMock()
        mock_processor = MockProcessor()

        collator = SelectiveLossCollator(
            base_collator=mock_base_collator,
            processor=mock_processor,
        )

        assert collator._is_structural_token("{", "", check_schema_keys=False) is True
        assert collator._is_structural_token("}", "", check_schema_keys=False) is True
        assert collator._is_structural_token("[", "", check_schema_keys=False) is True
        assert collator._is_structural_token("]", "", check_schema_keys=False) is True
        assert collator._is_structural_token(":", "", check_schema_keys=False) is True
        assert collator._is_structural_token(",", "", check_schema_keys=False) is True

    def test_is_structural_token_schema_keywords(self):
        """Test that schema keywords are detected as structural."""
        mock_base_collator = MagicMock()
        mock_processor = MockProcessor()

        collator = SelectiveLossCollator(
            base_collator=mock_base_collator,
            processor=mock_processor,
        )

        # Schema keywords should be structural
        assert collator._is_structural_token("type", "", check_schema_keys=False) is True
        assert collator._is_structural_token("properties", "", check_schema_keys=False) is True
        assert collator._is_structural_token("required", "", check_schema_keys=False) is True

    def test_is_structural_token_semantic_content(self):
        """Test that semantic content is not marked as structural."""
        mock_base_collator = MagicMock()
        mock_processor = MockProcessor()

        collator = SelectiveLossCollator(
            base_collator=mock_base_collator,
            processor=mock_processor,
        )

        # Semantic content should NOT be structural
        assert collator._is_structural_token("hello", "", check_schema_keys=False) is False
        assert collator._is_structural_token("world", "", check_schema_keys=False) is False
        assert collator._is_structural_token("John", "", check_schema_keys=False) is False
        assert collator._is_structural_token("123", "", check_schema_keys=False) is False


class TestDetectSchemaKeysFromDataset:
    """Tests for the detect_schema_keys_from_dataset function."""

    def test_detect_keys_from_simple_json(self):
        """Test detecting schema keys from a simple JSON dataset."""
        mock_processor = MockProcessor()

        # Create mock dataset with JSON responses
        dataset = [
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Extract info"}]},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": '{"name": "John", "age": 30}'}],
                    },
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Another request"}]},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": '{"name": "Jane", "city": "NYC"}'}],
                    },
                ]
            },
        ]

        keys = detect_schema_keys_from_dataset(
            dataset=dataset,
            processor=mock_processor,
            num_samples=2,
            threshold=0.5,
            verbose=False,
        )

        # "name" appears in both samples (100%), should be detected
        assert "name" in keys

    def test_detect_keys_empty_dataset(self):
        """Test detecting keys from empty dataset."""
        mock_processor = MockProcessor()

        keys = detect_schema_keys_from_dataset(
            dataset=[],
            processor=mock_processor,
            num_samples=10,
            threshold=0.3,
            verbose=False,
        )

        assert keys == set()

    def test_detect_keys_no_json_content(self):
        """Test detecting keys when responses are not JSON."""
        mock_processor = MockProcessor()

        dataset = [
            {"messages": [{"role": "assistant", "content": "Just plain text, no JSON here."}]}
        ]

        keys = detect_schema_keys_from_dataset(
            dataset=dataset,
            processor=mock_processor,
            num_samples=1,
            threshold=0.3,
            verbose=False,
        )

        assert keys == set()

    def test_detect_keys_respects_threshold(self):
        """Test that threshold parameter is respected."""
        mock_processor = MockProcessor()

        # Create dataset where "common" appears in 60% and "rare" in 20%
        dataset = [
            {
                "messages": [
                    {"role": "assistant", "content": [{"type": "text", "text": '{"common": 1}'}]}
                ]
            },
            {
                "messages": [
                    {"role": "assistant", "content": [{"type": "text", "text": '{"common": 2}'}]}
                ]
            },
            {
                "messages": [
                    {"role": "assistant", "content": [{"type": "text", "text": '{"common": 3}'}]}
                ]
            },
            {
                "messages": [
                    {"role": "assistant", "content": [{"type": "text", "text": '{"rare": 1}'}]}
                ]
            },
            {
                "messages": [
                    {"role": "assistant", "content": [{"type": "text", "text": '{"other": 1}'}]}
                ]
            },
        ]

        # With 50% threshold, only "common" should be detected
        keys = detect_schema_keys_from_dataset(
            dataset=dataset,
            processor=mock_processor,
            num_samples=5,
            threshold=0.5,
            verbose=False,
        )

        assert "common" in keys
        assert "rare" not in keys
        assert "other" not in keys

    def test_detect_keys_nested_json(self):
        """Test detecting keys from nested JSON structures."""
        mock_processor = MockProcessor()

        dataset = [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": '{"person": {"name": "John", "address": {"city": "NYC"}}}',
                            }
                        ],
                    }
                ]
            }
        ]

        keys = detect_schema_keys_from_dataset(
            dataset=dataset,
            processor=mock_processor,
            num_samples=1,
            threshold=0.0,  # Include all keys
            verbose=False,
        )

        # All nested keys should be detected
        assert "person" in keys
        assert "name" in keys
        assert "address" in keys
        assert "city" in keys
