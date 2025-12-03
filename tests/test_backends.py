"""Tests for model_garden.backends module."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from datasets import Dataset

from model_garden.training.backends.base import (
    TextTrainer,
    TrainingBackend,
    VisionTrainer,
)
from model_garden.training.backends.registry import (
    _BACKENDS,
    get_backend,
    is_backend_available,
    list_backends,
    register_backend,
)


class ConcreteTextTrainer(TextTrainer):
    """Concrete implementation of TextTrainer for testing."""

    def load_model(self) -> None:
        pass

    def prepare_for_training(
        self,
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: list[str] | None = None,
        use_rslora: bool = False,
        lora_bias: str = "none",
        task_type: str = "CAUSAL_LM",
        use_gradient_checkpointing: str | bool = "unsloth",
        random_state: int = 42,
        loftq_config: dict | None = None,
    ) -> None:
        pass

    def load_dataset_from_file(self, dataset_path: str) -> Dataset:
        return Dataset.from_dict({"text": ["sample"]})

    def load_dataset_from_hub(self, dataset_name: str, split: str = "train") -> Dataset:
        return Dataset.from_dict({"text": ["sample"]})

    def format_dataset(
        self,
        dataset: Dataset,
        instruction_field: str = "instruction",
        input_field: str = "input",
        output_field: str = "output",
        prompt_template: str | None = None,
    ) -> Dataset:
        return dataset

    def train(
        self,
        dataset: Dataset,
        output_dir: str,
        job_id: str | None = None,
        enable_carbon_tracking: bool = True,
        **kwargs,
    ) -> None:
        pass

    def save_model(
        self,
        output_dir: str,
        save_method: str = "merged_16bit",
        maximum_memory_usage: float = 0.75,
        max_shard_size: str = "5GB",
    ) -> None:
        pass


class ConcreteVisionTrainer(VisionTrainer):
    """Concrete implementation of VisionTrainer for testing."""

    def load_model(self) -> None:
        pass

    def prepare_for_training(
        self,
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: list[str] | None = None,
        use_rslora: bool = False,
        lora_bias: str = "none",
        task_type: str = "CAUSAL_LM",
        use_gradient_checkpointing: str | bool = "unsloth",
        random_state: int = 42,
        loftq_config: dict | None = None,
        finetune_vision_layers: bool = True,
        finetune_language_layers: bool = True,
        finetune_attention_modules: bool = True,
        finetune_mlp_modules: bool = True,
    ) -> None:
        pass

    def load_dataset_from_file(self, dataset_path: str) -> Dataset:
        return Dataset.from_dict({"text": ["sample"], "image": ["img.jpg"]})

    def load_dataset_from_hub(self, dataset_name: str, split: str = "train", **kwargs) -> Dataset:
        return Dataset.from_dict({"text": ["sample"], "image": ["img.jpg"]})

    def format_dataset(
        self,
        dataset: Dataset,
        text_field: str = "text",
        image_field: str = "image",
        system_message: str | None = None,
        messages_field: str | None = None,
    ) -> list[dict]:
        return [{"text": "sample", "image": "img.jpg"}]

    def train(
        self,
        dataset: Dataset | list[dict],
        output_dir: str,
        job_id: str | None = None,
        enable_carbon_tracking: bool = True,
        **kwargs,
    ) -> None:
        pass

    def save_model(
        self,
        output_dir: str,
        save_method: str = "merged_16bit",
        maximum_memory_usage: float = 0.75,
        max_shard_size: str = "5GB",
    ) -> None:
        pass


class ConcreteTrainingBackend(TrainingBackend):
    """Concrete implementation of TrainingBackend for testing."""

    @property
    def name(self) -> str:
        return "test-backend"

    @property
    def description(self) -> str:
        return "A test backend for unit testing"

    def supports_text_training(self) -> bool:
        return True

    def supports_vision_training(self) -> bool:
        return True

    def create_text_trainer(
        self,
        base_model: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: str | None = None,
    ) -> TextTrainer:
        return ConcreteTextTrainer(
            base_model=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            dtype=dtype,
        )

    def create_vision_trainer(
        self,
        base_model: str,
        max_seq_length: int = 16384,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: Any | None = None,
    ) -> VisionTrainer:
        return ConcreteVisionTrainer(
            base_model=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            dtype=dtype,
        )


class TextOnlyBackend(TrainingBackend):
    """Backend that only supports text training."""

    @property
    def name(self) -> str:
        return "text-only"

    @property
    def description(self) -> str:
        return "Text-only backend"

    def supports_text_training(self) -> bool:
        return True

    def supports_vision_training(self) -> bool:
        return False

    def create_text_trainer(
        self,
        base_model: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: str | None = None,
    ) -> TextTrainer:
        return ConcreteTextTrainer(
            base_model=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            dtype=dtype,
        )

    def create_vision_trainer(
        self,
        base_model: str,
        max_seq_length: int = 16384,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: Any | None = None,
    ) -> VisionTrainer:
        raise NotImplementedError("This backend does not support vision training")


class TestTextTrainer:
    """Tests for TextTrainer abstract base class."""

    def test_init(self):
        """Test TextTrainer initialization."""
        trainer = ConcreteTextTrainer(
            base_model="test/model",
            max_seq_length=4096,
            load_in_4bit=True,
        )
        assert trainer.base_model == "test/model"
        assert trainer.max_seq_length == 4096
        assert trainer.load_in_4bit is True
        assert trainer.model is None
        assert trainer.tokenizer is None

    def test_default_values(self):
        """Test default initialization values."""
        trainer = ConcreteTextTrainer(base_model="test/model")
        assert trainer.max_seq_length == 2048
        assert trainer.load_in_4bit is True
        assert trainer.load_in_8bit is False
        assert trainer.dtype is None


class TestVisionTrainer:
    """Tests for VisionTrainer abstract base class."""

    def test_init(self):
        """Test VisionTrainer initialization."""
        trainer = ConcreteVisionTrainer(
            base_model="test/vision-model",
            max_seq_length=8192,
            load_in_4bit=False,
            load_in_8bit=True,
        )
        assert trainer.base_model == "test/vision-model"
        assert trainer.max_seq_length == 8192
        assert trainer.load_in_4bit is False
        assert trainer.load_in_8bit is True
        assert trainer.processor is None

    def test_default_values(self):
        """Test default initialization values."""
        trainer = ConcreteVisionTrainer(base_model="test/vision-model")
        assert trainer.max_seq_length == 16384  # Larger default for vision
        assert trainer.load_in_4bit is True

    def test_load_dataset_convenience_method(self):
        """Test the load_dataset convenience method delegates correctly."""
        trainer = ConcreteVisionTrainer(base_model="test/model")

        # Mock the underlying methods to avoid dataset creation issues
        from unittest.mock import patch

        mock_file_dataset = MagicMock()
        mock_hub_dataset = MagicMock()

        with patch.object(
            trainer, "load_dataset_from_file", return_value=mock_file_dataset
        ) as mock_file:
            with patch.object(
                trainer, "load_dataset_from_hub", return_value=mock_hub_dataset
            ) as mock_hub:
                # Test loading from file
                result = trainer.load_dataset("/path/to/file.jsonl", from_hub=False)
                mock_file.assert_called_once_with("/path/to/file.jsonl")
                assert result == mock_file_dataset

                # Test loading from hub
                result = trainer.load_dataset("huggingface/dataset", from_hub=True, split="train")
                mock_hub.assert_called_once_with("huggingface/dataset", split="train")
                assert result == mock_hub_dataset


class TestTrainingBackend:
    """Tests for TrainingBackend abstract base class."""

    def test_backend_properties(self):
        """Test backend properties."""
        backend = ConcreteTrainingBackend()
        assert backend.name == "test-backend"
        assert backend.description == "A test backend for unit testing"
        assert backend.supports_text_training() is True
        assert backend.supports_vision_training() is True

    def test_create_text_trainer(self):
        """Test creating a text trainer from backend."""
        backend = ConcreteTrainingBackend()
        trainer = backend.create_text_trainer(
            base_model="test/model",
            max_seq_length=2048,
        )
        assert isinstance(trainer, TextTrainer)
        assert trainer.base_model == "test/model"

    def test_create_vision_trainer(self):
        """Test creating a vision trainer from backend."""
        backend = ConcreteTrainingBackend()
        trainer = backend.create_vision_trainer(
            base_model="test/vision-model",
        )
        assert isinstance(trainer, VisionTrainer)
        assert trainer.base_model == "test/vision-model"

    def test_text_only_backend(self):
        """Test backend that only supports text training."""
        backend = TextOnlyBackend()
        assert backend.supports_text_training() is True
        assert backend.supports_vision_training() is False

        # Should be able to create text trainer
        text_trainer = backend.create_text_trainer("test/model")
        assert isinstance(text_trainer, TextTrainer)

        # Should raise error for vision trainer
        with pytest.raises(NotImplementedError):
            backend.create_vision_trainer("test/vision-model")


class TestBackendRegistry:
    """Tests for backend registry functions."""

    @pytest.fixture(autouse=True)
    def save_and_restore_backends(self):
        """Save and restore the backend registry around each test."""
        original_backends = _BACKENDS.copy()
        yield
        _BACKENDS.clear()
        _BACKENDS.update(original_backends)

    def test_register_backend(self):
        """Test registering a backend."""
        # Clear existing test backends
        if "test" in _BACKENDS:
            del _BACKENDS["test"]

        register_backend("test", ConcreteTrainingBackend)
        assert "test" in _BACKENDS
        assert _BACKENDS["test"] == ConcreteTrainingBackend

    def test_register_backend_case_insensitive(self):
        """Test that backend registration is case-insensitive."""
        if "mybackend" in _BACKENDS:
            del _BACKENDS["mybackend"]

        register_backend("MyBackend", ConcreteTrainingBackend)
        assert "mybackend" in _BACKENDS

    def test_register_invalid_backend(self):
        """Test registering an invalid backend class."""

        class NotABackend:
            pass

        with pytest.raises(ValueError, match="must inherit from TrainingBackend"):
            register_backend("invalid", NotABackend)  # type: ignore[arg-type]

    def test_get_backend(self):
        """Test getting a backend by name."""
        register_backend("test-get", ConcreteTrainingBackend)
        backend = get_backend("test-get")
        assert isinstance(backend, ConcreteTrainingBackend)

    def test_get_backend_case_insensitive(self):
        """Test that getting backend is case-insensitive."""
        register_backend("TestCase", ConcreteTrainingBackend)
        backend = get_backend("TESTCASE")
        assert isinstance(backend, ConcreteTrainingBackend)

    def test_get_backend_not_found(self):
        """Test getting a non-existent backend."""
        with pytest.raises(ValueError, match="not found"):
            get_backend("nonexistent-backend")

    def test_list_backends(self):
        """Test listing all backends."""
        register_backend("list-test-1", ConcreteTrainingBackend)
        register_backend("list-test-2", TextOnlyBackend)

        backends = list_backends()

        # Find our test backends
        test_backends = [b for b in backends if b["name"].startswith("list-test")]
        assert len(test_backends) >= 2

        # Check structure
        for backend in test_backends:
            assert "name" in backend
            assert "description" in backend
            assert "supports_text" in backend
            assert "supports_vision" in backend

    def test_is_backend_available(self):
        """Test checking if backend is available."""
        register_backend("available-test", ConcreteTrainingBackend)

        assert is_backend_available("available-test") is True
        assert is_backend_available("AVAILABLE-TEST") is True  # Case insensitive
        assert is_backend_available("not-registered") is False


class TestRealBackends:
    """Tests for actual registered backends (integration tests)."""

    def test_unsloth_backend_registered(self):
        """Test that unsloth backend is registered."""
        # This tests the actual backends registered at module import
        assert is_backend_available("unsloth")

    def test_transformers_backend_registered(self):
        """Test that transformers backend is registered."""
        assert is_backend_available("transformers")

    def test_get_unsloth_backend(self):
        """Test getting the unsloth backend."""
        backend = get_backend("unsloth")
        assert backend.name == "unsloth"
        assert backend.supports_text_training()
        assert backend.supports_vision_training()

    def test_get_transformers_backend(self):
        """Test getting the transformers backend."""
        backend = get_backend("transformers")
        assert backend.name == "transformers"
        assert backend.supports_text_training()
        assert backend.supports_vision_training()

    def test_default_backend_is_unsloth(self):
        """Test that the default backend is unsloth."""
        backend = get_backend()  # No argument = default
        assert backend.name == "unsloth"
