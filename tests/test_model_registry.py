"""Tests for model_garden.model_registry module."""

import json
from pathlib import Path

import pytest

from model_garden.model_registry import (
    InferenceDefaults,
    ModelCapabilities,
    ModelCategory,
    ModelInfo,
    ModelRegistry,
    ModelRequirements,
    ModelStatus,
    get_model,
    get_registry,
    get_text_models,
    get_vision_models,
    validate_model_for_inference,
    validate_model_for_training,
)


class TestModelStatus:
    """Tests for ModelStatus enum."""

    def test_status_values(self):
        """Test ModelStatus enum values."""
        assert ModelStatus.STABLE == "stable"
        assert ModelStatus.EXPERIMENTAL == "experimental"
        assert ModelStatus.DEPRECATED == "deprecated"


class TestModelCategory:
    """Tests for ModelCategory enum."""

    def test_category_values(self):
        """Test ModelCategory enum values."""
        assert ModelCategory.TEXT_LLM == "text-llm"
        assert ModelCategory.VISION_VLM == "vision-vlm"


class TestModelCapabilities:
    """Tests for ModelCapabilities dataclass."""

    def test_create_capabilities(self):
        """Test creating model capabilities."""
        caps = ModelCapabilities(
            training=True,
            inference=True,
            vision=False,
            structured_outputs=True,
            streaming=True,
            function_calling=False,
        )
        assert caps.training is True
        assert caps.vision is False

    def test_default_function_calling(self):
        """Test default function_calling value."""
        caps = ModelCapabilities(
            training=True,
            inference=True,
            vision=False,
            structured_outputs=True,
            streaming=True,
        )
        assert caps.function_calling is False


class TestModelRequirements:
    """Tests for ModelRequirements dataclass."""

    def test_create_requirements(self):
        """Test creating model requirements."""
        reqs = ModelRequirements(
            min_vram_gb=4.0,
            recommended_vram_gb=8.0,
            min_ram_gb=16.0,
            cuda_compute_capability="7.0",
        )
        assert reqs.min_vram_gb == 4.0
        assert reqs.cuda_compute_capability == "7.0"

    def test_default_values(self):
        """Test default values."""
        reqs = ModelRequirements(
            min_vram_gb=4.0,
            recommended_vram_gb=8.0,
            min_ram_gb=16.0,
        )
        assert reqs.cuda_compute_capability is None
        assert reqs.min_gpus == 1


class TestInferenceDefaults:
    """Tests for InferenceDefaults dataclass."""

    def test_create_defaults(self):
        """Test creating inference defaults."""
        defaults = InferenceDefaults(
            max_model_len=4096,
            dtype="auto",
            gpu_memory_utilization=0.9,
            quantization="awq",
            tensor_parallel_size=2,
        )
        assert defaults.max_model_len == 4096
        assert defaults.quantization == "awq"

    def test_default_values(self):
        """Test default values."""
        defaults = InferenceDefaults(
            max_model_len=2048,
            dtype="float16",
            gpu_memory_utilization=0.8,
        )
        assert defaults.quantization is None
        assert defaults.tensor_parallel_size == 1
        assert defaults.max_num_seqs == 16
        assert defaults.enforce_eager is False
        assert defaults.limit_mm_per_prompt is None


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    @pytest.fixture
    def sample_registry_data(self) -> dict:
        """Create sample registry data."""
        return {
            "version": "1.0.0",
            "categories": {
                "text-llm": {
                    "name": "Text Models",
                    "description": "Text-only models",
                },
                "vision-vlm": {
                    "name": "Vision Models",
                    "description": "Vision-language models",
                },
            },
            "models": {
                "test/text-model": {
                    "id": "test/text-model",
                    "name": "Test Text Model",
                    "category": "text-llm",
                    "provider": "test",
                    "base_architecture": "llama",
                    "parameters": "1B",
                    "description": "A test text model",
                    "tags": ["test", "recommended"],
                    "status": "stable",
                    "quantization": {"method": None, "type": None},
                    "requirements": {
                        "min_vram_gb": 4,
                        "recommended_vram_gb": 8,
                        "min_ram_gb": 16,
                    },
                    "capabilities": {
                        "training": True,
                        "inference": True,
                        "vision": False,
                        "structured_outputs": True,
                        "streaming": True,
                    },
                    "training_defaults": {
                        "hyperparameters": {
                            "learning_rate": 2e-5,
                            "num_epochs": 3,
                        },
                        "lora_config": {
                            "r": 16,
                            "lora_alpha": 32,
                        },
                        "selective_loss": {
                            "supported": True,
                        },
                    },
                    "inference_defaults": {
                        "max_model_len": 2048,
                        "dtype": "auto",
                        "gpu_memory_utilization": 0.9,
                    },
                    "urls": {
                        "huggingface": "https://huggingface.co/test/text-model",
                    },
                },
                "test/vision-model": {
                    "id": "test/vision-model",
                    "name": "Test Vision Model",
                    "category": "vision-vlm",
                    "provider": "test",
                    "base_architecture": "qwen2-vl",
                    "parameters": "3B",
                    "description": "A test vision model",
                    "tags": ["test", "vision"],
                    "status": "stable",
                    "quantization": {"method": "4bit", "type": "bitsandbytes"},
                    "requirements": {
                        "min_vram_gb": 8,
                        "recommended_vram_gb": 16,
                        "min_ram_gb": 32,
                    },
                    "capabilities": {
                        "training": True,
                        "inference": True,
                        "vision": True,
                        "structured_outputs": True,
                        "streaming": True,
                    },
                    "training_defaults": {
                        "hyperparameters": {
                            "learning_rate": 1e-5,
                            "num_epochs": 2,
                        },
                        "lora_config": {
                            "r": 8,
                            "lora_alpha": 16,
                        },
                    },
                    "inference_defaults": {
                        "max_model_len": 4096,
                        "dtype": "bfloat16",
                        "gpu_memory_utilization": 0.85,
                        "limit_mm_per_prompt": {"image": 4},
                    },
                    "urls": {
                        "huggingface": "https://huggingface.co/test/vision-model",
                    },
                },
                "test/deprecated-model": {
                    "id": "test/deprecated-model",
                    "name": "Test Deprecated Model",
                    "category": "text-llm",
                    "provider": "test",
                    "base_architecture": "llama",
                    "parameters": "1B",
                    "description": "A deprecated model",
                    "tags": ["deprecated"],
                    "status": "deprecated",
                    "quantization": {"method": None, "type": None},
                    "requirements": {
                        "min_vram_gb": 4,
                        "recommended_vram_gb": 8,
                        "min_ram_gb": 16,
                    },
                    "capabilities": {
                        "training": True,
                        "inference": True,
                        "vision": False,
                        "structured_outputs": False,
                        "streaming": True,
                    },
                    "training_defaults": {
                        "hyperparameters": {},
                        "lora_config": {},
                    },
                    "inference_defaults": {
                        "max_model_len": 2048,
                        "dtype": "auto",
                        "gpu_memory_utilization": 0.9,
                    },
                    "urls": {},
                },
            },
        }

    @pytest.fixture
    def temp_registry(self, temp_dir: Path, sample_registry_data: dict) -> ModelRegistry:
        """Create a registry with temporary storage."""
        registry_path = temp_dir / "test_registry.json"
        with open(registry_path, "w") as f:
            json.dump(sample_registry_data, f)
        return ModelRegistry(registry_path=registry_path)

    def test_load_registry(self, temp_registry: ModelRegistry):
        """Test loading the registry."""
        models = temp_registry.get_all_models()
        assert len(models) == 3

    def test_get_model_existing(self, temp_registry: ModelRegistry):
        """Test getting an existing model."""
        model = temp_registry.get_model("test/text-model")
        assert model is not None
        assert model.name == "Test Text Model"
        assert model.category == "text-llm"

    def test_get_model_not_found(self, temp_registry: ModelRegistry):
        """Test getting a non-existent model."""
        model = temp_registry.get_model("nonexistent/model")
        assert model is None

    def test_get_models_by_category(self, temp_registry: ModelRegistry):
        """Test getting models by category."""
        text_models = temp_registry.get_models_by_category("text-llm")
        assert len(text_models) == 2  # text-model and deprecated-model

        vision_models = temp_registry.get_models_by_category("vision-vlm")
        assert len(vision_models) == 1

    def test_get_text_models(self, temp_registry: ModelRegistry):
        """Test getting text models."""
        text_models = temp_registry.get_text_models()
        assert len(text_models) == 2
        assert all(not m.is_vision_model for m in text_models)

    def test_get_vision_models(self, temp_registry: ModelRegistry):
        """Test getting vision models."""
        vision_models = temp_registry.get_vision_models()
        assert len(vision_models) == 1
        assert all(m.is_vision_model for m in vision_models)

    def test_get_models_by_tag(self, temp_registry: ModelRegistry):
        """Test getting models by tag."""
        recommended = temp_registry.get_models_by_tag("recommended")
        assert len(recommended) == 1
        assert recommended[0].id == "test/text-model"

    def test_get_stable_models(self, temp_registry: ModelRegistry):
        """Test getting stable models."""
        stable = temp_registry.get_stable_models()
        assert len(stable) == 2  # text-model and vision-model (not deprecated)
        assert all(m.status == "stable" for m in stable)

    def test_get_recommended_models(self, temp_registry: ModelRegistry):
        """Test getting recommended models."""
        recommended = temp_registry.get_recommended_models()
        assert len(recommended) == 1

        # With category filter
        recommended_text = temp_registry.get_recommended_models(category="text-llm")
        assert len(recommended_text) == 1

        recommended_vision = temp_registry.get_recommended_models(category="vision-vlm")
        assert len(recommended_vision) == 0

    def test_check_model_exists(self, temp_registry: ModelRegistry):
        """Test checking if model exists."""
        assert temp_registry.check_model_exists("test/text-model") is True
        assert temp_registry.check_model_exists("nonexistent") is False

    def test_get_categories(self, temp_registry: ModelRegistry):
        """Test getting categories."""
        categories = temp_registry.get_categories()
        assert "text-llm" in categories
        assert "vision-vlm" in categories

    def test_validate_model_for_training_valid(self, temp_registry: ModelRegistry):
        """Test validating a valid model for training."""
        is_valid, error = temp_registry.validate_model_for_training("test/text-model")
        assert is_valid is True
        assert error is None

    def test_validate_model_for_training_not_found(self, temp_registry: ModelRegistry):
        """Test validating a non-existent model for training."""
        is_valid, error = temp_registry.validate_model_for_training("nonexistent")
        assert is_valid is False
        assert "not found" in error

    def test_validate_model_for_training_deprecated(self, temp_registry: ModelRegistry):
        """Test validating a deprecated model for training."""
        is_valid, error = temp_registry.validate_model_for_training("test/deprecated-model")
        assert is_valid is False
        assert "deprecated" in error.lower()

    def test_validate_model_for_inference(self, temp_registry: ModelRegistry):
        """Test validating a model for inference."""
        is_valid, error = temp_registry.validate_model_for_inference("test/vision-model")
        assert is_valid is True
        assert error is None

    def test_get_model_list_for_ui(self, temp_registry: ModelRegistry):
        """Test getting model list for UI."""
        ui_list = temp_registry.get_model_list_for_ui()
        assert len(ui_list) == 3

        # Check structure
        first = ui_list[0]
        assert "id" in first
        assert "name" in first
        assert "is_vision" in first
        assert "is_quantized" in first
        assert "min_vram_gb" in first

    def test_get_model_list_for_ui_filtered(self, temp_registry: ModelRegistry):
        """Test getting filtered model list for UI."""
        ui_list = temp_registry.get_model_list_for_ui(category="vision-vlm")
        assert len(ui_list) == 1
        assert ui_list[0]["is_vision"] is True

    def test_registry_file_not_found(self, temp_dir: Path):
        """Test handling missing registry file."""
        registry = ModelRegistry(registry_path=temp_dir / "nonexistent.json")
        with pytest.raises(FileNotFoundError):
            registry.get_all_models()


class TestModelInfo:
    """Tests for ModelInfo properties."""

    @pytest.fixture
    def text_model(self, temp_dir: Path) -> ModelInfo:
        """Create a sample text model."""
        return ModelInfo(
            id="test/text-model",
            name="Test Model",
            category="text-llm",
            provider="test",
            base_architecture="llama",
            parameters="1B",
            description="Test model",
            tags=["test"],
            status="stable",
            quantization={"method": None, "type": None},
            requirements=ModelRequirements(min_vram_gb=4, recommended_vram_gb=8, min_ram_gb=16),
            capabilities=ModelCapabilities(
                training=True,
                inference=True,
                vision=False,
                structured_outputs=True,
                streaming=True,
            ),
            training_defaults={
                "hyperparameters": {"learning_rate": 2e-5},
                "lora_config": {"r": 16},
                "selective_loss": {"supported": True},
            },
            inference_defaults=InferenceDefaults(
                max_model_len=2048, dtype="auto", gpu_memory_utilization=0.9
            ),
            urls={"huggingface": "https://huggingface.co/test"},
        )

    @pytest.fixture
    def vision_model(self) -> ModelInfo:
        """Create a sample vision model."""
        return ModelInfo(
            id="test/vision-model",
            name="Test Vision",
            category="vision-vlm",
            provider="test",
            base_architecture="qwen2-vl",
            parameters="3B",
            description="Test vision model",
            tags=["vision"],
            status="stable",
            quantization={"method": "4bit", "type": "bnb"},
            requirements=ModelRequirements(min_vram_gb=8, recommended_vram_gb=16, min_ram_gb=32),
            capabilities=ModelCapabilities(
                training=True,
                inference=True,
                vision=True,
                structured_outputs=True,
                streaming=True,
            ),
            training_defaults={
                "hyperparameters": {"learning_rate": 1e-5},
                "lora_config": {"r": 8},
            },
            inference_defaults=InferenceDefaults(
                max_model_len=4096,
                dtype="bfloat16",
                gpu_memory_utilization=0.85,
                limit_mm_per_prompt={"image": 4},
            ),
            urls={},
        )

    def test_is_vision_model(self, text_model: ModelInfo, vision_model: ModelInfo):
        """Test is_vision_model property."""
        assert text_model.is_vision_model is False
        assert vision_model.is_vision_model is True

    def test_is_quantized(self, text_model: ModelInfo, vision_model: ModelInfo):
        """Test is_quantized property."""
        assert text_model.is_quantized is False
        assert vision_model.is_quantized is True

    def test_supports_selective_loss(self, text_model: ModelInfo, vision_model: ModelInfo):
        """Test supports_selective_loss property."""
        assert text_model.supports_selective_loss is True
        assert vision_model.supports_selective_loss is False

    def test_get_training_hyperparameters(self, text_model: ModelInfo):
        """Test getting training hyperparameters."""
        hyperparams = text_model.get_training_hyperparameters()
        assert "learning_rate" in hyperparams
        assert hyperparams["learning_rate"] == 2e-5

    def test_get_lora_config(self, text_model: ModelInfo):
        """Test getting LoRA config."""
        lora_config = text_model.get_lora_config()
        assert "r" in lora_config
        assert lora_config["r"] == 16

    def test_get_inference_config(self, text_model: ModelInfo, vision_model: ModelInfo):
        """Test getting inference config."""
        config = text_model.get_inference_config()
        assert config["max_model_len"] == 2048
        assert config["dtype"] == "auto"
        assert "limit_mm_per_prompt" not in config  # Not set for text model

        vision_config = vision_model.get_inference_config()
        assert "limit_mm_per_prompt" in vision_config
        assert vision_config["limit_mm_per_prompt"]["image"] == 4


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_registry(self):
        """Test getting the global registry."""
        registry = get_registry()
        assert isinstance(registry, ModelRegistry)

    def test_get_registry_singleton(self):
        """Test that get_registry returns singleton."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

    def test_get_model(self):
        """Test get_model convenience function."""
        # This tests against the real registry
        model = get_model("unsloth/tinyllama-bnb-4bit")
        # May be None if registry doesn't exist, but shouldn't crash
        if model is not None:
            assert model.id == "unsloth/tinyllama-bnb-4bit"

    def test_get_text_models_function(self):
        """Test get_text_models convenience function."""
        models = get_text_models()
        assert isinstance(models, list)

    def test_get_vision_models_function(self):
        """Test get_vision_models convenience function."""
        models = get_vision_models()
        assert isinstance(models, list)

    def test_validate_model_for_training_function(self):
        """Test validate_model_for_training convenience function."""
        is_valid, error = validate_model_for_training("nonexistent/model")
        assert is_valid is False

    def test_validate_model_for_inference_function(self):
        """Test validate_model_for_inference convenience function."""
        is_valid, error = validate_model_for_inference("nonexistent/model")
        assert is_valid is False
