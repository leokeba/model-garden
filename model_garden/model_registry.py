"""Model Registry - Central management for supported models.

This module provides a single source of truth for all supported models,
their configurations, requirements, and capabilities.
"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

# Path to the registry file
REGISTRY_PATH = Path(__file__).parent.parent / "storage" / "models_registry.json"


class ModelStatus(str, Enum):
    """Model support status."""
    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"


class ModelCategory(str, Enum):
    """Model categories."""
    TEXT_LLM = "text-llm"
    VISION_VLM = "vision-vlm"


@dataclass
class ModelCapabilities:
    """Model capabilities."""
    training: bool
    inference: bool
    vision: bool
    structured_outputs: bool
    streaming: bool
    function_calling: bool = False


@dataclass
class ModelRequirements:
    """Hardware and software requirements."""
    min_vram_gb: float
    recommended_vram_gb: float
    min_ram_gb: float
    cuda_compute_capability: str | None = None
    min_gpus: int = 1


@dataclass
class HyperparametersDefaults:
    """Default training hyperparameters."""
    learning_rate: float
    num_epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    max_seq_length: int
    optim: str
    lr_scheduler_type: str


@dataclass
class LoRADefaults:
    """Default LoRA configuration."""
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    use_rslora: bool = False


@dataclass
class InferenceDefaults:
    """Default inference settings."""
    max_model_len: int
    dtype: str
    gpu_memory_utilization: float
    quantization: str | None = None
    tensor_parallel_size: int = 1
    max_num_seqs: int = 16  # Max concurrent sequences (reduce for memory-constrained GPUs)
    enforce_eager: bool = False  # Disable CUDA graphs (True saves ~2GB memory)
    limit_mm_per_prompt: dict[str, int] | None = None  # Limit multimodal inputs e.g. {"image": 2, "video": 0}


@dataclass
class ModelInfo:
    """Complete model information."""
    id: str
    name: str
    category: str
    provider: str
    base_architecture: str
    parameters: str
    description: str
    tags: list[str]
    status: str
    quantization: dict[str, str | None]
    requirements: ModelRequirements
    capabilities: ModelCapabilities
    training_defaults: dict[str, Any]
    inference_defaults: InferenceDefaults
    urls: dict[str, str | None]

    @property
    def is_vision_model(self) -> bool:
        """Check if this is a vision-language model."""
        return self.capabilities.vision

    @property
    def is_quantized(self) -> bool:
        """Check if this model is pre-quantized."""
        return self.quantization.get("method") is not None

    @property
    def supports_selective_loss(self) -> bool:
        """Check if model supports selective loss for structured outputs."""
        selective_loss = self.training_defaults.get("selective_loss", {})
        return selective_loss.get("supported", False)

    def get_training_hyperparameters(self) -> dict[str, Any]:
        """Get default training hyperparameters."""
        return self.training_defaults["hyperparameters"]

    def get_lora_config(self) -> dict[str, Any]:
        """Get default LoRA configuration."""
        return self.training_defaults["lora_config"]

    def get_inference_config(self) -> dict[str, Any]:
        """Get default inference configuration."""
        config = {
            "max_model_len": self.inference_defaults.max_model_len,
            "dtype": self.inference_defaults.dtype,
            "gpu_memory_utilization": self.inference_defaults.gpu_memory_utilization,
            "quantization": self.inference_defaults.quantization,
            "tensor_parallel_size": self.inference_defaults.tensor_parallel_size,
            "max_num_seqs": self.inference_defaults.max_num_seqs,
            "enforce_eager": self.inference_defaults.enforce_eager,
        }
        # Only include limit_mm_per_prompt if set (for vision models)
        if self.inference_defaults.limit_mm_per_prompt:
            config["limit_mm_per_prompt"] = self.inference_defaults.limit_mm_per_prompt
        return config


class ModelRegistry:
    """Central registry for supported models."""

    def __init__(self, registry_path: Path | None = None):
        """Initialize the registry.

        Args:
            registry_path: Path to the registry JSON file (default: storage/models_registry.json)
        """
        self.registry_path = registry_path or REGISTRY_PATH
        self._registry_data: dict | None = None
        self._models_cache: dict[str, ModelInfo] | None = None

    def _load_registry(self) -> dict:
        """Load the registry from disk."""
        if self._registry_data is None:
            if not self.registry_path.exists():
                raise FileNotFoundError(f"Registry not found: {self.registry_path}")

            with open(self.registry_path) as f:
                self._registry_data = json.load(f)

        assert self._registry_data is not None
        return self._registry_data

    def _parse_model(self, model_id: str, data: dict) -> ModelInfo:
        """Parse model data into ModelInfo object."""
        return ModelInfo(
            id=data["id"],
            name=data["name"],
            category=data["category"],
            provider=data["provider"],
            base_architecture=data["base_architecture"],
            parameters=data["parameters"],
            description=data["description"],
            tags=data["tags"],
            status=data["status"],
            quantization=data["quantization"],
            requirements=ModelRequirements(**data["requirements"]),
            capabilities=ModelCapabilities(**data["capabilities"]),
            training_defaults=data["training_defaults"],
            inference_defaults=InferenceDefaults(**data["inference_defaults"]),
            urls=data["urls"],
        )

    def get_all_models(self) -> dict[str, ModelInfo]:
        """Get all models from the registry.

        Returns:
            Dictionary mapping model IDs to ModelInfo objects
        """
        if self._models_cache is None:
            registry = self._load_registry()
            self._models_cache = {
                model_id: self._parse_model(model_id, data)
                for model_id, data in registry["models"].items()
            }

        return self._models_cache

    def get_model(self, model_id: str) -> ModelInfo | None:
        """Get a specific model by ID.

        Args:
            model_id: Model identifier (HuggingFace model ID)

        Returns:
            ModelInfo object or None if not found
        """
        models = self.get_all_models()
        return models.get(model_id)

    def get_models_by_category(self, category: str) -> list[ModelInfo]:
        """Get all models in a specific category.

        Args:
            category: Category name (e.g., "text-llm", "vision-vlm")

        Returns:
            List of ModelInfo objects
        """
        models = self.get_all_models()
        return [model for model in models.values() if model.category == category]

    def get_text_models(self) -> list[ModelInfo]:
        """Get all text-only models.

        Returns:
            List of text-only ModelInfo objects
        """
        return self.get_models_by_category(ModelCategory.TEXT_LLM.value)

    def get_vision_models(self) -> list[ModelInfo]:
        """Get all vision-language models.

        Returns:
            List of vision-language ModelInfo objects
        """
        return self.get_models_by_category(ModelCategory.VISION_VLM.value)

    def get_models_by_tag(self, tag: str) -> list[ModelInfo]:
        """Get all models with a specific tag.

        Args:
            tag: Tag to filter by (e.g., "quantized", "recommended")

        Returns:
            List of ModelInfo objects
        """
        models = self.get_all_models()
        return [model for model in models.values() if tag in model.tags]

    def get_stable_models(self) -> list[ModelInfo]:
        """Get all stable (production-ready) models.

        Returns:
            List of stable ModelInfo objects
        """
        models = self.get_all_models()
        return [model for model in models.values() if model.status == ModelStatus.STABLE.value]

    def get_recommended_models(self, category: str | None = None) -> list[ModelInfo]:
        """Get recommended models, optionally filtered by category.

        Args:
            category: Optional category filter

        Returns:
            List of recommended ModelInfo objects
        """
        models = self.get_models_by_tag("recommended")
        if category:
            models = [m for m in models if m.category == category]
        return models

    def check_model_exists(self, model_id: str) -> bool:
        """Check if a model is in the registry.

        Args:
            model_id: Model identifier

        Returns:
            True if model exists in registry
        """
        return model_id in self.get_all_models()

    def get_categories(self) -> dict[str, dict[str, str]]:
        """Get all available categories.

        Returns:
            Dictionary of categories with their metadata
        """
        registry = self._load_registry()
        return registry["categories"]

    def validate_model_for_training(self, model_id: str) -> tuple[bool, str | None]:
        """Validate if a model can be used for training.

        Args:
            model_id: Model identifier

        Returns:
            Tuple of (is_valid, error_message)
        """
        model = self.get_model(model_id)

        if model is None:
            return False, f"Model '{model_id}' not found in registry"

        if not model.capabilities.training:
            return False, f"Model '{model_id}' does not support training"

        if model.status == ModelStatus.DEPRECATED.value:
            return False, f"Model '{model_id}' is deprecated"

        return True, None

    def validate_model_for_inference(self, model_id: str) -> tuple[bool, str | None]:
        """Validate if a model can be used for inference.

        Args:
            model_id: Model identifier

        Returns:
            Tuple of (is_valid, error_message)
        """
        model = self.get_model(model_id)

        if model is None:
            return False, f"Model '{model_id}' not found in registry"

        if not model.capabilities.inference:
            return False, f"Model '{model_id}' does not support inference"

        if model.status == ModelStatus.DEPRECATED.value:
            return False, f"Model '{model_id}' is deprecated"

        return True, None

    def get_model_list_for_ui(self, category: str | None = None) -> list[dict[str, Any]]:
        """Get simplified model list for UI consumption.

        Args:
            category: Optional category filter

        Returns:
            List of model dictionaries with essential info
        """
        models = self.get_all_models().values()

        if category:
            models = [m for m in models if m.category == category]

        return [
            {
                "id": model.id,
                "name": model.name,
                "category": model.category,
                "parameters": model.parameters,
                "description": model.description,
                "tags": model.tags,
                "status": model.status,
                "is_vision": model.is_vision_model,
                "is_quantized": model.is_quantized,
                "min_vram_gb": model.requirements.min_vram_gb,
                "recommended_vram_gb": model.requirements.recommended_vram_gb,
            }
            for model in models
        ]


# Global registry instance
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Get the global model registry instance.

    Returns:
        ModelRegistry singleton instance
    """
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


# Convenience functions
def get_model(model_id: str) -> ModelInfo | None:
    """Get model info by ID."""
    return get_registry().get_model(model_id)


def get_text_models() -> list[ModelInfo]:
    """Get all text-only models."""
    return get_registry().get_text_models()


def get_vision_models() -> list[ModelInfo]:
    """Get all vision-language models."""
    return get_registry().get_vision_models()


def validate_model_for_training(model_id: str) -> tuple[bool, str | None]:
    """Validate model for training."""
    return get_registry().validate_model_for_training(model_id)


def validate_model_for_inference(model_id: str) -> tuple[bool, str | None]:
    """Validate model for inference."""
    return get_registry().validate_model_for_inference(model_id)
