# Training package
"""
Training components for Model Garden:
- config.py: Training configuration dataclasses
- mixins.py: Shared trainer mixin with common functionality
- backends/: Training backends (Unsloth, Transformers)
  - base.py: Abstract base classes (TextTrainer, VisionTrainer, TrainingBackend)
  - registry.py: Backend registration system
  - unsloth_backend.py: Unsloth-optimized backend
  - unsloth_text_trainer.py: Unsloth text trainer (ModelTrainer)
  - unsloth_vision_trainer.py: Unsloth vision trainer (VisionLanguageTrainer)
  - transformers_backend.py: Standard HuggingFace + PEFT backend
- callbacks/: Training callbacks package (consolidated)
  - metrics.py: TrainingMetricsCallback
  - progress.py: ProgressEstimationCallback
  - early_stopping.py: EarlyStoppingCallback
  - memory.py: MemoryMonitorCallback
- protocols.py: Protocol-based interfaces for duck typing
- pipeline.py: Unified training pipeline
- selective_loss.py: Selective loss for structured outputs
- weighted_loss.py: Weighted loss trainers
- subprocess_runner.py: Subprocess-based training execution
- dataset_formats.py: Dataset format detection and conversion
- chat_template.py: Chat template detection utilities

NOTE: ModelTrainer and VisionLanguageTrainer are Unsloth-specific and have been
moved to the backends folder. Use create_text_trainer() and create_vision_trainer()
with backend selection, or import directly from backends for Unsloth-specific usage.
"""

# Training backends
from .backends import (
    TextTrainer,
    TrainingBackend,
    VisionTrainer,
    get_backend,
    get_default_backend,
    list_backends,
    register_backend,
)

# Configuration dataclasses
# Callbacks (consolidated in callbacks package)
from .callbacks import (
    EarlyStoppingCallback,
    MemoryMonitorCallback,
    ProgressEstimate,
    ProgressEstimationCallback,
    TrainingMetrics,
    TrainingMetricsCallback,
)

# Chat template utilities
from .chat_template import FALLBACK_MARKERS, ChatTemplateDetector
from .config import (
    LoRAConfig,
    ModelConfig,
    SelectiveLossConfig,
    TrainingConfig,
    VisionLoRAConfig,
    VisionModelConfig,
    VisionTrainingConfig,
)

# Dataset format utilities
from .dataset_formats import DatasetFormatConverter

# Lazy dataset for vision models
from .lazy_dataset import LazyVisionDataset, LazyVisionDatasetWithMultipleImages

# Shared mixin and utilities (consolidated location)
from .mixins import (
    TrainerMixin,
    cleanup_memory,
    detect_model_dtype,
    get_training_precision_config,
)

# Unified training pipeline and factory functions (backend-agnostic)
from .pipeline import (
    TrainingResult,
    create_text_trainer,
    create_vision_trainer,
    is_vision_model,
    train,
    train_text,
    train_vision,
)

# Protocol-based interfaces
from .protocols import (
    TextTrainerProtocol,
    TrainingBackendProtocol,
    VisionTrainerProtocol,
    is_text_trainer,
    is_training_backend,
    is_vision_trainer,
)

# Selective loss (generic parts only - Unsloth-specific parts are in backends)
from .selective_loss import (
    SelectiveLossCollator,
    SelectiveLossMixin,
    create_selective_loss_collator,
    detect_schema_keys_from_dataset,
)

# Custom SFT trainer
from .sft_trainer import ConsistentLossSFTTrainer, FixedSFTTrainer

# Subprocess execution
from .subprocess_runner import (
    execute_training_job_in_subprocess,
    run_training_in_subprocess,
)

# Weighted loss trainers
from .weighted_loss import WeightedLossTrainer, WeightedLossTrainerWithMetrics

# =============================================================================
# Backwards Compatibility: Lazy imports for ModelTrainer and VisionLanguageTrainer
# These are Unsloth-specific and now live in backends/
# =============================================================================


def __getattr__(name: str):
    """Lazy import for backwards compatibility with Unsloth-specific trainers."""
    if name == "ModelTrainer":
        # Backwards compatibility - import from Unsloth backend (requires Unsloth)
        from model_garden.utils.optional_deps import require_unsloth

        require_unsloth("ModelTrainer is an Unsloth-specific class")
        from model_garden.training.backends.unsloth_text_trainer import ModelTrainer

        return ModelTrainer
    elif name == "VisionLanguageTrainer":
        # Backwards compatibility - import from Unsloth backend (requires Unsloth)
        from model_garden.utils.optional_deps import require_unsloth

        require_unsloth("VisionLanguageTrainer is an Unsloth-specific class")
        from model_garden.training.backends.unsloth_vision_trainer import VisionLanguageTrainer

        return VisionLanguageTrainer
    elif name == "create_sample_dataset":
        # Unsloth-specific sample dataset creator
        from model_garden.utils.optional_deps import require_unsloth

        require_unsloth("create_sample_dataset is an Unsloth-specific function")
        from model_garden.training.backends.unsloth_text_trainer import create_sample_dataset

        return create_sample_dataset
    elif name == "create_vision_sample_dataset":
        # Unsloth-specific sample dataset creator
        from model_garden.utils.optional_deps import require_unsloth

        require_unsloth("create_vision_sample_dataset is an Unsloth-specific function")
        from model_garden.training.backends.unsloth_vision_trainer import (
            create_vision_sample_dataset,
        )

        return create_vision_sample_dataset
    elif name == "merge_vision_lora_adapter":
        # Unsloth-specific function
        from model_garden.utils.optional_deps import require_unsloth

        require_unsloth("merge_vision_lora_adapter is an Unsloth-specific function")
        from model_garden.training.backends.unsloth_vision_trainer import merge_vision_lora_adapter

        return merge_vision_lora_adapter
    # Selective loss Unsloth-specific exports (backwards compat)
    elif name == "SelectiveLossUnslothCollator":
        from model_garden.training.selective_loss import SelectiveLossUnslothCollator

        return SelectiveLossUnslothCollator
    elif name == "SelectiveLossVisionCollator":
        from model_garden.training.selective_loss import SelectiveLossVisionCollator

        return SelectiveLossVisionCollator
    elif name == "is_unsloth_available":
        from model_garden.utils.optional_deps import is_unsloth_installed

        return is_unsloth_installed
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Configuration
    "TrainingConfig",
    "VisionTrainingConfig",
    "SelectiveLossConfig",
    "LoRAConfig",
    "VisionLoRAConfig",
    "ModelConfig",
    "VisionModelConfig",
    # Trainers (backwards compat - lazy loaded)
    "ModelTrainer",
    "VisionLanguageTrainer",
    "TrainerMixin",
    "LazyVisionDataset",
    "LazyVisionDatasetWithMultipleImages",
    "FixedSFTTrainer",
    "ConsistentLossSFTTrainer",
    "create_text_trainer",
    "create_vision_trainer",
    "create_sample_dataset",
    "create_vision_sample_dataset",
    "merge_vision_lora_adapter",
    # Callbacks
    "EarlyStoppingCallback",
    "TrainingMetricsCallback",
    "TrainingMetrics",
    "ProgressEstimationCallback",
    "ProgressEstimate",
    # Protocols
    "TextTrainerProtocol",
    "VisionTrainerProtocol",
    "TrainingBackendProtocol",
    "is_text_trainer",
    "is_vision_trainer",
    "is_training_backend",
    # Unified pipeline
    "train",
    "train_text",
    "train_vision",
    "TrainingResult",
    "is_vision_model",
    # Selective loss
    "SelectiveLossMixin",
    "SelectiveLossCollator",
    "SelectiveLossUnslothCollator",
    "SelectiveLossVisionCollator",  # Backwards compatibility alias
    "create_selective_loss_collator",
    "detect_schema_keys_from_dataset",
    "is_unsloth_available",
    # Weighted loss
    "WeightedLossTrainer",
    "WeightedLossTrainerWithMetrics",
    # Utilities
    "detect_model_dtype",
    "get_training_precision_config",
    "MemoryMonitorCallback",
    "cleanup_memory",
    # Subprocess
    "run_training_in_subprocess",
    "execute_training_job_in_subprocess",
    # Dataset format utilities
    "DatasetFormatConverter",
    # Chat template utilities
    "ChatTemplateDetector",
    "FALLBACK_MARKERS",
    # Training backends
    "TextTrainer",
    "VisionTrainer",
    "TrainingBackend",
    "get_backend",
    "get_default_backend",
    "list_backends",
    "register_backend",
]
