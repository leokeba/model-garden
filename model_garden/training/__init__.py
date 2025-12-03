# Training package
"""
Training components for Model Garden:
- trainer.py: Text model training (ModelTrainer)
- vision_trainer.py: Vision-language model training (VisionLanguageTrainer)
- config.py: Training configuration dataclasses
- mixins.py: Shared trainer mixin with common functionality
- callbacks.py: Training callbacks (metrics, progress estimation)
- protocols.py: Protocol-based interfaces for duck typing
- pipeline.py: Unified training pipeline
- selective_loss.py: Selective loss for structured outputs
- early_stopping.py: Early stopping callback
- weighted_loss.py: Weighted loss trainers
- utils.py: Training utilities (deprecated, use mixins.py)
- subprocess_runner.py: Subprocess-based training execution
"""

# Configuration dataclasses
from .config import (
    LoRAConfig,
    ModelConfig,
    TrainingConfig,
    VisionLoRAConfig,
    VisionModelConfig,
    VisionTrainingConfig,
)

# Callbacks
from .callbacks import (
    ProgressEstimate,
    ProgressEstimationCallback,
    TrainingMetrics,
    TrainingMetricsCallback,
)
from .early_stopping import EarlyStoppingCallback

# Shared mixin and utilities (consolidated location)
from .mixins import (
    MemoryMonitorCallback,
    TrainerMixin,
    cleanup_memory,
    detect_model_dtype,
    get_training_precision_config,
    retry_with_backoff,
)

# Unified training pipeline
from .pipeline import (
    TrainingResult,
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

# Selective loss
from .selective_loss import (
    SelectiveLossVisionCollator,
    create_selective_loss_collator,
    detect_schema_keys_from_dataset,
)

# Subprocess execution
from .subprocess_runner import (
    execute_training_job_in_subprocess,
    run_training_in_subprocess,
)

# Main trainers
from .trainer import ModelTrainer, create_sample_dataset, create_text_trainer
from .vision_trainer import (
    LazyVisionDataset,
    VisionLanguageTrainer,
    create_vision_sample_dataset,
    create_vision_trainer,
    merge_vision_lora_adapter,
)

# Weighted loss trainers
from .weighted_loss import WeightedLossTrainer, WeightedLossTrainerWithMetrics

__all__ = [
    # Configuration
    "TrainingConfig",
    "VisionTrainingConfig",
    "LoRAConfig",
    "VisionLoRAConfig",
    "ModelConfig",
    "VisionModelConfig",
    # Trainers
    "ModelTrainer",
    "VisionLanguageTrainer",
    "TrainerMixin",
    "LazyVisionDataset",
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
    "SelectiveLossVisionCollator",
    "create_selective_loss_collator",
    "detect_schema_keys_from_dataset",
    # Weighted loss
    "WeightedLossTrainer",
    "WeightedLossTrainerWithMetrics",
    # Utilities
    "detect_model_dtype",
    "get_training_precision_config",
    "MemoryMonitorCallback",
    "cleanup_memory",
    "retry_with_backoff",
    # Subprocess
    "run_training_in_subprocess",
    "execute_training_job_in_subprocess",
]
