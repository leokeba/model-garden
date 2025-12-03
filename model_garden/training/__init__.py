# Training package
"""
Training components for Model Garden:
- trainer.py: Text model training (ModelTrainer)
- vision_trainer.py: Vision-language model training (VisionLanguageTrainer)
- config.py: Training configuration dataclasses
- mixins.py: Shared trainer mixin with common functionality
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
- model_loading.py: Model loading with retry support
"""

# Configuration dataclasses
from .config import (
    LoRAConfig,
    ModelConfig,
    SelectiveLossConfig,
    TrainingConfig,
    VisionLoRAConfig,
    VisionModelConfig,
    VisionTrainingConfig,
)

# Callbacks (consolidated in callbacks package)
from .callbacks import (
    EarlyStoppingCallback,
    MemoryMonitorCallback,
    ProgressEstimate,
    ProgressEstimationCallback,
    TrainingMetrics,
    TrainingMetricsCallback,
)

# Shared mixin and utilities (consolidated location)
from .mixins import (
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
    VisionLanguageTrainer,
    create_vision_sample_dataset,
    create_vision_trainer,
    merge_vision_lora_adapter,
)

# Lazy dataset for vision models
from .lazy_dataset import LazyVisionDataset, LazyVisionDatasetWithMultipleImages

# Custom SFT trainer
from .sft_trainer import FixedSFTTrainer, ConsistentLossSFTTrainer

# Weighted loss trainers
from .weighted_loss import WeightedLossTrainer, WeightedLossTrainerWithMetrics

# Dataset format utilities
from .dataset_formats import DatasetFormatConverter

# Chat template utilities
from .chat_template import ChatTemplateDetector, FALLBACK_MARKERS

# Model loading utilities
from .model_loading import ModelLoaderWithFallback, ModelLoadingError

__all__ = [
    # Configuration
    "TrainingConfig",
    "VisionTrainingConfig",
    "SelectiveLossConfig",
    "LoRAConfig",
    "VisionLoRAConfig",
    "ModelConfig",
    "VisionModelConfig",
    # Trainers
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
    # Dataset format utilities
    "DatasetFormatConverter",
    # Chat template utilities
    "ChatTemplateDetector",
    "FALLBACK_MARKERS",
    # Model loading utilities
    "ModelLoaderWithFallback",
    "ModelLoadingError",
]
