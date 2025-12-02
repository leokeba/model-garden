# Training package
"""
Training components for Model Garden:
- trainer.py: Text model training (ModelTrainer)
- vision_trainer.py: Vision-language model training (VisionLanguageTrainer)
- selective_loss.py: Selective loss for structured outputs
- early_stopping.py: Early stopping callback
- weighted_loss.py: Weighted loss trainers
- utils.py: Training utilities
- subprocess_runner.py: Subprocess-based training execution
"""

# Main trainers
# Callbacks
from .early_stopping import EarlyStoppingCallback

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
from .trainer import ModelTrainer, create_sample_dataset, create_text_trainer

# Utilities
from .utils import (
    MemoryMonitorCallback,
    detect_model_dtype,
    get_training_precision_config,
)
from .vision_trainer import (
    VisionLanguageTrainer,
    create_vision_sample_dataset,
    create_vision_trainer,
    merge_vision_lora_adapter,
)

# Weighted loss trainers
from .weighted_loss import WeightedLossTrainer, WeightedLossTrainerWithMetrics

__all__ = [
    # Trainers
    "ModelTrainer",
    "VisionLanguageTrainer",
    "create_text_trainer",
    "create_vision_trainer",
    "create_sample_dataset",
    "create_vision_sample_dataset",
    "merge_vision_lora_adapter",
    # Callbacks
    "EarlyStoppingCallback",
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
    # Subprocess
    "run_training_in_subprocess",
    "execute_training_job_in_subprocess",
]
