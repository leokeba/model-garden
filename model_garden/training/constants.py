"""Training constants and default values.

This module centralizes all magic numbers and default values used in training,
making them easy to find, document, and adjust.
"""

from typing import Final

# =============================================================================
# Retry Configuration
# =============================================================================

# Default number of retry attempts for network operations
DEFAULT_RETRY_ATTEMPTS: Final[int] = 3

# Base delay between retries (seconds)
RETRY_BASE_DELAY_SECONDS: Final[float] = 1.0

# Maximum delay between retries (seconds)
RETRY_MAX_DELAY_SECONDS: Final[float] = 10.0

# Exponential backoff multiplier
RETRY_EXPONENTIAL_BACKOFF: Final[float] = 2.0

# =============================================================================
# Learning Rates
# =============================================================================

# Default learning rate for text models
TEXT_DEFAULT_LEARNING_RATE: Final[float] = 2e-4

# Default learning rate for vision models (lower for stability)
VISION_DEFAULT_LEARNING_RATE: Final[float] = 2e-5

# Minimum recommended learning rate
MIN_LEARNING_RATE: Final[float] = 1e-6

# Maximum recommended learning rate
MAX_LEARNING_RATE: Final[float] = 1e-3


# =============================================================================
# Batch Sizes
# =============================================================================

# Default batch size for text models
TEXT_DEFAULT_BATCH_SIZE: Final[int] = 2

# Default batch size for vision models (smaller due to memory)
VISION_DEFAULT_BATCH_SIZE: Final[int] = 1

# Default gradient accumulation steps for text models
TEXT_DEFAULT_GRADIENT_ACCUMULATION: Final[int] = 4

# Default gradient accumulation steps for vision models (larger to compensate)
VISION_DEFAULT_GRADIENT_ACCUMULATION: Final[int] = 8


# =============================================================================
# Sequence Lengths
# =============================================================================

# Default max sequence length for text models
TEXT_DEFAULT_MAX_SEQ_LENGTH: Final[int] = 2048

# Default max sequence length for vision models (larger for image tokens)
VISION_DEFAULT_MAX_SEQ_LENGTH: Final[int] = 16384

# Minimum recommended sequence length for vision models
# (images use ~1500+ tokens, need room for prompts/responses)
VISION_MIN_RECOMMENDED_SEQ_LENGTH: Final[int] = 4096


# =============================================================================
# Training Steps
# =============================================================================

# Default warmup steps
DEFAULT_WARMUP_STEPS: Final[int] = 10

# Default logging steps
DEFAULT_LOGGING_STEPS: Final[int] = 10

# Default save steps
DEFAULT_SAVE_STEPS: Final[int] = 100

# Default checkpoint save limit
DEFAULT_SAVE_TOTAL_LIMIT: Final[int] = 3

# Default number of training epochs
DEFAULT_NUM_EPOCHS: Final[int] = 3


# =============================================================================
# Optimizer Settings
# =============================================================================

# Default optimizer
DEFAULT_OPTIMIZER: Final[str] = "adamw_8bit"

# Quality mode optimizer (better but more memory)
QUALITY_OPTIMIZER: Final[str] = "adamw_torch_fused"

# Default weight decay
DEFAULT_WEIGHT_DECAY: Final[float] = 0.01

# Default max gradient norm for clipping
DEFAULT_MAX_GRAD_NORM: Final[float] = 1.0

# Adam optimizer defaults
DEFAULT_ADAM_BETA1: Final[float] = 0.9
DEFAULT_ADAM_BETA2: Final[float] = 0.999
DEFAULT_ADAM_EPSILON: Final[float] = 1e-8


# =============================================================================
# Learning Rate Schedulers
# =============================================================================

# Default scheduler for text models
TEXT_DEFAULT_LR_SCHEDULER: Final[str] = "linear"

# Default scheduler for vision models (cosine works better)
VISION_DEFAULT_LR_SCHEDULER: Final[str] = "cosine"

# Available scheduler types
AVAILABLE_LR_SCHEDULERS: Final[tuple[str, ...]] = (
    "linear",
    "cosine",
    "constant",
    "constant_with_warmup",
    "polynomial",
)


# =============================================================================
# LoRA Defaults
# =============================================================================

# Default LoRA rank
DEFAULT_LORA_R: Final[int] = 16

# Default LoRA alpha (typically equal to r)
DEFAULT_LORA_ALPHA: Final[int] = 16

# Default LoRA dropout
DEFAULT_LORA_DROPOUT: Final[float] = 0.0

# Default LoRA bias setting
DEFAULT_LORA_BIAS: Final[str] = "none"

# Threshold above which RSLoRA is recommended
RSLORA_RECOMMENDED_RANK_THRESHOLD: Final[int] = 32

# Default target modules for most transformer models
DEFAULT_LORA_TARGET_MODULES: Final[tuple[str, ...]] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


# =============================================================================
# Memory and Performance
# =============================================================================

# Default number of dataloader workers
DEFAULT_DATALOADER_NUM_WORKERS: Final[int] = 0

# Default pin memory setting for text models
TEXT_DEFAULT_PIN_MEMORY: Final[bool] = True

# Default pin memory setting for vision models (disabled to prevent RAM accumulation)
VISION_DEFAULT_PIN_MEMORY: Final[bool] = False

# Memory monitor logging interval (steps)
MEMORY_MONITOR_LOG_INTERVAL: Final[int] = 10

# Maximum memory utilization for model saving
DEFAULT_MAX_MEMORY_UTILIZATION: Final[float] = 0.75

# Default shard size for model saving
DEFAULT_MAX_SHARD_SIZE: Final[str] = "5GB"


# =============================================================================
# Selective Loss Defaults
# =============================================================================

# Default selective loss level
DEFAULT_SELECTIVE_LOSS_LEVEL: Final[str] = "conservative"

# Default masking strategy
DEFAULT_SELECTIVE_LOSS_STRATEGY: Final[str] = "epoch_based"

# Default masking start epoch (0.0 = immediate)
DEFAULT_SELECTIVE_LOSS_START_EPOCH: Final[float] = 0.0

# Alternating strategy defaults
DEFAULT_MASK_EVERY_N_STEPS: Final[int] = 100
DEFAULT_MASK_FOR_N_STEPS: Final[int] = 50

# Weighted strategy default structural weight
DEFAULT_STRUCTURAL_WEIGHT: Final[float] = 0.1


# =============================================================================
# Evaluation Defaults
# =============================================================================

# Default evaluation strategy
DEFAULT_EVAL_STRATEGY: Final[str] = "steps"

# Default metric for best model selection
DEFAULT_METRIC_FOR_BEST_MODEL: Final[str] = "eval_loss"


# =============================================================================
# Carbon Tracking
# =============================================================================

# Maximum retry attempts for carbon tracking initialization
CARBON_TRACKING_MAX_RETRIES: Final[int] = 3

# Delay between retry attempts (seconds)
CARBON_TRACKING_RETRY_DELAY: Final[float] = 1.0


# =============================================================================
# Model Loading
# =============================================================================

# Maximum retry attempts for model loading from HuggingFace Hub
MODEL_LOADING_MAX_RETRIES: Final[int] = 3

# Delay between retry attempts (seconds)
MODEL_LOADING_RETRY_DELAY: Final[float] = 2.0

# Timeout for model loading operations (seconds)
MODEL_LOADING_TIMEOUT: Final[int] = 600  # 10 minutes


# =============================================================================
# Random Seeds
# =============================================================================

# Default random seed for reproducibility
DEFAULT_RANDOM_SEED: Final[int] = 42
