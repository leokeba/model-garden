"""Training configuration dataclasses.

This module provides dataclasses for training configuration, reducing the parameter
sprawl in train() methods and providing a single source of truth for defaults.

Configuration is organized hierarchically:
- ModelConfig / VisionModelConfig: Model loading settings
- LoRAConfig / VisionLoRAConfig: LoRA adapter settings
- TrainingConfig: Base training hyperparameters
- SelectiveLossConfig: Selective loss masking settings (vision)
- VisionTrainingConfig: Vision training with composed configs

This modular approach allows reusing individual configs and makes the
API cleaner with related settings grouped together.
"""

from dataclasses import dataclass
from typing import Any, Literal

from model_garden.training.constants import (
    DEFAULT_ADAM_BETA1,
    DEFAULT_ADAM_BETA2,
    DEFAULT_ADAM_EPSILON,
    DEFAULT_DATALOADER_NUM_WORKERS,
    DEFAULT_EVAL_STRATEGY,
    DEFAULT_LOGGING_STEPS,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_BIAS,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_MASK_EVERY_N_STEPS,
    DEFAULT_MASK_FOR_N_STEPS,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_OPTIMIZER,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SAVE_STEPS,
    DEFAULT_SAVE_TOTAL_LIMIT,
    DEFAULT_SELECTIVE_LOSS_LEVEL,
    DEFAULT_SELECTIVE_LOSS_STRATEGY,
    DEFAULT_STRUCTURAL_WEIGHT,
    DEFAULT_WARMUP_STEPS,
    DEFAULT_WEIGHT_DECAY,
    TEXT_DEFAULT_BATCH_SIZE,
    TEXT_DEFAULT_GRADIENT_ACCUMULATION,
    TEXT_DEFAULT_LEARNING_RATE,
    TEXT_DEFAULT_LR_SCHEDULER,
    TEXT_DEFAULT_MAX_SEQ_LENGTH,
    VISION_DEFAULT_BATCH_SIZE,
    VISION_DEFAULT_GRADIENT_ACCUMULATION,
    VISION_DEFAULT_LEARNING_RATE,
    VISION_DEFAULT_LR_SCHEDULER,
    VISION_DEFAULT_MAX_SEQ_LENGTH,
)


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning.

    LoRA reduces memory requirements by training small adapter matrices instead
    of full model weights. This config controls the adapter architecture.

    Attributes:
        r: LoRA rank - higher values = more parameters, better quality but slower.
            Typical values: 8, 16, 32, 64. Start with 16 for most tasks.
        lora_alpha: Scaling factor for LoRA weights. Typically equal to r.
            Higher values = stronger adaptation effect.
        lora_dropout: Dropout rate for LoRA layers (0.0 to 0.3).
            Higher values = more regularization, helps prevent overfitting.
        target_modules: Which model modules to apply LoRA to.
            None = auto-detect based on model architecture.
            Common modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        use_rslora: Use Rank-Stabilized LoRA. Better for high ranks (r > 32).
        bias: How to handle bias terms. Options: "none", "all", "lora_only".
        task_type: Type of task. Options: "CAUSAL_LM", "SEQ_2_SEQ_LM".
        use_gradient_checkpointing: Gradient checkpointing mode.
            "unsloth" = most memory efficient (30% less VRAM), minor quality loss.
            True = standard checkpointing, better quality.
            False = no checkpointing, best quality but most memory.
        random_state: Random seed for reproducibility.
        loftq_config: LoftQ quantization config (advanced, None to disable).

    Example:
        >>> config = LoRAConfig(r=32, lora_alpha=32, use_rslora=True)
        >>> trainer.prepare_for_training(**config.to_dict())
    """

    r: int = DEFAULT_LORA_R
    lora_alpha: int = DEFAULT_LORA_ALPHA
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    target_modules: list[str] | None = None
    use_rslora: bool = False
    bias: Literal["none", "all", "lora_only"] = DEFAULT_LORA_BIAS  # type: ignore[assignment]
    task_type: str = "CAUSAL_LM"
    use_gradient_checkpointing: str | bool = "unsloth"
    random_state: int = DEFAULT_RANDOM_SEED
    loftq_config: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for passing to trainer methods."""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "use_rslora": self.use_rslora,
            "lora_bias": self.bias,  # Note: trainer uses lora_bias, not bias
            "task_type": self.task_type,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "random_state": self.random_state,
            "loftq_config": self.loftq_config,
        }


@dataclass
class VisionLoRAConfig(LoRAConfig):
    """Extended LoRA config for vision-language models.

    Adds vision-specific options for selective layer fine-tuning.

    Attributes:
        finetune_vision_layers: Whether to fine-tune vision encoder layers.
        finetune_language_layers: Whether to fine-tune language model layers.
        finetune_attention_modules: Whether to fine-tune attention layers.
        finetune_mlp_modules: Whether to fine-tune MLP layers.

    Example:
        >>> config = VisionLoRAConfig(
        ...     r=16,
        ...     finetune_vision_layers=True,
        ...     finetune_language_layers=True
        ... )
    """

    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for passing to trainer methods."""
        base = super().to_dict()
        base.update(
            {
                "finetune_vision_layers": self.finetune_vision_layers,
                "finetune_language_layers": self.finetune_language_layers,
                "finetune_attention_modules": self.finetune_attention_modules,
                "finetune_mlp_modules": self.finetune_mlp_modules,
            }
        )
        return base


@dataclass
class TrainingConfig:
    """Configuration for model training.

    Consolidates all training hyperparameters into a single config object,
    reducing parameter sprawl in train() methods.

    Attributes:
        output_dir: Directory to save checkpoints and final model.
        num_epochs: Number of training epochs.
        batch_size: Batch size per device.
        gradient_accumulation_steps: Steps to accumulate gradients before update.
            Effective batch size = batch_size * gradient_accumulation_steps.
        learning_rate: Initial learning rate.
        warmup_steps: Number of warmup steps for learning rate scheduler.
        max_steps: Maximum training steps (-1 = use num_epochs).
        logging_steps: Log metrics every N steps.
        save_steps: Save checkpoint every N steps.
        optim: Optimizer to use. Options: "adamw_8bit", "adamw_torch", "adafactor".
        weight_decay: Weight decay for regularization.
        lr_scheduler_type: Learning rate scheduler. Options: "linear", "cosine", "constant".
        max_grad_norm: Maximum gradient norm for clipping.
        adam_beta1: Beta1 parameter for Adam optimizer.
        adam_beta2: Beta2 parameter for Adam optimizer.
        adam_epsilon: Epsilon parameter for Adam optimizer.
        dataloader_num_workers: Number of dataloader workers (0 = main process).
        dataloader_pin_memory: Whether to pin memory in dataloader.
        eval_strategy: Evaluation strategy. Options: "no", "steps", "epoch".
        eval_steps: Evaluate every N steps (if eval_strategy="steps").
        load_best_model_at_end: Load best model at end of training.
        metric_for_best_model: Metric to use for best model selection.
        save_total_limit: Maximum number of checkpoints to keep.

    Example:
        >>> config = TrainingConfig(
        ...     output_dir="./models/my-model",
        ...     num_epochs=3,
        ...     batch_size=4,
        ...     learning_rate=2e-4
        ... )
        >>> trainer.train(dataset, config=config)
    """

    output_dir: str = "./output"
    num_epochs: int = DEFAULT_NUM_EPOCHS
    batch_size: int = TEXT_DEFAULT_BATCH_SIZE
    gradient_accumulation_steps: int = TEXT_DEFAULT_GRADIENT_ACCUMULATION
    learning_rate: float = TEXT_DEFAULT_LEARNING_RATE
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    max_steps: int = -1
    logging_steps: int = DEFAULT_LOGGING_STEPS
    save_steps: int = DEFAULT_SAVE_STEPS
    optim: str = DEFAULT_OPTIMIZER
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    lr_scheduler_type: str = TEXT_DEFAULT_LR_SCHEDULER
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    adam_beta1: float = DEFAULT_ADAM_BETA1
    adam_beta2: float = DEFAULT_ADAM_BETA2
    adam_epsilon: float = DEFAULT_ADAM_EPSILON
    dataloader_num_workers: int = DEFAULT_DATALOADER_NUM_WORKERS
    dataloader_pin_memory: bool = True
    eval_strategy: str = DEFAULT_EVAL_STRATEGY
    eval_steps: int | None = None
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    save_total_limit: int = DEFAULT_SAVE_TOTAL_LIMIT

    # Dataset statistics (for BoAmps reporting)
    dataset_size: int | None = None
    dataset_num_samples: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for passing to trainer methods."""
        return {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_epochs,
            "per_device_train_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "optim": self.optim,
            "weight_decay": self.weight_decay,
            "lr_scheduler_type": self.lr_scheduler_type,
            "max_grad_norm": self.max_grad_norm,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "save_total_limit": self.save_total_limit,
            "dataset_size": self.dataset_size,
            "dataset_num_samples": self.dataset_num_samples,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary (e.g., from job config)."""
        # Map from various key formats to our canonical names
        key_mapping = {
            "num_train_epochs": "num_epochs",
            "per_device_train_batch_size": "batch_size",
            "epochs": "num_epochs",
        }

        normalized = {}
        for key, value in d.items():
            canonical_key = key_mapping.get(key, key)
            if canonical_key in cls.__dataclass_fields__:
                normalized[canonical_key] = value

        return cls(**normalized)


@dataclass
class SelectiveLossConfig:
    """Configuration for selective loss masking.

    Selective loss masking improves structured output quality by reducing
    the loss contribution from structural tokens (JSON syntax, schema keys)
    so the model focuses on learning content values rather than formatting.

    This configuration groups all selective loss related settings, making
    the VisionTrainingConfig cleaner and enabling reuse of selective loss
    settings across different training configurations.

    Attributes:
        enabled: Whether to enable selective loss masking.
        level: Masking aggressiveness level.
            - "conservative": Mask only obvious structural tokens
            - "moderate": Mask structural tokens + common patterns
            - "aggressive": Maximum masking, may affect content learning
        schema_keys: List of JSON schema keys to mask. If None, auto-detected
            from the dataset.
        masking_strategy: How to apply masking over time.
            - "epoch_based": Start masking after masking_start_epoch
            - "alternating": Cycle between masked/unmasked every N steps
            - "weighted": Use structural_weight instead of full masking
        masking_start_epoch: For epoch_based strategy, delay masking until
            this epoch to let the model learn basic structure first.
        mask_every_n_steps: For alternating strategy, the cycle length.
        mask_for_n_steps: For alternating strategy, steps with masking ON
            per cycle.
        structural_weight: For weighted strategy, the loss weight for
            structural tokens (0.0 = ignore, 1.0 = full weight).
        verbose: Print masking statistics during training for debugging.

    Example:
        >>> # Conservative masking starting from epoch 1
        >>> config = SelectiveLossConfig(
        ...     enabled=True,
        ...     level="conservative",
        ...     masking_strategy="epoch_based",
        ...     masking_start_epoch=1.0,
        ... )

        >>> # Aggressive masking with alternating cycles
        >>> config = SelectiveLossConfig(
        ...     enabled=True,
        ...     level="aggressive",
        ...     masking_strategy="alternating",
        ...     mask_every_n_steps=100,
        ...     mask_for_n_steps=80,
        ... )

        >>> # Weighted masking (soft)
        >>> config = SelectiveLossConfig(
        ...     enabled=True,
        ...     masking_strategy="weighted",
        ...     structural_weight=0.1,  # 10% weight for structural tokens
        ... )
    """

    enabled: bool = False
    level: Literal["conservative", "moderate", "aggressive"] = DEFAULT_SELECTIVE_LOSS_LEVEL  # type: ignore[assignment]
    schema_keys: list[str] | None = None
    masking_strategy: Literal["epoch_based", "alternating", "weighted"] = (
        DEFAULT_SELECTIVE_LOSS_STRATEGY  # type: ignore[assignment]
    )
    masking_start_epoch: float = 0.0
    mask_every_n_steps: int = DEFAULT_MASK_EVERY_N_STEPS
    mask_for_n_steps: int = DEFAULT_MASK_FOR_N_STEPS
    structural_weight: float = DEFAULT_STRUCTURAL_WEIGHT
    verbose: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with prefixed keys for VisionTrainingConfig."""
        return {
            "selective_loss": self.enabled,
            "selective_loss_level": self.level,
            "selective_loss_schema_keys": self.schema_keys,
            "selective_loss_masking_strategy": self.masking_strategy,
            "selective_loss_masking_start_epoch": self.masking_start_epoch,
            "selective_loss_mask_every_n_steps": self.mask_every_n_steps,
            "selective_loss_mask_for_n_steps": self.mask_for_n_steps,
            "selective_loss_structural_weight": self.structural_weight,
            "selective_loss_verbose": self.verbose,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SelectiveLossConfig":
        """Create config from dictionary (handles both prefixed and unprefixed keys)."""
        # Handle both "selective_loss_level" and "level" style keys
        prefix = "selective_loss_"
        normalized = {}

        for key, value in d.items():
            # Remove prefix if present
            if key.startswith(prefix) and key != "selective_loss":
                clean_key = key[len(prefix) :]
            elif key == "selective_loss":
                clean_key = "enabled"
            else:
                clean_key = key

            if clean_key in cls.__dataclass_fields__:
                normalized[clean_key] = value

        return cls(**normalized)


@dataclass
class VisionTrainingConfig(TrainingConfig):
    """Extended training config for vision-language models.

    Vision models have different defaults and additional options for
    selective loss masking during structured output training.

    Supports two configuration styles:
    1. Flat style (backwards compatible): Set individual selective_loss_* fields
    2. Composed style (recommended): Pass a SelectiveLossConfig object

    Attributes:
        batch_size: Smaller default (1) for vision models due to memory.
        gradient_accumulation_steps: Larger default (8) to compensate.
        learning_rate: Lower default (2e-5) for vision models.
        lr_scheduler_type: Cosine scheduler works better for vision.
        dataloader_pin_memory: Disabled by default to prevent RAM accumulation.
        lazy_loading: Load images on-demand instead of all at once (saves memory).
        selective_loss_config: Composed config object (preferred).
        selective_loss: Enable selective loss masking (flat style).
        selective_loss_level: Masking level (flat style).
        ... (other selective_loss_* fields for backwards compatibility)

    Example (composed style - recommended):
        >>> sl_config = SelectiveLossConfig(
        ...     enabled=True,
        ...     level="aggressive",
        ...     masking_strategy="epoch_based",
        ...     masking_start_epoch=1.0,
        ... )
        >>> config = VisionTrainingConfig(
        ...     output_dir="./models/vision-model",
        ...     selective_loss_config=sl_config,
        ...     lazy_loading=True
        ... )

    Example (flat style - backwards compatible):
        >>> config = VisionTrainingConfig(
        ...     output_dir="./models/vision-model",
        ...     selective_loss=True,
        ...     selective_loss_level="aggressive",
        ...     lazy_loading=True
        ... )
    """

    # Override defaults for vision models
    batch_size: int = VISION_DEFAULT_BATCH_SIZE
    gradient_accumulation_steps: int = VISION_DEFAULT_GRADIENT_ACCUMULATION
    learning_rate: float = VISION_DEFAULT_LEARNING_RATE
    lr_scheduler_type: str = VISION_DEFAULT_LR_SCHEDULER
    dataloader_pin_memory: bool = False

    # Memory optimization
    lazy_loading: bool = False  # Load images on-demand to prevent memory exhaustion

    # Composed selective loss config (preferred)
    selective_loss_config: SelectiveLossConfig | None = None

    # Flat selective loss options (for backwards compatibility)
    selective_loss: bool = False
    selective_loss_level: str = DEFAULT_SELECTIVE_LOSS_LEVEL
    selective_loss_schema_keys: list[str] | None = None
    selective_loss_masking_strategy: str = DEFAULT_SELECTIVE_LOSS_STRATEGY
    selective_loss_masking_start_epoch: float = 0.0
    selective_loss_mask_every_n_steps: int = DEFAULT_MASK_EVERY_N_STEPS
    selective_loss_mask_for_n_steps: int = DEFAULT_MASK_FOR_N_STEPS
    selective_loss_structural_weight: float = DEFAULT_STRUCTURAL_WEIGHT
    selective_loss_verbose: bool = False

    def __post_init__(self):
        """Sync composed config with flat fields."""
        if self.selective_loss_config is not None:
            # Composed config takes priority - sync flat fields from it
            self.selective_loss = self.selective_loss_config.enabled
            self.selective_loss_level = self.selective_loss_config.level
            self.selective_loss_schema_keys = self.selective_loss_config.schema_keys
            self.selective_loss_masking_strategy = self.selective_loss_config.masking_strategy
            self.selective_loss_masking_start_epoch = self.selective_loss_config.masking_start_epoch
            self.selective_loss_mask_every_n_steps = self.selective_loss_config.mask_every_n_steps
            self.selective_loss_mask_for_n_steps = self.selective_loss_config.mask_for_n_steps
            self.selective_loss_structural_weight = self.selective_loss_config.structural_weight
            self.selective_loss_verbose = self.selective_loss_config.verbose

    def get_selective_loss_config(self) -> SelectiveLossConfig:
        """Get a SelectiveLossConfig from this training config.

        Returns the composed config if set, otherwise creates one from flat fields.
        """
        if self.selective_loss_config is not None:
            return self.selective_loss_config

        return SelectiveLossConfig(
            enabled=self.selective_loss,
            level=self.selective_loss_level,  # type: ignore[arg-type]
            schema_keys=self.selective_loss_schema_keys,
            masking_strategy=self.selective_loss_masking_strategy,  # type: ignore[arg-type]
            masking_start_epoch=self.selective_loss_masking_start_epoch,
            mask_every_n_steps=self.selective_loss_mask_every_n_steps,
            mask_for_n_steps=self.selective_loss_mask_for_n_steps,
            structural_weight=self.selective_loss_structural_weight,
            verbose=self.selective_loss_verbose,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for passing to trainer methods."""
        base = super().to_dict()
        base.update(
            {
                "lazy_loading": self.lazy_loading,
                "selective_loss": self.selective_loss,
                "selective_loss_level": self.selective_loss_level,
                "selective_loss_schema_keys": self.selective_loss_schema_keys,
                "selective_loss_masking_strategy": self.selective_loss_masking_strategy,
                "selective_loss_masking_start_epoch": self.selective_loss_masking_start_epoch,
                "selective_loss_mask_every_n_steps": self.selective_loss_mask_every_n_steps,
                "selective_loss_mask_for_n_steps": self.selective_loss_mask_for_n_steps,
                "selective_loss_structural_weight": self.selective_loss_structural_weight,
                "selective_loss_verbose": self.selective_loss_verbose,
            }
        )
        return base


@dataclass
class ModelConfig:
    """Configuration for model loading.

    Attributes:
        base_model: HuggingFace model identifier or local path.
        max_seq_length: Maximum sequence length.
        load_in_4bit: Load in 4-bit quantization (memory efficient, ~95% quality).
        load_in_8bit: Load in 8-bit quantization (balanced, ~98% quality).
        dtype: Data type (None for auto-detection).

    Note:
        If both load_in_4bit and load_in_8bit are False, uses 16-bit precision.
        8-bit takes priority over 4-bit if both are True.

    Example:
        >>> config = ModelConfig(
        ...     base_model="unsloth/tinyllama-bnb-4bit",
        ...     max_seq_length=2048,
        ...     load_in_4bit=True
        ... )
    """

    base_model: str = ""
    max_seq_length: int = TEXT_DEFAULT_MAX_SEQ_LENGTH
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    dtype: str | None = None

    def __post_init__(self):
        """Validate and normalize config."""
        # 8-bit takes priority over 4-bit
        if self.load_in_8bit:
            self.load_in_4bit = False

    @property
    def precision_description(self) -> str:
        """Get human-readable precision description."""
        if self.load_in_8bit:
            return "8-bit (balanced quality/memory)"
        elif self.load_in_4bit:
            return "4-bit (memory efficient)"
        else:
            return "16-bit (full quality)"


@dataclass
class VisionModelConfig(ModelConfig):
    """Configuration for vision-language model loading.

    Vision models typically need larger sequence lengths to accommodate
    image tokens plus text.

    Attributes:
        max_seq_length: Default 16384 for vision models (images use ~1500+ tokens).

    Example:
        >>> config = VisionModelConfig(
        ...     base_model="Qwen/Qwen2.5-VL-3B-Instruct",
        ...     max_seq_length=16384
        ... )
    """

    max_seq_length: int = VISION_DEFAULT_MAX_SEQ_LENGTH
