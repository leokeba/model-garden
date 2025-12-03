"""Training configuration dataclasses.

This module provides dataclasses for training configuration, reducing the parameter
sprawl in train() methods and providing a single source of truth for defaults.
"""

from dataclasses import dataclass
from typing import Any, Literal


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

    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list[str] | None = None
    use_rslora: bool = False
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"
    use_gradient_checkpointing: str | bool = "unsloth"
    random_state: int = 42
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
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    max_steps: int = -1
    logging_steps: int = 10
    save_steps: int = 100
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    eval_strategy: str = "steps"
    eval_steps: int | None = None
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    save_total_limit: int = 3

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
class VisionTrainingConfig(TrainingConfig):
    """Extended training config for vision-language models.

    Vision models have different defaults and additional options for
    selective loss masking during structured output training.

    Attributes:
        batch_size: Smaller default (1) for vision models due to memory.
        gradient_accumulation_steps: Larger default (8) to compensate.
        learning_rate: Lower default (2e-5) for vision models.
        lr_scheduler_type: Cosine scheduler works better for vision.
        dataloader_pin_memory: Disabled by default to prevent RAM accumulation.
        selective_loss: Enable selective loss masking for structured outputs.
        selective_loss_level: Masking level ("conservative", "moderate", "aggressive").
        selective_loss_schema_keys: Schema keys to mask (auto-detected if None).
        selective_loss_masking_strategy: Strategy ("epoch_based", "alternating", "weighted").
        selective_loss_masking_start_epoch: Delay masking until this epoch.
        selective_loss_mask_every_n_steps: Cycle length for alternating strategy.
        selective_loss_mask_for_n_steps: Steps with masking ON per cycle.
        selective_loss_structural_weight: Weight for structural tokens (weighted strategy).
        selective_loss_verbose: Print masking statistics during training.

    Example:
        >>> config = VisionTrainingConfig(
        ...     output_dir="./models/vision-model",
        ...     selective_loss=True,
        ...     selective_loss_level="aggressive"
        ... )
    """

    # Override defaults for vision models
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    dataloader_pin_memory: bool = False

    # Selective loss options
    selective_loss: bool = False
    selective_loss_level: str = "conservative"
    selective_loss_schema_keys: list[str] | None = None
    selective_loss_masking_strategy: str = "epoch_based"
    selective_loss_masking_start_epoch: float = 0.0
    selective_loss_mask_every_n_steps: int = 100
    selective_loss_mask_for_n_steps: int = 50
    selective_loss_structural_weight: float = 0.1
    selective_loss_verbose: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for passing to trainer methods."""
        base = super().to_dict()
        base.update(
            {
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
    max_seq_length: int = 2048
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

    max_seq_length: int = 16384
