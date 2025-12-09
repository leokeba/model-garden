"""Pydantic models for training-related API endpoints."""

from pydantic import BaseModel


class TrainingJobRequest(BaseModel):
    """Request to create a training job."""

    name: str
    base_model: str
    dataset_path: str
    validation_dataset_path: str | None = None  # Optional validation dataset
    output_dir: str
    hyperparameters: dict | None = None
    lora_config: dict | None = None
    from_hub: bool = False
    validation_from_hub: bool = False  # Separate flag for validation dataset
    is_vision: bool = False  # Flag for vision-language models
    model_type: str | None = None  # 'text' or 'vision'
    save_method: str = "merged_16bit"  # How to save: 'lora', 'merged_16bit', 'merged_4bit'
    backend: str = "unsloth"  # Training backend to use
    selective_loss: bool = False  # Enable selective loss for structured outputs
    selective_loss_level: str = "conservative"  # Level: conservative, moderate, aggressive
    selective_loss_schema_keys: list[str] | None = None  # Schema keys to mask
    selective_loss_masking_strategy: str = (
        "epoch_based"  # Strategy: epoch_based, alternating, or weighted
    )
    selective_loss_masking_start_epoch: float = 0.0  # [epoch_based] Delay masking until this epoch
    selective_loss_mask_every_n_steps: int = 100  # [alternating] Cycle length in steps
    selective_loss_mask_for_n_steps: int = 50  # [alternating] Steps with masking ON per cycle
    selective_loss_structural_weight: float = (
        0.1  # [weighted] Weight for structural tokens (0.0-1.0)
    )
    selective_loss_verbose: bool = False  # Print masking statistics
    # Early stopping
    early_stopping_enabled: bool = False  # Enable early stopping
    early_stopping_patience: int = 3  # Number of evals with no improvement before stopping
    early_stopping_threshold: float = 0.0  # Minimum improvement to count
    # Quality settings
    quality_mode: bool = False  # Enable quality-optimized settings (16-bit, better optimizer, etc.)
    load_in_16bit: bool = False  # Load model in 16-bit precision (better quality, 4x more memory)
    load_in_8bit: bool = False  # Load model in 8-bit precision (balanced quality/memory)
    # Note: load_in_4bit is derived server-side as the default when neither 16-bit nor 8-bit is set


class TrainingJobInfo(BaseModel):
    """Training job information."""

    id: str
    name: str
    status: str
    base_model: str
    dataset_path: str
    validation_dataset_path: str | None = None
    output_dir: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    progress: dict | None = None
    error_message: str | None = None
    hyperparameters: dict | None = None
    lora_config: dict | None = None
    from_hub: bool | None = False
    validation_from_hub: bool | None = False
    is_vision: bool | None = False
    model_type: str | None = None
    current_step: int | None = None
    total_steps: int | None = None
    current_epoch: float | None = None
    save_method: str | None = "merged_16bit"
    backend: str | None = "unsloth"
    metrics: dict | None = None  # Training and validation metrics history
    # Selective loss settings
    selective_loss: bool | None = False
    selective_loss_level: str | None = "conservative"
    selective_loss_schema_keys: list[str] | None = None
    selective_loss_masking_strategy: str | None = "epoch_based"
    selective_loss_masking_start_epoch: float | None = 0.0
    selective_loss_mask_every_n_steps: int | None = 100
    selective_loss_mask_for_n_steps: int | None = 50
    selective_loss_structural_weight: float | None = 0.1
    selective_loss_verbose: bool | None = False
    # Quality settings
    quality_mode: bool | None = False
    load_in_16bit: bool | None = False
    load_in_8bit: bool | None = False
    load_in_4bit: bool | None = True
    # Early stopping settings
    early_stopping_enabled: bool | None = False
    early_stopping_patience: int | None = 3
    early_stopping_threshold: float | None = 0.0
    # Rerun metadata
    rerun_from: str | None = None
    rerun_from_name: str | None = None
    queue_position: int | None = None
