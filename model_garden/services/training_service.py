"""Training service - backend-agnostic training orchestration.

This module provides the TrainingService class and TrainingRequest dataclass
that serve as the single entry point for all training operations.

Both the CLI and API use this service, ensuring:
- No duplicated business logic
- Consistent behavior across interfaces
- Single place to implement features like quality mode, validation, etc.

Example:
    >>> from model_garden.services import TrainingService, TrainingRequest
    >>> from model_garden.training.config import TrainingConfig, LoRAConfig
    >>>
    >>> request = TrainingRequest(
    ...     name="my-model",
    ...     base_model="unsloth/tinyllama-bnb-4bit",
    ...     dataset_path="./data/train.jsonl",
    ...     output_dir="./models/my-model",
    ...     training_config=TrainingConfig(num_epochs=3, learning_rate=2e-4),
    ...     lora_config=LoRAConfig(r=16, lora_alpha=16),
    ... )
    >>>
    >>> service = TrainingService()
    >>> result = service.train(request)
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from model_garden.training.config import (
    LoRAConfig,
    TrainingConfig,
    VisionLoRAConfig,
    VisionTrainingConfig,
)
from model_garden.training.pipeline import TrainingResult


@dataclass
class TrainingRequest:
    """Unified training request - the single source of truth for training parameters.

    This dataclass consolidates all training parameters in one place. Both the CLI
    and API convert their inputs to a TrainingRequest before calling TrainingService.

    Attributes:
        name: Human-readable name for this training job
        base_model: HuggingFace model ID or local path
        dataset_path: Path to dataset file or HuggingFace dataset ID
        output_dir: Directory to save the trained model
        validation_dataset_path: Optional path to validation dataset
        is_vision: Whether this is a vision-language model
        from_hub: Load dataset from HuggingFace Hub
        validation_from_hub: Load validation dataset from HuggingFace Hub
        training_config: Training hyperparameters (TrainingConfig or VisionTrainingConfig)
        lora_config: LoRA adapter configuration
        quality_mode: Enable quality-optimized settings (16-bit, better optimizer)
        load_in_4bit: Load model in 4-bit quantization
        load_in_8bit: Load model in 8-bit quantization
        save_method: How to save model ('lora', 'merged_16bit', 'merged_4bit')
        backend: Training backend ('unsloth', 'transformers', or None for auto)
        job_id: Optional job ID for tracking (used by API)
        enable_carbon_tracking: Whether to track carbon emissions
        callbacks: Optional list of training callbacks
        early_stopping_enabled: Enable early stopping
        early_stopping_patience: Evaluations before early stop
        early_stopping_threshold: Minimum improvement threshold
        warning_callback: Optional callback for warnings (used by API/UI)

    Example:
        >>> request = TrainingRequest(
        ...     name="my-model",
        ...     base_model="unsloth/tinyllama-bnb-4bit",
        ...     dataset_path="./data/train.jsonl",
        ...     output_dir="./models/my-model",
        ...     quality_mode=True,  # Auto-applies 16-bit, better optimizer, etc.
        ... )
    """

    # Required fields
    name: str
    base_model: str
    dataset_path: str
    output_dir: str

    # Dataset options
    validation_dataset_path: str | None = None
    from_hub: bool = False
    validation_from_hub: bool = False

    # Model type
    is_vision: bool = False

    # Configurations (use existing config dataclasses)
    training_config: TrainingConfig | VisionTrainingConfig | None = None
    lora_config: LoRAConfig | VisionLoRAConfig | None = None

    # Quality/precision settings
    quality_mode: bool = False
    load_in_4bit: bool = True
    load_in_8bit: bool = False

    # Save settings
    save_method: Literal["lora", "merged_16bit", "merged_4bit"] = "merged_16bit"

    # Backend selection
    backend: str | None = None

    # Job tracking (for API usage)
    job_id: str | None = None
    enable_carbon_tracking: bool = True

    # Callbacks
    callbacks: list[Any] | None = None
    warning_callback: Callable[[str], None] | None = None

    # Early stopping
    early_stopping_enabled: bool = False
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0

    # Dataset field mappings (for text models)
    instruction_field: str = "instruction"
    input_field: str = "input"
    output_field: str = "output"

    # Dataset field mappings (for vision models)
    text_field: str = "text"
    image_field: str = "image"

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        # Ensure output_dir is set in training config
        if self.training_config is not None:
            self.training_config.output_dir = self.output_dir

        # Create default configs if not provided
        if self.training_config is None:
            if self.is_vision:
                self.training_config = VisionTrainingConfig(output_dir=self.output_dir)
            else:
                self.training_config = TrainingConfig(output_dir=self.output_dir)

        if self.lora_config is None:
            if self.is_vision:
                self.lora_config = VisionLoRAConfig()
            else:
                self.lora_config = LoRAConfig()

        # Validate precision settings
        if self.load_in_8bit and self.load_in_4bit:
            # 8-bit takes priority
            self.load_in_4bit = False

    def apply_quality_mode(self) -> TrainingRequest:
        """Apply quality mode transformations.

        Quality mode optimizes for training quality over memory efficiency:
        - Uses 16-bit precision (full quality)
        - Uses standard gradient checkpointing (better than unsloth variant)
        - Uses adamw_torch optimizer (better than 8-bit)
        - Enables RSLoRA for high ranks (r >= 32)

        Returns:
            Self with quality mode transformations applied

        Note:
            This is implemented as a method on TrainingRequest so the logic
            exists in ONE place, used by both CLI and API.
        """
        if not self.quality_mode:
            return self

        # Create a copy to avoid mutating the original
        result = copy.deepcopy(self)

        # 16-bit precision
        result.load_in_4bit = False
        result.load_in_8bit = False

        # Update LoRA config
        if result.lora_config is not None:
            # Standard gradient checkpointing (better quality than unsloth variant)
            if result.lora_config.use_gradient_checkpointing == "unsloth":
                result.lora_config.use_gradient_checkpointing = True

            # RSLoRA for high ranks
            if result.lora_config.r >= 32:
                result.lora_config.use_rslora = True

        # Update training config
        if result.training_config is not None:
            # Better optimizer
            if result.training_config.optim == "adamw_8bit":
                result.training_config.optim = "adamw_torch"

        return result

    def get_precision_description(self) -> str:
        """Get human-readable description of precision settings."""
        if self.load_in_8bit:
            return "8-bit (balanced quality/memory)"
        elif self.load_in_4bit:
            return "4-bit (memory efficient)"
        else:
            return "16-bit (full quality)"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization (e.g., job queue).

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "name": self.name,
            "base_model": self.base_model,
            "dataset_path": self.dataset_path,
            "output_dir": self.output_dir,
            "validation_dataset_path": self.validation_dataset_path,
            "from_hub": self.from_hub,
            "validation_from_hub": self.validation_from_hub,
            "is_vision": self.is_vision,
            "hyperparameters": self.training_config.to_dict() if self.training_config else {},
            "lora_config": self.lora_config.to_dict() if self.lora_config else {},
            "quality_mode": self.quality_mode,
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "save_method": self.save_method,
            "backend": self.backend,
            "job_id": self.job_id,
            "enable_carbon_tracking": self.enable_carbon_tracking,
            "early_stopping_enabled": self.early_stopping_enabled,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_threshold": self.early_stopping_threshold,
            "instruction_field": self.instruction_field,
            "input_field": self.input_field,
            "output_field": self.output_field,
            "text_field": self.text_field,
            "image_field": self.image_field,
            # Selective loss settings (from VisionTrainingConfig)
            "selective_loss": getattr(self.training_config, "selective_loss", False),
            "selective_loss_level": getattr(
                self.training_config, "selective_loss_level", "conservative"
            ),
            "selective_loss_schema_keys": getattr(
                self.training_config, "selective_loss_schema_keys", None
            ),
            "selective_loss_masking_strategy": getattr(
                self.training_config, "selective_loss_masking_strategy", "epoch_based"
            ),
            "selective_loss_masking_start_epoch": getattr(
                self.training_config, "selective_loss_masking_start_epoch", 0.0
            ),
            "selective_loss_mask_every_n_steps": getattr(
                self.training_config, "selective_loss_mask_every_n_steps", 100
            ),
            "selective_loss_mask_for_n_steps": getattr(
                self.training_config, "selective_loss_mask_for_n_steps", 50
            ),
            "selective_loss_structural_weight": getattr(
                self.training_config, "selective_loss_structural_weight", 0.1
            ),
            "selective_loss_verbose": getattr(
                self.training_config, "selective_loss_verbose", False
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingRequest:
        """Create a TrainingRequest from a dictionary (e.g., from job queue).

        Args:
            data: Dictionary with training parameters

        Returns:
            TrainingRequest instance
        """
        is_vision = data.get("is_vision", False)

        # Reconstruct training config
        hyperparams = data.get("hyperparameters", {})
        if is_vision:
            training_config = VisionTrainingConfig(
                output_dir=data.get("output_dir", "./output"),
                num_epochs=hyperparams.get("num_train_epochs", hyperparams.get("num_epochs", 3)),
                batch_size=hyperparams.get(
                    "per_device_train_batch_size", hyperparams.get("batch_size", 1)
                ),
                gradient_accumulation_steps=hyperparams.get("gradient_accumulation_steps", 8),
                learning_rate=hyperparams.get("learning_rate", 2e-5),
                warmup_steps=hyperparams.get("warmup_steps", 10),
                max_steps=hyperparams.get("max_steps", -1),
                logging_steps=hyperparams.get("logging_steps", 10),
                save_steps=hyperparams.get("save_steps", 100),
                eval_steps=hyperparams.get("eval_steps"),
                optim=hyperparams.get("optim", "adamw_8bit"),
                weight_decay=hyperparams.get("weight_decay", 0.01),
                lr_scheduler_type=hyperparams.get("lr_scheduler_type", "cosine"),
                max_grad_norm=hyperparams.get("max_grad_norm", 1.0),
                # Selective loss settings
                selective_loss=data.get("selective_loss", False),
                selective_loss_level=data.get("selective_loss_level", "conservative"),
                selective_loss_schema_keys=data.get("selective_loss_schema_keys"),
                selective_loss_masking_strategy=data.get(
                    "selective_loss_masking_strategy", "epoch_based"
                ),
                selective_loss_masking_start_epoch=data.get(
                    "selective_loss_masking_start_epoch", 0.0
                ),
                selective_loss_mask_every_n_steps=data.get(
                    "selective_loss_mask_every_n_steps", 100
                ),
                selective_loss_mask_for_n_steps=data.get("selective_loss_mask_for_n_steps", 50),
                selective_loss_structural_weight=data.get("selective_loss_structural_weight", 0.1),
                selective_loss_verbose=data.get("selective_loss_verbose", False),
            )
        else:
            training_config = TrainingConfig(
                output_dir=data.get("output_dir", "./output"),
                num_epochs=hyperparams.get("num_train_epochs", hyperparams.get("num_epochs", 3)),
                batch_size=hyperparams.get(
                    "per_device_train_batch_size", hyperparams.get("batch_size", 2)
                ),
                gradient_accumulation_steps=hyperparams.get("gradient_accumulation_steps", 4),
                learning_rate=hyperparams.get("learning_rate", 2e-4),
                warmup_steps=hyperparams.get("warmup_steps", 10),
                max_steps=hyperparams.get("max_steps", -1),
                logging_steps=hyperparams.get("logging_steps", 10),
                save_steps=hyperparams.get("save_steps", 100),
                eval_steps=hyperparams.get("eval_steps"),
                optim=hyperparams.get("optim", "adamw_8bit"),
                weight_decay=hyperparams.get("weight_decay", 0.01),
                lr_scheduler_type=hyperparams.get("lr_scheduler_type", "linear"),
                max_grad_norm=hyperparams.get("max_grad_norm", 1.0),
            )

        # Reconstruct LoRA config
        lora_data = data.get("lora_config", {})
        if is_vision:
            lora_config = VisionLoRAConfig(
                r=lora_data.get("r", 16),
                lora_alpha=lora_data.get("lora_alpha", 16),
                lora_dropout=lora_data.get("lora_dropout", 0.0),
                bias=lora_data.get("lora_bias", lora_data.get("bias", "none")),
                use_rslora=lora_data.get("use_rslora", False),
                use_gradient_checkpointing=lora_data.get("use_gradient_checkpointing", "unsloth"),
                random_state=lora_data.get("random_state", 42),
                finetune_vision_layers=lora_data.get("finetune_vision_layers", True),
                finetune_language_layers=lora_data.get("finetune_language_layers", True),
                finetune_attention_modules=lora_data.get("finetune_attention_modules", True),
                finetune_mlp_modules=lora_data.get("finetune_mlp_modules", True),
            )
        else:
            lora_config = LoRAConfig(
                r=lora_data.get("r", 16),
                lora_alpha=lora_data.get("lora_alpha", 16),
                lora_dropout=lora_data.get("lora_dropout", 0.0),
                bias=lora_data.get("lora_bias", lora_data.get("bias", "none")),
                use_rslora=lora_data.get("use_rslora", False),
                use_gradient_checkpointing=lora_data.get("use_gradient_checkpointing", "unsloth"),
                random_state=lora_data.get("random_state", 42),
            )

        return cls(
            name=data.get("name", ""),
            base_model=data.get("base_model", ""),
            dataset_path=data.get("dataset_path", ""),
            output_dir=data.get("output_dir", "./output"),
            validation_dataset_path=data.get("validation_dataset_path"),
            from_hub=data.get("from_hub", False),
            validation_from_hub=data.get("validation_from_hub", False),
            is_vision=is_vision,
            training_config=training_config,
            lora_config=lora_config,
            quality_mode=data.get("quality_mode", False),
            load_in_4bit=data.get("load_in_4bit", True),
            load_in_8bit=data.get("load_in_8bit", False),
            save_method=data.get("save_method", "merged_16bit"),
            backend=data.get("backend"),
            job_id=data.get("job_id"),
            enable_carbon_tracking=data.get("enable_carbon_tracking", True),
            early_stopping_enabled=data.get("early_stopping_enabled", False),
            early_stopping_patience=data.get("early_stopping_patience", 3),
            early_stopping_threshold=data.get("early_stopping_threshold", 0.0),
            instruction_field=data.get("instruction_field", "instruction"),
            input_field=data.get("input_field", "input"),
            output_field=data.get("output_field", "output"),
            text_field=data.get("text_field", "text"),
            image_field=data.get("image_field", "image"),
        )

    @classmethod
    def from_api_request(cls, api_request: Any) -> TrainingRequest:
        """Create a TrainingRequest from an API TrainingJobRequest.

        Args:
            api_request: Pydantic model from API

        Returns:
            TrainingRequest instance
        """
        # Extract hyperparameters
        hyperparams = api_request.hyperparameters or {}
        lora_data = api_request.lora_config or {}

        is_vision = api_request.is_vision

        # Build training config
        if is_vision:
            training_config = VisionTrainingConfig(
                output_dir=api_request.output_dir,
                num_epochs=hyperparams.get("num_epochs", 3),
                batch_size=hyperparams.get("batch_size", 1),
                gradient_accumulation_steps=hyperparams.get("gradient_accumulation_steps", 8),
                learning_rate=hyperparams.get("learning_rate", 2e-5),
                warmup_steps=hyperparams.get("warmup_steps", 10),
                max_steps=hyperparams.get("max_steps", -1),
                logging_steps=hyperparams.get("logging_steps", 10),
                save_steps=hyperparams.get("save_steps", 100),
                eval_steps=hyperparams.get("eval_steps"),
                optim=hyperparams.get("optim", "adamw_8bit"),
                weight_decay=hyperparams.get("weight_decay", 0.01),
                lr_scheduler_type=hyperparams.get("lr_scheduler_type", "cosine"),
                # Selective loss
                selective_loss=api_request.selective_loss,
                selective_loss_level=api_request.selective_loss_level,
                selective_loss_schema_keys=api_request.selective_loss_schema_keys,
                selective_loss_masking_strategy=api_request.selective_loss_masking_strategy,
                selective_loss_masking_start_epoch=api_request.selective_loss_masking_start_epoch,
                selective_loss_mask_every_n_steps=api_request.selective_loss_mask_every_n_steps,
                selective_loss_mask_for_n_steps=api_request.selective_loss_mask_for_n_steps,
                selective_loss_structural_weight=api_request.selective_loss_structural_weight,
                selective_loss_verbose=api_request.selective_loss_verbose,
            )
            lora_config = VisionLoRAConfig(
                r=lora_data.get("r", 16),
                lora_alpha=lora_data.get("lora_alpha", 16),
                lora_dropout=lora_data.get("lora_dropout", 0.0),
                bias=lora_data.get("lora_bias", lora_data.get("bias", "none")),
                use_rslora=lora_data.get("use_rslora", False),
                use_gradient_checkpointing=lora_data.get("use_gradient_checkpointing", "unsloth"),
                finetune_vision_layers=lora_data.get("finetune_vision_layers", True),
                finetune_language_layers=lora_data.get("finetune_language_layers", True),
                finetune_attention_modules=lora_data.get("finetune_attention_modules", True),
                finetune_mlp_modules=lora_data.get("finetune_mlp_modules", True),
            )
        else:
            training_config = TrainingConfig(
                output_dir=api_request.output_dir,
                num_epochs=hyperparams.get("num_epochs", 3),
                batch_size=hyperparams.get("batch_size", 2),
                gradient_accumulation_steps=hyperparams.get("gradient_accumulation_steps", 4),
                learning_rate=hyperparams.get("learning_rate", 2e-4),
                warmup_steps=hyperparams.get("warmup_steps", 10),
                max_steps=hyperparams.get("max_steps", -1),
                logging_steps=hyperparams.get("logging_steps", 10),
                save_steps=hyperparams.get("save_steps", 100),
                eval_steps=hyperparams.get("eval_steps"),
                optim=hyperparams.get("optim", "adamw_8bit"),
                weight_decay=hyperparams.get("weight_decay", 0.01),
                lr_scheduler_type=hyperparams.get("lr_scheduler_type", "linear"),
            )
            lora_config = LoRAConfig(
                r=lora_data.get("r", 16),
                lora_alpha=lora_data.get("lora_alpha", 16),
                lora_dropout=lora_data.get("lora_dropout", 0.0),
                bias=lora_data.get("lora_bias", lora_data.get("bias", "none")),
                use_rslora=lora_data.get("use_rslora", False),
                use_gradient_checkpointing=lora_data.get("use_gradient_checkpointing", "unsloth"),
            )

        return cls(
            name=api_request.name,
            base_model=api_request.base_model,
            dataset_path=api_request.dataset_path,
            output_dir=api_request.output_dir,
            validation_dataset_path=api_request.validation_dataset_path,
            from_hub=api_request.from_hub,
            validation_from_hub=api_request.validation_from_hub,
            is_vision=is_vision,
            training_config=training_config,
            lora_config=lora_config,
            quality_mode=api_request.quality_mode,
            load_in_4bit=not (api_request.load_in_16bit or api_request.load_in_8bit),
            load_in_8bit=api_request.load_in_8bit,
            save_method=api_request.save_method,
            backend=api_request.backend,
            early_stopping_enabled=api_request.early_stopping_enabled,
            early_stopping_patience=api_request.early_stopping_patience,
            early_stopping_threshold=api_request.early_stopping_threshold,
            instruction_field=hyperparams.get("instruction_field", "instruction"),
            input_field=hyperparams.get("input_field", "input"),
            output_field=hyperparams.get("output_field", "output"),
            text_field=hyperparams.get("text_field", "text"),
            image_field=hyperparams.get("image_field", "image"),
        )


class TrainingService:
    """Backend-agnostic training service.

    This service encapsulates all training logic and serves as the single
    entry point for both CLI and API. It handles:
    - Quality mode transformations
    - Backend selection
    - Model loading and preparation
    - Dataset loading and formatting
    - Training execution
    - Model saving
    - Resource cleanup

    Example:
        >>> service = TrainingService()
        >>> request = TrainingRequest(
        ...     name="my-model",
        ...     base_model="unsloth/tinyllama-bnb-4bit",
        ...     dataset_path="./data/train.jsonl",
        ...     output_dir="./models/my-model"
        ... )
        >>> result = service.train(request)
        >>> print(f"Success: {result.success}")
    """

    def __init__(self):
        """Initialize the training service."""
        self._trainer = None
        self._train_dataset = None
        self._eval_dataset = None

    def train(
        self,
        request: TrainingRequest,
        callbacks: list[Any] | None = None,
        progress_callback: Any = None,
    ) -> TrainingResult:
        """Execute a training job.

        This is the main entry point for training. It handles:
        1. Applying quality mode transformations
        2. Creating the appropriate trainer (text or vision)
        3. Loading and preparing the model
        4. Loading and formatting datasets
        5. Running training with callbacks
        6. Saving the trained model
        7. Cleaning up resources

        Args:
            request: TrainingRequest with all parameters
            callbacks: Optional list of additional callbacks
            progress_callback: Optional callback for progress updates

        Returns:
            TrainingResult with outcome and metrics

        Raises:
            RuntimeError: If training fails
        """
        import time

        from model_garden.utils.console import console

        start_time = time.time()

        # Apply quality mode transformations (single implementation!)
        request = request.apply_quality_mode()

        model_type = "vision" if request.is_vision else "text"
        console.print(f"\n[bold cyan]🌱 Model Garden - {model_type.title()} Training[/bold cyan]\n")
        console.print(f"[cyan]Base model: {request.base_model}[/cyan]")
        console.print(f"[cyan]Precision: {request.get_precision_description()}[/cyan]")

        if request.quality_mode:
            console.print("[yellow]🎯 Quality mode enabled[/yellow]")
            console.print("[yellow]⚠️  Warning: Uses ~4x more VRAM than default[/yellow]\n")

        try:
            if request.is_vision:
                result = self._train_vision(request, callbacks, progress_callback)
            else:
                result = self._train_text(request, callbacks, progress_callback)

            result.training_time_seconds = time.time() - start_time

            if result.success:
                console.print("\n[bold green]✨ Training completed successfully![/bold green]")
                console.print(f"[green]Model saved to: {request.output_dir}[/green]\n")

            return result

        except Exception as e:
            console.print(f"\n[bold red]❌ Training failed: {e}[/bold red]\n")
            import traceback

            traceback.print_exc()

            return TrainingResult(
                success=False,
                output_dir=request.output_dir,
                model_type=model_type,
                base_model=request.base_model,
                training_time_seconds=time.time() - start_time,
                error=str(e),
            )
        finally:
            self._cleanup()

    def _train_text(
        self,
        request: TrainingRequest,
        callbacks: list[Any] | None,
        progress_callback: Any,
    ) -> TrainingResult:
        """Execute text model training."""
        from model_garden.training import create_text_trainer

        # Create trainer
        self._trainer = create_text_trainer(
            base_model=request.base_model,
            max_seq_length=getattr(request.training_config, "max_seq_length", 2048)
            if hasattr(request.training_config, "max_seq_length")
            else request.to_dict().get("hyperparameters", {}).get("max_seq_length", 2048),
            load_in_4bit=request.load_in_4bit,
            load_in_8bit=request.load_in_8bit,
            backend=request.backend,
        )

        # Load model
        self._trainer.load_model()

        # Prepare for training with LoRA
        lora_params = request.lora_config.to_dict() if request.lora_config else {}
        self._trainer.prepare_for_training(**lora_params)

        # Load dataset
        if request.from_hub:
            self._train_dataset = self._trainer.load_dataset_from_hub(
                request.dataset_path, split="train"
            )
        else:
            self._train_dataset = self._trainer.load_dataset_from_file(request.dataset_path)

        # Format dataset
        self._train_dataset = self._trainer.format_dataset(
            self._train_dataset,
            instruction_field=request.instruction_field,
            input_field=request.input_field,
            output_field=request.output_field,
        )

        # Load validation dataset if provided
        if request.validation_dataset_path:
            if request.validation_from_hub:
                self._eval_dataset = self._trainer.load_dataset_from_hub(
                    request.validation_dataset_path, split="validation"
                )
            else:
                self._eval_dataset = self._trainer.load_dataset_from_file(
                    request.validation_dataset_path
                )
            self._eval_dataset = self._trainer.format_dataset(
                self._eval_dataset,
                instruction_field=request.instruction_field,
                input_field=request.input_field,
                output_field=request.output_field,
            )

        # Build callbacks list
        all_callbacks = list(callbacks or [])
        if progress_callback:
            all_callbacks.append(progress_callback)

        if request.early_stopping_enabled:
            from model_garden.training import EarlyStoppingCallback

            all_callbacks.append(
                EarlyStoppingCallback(
                    patience=request.early_stopping_patience,
                    threshold=request.early_stopping_threshold,
                )
            )

        # Train
        self._trainer.train(
            dataset=self._train_dataset,
            config=request.training_config,
            eval_dataset=self._eval_dataset,
            callbacks=all_callbacks if all_callbacks else None,
        )

        # Save model
        if request.save_method != "lora":
            self._trainer.save_model(request.output_dir, save_method=request.save_method)

        return TrainingResult(
            success=True,
            output_dir=request.output_dir,
            model_type="text",
            base_model=request.base_model,
        )

    def _train_vision(
        self,
        request: TrainingRequest,
        callbacks: list[Any] | None,
        progress_callback: Any,
    ) -> TrainingResult:
        """Execute vision model training."""
        from model_garden.training import create_vision_trainer

        # Create trainer
        self._trainer = create_vision_trainer(
            base_model=request.base_model,
            max_seq_length=getattr(request.training_config, "max_seq_length", 16384)
            if hasattr(request.training_config, "max_seq_length")
            else request.to_dict().get("hyperparameters", {}).get("max_seq_length", 16384),
            load_in_4bit=request.load_in_4bit,
            load_in_8bit=request.load_in_8bit,
            backend=request.backend,
        )

        # Set warning callback if provided
        if request.warning_callback:
            self._trainer.warning_callback = request.warning_callback

        # Load model
        self._trainer.load_model()

        # Prepare for training with LoRA
        lora_params = request.lora_config.to_dict() if request.lora_config else {}
        self._trainer.prepare_for_training(**lora_params)

        # Load dataset
        self._train_dataset = self._trainer.load_dataset(
            dataset_path=request.dataset_path,
            from_hub=request.from_hub,
            split="train",
        )

        # Format dataset
        self._train_dataset = self._trainer.format_dataset(
            self._train_dataset,
            text_field=request.text_field,
            image_field=request.image_field,
        )

        # Load validation dataset if provided
        if request.validation_dataset_path:
            self._eval_dataset = self._trainer.load_dataset(
                dataset_path=request.validation_dataset_path,
                from_hub=request.validation_from_hub,
                split="validation",
            )
            self._eval_dataset = self._trainer.format_dataset(
                self._eval_dataset,
                text_field=request.text_field,
                image_field=request.image_field,
            )

        # Build callbacks list
        all_callbacks = list(callbacks or [])
        if progress_callback:
            all_callbacks.append(progress_callback)

        if request.early_stopping_enabled:
            from model_garden.training import EarlyStoppingCallback

            all_callbacks.append(
                EarlyStoppingCallback(
                    patience=request.early_stopping_patience,
                    threshold=request.early_stopping_threshold,
                )
            )

        # Train
        self._trainer.train(
            dataset=self._train_dataset,
            config=request.training_config,
            eval_dataset=self._eval_dataset,
            callbacks=all_callbacks if all_callbacks else None,
        )

        # Save model
        self._trainer.save_model(request.output_dir, save_method=request.save_method)

        return TrainingResult(
            success=True,
            output_dir=request.output_dir,
            model_type="vision",
            base_model=request.base_model,
        )

    def _cleanup(self):
        """Clean up training resources."""
        import gc

        import torch

        # Clean up trainer resources
        if self._trainer is not None:
            try:
                if hasattr(self._trainer, "model") and self._trainer.model is not None:
                    try:
                        self._trainer.model.to("cpu")
                    except Exception:
                        pass
                    self._trainer.model = None

                if hasattr(self._trainer, "tokenizer"):
                    self._trainer.tokenizer = None

                if hasattr(self._trainer, "processor"):
                    self._trainer.processor = None
            except Exception:
                pass

            self._trainer = None

        # Clean up datasets
        self._train_dataset = None
        self._eval_dataset = None

        # Force garbage collection
        for _ in range(5):
            gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            for device_id in range(torch.cuda.device_count()):
                with torch.cuda.device(device_id):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

        # Final GC passes
        for _ in range(3):
            gc.collect()
