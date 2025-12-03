"""Unified training pipeline for Model Garden.

This module provides a single entry point for all training workflows,
automatically handling both text and vision models based on configuration.

Benefits:
- Simplified API: One function to rule them all
- Auto-detection: Determines model type from config or model name
- Consistent interface: Same workflow for text and vision
- Full control: All options available through config objects

Example:
    >>> from model_garden.training.pipeline import train
    >>> from model_garden.training.config import TrainingConfig, VisionTrainingConfig
    >>>
    >>> # Text training
    >>> config = TrainingConfig(output_dir="./models/text-model")
    >>> result = train(
    ...     base_model="unsloth/tinyllama-bnb-4bit",
    ...     dataset_path="./data/dataset.jsonl",
    ...     config=config
    ... )
    >>>
    >>> # Vision training (auto-detected from model name)
    >>> config = VisionTrainingConfig(output_dir="./models/vision-model")
    >>> result = train(
    ...     base_model="Qwen/Qwen2.5-VL-3B-Instruct",
    ...     dataset_path="./data/vision_dataset.jsonl",
    ...     config=config
    ... )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from datasets import Dataset

from model_garden.training.config import (
    LoRAConfig,
    TrainingConfig,
    VisionLoRAConfig,
    VisionTrainingConfig,
)
from model_garden.utils.console import console


@dataclass
class TrainingResult:
    """Result of a training run.

    Attributes:
        success: Whether training completed successfully
        output_dir: Directory where model was saved
        model_type: Type of model trained ("text" or "vision")
        base_model: Base model used
        total_steps: Total training steps completed
        final_loss: Final training loss (if available)
        carbon_emissions_kg: Carbon emissions in kg CO2 (if tracked)
        training_time_seconds: Total training time
        error: Error message if training failed
        metrics: Additional metrics from training
    """

    success: bool = True
    output_dir: str = ""
    model_type: Literal["text", "vision"] = "text"
    base_model: str = ""
    total_steps: int = 0
    final_loss: float | None = None
    carbon_emissions_kg: float | None = None
    training_time_seconds: float = 0.0
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "output_dir": self.output_dir,
            "model_type": self.model_type,
            "base_model": self.base_model,
            "total_steps": self.total_steps,
            "final_loss": self.final_loss,
            "carbon_emissions_kg": self.carbon_emissions_kg,
            "training_time_seconds": self.training_time_seconds,
            "error": self.error,
            "metrics": self.metrics,
        }


def is_vision_model(model_name: str) -> bool:
    """Detect if a model is a vision-language model based on its name.

    Args:
        model_name: HuggingFace model identifier or local path

    Returns:
        True if the model appears to be a vision-language model
    """
    # Common vision model indicators
    vision_indicators = [
        "VL",  # Qwen2.5-VL, InternVL
        "vision",
        "vit",  # Vision Transformer
        "clip",
        "llava",
        "bakllava",
        "idefics",
        "paligemma",
        "fuyu",
        "kosmos",
        "cogvlm",
        "minicpm-v",
        "phi-3-vision",
        "phi-3.5-vision",
    ]

    model_lower = model_name.lower()
    return any(indicator.lower() in model_lower for indicator in vision_indicators)


def train(
    base_model: str,
    dataset_path: str | None = None,
    dataset: Dataset | list[dict] | None = None,
    config: TrainingConfig | VisionTrainingConfig | None = None,
    lora_config: LoRAConfig | VisionLoRAConfig | None = None,
    output_dir: str | None = None,
    # Model loading options
    max_seq_length: int | None = None,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    # Dataset options
    from_hub: bool = False,
    dataset_split: str = "train",
    # Training options
    job_id: str | None = None,
    enable_carbon_tracking: bool = True,
    callbacks: list | None = None,
    eval_dataset: Dataset | list[dict] | None = None,
    eval_dataset_path: str | None = None,
    # Save options
    save_method: str = "merged_16bit",
    save_model: bool = True,
    # Force model type (override auto-detection)
    force_vision: bool | None = None,
    # Backend selection
    backend: str = "unsloth",
) -> TrainingResult:
    """Unified training entry point for text and vision models.

    This function automatically detects the model type (text or vision) and
    uses the appropriate trainer. It provides a simplified interface while
    still allowing full control through config objects.

    Args:
        base_model: HuggingFace model identifier or local path
        dataset_path: Path to dataset file or HuggingFace dataset ID
        dataset: Pre-loaded dataset (alternative to dataset_path)
        config: Training configuration (TrainingConfig or VisionTrainingConfig).
               If None, defaults are created based on model type.
        lora_config: LoRA configuration. If None, uses defaults.
        output_dir: Override output directory from config
        max_seq_length: Override max sequence length (auto-set based on model type)
        load_in_4bit: Load model in 4-bit quantization
        load_in_8bit: Load model in 8-bit quantization
        from_hub: Load dataset from HuggingFace Hub
        dataset_split: Dataset split to use (for Hub datasets)
        job_id: Job identifier for carbon tracking
        enable_carbon_tracking: Whether to track carbon emissions
        callbacks: Additional training callbacks
        eval_dataset: Pre-loaded evaluation dataset
        eval_dataset_path: Path to evaluation dataset
        save_method: How to save model ("lora", "merged_16bit", "merged_4bit")
        save_model: Whether to save the model after training
        force_vision: Force vision training (True) or text training (False).
                     If None, auto-detects from model name.
        backend: Training backend to use ("unsloth" or "transformers")

    Returns:
        TrainingResult with training outcomes and metrics

    Raises:
        ValueError: If neither dataset nor dataset_path is provided
        RuntimeError: If training fails

    Example:
        >>> # Simple text training
        >>> result = train(
        ...     base_model="unsloth/tinyllama-bnb-4bit",
        ...     dataset_path="./data/dataset.jsonl",
        ...     output_dir="./models/my-model"
        ... )
        >>>
        >>> # Vision training with custom config
        >>> config = VisionTrainingConfig(
        ...     num_epochs=5,
        ...     learning_rate=1e-5,
        ...     selective_loss=True
        ... )
        >>> result = train(
        ...     base_model="Qwen/Qwen2.5-VL-3B-Instruct",
        ...     dataset_path="./data/vision.jsonl",
        ...     config=config
        ... )
    """
    import time

    start_time = time.time()

    # Determine model type
    is_vision = force_vision if force_vision is not None else is_vision_model(base_model)
    model_type: Literal["text", "vision"] = "vision" if is_vision else "text"

    console.print(f"[bold cyan]🚀 Starting {model_type} model training[/bold cyan]")
    console.print(f"[cyan]Base model: {base_model}[/cyan]")

    # Validate inputs
    if dataset is None and dataset_path is None:
        return TrainingResult(
            success=False,
            model_type=model_type,
            base_model=base_model,
            error="Either dataset or dataset_path must be provided",
        )

    try:
        # Create default configs if not provided
        if config is None:
            if is_vision:
                config = VisionTrainingConfig()
            else:
                config = TrainingConfig()

        # Override output_dir if provided
        if output_dir is not None:
            config.output_dir = output_dir

        # Ensure output directory exists
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Get the appropriate backend
        from model_garden.training.backends.registry import get_backend

        training_backend = get_backend(backend)

        if is_vision:
            result = _train_vision(
                backend=training_backend,
                base_model=base_model,
                dataset_path=dataset_path,
                dataset=dataset,
                config=config
                if isinstance(config, VisionTrainingConfig)
                else VisionTrainingConfig(**config.to_dict()),
                lora_config=lora_config
                if isinstance(lora_config, VisionLoRAConfig)
                else VisionLoRAConfig(**(lora_config.to_dict() if lora_config else {})),
                max_seq_length=max_seq_length or 16384,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                from_hub=from_hub,
                dataset_split=dataset_split,
                job_id=job_id,
                enable_carbon_tracking=enable_carbon_tracking,
                callbacks=callbacks,
                eval_dataset=eval_dataset,
                eval_dataset_path=eval_dataset_path,
                save_method=save_method,
                save_model_flag=save_model,
            )
        else:
            result = _train_text(
                backend=training_backend,
                base_model=base_model,
                dataset_path=dataset_path,
                dataset=dataset,
                config=config,
                lora_config=lora_config or LoRAConfig(),
                max_seq_length=max_seq_length or 2048,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                from_hub=from_hub,
                dataset_split=dataset_split,
                job_id=job_id,
                enable_carbon_tracking=enable_carbon_tracking,
                callbacks=callbacks,
                eval_dataset=eval_dataset,
                eval_dataset_path=eval_dataset_path,
                save_method=save_method,
                save_model_flag=save_model,
            )

        # Update timing
        result.training_time_seconds = time.time() - start_time

        return result

    except Exception as e:
        console.print(f"[red]❌ Training failed: {e}[/red]")
        return TrainingResult(
            success=False,
            output_dir=config.output_dir if config else "",
            model_type=model_type,
            base_model=base_model,
            training_time_seconds=time.time() - start_time,
            error=str(e),
        )


def _train_text(
    backend: Any,
    base_model: str,
    dataset_path: str | None,
    dataset: Dataset | list[dict] | None,
    config: TrainingConfig,
    lora_config: LoRAConfig,
    max_seq_length: int,
    load_in_4bit: bool,
    load_in_8bit: bool,
    from_hub: bool,
    dataset_split: str,
    job_id: str | None,
    enable_carbon_tracking: bool,
    callbacks: list | None,
    eval_dataset: Dataset | list[dict] | None,
    eval_dataset_path: str | None,
    save_method: str,
    save_model_flag: bool,
) -> TrainingResult:
    """Internal function for text model training."""
    # Create trainer
    trainer = backend.create_text_trainer(
        base_model=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

    # Load model
    trainer.load_model()

    # Prepare for training (configure LoRA)
    trainer.prepare_for_training(**lora_config.to_dict())

    # Load dataset
    if dataset is None:
        if from_hub:
            dataset = trainer.load_dataset_from_hub(dataset_path, split=dataset_split)  # type: ignore
        else:
            dataset = trainer.load_dataset_from_file(dataset_path)  # type: ignore

    # Format dataset
    formatted_dataset = trainer.format_dataset(dataset)

    # Load eval dataset if path provided
    if eval_dataset is None and eval_dataset_path is not None:
        if from_hub:
            eval_dataset = trainer.load_dataset_from_hub(eval_dataset_path, split=dataset_split)
        else:
            eval_dataset = trainer.load_dataset_from_file(eval_dataset_path)
        eval_dataset = trainer.format_dataset(eval_dataset)

    # Train
    trainer.train(
        dataset=formatted_dataset,
        config=config,
        job_id=job_id,
        enable_carbon_tracking=enable_carbon_tracking,
        callbacks=callbacks,
        eval_dataset=eval_dataset,
    )

    # Save model
    if save_model_flag:
        trainer.save_model(
            output_dir=config.output_dir,
            save_method=save_method,
        )

    return TrainingResult(
        success=True,
        output_dir=config.output_dir,
        model_type="text",
        base_model=base_model,
    )


def _train_vision(
    backend: Any,
    base_model: str,
    dataset_path: str | None,
    dataset: Dataset | list[dict] | None,
    config: VisionTrainingConfig,
    lora_config: VisionLoRAConfig,
    max_seq_length: int,
    load_in_4bit: bool,
    load_in_8bit: bool,
    from_hub: bool,
    dataset_split: str,
    job_id: str | None,
    enable_carbon_tracking: bool,
    callbacks: list | None,
    eval_dataset: Dataset | list[dict] | None,
    eval_dataset_path: str | None,
    save_method: str,
    save_model_flag: bool,
) -> TrainingResult:
    """Internal function for vision model training."""
    # Create trainer
    trainer = backend.create_vision_trainer(
        base_model=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

    # Load model
    trainer.load_model()

    # Prepare for training (configure LoRA)
    trainer.prepare_for_training(**lora_config.to_dict())

    # Load dataset
    if dataset is None:
        if from_hub:
            dataset = trainer.load_dataset_from_hub(dataset_path, split=dataset_split)  # type: ignore
        else:
            dataset = trainer.load_dataset_from_file(dataset_path)  # type: ignore

    # Format dataset (with lazy loading if configured)
    formatted_dataset = trainer.format_dataset(
        dataset,
        lazy_loading=config.lazy_loading,
    )

    # Load eval dataset if path provided
    if eval_dataset is None and eval_dataset_path is not None:
        if from_hub:
            eval_ds = trainer.load_dataset_from_hub(eval_dataset_path, split=dataset_split)
        else:
            eval_ds = trainer.load_dataset_from_file(eval_dataset_path)
        eval_dataset = trainer.format_dataset(eval_ds, lazy_loading=config.lazy_loading)

    # Train
    trainer.train(
        dataset=formatted_dataset,
        config=config,
        job_id=job_id,
        enable_carbon_tracking=enable_carbon_tracking,
        callbacks=callbacks,
        eval_dataset=eval_dataset,
    )

    # Save model
    if save_model_flag:
        trainer.save_model(
            output_dir=config.output_dir,
            save_method=save_method,
        )

    return TrainingResult(
        success=True,
        output_dir=config.output_dir,
        model_type="vision",
        base_model=base_model,
    )


# Convenience aliases
def train_text(
    base_model: str,
    dataset_path: str | None = None,
    dataset: Dataset | None = None,
    config: TrainingConfig | None = None,
    **kwargs,
) -> TrainingResult:
    """Train a text-only model.

    Convenience wrapper around train() with force_vision=False.
    See train() for full documentation.
    """
    return train(
        base_model=base_model,
        dataset_path=dataset_path,
        dataset=dataset,
        config=config,
        force_vision=False,
        **kwargs,
    )


def train_vision(
    base_model: str,
    dataset_path: str | None = None,
    dataset: Dataset | list[dict] | None = None,
    config: VisionTrainingConfig | None = None,
    **kwargs,
) -> TrainingResult:
    """Train a vision-language model.

    Convenience wrapper around train() with force_vision=True.
    See train() for full documentation.
    """
    return train(
        base_model=base_model,
        dataset_path=dataset_path,
        dataset=dataset,
        config=config,
        force_vision=True,
        **kwargs,
    )
