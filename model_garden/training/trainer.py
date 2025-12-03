"""Core training module using Unsloth for efficient fine-tuning.

Note: You may see non-critical warnings:
- TorchAO compatibility warning: Safe to ignore unless you need C++/CUDA kernels
- PyTorch CUDA allocator deprecation: Will be resolved in future PyTorch versions
"""

import json
from pathlib import Path
from typing import Any

# Configure HuggingFace cache BEFORE importing HF libraries
from model_garden.utils.hf_cache import configure_hf_cache, configure_pytorch_memory

configure_hf_cache()
configure_pytorch_memory()

# CRITICAL: Import unsloth BEFORE any other ML libraries (datasets, transformers, trl, peft)
# This ensures Unsloth's PyTorch patches are applied correctly for optimal performance

# Then import other ML libraries AFTER unsloth
from datasets import Dataset
from rich.progress import Progress, SpinnerColumn, TextColumn
from trl.trainer.sft_trainer import SFTTrainer
from unsloth import FastLanguageModel

# Import backend base class
from model_garden.training.backends.base import TextTrainer

# Import configuration dataclasses
from model_garden.training.config import TrainingConfig

# Import shared training mixin and utilities
from model_garden.training.mixins import TrainerMixin

# Import centralized console
from model_garden.utils.console import console


class ModelTrainer(TrainerMixin, TextTrainer):
    """Handles model fine-tuning using Unsloth.

    This trainer uses Unsloth's optimized kernels for 2x faster training
    and 60% less memory usage compared to standard HuggingFace training.

    Inherits shared functionality from TrainerMixin:
    - Dataset loading (load_dataset_from_file, load_dataset_from_hub)
    - Carbon tracking (_start_carbon_tracking, _stop_carbon_tracking)
    - Training arguments creation (_create_training_args)
    - Memory management (cleanup_memory, _cleanup_after_training)
    """

    # Processor is not used for text-only models, but defined for mixin compatibility
    processor: Any | None = None

    def __init__(
        self,
        base_model: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: str | None = None,
    ):
        """Initialize the trainer.

        Args:
            base_model: HuggingFace model identifier or local path
            max_seq_length: Maximum sequence length
            load_in_4bit: Whether to load model in 4-bit quantization (memory efficient, ~95% quality)
            load_in_8bit: Whether to load model in 8-bit quantization (balanced, ~98% quality, 2x memory vs 4-bit)
            dtype: Data type (None for auto-detection, used for 16-bit precision when both quantizations are False)

        Note on quantization priority:
            - If load_in_8bit=True: Uses 8-bit quantization (overrides load_in_4bit)
            - If load_in_4bit=True and load_in_8bit=False: Uses 4-bit quantization
            - If both False: Uses 16-bit precision (full quality, 4x memory vs 4-bit)
        """
        self.base_model = base_model
        self.max_seq_length = max_seq_length
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit and not load_in_8bit  # 8-bit takes priority
        self.dtype = dtype
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """Load the base model with Unsloth optimizations.

        Supports 4-bit, 8-bit, and 16-bit (full precision) loading.
        Uses Unsloth's FastLanguageModel for optimized inference and training.
        """
        console.print(f"[cyan]Loading base model: {self.base_model}[/cyan]")
        console.print(f"[cyan]Precision: {self._get_precision_description()}[/cyan]")

        # Get HuggingFace token from environment for private models
        hf_token = self._get_hf_token()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(description="Loading model...", total=None)

            # Unsloth supports both 4-bit and 8-bit quantization
            # Note: For 16-bit, set both load_in_4bit and load_in_8bit to False
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model,
                max_seq_length=self.max_seq_length,
                dtype=self.dtype,
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                token=hf_token,
            )

        console.print("[green]✓[/green] Model loaded successfully")

    def prepare_for_training(
        self,
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: list[str] | None = None,
        use_rslora: bool = False,
        lora_bias: str = "none",
        task_type: str = "CAUSAL_LM",
        use_gradient_checkpointing: str | bool = "unsloth",
        random_state: int = 42,
        loftq_config: dict | None = None,
    ) -> None:
        """Prepare model for LoRA fine-tuning.

        Args:
            r: LoRA rank (higher = more parameters, better quality but slower)
            lora_alpha: LoRA alpha parameter (scaling factor, typically equal to r)
            lora_dropout: LoRA dropout rate (0.0 to 0.3, higher = more regularization)
            target_modules: Modules to apply LoRA to (None for auto-detection)
            use_rslora: Whether to use rank-stabilized LoRA (better for high ranks)
            lora_bias: How to handle bias ("none", "all", "lora_only")
            task_type: Type of task ("CAUSAL_LM", "SEQ_2_SEQ_LM", etc.)
            use_gradient_checkpointing: Gradient checkpointing mode:
                - "unsloth": Most memory efficient (30% less VRAM), minor quality loss
                - True: Standard gradient checkpointing, better quality
                - False: No gradient checkpointing, best quality but most memory
            random_state: Random seed for reproducibility
            loftq_config: LoftQ quantization config (None to disable)
        """
        console.print("[cyan]Configuring LoRA adapters...[/cyan]")

        if target_modules is None:
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            use_gradient_checkpointing=use_gradient_checkpointing,  # type: ignore
            random_state=random_state,
            use_rslora=use_rslora,
            loftq_config=loftq_config,
        )

        console.print("[green]✓[/green] LoRA adapters configured")

    # NOTE: load_dataset_from_file and load_dataset_from_hub are inherited from TrainerMixin

    def format_dataset(
        self,
        dataset: Dataset,
        instruction_field: str = "instruction",
        input_field: str = "input",
        output_field: str = "output",
        prompt_template: str | None = None,
    ) -> Dataset:
        """Format dataset for instruction fine-tuning.

        Delegates to TrainerMixin.format_text_dataset() for the actual formatting.
        This method exists for interface compatibility with TextTrainer base class.

        Args:
            dataset: Input dataset
            instruction_field: Field name for instructions
            input_field: Field name for inputs (optional)
            output_field: Field name for outputs
            prompt_template: Custom prompt template

        Returns:
            Formatted dataset with 'text' field
        """
        return self.format_text_dataset(
            dataset=dataset,
            instruction_field=instruction_field,
            input_field=input_field,
            output_field=output_field,
            prompt_template=prompt_template,
        )

    def train(
        self,
        dataset: Dataset,
        config: TrainingConfig,
        job_id: str | None = None,
        enable_carbon_tracking: bool = True,
        callbacks: list | None = None,
        eval_dataset: Dataset | None = None,
    ) -> None:
        """Train the model.

        Args:
            dataset: Training dataset (should have 'text' field)
            config: Training configuration (hyperparameters, output directory, etc.)
            job_id: Optional job identifier for carbon tracking
            enable_carbon_tracking: Whether to track carbon emissions
            callbacks: Optional list of TrainerCallback instances
            eval_dataset: Optional validation dataset for evaluation

        Example:
            >>> config = TrainingConfig(
            ...     output_dir="./models/my-model",
            ...     num_epochs=3,
            ...     batch_size=4,
            ...     learning_rate=2e-4
            ... )
            >>> trainer.train(dataset, config)
        """
        console.print("[cyan]Starting training...[/cyan]")

        # Ensure model is loaded
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Start carbon tracking (uses mixin helper)
        if enable_carbon_tracking:
            self._start_carbon_tracking(config.output_dir, job_id, "training")

        # Create training arguments using mixin helper
        training_args = self._create_training_args_from_config(config, eval_dataset=eval_dataset)

        # Get callbacks (uses mixin helper)
        all_callbacks = self._get_default_callbacks()
        if callbacks:
            all_callbacks.extend(callbacks)

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,  # type: ignore
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            callbacks=all_callbacks,
        )

        # Train
        trainer.train()
        console.print("[green]✓[/green] Training completed")

        # Stop carbon tracking (uses mixin helper)
        self._stop_carbon_tracking()

        # Save final model
        console.print(f"[cyan]Saving model to: {config.output_dir}[/cyan]")
        trainer.save_model(config.output_dir)

        # Ensure tokenizer is available
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available.")
        self.tokenizer.save_pretrained(config.output_dir)
        console.print("[green]✓[/green] Model saved successfully")

    def save_model(
        self,
        output_dir: str,
        save_method: str = "merged_16bit",
        maximum_memory_usage: float = 0.75,
        max_shard_size: str = "5GB",
    ) -> None:
        """Save the trained model.

        Args:
            output_dir: Directory to save the model
            save_method: How to save the model:
                - 'merged_16bit': Merge LoRA and save in 16-bit
                - 'merged_4bit': Merge LoRA and save in 4-bit (not recommended for GGUF conversion)
                - 'lora': Save only LoRA adapters
            maximum_memory_usage: Maximum RAM usage ratio (0.0-0.95, lower = less RAM, default: 0.75)
                                  Reduce this (e.g., 0.5) if you run out of memory during merge
            max_shard_size: Maximum size per shard file (e.g., "1GB", "2GB", "5GB")
                           Smaller values use less peak memory during save
        """
        console.print(f"[cyan]Saving model with method: {save_method}[/cyan]")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Ensure model and tokenizer are available
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available.")

        if save_method == "lora":
            # Save only LoRA adapters
            self.model.save_pretrained(str(output_path))
            self.tokenizer.save_pretrained(str(output_path))
        elif save_method == "merged_16bit":
            # Merge and save in 16-bit
            console.print(
                f"[cyan]Memory settings: max_usage={maximum_memory_usage}, shard_size={max_shard_size}[/cyan]"
            )
            self.model.save_pretrained_merged(
                str(output_path),
                self.tokenizer,
                save_method="merged_16bit",
                maximum_memory_usage=maximum_memory_usage,
                max_shard_size=max_shard_size,
            )
        elif save_method == "merged_4bit":
            # Merge and save in 4-bit
            console.print(
                f"[cyan]Memory settings: max_usage={maximum_memory_usage}, shard_size={max_shard_size}[/cyan]"
            )
            console.print(
                "[yellow]⚠️  Warning: 4-bit merge may reduce accuracy for GGUF conversion[/yellow]"
            )
            self.model.save_pretrained_merged(
                str(output_path),
                self.tokenizer,
                save_method="merged_4bit_forced",
                maximum_memory_usage=maximum_memory_usage,
                max_shard_size=max_shard_size,
            )
        else:
            raise ValueError(f"Unknown save method: {save_method}")

        console.print(f"[green]✓[/green] Model saved to {output_path}")


def create_sample_dataset(output_path: str, num_examples: int = 100) -> None:
    """Create a sample dataset for testing.

    Args:
        output_path: Path to save the dataset
        num_examples: Number of examples to generate
    """
    console.print(f"[cyan]Creating sample dataset with {num_examples} examples...[/cyan]")

    examples = []
    for i in range(num_examples):
        examples.append(
            {
                "instruction": f"Sample instruction {i}",
                "input": f"Sample input {i}",
                "output": f"Sample output {i}",
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    console.print(f"[green]✓[/green] Sample dataset created at {output_path}")


def create_text_trainer(
    base_model: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    dtype: str | None = None,
    backend: str = "unsloth",
) -> TextTrainer:
    """Create a text trainer using the specified backend.

    This is a convenience function that creates a text trainer through the backend system.
    It allows for backend selection while maintaining backward compatibility.

    Args:
        base_model: HuggingFace model identifier or local path
        max_seq_length: Maximum sequence length
        load_in_4bit: Whether to load model in 4-bit quantization
        load_in_8bit: Whether to load model in 8-bit quantization
        dtype: Data type (None for auto-detection)
        backend: Backend to use ('unsloth', etc.)

    Returns:
        A text trainer instance

    Example:
        >>> trainer = create_text_trainer("unsloth/tinyllama-bnb-4bit", backend="unsloth")
        >>> trainer.load_model()
    """
    from model_garden.training.backends import get_backend

    backend_instance = get_backend(backend)
    return backend_instance.create_text_trainer(
        base_model=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        dtype=dtype,
    )
