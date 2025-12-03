"""Shared mixin for training backends.

This module provides a mixin class that consolidates shared logic used by all
trainers (Unsloth, Transformers, etc.), eliminating code duplication.

The mixin provides:
- Dataset loading (local files and HuggingFace Hub)
- Carbon tracking setup and teardown
- Training arguments creation
- LoRA configuration
- Memory cleanup utilities
- Precision detection and configuration
"""

import gc
import time
from pathlib import Path
from typing import Any, Literal, cast

import torch
from datasets import Dataset, load_dataset
from transformers import TrainingArguments

# Re-export MemoryMonitorCallback from callbacks package for backwards compatibility
from model_garden.training.callbacks.memory import MemoryMonitorCallback
from model_garden.utils.console import console
from model_garden.utils.hf_cache import get_hf_token


def detect_model_dtype(
    model: Any,
    load_in_4bit: bool,
    load_in_8bit: bool,
) -> torch.dtype:
    """Detect the actual dtype of a model's parameters.

    This function reliably detects the precision of model parameters by checking
    the actual parameter tensors, not just model attributes which can be misleading.

    Why this matters:
    - Many modern models (e.g., Qwen2.5-VL) use bfloat16 natively
    - The model's .dtype attribute may return float32 (default) even when parameters are bfloat16
    - We need to match training precision (fp16/bf16) to actual model precision

    Detection strategy:
    1. For quantized models (4-bit/8-bit): Always return bfloat16 (standard practice)
    2. For 16-bit models: Check actual parameter dtype (most reliable)
    3. Fallback: Check model attributes if parameters not accessible

    Args:
        model: The model to check (can be wrapped in PeftModel, etc.)
        load_in_4bit: Whether model was loaded with 4-bit quantization
        load_in_8bit: Whether model was loaded with 8-bit quantization

    Returns:
        torch.dtype: The detected dtype (e.g., torch.bfloat16, torch.float16, torch.float32)
    """
    # For quantized models, always use bfloat16 for training
    if load_in_4bit or load_in_8bit:
        return torch.bfloat16

    if model is None:
        return torch.float32

    # Method 1: Check actual parameter dtypes (MOST RELIABLE)
    try:
        first_param = next(model.parameters())
        if first_param.device.type != "meta":
            return first_param.dtype
    except (StopIteration, AttributeError):
        pass

    # Method 2: For PEFT wrapped models, check the base model's parameters
    if hasattr(model, "base_model"):
        try:
            first_param = next(model.base_model.parameters())
            if first_param.device.type != "meta":
                return first_param.dtype
        except (StopIteration, AttributeError):
            pass

    # Method 3: Check model.dtype attribute
    if hasattr(model, "dtype") and model.dtype is not None:
        return model.dtype

    # Method 4: Check config as last resort
    if hasattr(model, "config") and hasattr(model.config, "torch_dtype"):
        dtype = model.config.torch_dtype
        if dtype is not None and dtype != torch.float32:
            return dtype

    # Method 5: Model-specific defaults for known architectures
    if hasattr(model, "config") and hasattr(model.config, "model_type"):
        model_type = model.config.model_type
        if model_type in ["qwen2_vl", "qwen2_5_vl", "qwen2_audio_vl"]:
            return torch.bfloat16

    # Check base_model for PEFT wrapped models
    if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        if hasattr(model.base_model.config, "model_type"):
            model_type = model.base_model.config.model_type
            if model_type in ["qwen2_vl", "qwen2_5_vl", "qwen2_audio_vl"]:
                return torch.bfloat16

    # Default to float32 as safe fallback
    return torch.float32


def get_training_precision_config(
    model: Any,
    load_in_4bit: bool,
    load_in_8bit: bool,
) -> dict[str, bool]:
    """Get fp16/bf16 configuration for training based on model dtype.

    Args:
        model: The model to check
        load_in_4bit: Whether model was loaded with 4-bit quantization
        load_in_8bit: Whether model was loaded with 8-bit quantization

    Returns:
        dict: Dictionary with 'fp16' and 'bf16' keys set appropriately
    """
    model_dtype = detect_model_dtype(model, load_in_4bit, load_in_8bit)
    is_bfloat16 = model_dtype == torch.bfloat16

    return {
        "fp16": not is_bfloat16,
        "bf16": is_bfloat16,
    }


def cleanup_memory() -> None:
    """Clean up GPU and system memory.

    Performs:
    - Multiple garbage collection passes
    - GPU cache clearing
    - Memory synchronization
    """
    # Multiple passes of garbage collection to ensure all cycles are broken
    for _ in range(3):
        gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Reset peak memory stats to get accurate readings for next operation
        torch.cuda.reset_peak_memory_stats()


class TrainerMixin:
    """Mixin providing shared functionality for all trainers.

    This mixin consolidates common logic used by both Unsloth and Transformers
    trainers, eliminating code duplication while allowing backend-specific
    customization.

    Subclasses should define:
    - self.model: The model being trained
    - self.tokenizer: The tokenizer
    - self.processor: Optional processor (for vision models)
    - self.base_model: Base model identifier
    - self.max_seq_length: Maximum sequence length
    - self.load_in_4bit: Whether 4-bit quantization is used
    - self.load_in_8bit: Whether 8-bit quantization is used
    """

    # Instance attributes to be defined by subclasses
    model: Any
    tokenizer: Any
    processor: Any | None
    base_model: str
    max_seq_length: int
    load_in_4bit: bool
    load_in_8bit: bool
    dtype: Any

    # Carbon tracker instance
    _carbon_tracker: Any | None = None

    def _get_hf_token(self) -> str | None:
        """Get HuggingFace token from environment."""
        return get_hf_token()

    def _get_torch_dtype(self) -> torch.dtype:
        """Get appropriate torch dtype based on hardware support."""
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def _get_precision_description(self) -> str:
        """Get human-readable precision description."""
        if self.load_in_8bit:
            return "8-bit (balanced quality/memory)"
        elif self.load_in_4bit:
            return "4-bit (memory efficient)"
        else:
            return "16-bit (full quality)"

    def _get_quantization_config(self) -> Any:
        """Get BitsAndBytes quantization configuration if needed."""
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._get_torch_dtype(),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        return None

    # =========================================================================
    # Dataset Loading
    # =========================================================================

    def load_dataset_from_file(self, dataset_path: str) -> Dataset:
        """Load dataset from a local file.

        Supports JSONL, JSON, CSV, and Parquet formats.

        Args:
            dataset_path: Path to dataset file

        Returns:
            Loaded dataset

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If file format is not supported
        """
        console.print(f"[cyan]Loading dataset from: {dataset_path}[/cyan]")

        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        # Determine file format
        suffix = path.suffix.lower()
        format_map = {
            ".jsonl": "json",
            ".json": "json",
            ".csv": "csv",
            ".parquet": "parquet",
        }

        if suffix not in format_map:
            raise ValueError(
                f"Unsupported file format: {suffix}. Use .json, .jsonl, .csv, or .parquet"
            )

        dataset = load_dataset(format_map[suffix], data_files=str(path), split="train")

        try:
            dataset_len = len(dataset)  # type: ignore
            console.print(f"[green]✓[/green] Loaded {dataset_len} examples")
        except (TypeError, AttributeError):
            console.print("[green]✓[/green] Loaded dataset (streaming)")

        return cast(Dataset, dataset)

    def load_dataset_from_hub(self, dataset_name: str, split: str = "train", **kwargs) -> Dataset:
        """Load dataset from HuggingFace Hub.

        Supports loading specific files using '::' separator.

        Args:
            dataset_name: Dataset identifier on HuggingFace Hub
                         Can include specific file with '::' separator
                         (e.g., 'user/repo::train.jsonl')
            split: Dataset split to load (ignored if specific file is provided)
            **kwargs: Additional arguments passed to load_dataset

        Returns:
            Loaded dataset
        """
        hf_token = self._get_hf_token()

        # Check if dataset_name includes a specific file
        if "::" in dataset_name:
            repo_name, file_name = dataset_name.split("::", 1)
            console.print(f"[cyan]Loading dataset from Hub: {repo_name} (file: {file_name})[/cyan]")
            dataset = load_dataset(
                repo_name, data_files=file_name, split="train", token=hf_token, **kwargs
            )
        else:
            console.print(f"[cyan]Loading dataset from Hub: {dataset_name} (split: {split})[/cyan]")
            dataset = load_dataset(dataset_name, split=split, token=hf_token, **kwargs)

        try:
            dataset_len = len(dataset)  # type: ignore
            console.print(f"[green]✓[/green] Loaded {dataset_len} examples")
        except (TypeError, AttributeError):
            console.print("[green]✓[/green] Loaded dataset (streaming)")

        return cast(Dataset, dataset)

    # =========================================================================
    # Dataset Formatting
    # =========================================================================

    # Default Alpaca-style prompt template for instruction fine-tuning
    DEFAULT_PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

    def format_text_dataset(
        self,
        dataset: Dataset,
        instruction_field: str = "instruction",
        input_field: str = "input",
        output_field: str = "output",
        prompt_template: str | None = None,
    ) -> Dataset:
        """Format dataset for instruction fine-tuning (text models).

        Applies a prompt template to each example in the dataset, creating a
        'text' field that can be used for training.

        Args:
            dataset: Input dataset with instruction/input/output fields
            instruction_field: Field name for instructions
            input_field: Field name for inputs (optional context)
            output_field: Field name for outputs/responses
            prompt_template: Custom prompt template with {instruction}, {input},
                           {output} placeholders. If None, uses Alpaca-style default.

        Returns:
            Formatted dataset with 'text' field suitable for training
        """
        console.print("[cyan]Formatting dataset...[/cyan]")

        if prompt_template is None:
            prompt_template = self.DEFAULT_PROMPT_TEMPLATE

        def format_example(example):
            instruction = example.get(instruction_field, "")
            input_text = example.get(input_field, "")
            output = example.get(output_field, "")

            text = prompt_template.format(
                instruction=instruction,
                input=input_text,
                output=output,
            )
            return {"text": text}

        formatted_dataset = dataset.map(format_example)
        console.print("[green]✓[/green] Dataset formatted")
        return formatted_dataset

    # =========================================================================
    # Carbon Tracking
    # =========================================================================

    def _start_carbon_tracking(
        self,
        output_dir: str,
        job_id: str | None = None,
        job_type: str = "training",
        max_retries: int = 3,
    ) -> Any | None:
        """Start carbon tracking for training with retry logic.

        Carbon tracking may fail due to network issues (fetching carbon intensity
        data) or hardware detection problems. This method retries with exponential
        backoff to handle transient failures gracefully.

        Args:
            output_dir: Directory for saving logs
            job_id: Optional job identifier (auto-generated if None)
            job_type: Type of job ("training", "vision-training", etc.)
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            CarbonTracker instance or None if tracking failed to start
        """
        from model_garden.carbon import CarbonTracker
        from model_garden.training.constants import (
            RETRY_BASE_DELAY_SECONDS,
            RETRY_EXPONENTIAL_BACKOFF,
            RETRY_MAX_DELAY_SECONDS,
        )

        if job_id is None:
            job_id = f"{job_type}-{int(time.time())}"

        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                carbon_tracker = CarbonTracker(
                    job_id=job_id,
                    job_type=job_type,
                    output_dir=Path(output_dir) / ".." / "logs" / job_id,
                    model_name=Path(output_dir).name,
                    base_model=self.base_model,
                )
                carbon_tracker.start()
                self._carbon_tracker = carbon_tracker
                return carbon_tracker

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Calculate delay with exponential backoff
                    delay = min(
                        RETRY_BASE_DELAY_SECONDS * (RETRY_EXPONENTIAL_BACKOFF**attempt),
                        RETRY_MAX_DELAY_SECONDS,
                    )
                    console.print(
                        f"[yellow]⚠️  Carbon tracking attempt {attempt + 1}/{max_retries} failed: {e}[/yellow]"
                    )
                    console.print(f"[yellow]   Retrying in {delay:.1f}s...[/yellow]")
                    time.sleep(delay)

        # All retries failed
        console.print(
            f"[yellow]⚠️  Failed to start carbon tracking after {max_retries} attempts: {last_error}[/yellow]"
        )
        console.print("[yellow]Continuing training without carbon tracking...[/yellow]")
        return None

    def _stop_carbon_tracking(self) -> dict | None:
        """Stop carbon tracking and return emissions data.

        Returns:
            Emissions data dictionary or None if tracking wasn't active
        """
        if self._carbon_tracker is None:
            return None

        try:
            emissions_data = self._carbon_tracker.stop()
            if emissions_data:
                emissions = emissions_data.get("emissions_kg_co2") or emissions_data.get(
                    "emissions", 0
                )
                console.print(f"[green]🌍 Carbon emissions: {emissions:.6f} kg CO2[/green]")
            self._carbon_tracker = None
            return emissions_data
        except Exception as e:
            console.print(f"[yellow]⚠️  Failed to stop carbon tracking: {e}[/yellow]")
            self._carbon_tracker = None
            return None

    # =========================================================================
    # Training Arguments
    # =========================================================================

    def _create_training_args_from_config(
        self,
        config: Any,
        eval_dataset: Any = None,
        **kwargs,
    ) -> TrainingArguments:
        """Create training arguments from a TrainingConfig object.

        This is a convenience wrapper around _create_training_args that extracts
        values from a TrainingConfig dataclass, reducing boilerplate in train() methods.

        Args:
            config: TrainingConfig or VisionTrainingConfig instance
            eval_dataset: Optional evaluation dataset (for setting eval strategy)
            **kwargs: Additional arguments to override config values

        Returns:
            Configured TrainingArguments instance
        """
        return self._create_training_args(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            optim=config.optim,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            max_grad_norm=config.max_grad_norm,
            adam_beta1=config.adam_beta1,
            adam_beta2=config.adam_beta2,
            adam_epsilon=config.adam_epsilon,
            dataloader_num_workers=config.dataloader_num_workers,
            dataloader_pin_memory=config.dataloader_pin_memory,
            save_total_limit=config.save_total_limit,
            eval_dataset=eval_dataset,
            eval_strategy=config.eval_strategy,
            eval_steps=config.eval_steps,
            load_best_model_at_end=config.load_best_model_at_end,
            metric_for_best_model=config.metric_for_best_model,
            **kwargs,
        )

    def _create_training_args(
        self,
        output_dir: str,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 10,
        max_steps: int = -1,
        logging_steps: int = 10,
        save_steps: int = 100,
        optim: str = "adamw_8bit",
        weight_decay: float = 0.01,
        lr_scheduler_type: str = "linear",
        max_grad_norm: float = 1.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        dataloader_num_workers: int = 0,
        dataloader_pin_memory: bool = True,
        save_total_limit: int = 3,
        eval_dataset: Any = None,
        eval_strategy: str = "steps",
        eval_steps: int | None = None,
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "eval_loss",
        remove_unused_columns: bool = True,
        report_to: str = "none",
        seed: int = 42,
        **kwargs,
    ) -> TrainingArguments:
        """Create HuggingFace TrainingArguments with proper defaults.

        This method handles precision detection and evaluation strategy setup.

        Args:
            output_dir: Directory to save checkpoints
            ... (see TrainingConfig for parameter descriptions)
            **kwargs: Additional arguments passed to TrainingArguments

        Returns:
            Configured TrainingArguments instance
        """
        # Set evaluation strategy
        final_eval_strategy = eval_strategy if eval_dataset is not None else "no"
        eval_steps_value = eval_steps if eval_steps is not None else save_steps
        final_load_best = load_best_model_at_end and eval_dataset is not None
        final_metric = metric_for_best_model if eval_dataset is not None else None

        # Detect model dtype and set precision
        precision_config = get_training_precision_config(
            self.model, self.load_in_4bit, self.load_in_8bit
        )
        model_dtype = detect_model_dtype(self.model, self.load_in_4bit, self.load_in_8bit)

        console.print(f"[cyan]🔍 Detected model dtype: {model_dtype}[/cyan]")
        console.print(
            f"[cyan]📊 Training precision: {'bf16' if precision_config['bf16'] else 'fp16'}[/cyan]"
        )

        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            fp16=precision_config["fp16"],
            bf16=precision_config["bf16"],
            logging_steps=logging_steps,
            optim=optim,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            max_grad_norm=max_grad_norm,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_pin_memory=dataloader_pin_memory,
            seed=seed,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            report_to=report_to,
            do_eval=eval_dataset is not None,
            eval_strategy=final_eval_strategy,
            eval_steps=eval_steps_value if eval_dataset else None,
            load_best_model_at_end=final_load_best,
            metric_for_best_model=final_metric,
            remove_unused_columns=remove_unused_columns,
            **kwargs,
        )

    # =========================================================================
    # LoRA Configuration
    # =========================================================================

    def _configure_lora_peft(
        self,
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: list[str] | None = None,
        use_rslora: bool = False,
        lora_bias: str = "none",
        task_type: str = "CAUSAL_LM",
        use_gradient_checkpointing: str | bool = "unsloth",
    ) -> None:
        """Configure LoRA adapters using PEFT.

        This is a fallback implementation for when Unsloth's get_peft_model
        is not available. Unsloth trainers should override this with their
        optimized version.

        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha (scaling factor)
            lora_dropout: Dropout rate for LoRA layers
            target_modules: Modules to apply LoRA to (None for auto-detect)
            use_rslora: Use rank-stabilized LoRA
            lora_bias: How to handle bias ("none", "all", "lora_only")
            task_type: Task type ("CAUSAL_LM", "SEQ_2_SEQ_LM")
            use_gradient_checkpointing: Gradient checkpointing mode
        """
        from peft import LoraConfig, TaskType, get_peft_model

        console.print("[cyan]Configuring LoRA adapters (PEFT)...[/cyan]")

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

        # Convert task_type string to TaskType enum
        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        }
        peft_task_type = task_type_map.get(task_type, TaskType.CAUSAL_LM)

        # Create PEFT config
        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=cast(Literal["none", "all", "lora_only"], lora_bias),
            task_type=peft_task_type,
            use_rslora=use_rslora,
        )

        # Apply PEFT
        self.model = get_peft_model(self.model, peft_config)

        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing and use_gradient_checkpointing != "false":
            self.model.enable_input_require_grads()
            if hasattr(self.model.base_model, "gradient_checkpointing_enable"):
                self.model.base_model.gradient_checkpointing_enable()

        console.print("[green]✓[/green] LoRA adapters configured")

    # =========================================================================
    # Callbacks
    # =========================================================================

    def _get_default_callbacks(self, extra_callbacks: list | None = None) -> list[Any]:
        """Get training callbacks including memory monitor.

        Args:
            extra_callbacks: Additional callbacks to include

        Returns:
            List of callback instances
        """
        memory_monitor = MemoryMonitorCallback()
        all_callbacks: list[Any] = [memory_monitor]
        if extra_callbacks:
            all_callbacks.extend(extra_callbacks)
        console.print("[cyan]💡 Memory monitoring enabled[/cyan]")
        return all_callbacks

    # =========================================================================
    # Model Saving
    # =========================================================================

    def _save_model_merged(
        self,
        output_dir: str,
        save_method: str = "merged_16bit",
        max_shard_size: str = "5GB",
    ) -> None:
        """Save the trained model by merging LoRA weights.

        This is a fallback implementation using PEFT's merge_and_unload.
        Unsloth trainers should use save_pretrained_merged for better
        memory efficiency.

        Args:
            output_dir: Directory to save the model
            save_method: How to save ("lora", "merged_16bit", "merged_4bit")
            max_shard_size: Maximum size per shard file
        """
        from peft import PeftModel

        console.print(f"[cyan]Saving model with method: {save_method}[/cyan]")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Determine which tokenizer/processor to save
        saver = self.processor if hasattr(self, "processor") and self.processor else self.tokenizer
        if saver is None:
            raise RuntimeError("Tokenizer/processor not available.")

        if save_method == "lora":
            # Save only LoRA adapters
            if isinstance(self.model, PeftModel):
                self.model.save_pretrained(str(output_path))
                saver.save_pretrained(str(output_path))
            else:
                raise ValueError("Model is not a PEFT model, cannot save LoRA adapters")
        elif save_method in ["merged_16bit", "merged_4bit"]:
            # Merge LoRA weights into base model
            console.print("[cyan]Merging LoRA weights...[/cyan]")
            if isinstance(self.model, PeftModel):
                # Type guard: ensure model has merge_and_unload method
                merge_fn = getattr(self.model, "merge_and_unload", None)
                if merge_fn is None or not callable(merge_fn):
                    raise RuntimeError("PeftModel instance missing merge_and_unload method")
                # Call the method directly on self.model to avoid type issues
                merged_model = self.model.merge_and_unload()  # type: ignore[operator]
                merged_model.save_pretrained(str(output_path), max_shard_size=max_shard_size)
            else:
                self.model.save_pretrained(str(output_path), max_shard_size=max_shard_size)
            saver.save_pretrained(str(output_path))
        else:
            raise ValueError(f"Unknown save method: {save_method}")

        console.print(f"[green]✓[/green] Model saved to {output_path}")

    # =========================================================================
    # Memory Cleanup
    # =========================================================================

    def _cleanup_after_training(self) -> None:
        """Clean up resources after training completes.

        Clears model, tokenizer, and processor references to enable
        garbage collection, then runs memory cleanup.
        """
        console.print("[cyan]🧹 Cleaning up training resources...[/cyan]")

        # Clear references
        self.model = None
        self.tokenizer = None
        if hasattr(self, "processor"):
            self.processor = None

        # Run memory cleanup
        cleanup_memory()

        console.print("[green]✓[/green] Cleanup complete")

    def _cleanup_trainer_datasets(self, trainer: Any) -> None:
        """Clear dataset references from trainer to enable garbage collection.

        Vision models keep PIL images in RAM which can accumulate across
        multiple training runs if not properly cleaned up.

        Args:
            trainer: The trainer instance to clean up
        """
        console.print("[cyan]🧹 Clearing dataset references from trainer...[/cyan]")
        try:
            if hasattr(trainer, "train_dataset"):
                trainer.train_dataset = None
            if hasattr(trainer, "eval_dataset"):
                trainer.eval_dataset = None
            if hasattr(trainer, "data_collator"):
                trainer.data_collator = None
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Failed to clear trainer datasets: {e}[/yellow]")
