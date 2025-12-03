"""Base mixin for Transformers training backends.

This module provides shared functionality between TransformersTextTrainer and
TransformersVisionTrainer to reduce code duplication.
"""

import time
from pathlib import Path
from typing import Any, Literal, cast

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import TrainingArguments

from model_garden.carbon import CarbonTracker
from model_garden.training.utils import (
    MemoryMonitorCallback,
    detect_model_dtype,
    get_training_precision_config,
)
from model_garden.utils.console import console
from model_garden.utils.hf_cache import get_hf_token


class TransformersTrainerMixin:
    """Mixin providing shared functionality for Transformers-based trainers.

    This mixin contains methods that are identical or nearly identical between
    TransformersTextTrainer and TransformersVisionTrainer.
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

    def _get_hf_token(self) -> str | None:
        """Get HuggingFace token from environment."""
        return get_hf_token()

    def _get_torch_dtype(self) -> torch.dtype:
        """Get appropriate torch dtype based on hardware support."""
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def _get_quantization_config(self) -> Any:
        """Get quantization configuration if needed."""
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._get_torch_dtype(),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        return None

    def _get_precision_description(self) -> str:
        """Get human-readable precision description."""
        if self.load_in_8bit:
            return "8-bit (balanced quality/memory)"
        elif self.load_in_4bit:
            return "4-bit (memory efficient)"
        else:
            return "16-bit (full quality)"

    def _configure_lora(
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

        This is the shared implementation used by both text and vision trainers.
        """
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
        self.model = get_peft_model(self.model, peft_config)  # type: ignore

        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing and use_gradient_checkpointing != "false":
            self.model.enable_input_require_grads()  # type: ignore
            if hasattr(self.model.base_model, "gradient_checkpointing_enable"):
                self.model.base_model.gradient_checkpointing_enable()  # type: ignore

        console.print("[green]✓[/green] LoRA adapters configured")

    def _load_dataset_from_local_file(self, dataset_path: str) -> Dataset:
        """Load dataset from a local file (shared implementation)."""
        console.print(f"[cyan]Loading dataset from: {dataset_path}[/cyan]")

        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        # Determine file format
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            dataset = load_dataset("json", data_files=str(path), split="train")
        elif suffix == ".json":
            dataset = load_dataset("json", data_files=str(path), split="train")
        elif suffix == ".csv":
            dataset = load_dataset("csv", data_files=str(path), split="train")
        elif suffix == ".parquet":
            dataset = load_dataset("parquet", data_files=str(path), split="train")
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. Use .json, .jsonl, .csv, or .parquet"
            )

        dataset_len = len(dataset)  # type: ignore
        console.print(f"[green]✓[/green] Loaded {dataset_len} examples")
        return cast(Dataset, dataset)

    def _load_dataset_from_hf_hub(
        self, dataset_name: str, split: str = "train", **kwargs
    ) -> Dataset:
        """Load dataset from HuggingFace Hub (shared implementation)."""
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

        dataset_len = len(dataset)  # type: ignore
        console.print(f"[green]✓[/green] Loaded {dataset_len} examples")
        return cast(Dataset, dataset)

    def _setup_carbon_tracking(
        self, output_dir: str, job_id: str | None, job_type: str = "training"
    ) -> CarbonTracker | None:
        """Set up carbon tracking for training."""
        if job_id is None:
            job_id = f"{job_type}-{int(time.time())}"

        try:
            carbon_tracker = CarbonTracker(
                job_id=job_id,
                job_type=job_type,
                output_dir=Path(output_dir) / ".." / "logs" / job_id,
                model_name=Path(output_dir).name,  # Use output dir name as model name
                base_model=self.base_model,
            )
            carbon_tracker.start()
            return carbon_tracker
        except Exception as e:
            console.print(f"[yellow]⚠️  Failed to start carbon tracking: {e}[/yellow]")
            return None

    def _stop_carbon_tracking(self, carbon_tracker: CarbonTracker | None) -> None:
        """Stop carbon tracking and report emissions."""
        if carbon_tracker is not None:
            try:
                emissions_data = carbon_tracker.stop()
                if emissions_data:
                    # Handle both old and new emission data formats
                    emissions = emissions_data.get("emissions_kg_co2") or emissions_data.get(
                        "emissions", 0
                    )
                    console.print(f"[green]🌍 Carbon emissions: {emissions:.6f} kg CO2[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️  Failed to stop carbon tracking: {e}[/yellow]")

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
    ) -> TrainingArguments:
        """Create training arguments (shared implementation)."""
        # Set evaluation strategy
        final_eval_strategy = eval_strategy if eval_dataset is not None else "no"
        eval_steps_value = eval_steps if eval_steps is not None else save_steps
        final_load_best = load_best_model_at_end and eval_dataset is not None
        final_metric = metric_for_best_model if eval_dataset is not None else None

        # Detect model dtype and set precision
        model_dtype = detect_model_dtype(self.model, self.load_in_4bit, self.load_in_8bit)
        precision_config = get_training_precision_config(
            self.model, self.load_in_4bit, self.load_in_8bit
        )

        console.print(f"[cyan]🔍 Detected model dtype: {model_dtype}[/cyan]")
        console.print(
            f"[cyan]📊 Training precision: {'bf16' if precision_config['bf16'] else 'fp16'}[/cyan]"
        )

        return TrainingArguments(  # type: ignore
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
            seed=42,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            report_to="none",
            do_eval=eval_dataset is not None,
            eval_strategy=final_eval_strategy,
            eval_steps=eval_steps_value if eval_dataset else None,
            load_best_model_at_end=final_load_best,
            metric_for_best_model=final_metric,
            remove_unused_columns=remove_unused_columns,
        )

    def _get_callbacks(self, extra_callbacks: list | None = None) -> list[Any]:
        """Get training callbacks including memory monitor."""
        memory_monitor = MemoryMonitorCallback()
        all_callbacks: list[Any] = [memory_monitor]
        if extra_callbacks:
            all_callbacks.extend(extra_callbacks)
        console.print("[cyan]💡 Memory monitoring enabled[/cyan]")
        return all_callbacks

    def _save_model_internal(
        self,
        output_dir: str,
        save_method: str = "merged_16bit",
        max_shard_size: str = "5GB",
    ) -> None:
        """Save the trained model (shared implementation)."""
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
                merged_model = self.model.merge_and_unload()  # type: ignore
                merged_model.save_pretrained(str(output_path), max_shard_size=max_shard_size)
            else:
                self.model.save_pretrained(str(output_path), max_shard_size=max_shard_size)
            saver.save_pretrained(str(output_path))
        else:
            raise ValueError(f"Unknown save method: {save_method}")

        console.print(f"[green]✓[/green] Model saved to {output_path}")
