"""Core training module using Unsloth for efficient fine-tuning.

Note: You may see non-critical warnings:
- TorchAO compatibility warning: Safe to ignore unless you need C++/CUDA kernels
- PyTorch CUDA allocator deprecation: Will be resolved in future PyTorch versions
"""

import json
import os
import warnings
from pathlib import Path
from typing import Optional, List

# Configure HuggingFace cache from environment before importing HF libraries
from dotenv import load_dotenv
load_dotenv()

HF_HOME = os.getenv('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))
os.environ['HF_HOME'] = HF_HOME
os.environ['TRANSFORMERS_CACHE'] = str(Path(HF_HOME) / 'hub')
os.environ['HF_DATASETS_CACHE'] = str(Path(HF_HOME) / 'datasets')

# Suppress non-critical warnings
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# CRITICAL: Import unsloth BEFORE any other ML libraries (datasets, transformers, trl, peft)
# This ensures Unsloth's PyTorch patches are applied correctly for optimal performance
from unsloth import FastLanguageModel

# Then import other ML libraries AFTER unsloth
from datasets import Dataset, load_dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from trl.trainer.sft_trainer import SFTTrainer
from transformers import TrainingArguments
from typing import cast

# Import carbon tracking
from model_garden.carbon import CarbonTracker

console = Console()


class ModelTrainer:
    """Handles model fine-tuning using Unsloth."""

    def __init__(
        self,
        base_model: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        dtype: Optional[str] = None,
    ):
        """Initialize the trainer.

        Args:
            base_model: HuggingFace model identifier or local path
            max_seq_length: Maximum sequence length
            load_in_4bit: Whether to load model in 4-bit quantization
            dtype: Data type (None for auto-detection)
        """
        self.base_model = base_model
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.dtype = dtype
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """Load the base model with Unsloth optimizations."""
        console.print(f"[cyan]Loading base model: {self.base_model}[/cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(description="Loading model...", total=None)

            # Get HuggingFace token from environment for private models
            hf_token = os.getenv('HF_TOKEN')
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model,
                max_seq_length=self.max_seq_length,
                dtype=self.dtype,
                load_in_4bit=self.load_in_4bit,
                token=hf_token,
            )

        console.print("[green]✓[/green] Model loaded successfully")

    def prepare_for_training(
        self,
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: Optional[list[str]] = None,
        use_rslora: bool = False,
        lora_bias: str = "none",
        task_type: str = "CAUSAL_LM",
        use_gradient_checkpointing: str = "unsloth",
        random_state: int = 42,
        loftq_config: Optional[dict] = None,
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
            use_gradient_checkpointing: Gradient checkpointing mode ("unsloth", True, False)
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
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
            use_rslora=use_rslora,
            loftq_config=loftq_config,
        )

        console.print("[green]✓[/green] LoRA adapters configured")

    def load_dataset_from_file(self, dataset_path: str) -> Dataset:
        """Load dataset from a local file.

        Args:
            dataset_path: Path to dataset file (JSONL, JSON, CSV, etc.)

        Returns:
            Loaded dataset
        """
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
            raise ValueError(f"Unsupported file format: {suffix}")

        # Handle dataset types - cast to Dataset for type safety
        try:
            dataset_len = len(dataset)  # type: ignore
            console.print(f"[green]✓[/green] Loaded {dataset_len} examples")
        except (TypeError, AttributeError):
            console.print("[green]✓[/green] Loaded dataset (streaming)")
        return cast(Dataset, dataset)

    def load_dataset_from_hub(self, dataset_name: str, split: str = "train") -> Dataset:
        """Load dataset from HuggingFace Hub.

        Args:
            dataset_name: Dataset identifier on HuggingFace Hub
                         Can include specific file with '::' separator (e.g., 'user/repo::train.jsonl')
            split: Dataset split to load (ignored if specific file is provided)

        Returns:
            Loaded dataset
        """
        # Get HuggingFace token from environment for private datasets
        hf_token = os.getenv('HF_TOKEN')
        
        # Check if dataset_name includes a specific file
        if "::" in dataset_name:
            repo_name, file_name = dataset_name.split("::", 1)
            console.print(f"[cyan]Loading dataset from Hub: {repo_name} (file: {file_name})[/cyan]")
            
            # Load specific file from the repo
            dataset = load_dataset(repo_name, data_files=file_name, split="train", token=hf_token)
        else:
            console.print(f"[cyan]Loading dataset from Hub: {dataset_name} (split: {split})[/cyan]")
            
            # Load standard split
            dataset = load_dataset(dataset_name, split=split, token=hf_token)
        
        # Handle dataset types - cast to Dataset for type safety
        try:
            dataset_len = len(dataset)  # type: ignore
            console.print(f"[green]✓[/green] Loaded {dataset_len} examples")
        except (TypeError, AttributeError):
            console.print("[green]✓[/green] Loaded dataset (streaming)")
        return cast(Dataset, dataset)

    def format_dataset(
        self,
        dataset: Dataset,
        instruction_field: str = "instruction",
        input_field: str = "input",
        output_field: str = "output",
        prompt_template: Optional[str] = None,
    ) -> Dataset:
        """Format dataset for instruction fine-tuning.

        Args:
            dataset: Input dataset
            instruction_field: Field name for instructions
            input_field: Field name for inputs (optional)
            output_field: Field name for outputs
            prompt_template: Custom prompt template

        Returns:
            Formatted dataset with 'text' field
        """
        console.print("[cyan]Formatting dataset...[/cyan]")

        if prompt_template is None:
            # Default Alpaca-style prompt
            prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

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

    def train(
        self,
        dataset: Dataset,
        output_dir: str,
        job_id: Optional[str] = None,
        enable_carbon_tracking: bool = True,
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
        eval_strategy: str = "steps",
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "eval_loss",
        save_total_limit: int = 3,
        callbacks: Optional[List] = None,
        eval_dataset: Optional[Dataset] = None,
        eval_steps: Optional[int] = None,
    ) -> None:
        """Train the model.

        Args:
            dataset: Training dataset (should have 'text' field)
            output_dir: Directory to save checkpoints
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            max_steps: Maximum training steps (-1 for full epochs)
            logging_steps: Log every N steps
            save_steps: Save checkpoint every N steps
            optim: Optimizer to use (adamw_8bit, adamw_torch, adafactor, etc.)
            weight_decay: Weight decay for regularization
            lr_scheduler_type: Learning rate scheduler (linear, cosine, constant, etc.)
            max_grad_norm: Maximum gradient norm for clipping
            adam_beta1: Beta1 parameter for Adam optimizer
            adam_beta2: Beta2 parameter for Adam optimizer
            adam_epsilon: Epsilon parameter for Adam optimizer
            dataloader_num_workers: Number of dataloader workers
            dataloader_pin_memory: Whether to pin memory in dataloader
            eval_strategy: Evaluation strategy ('no', 'steps', 'epoch')
            load_best_model_at_end: Load best model at end of training
            metric_for_best_model: Metric to use for best model selection
            save_total_limit: Maximum number of checkpoints to keep
            callbacks: Optional list of TrainerCallback instances
            eval_dataset: Optional validation dataset for evaluation
            eval_steps: Optional number of steps between evaluations (defaults to save_steps)
        """
        console.print("[cyan]Starting training...[/cyan]")
        
        # Initialize carbon tracker
        carbon_tracker = None
        emissions_data = None
        if enable_carbon_tracking:
            # Generate job_id if not provided
            if job_id is None:
                import time
                job_id = f"training-{int(time.time())}"
            
            try:
                carbon_tracker = CarbonTracker(
                    job_id=job_id,
                    job_type="training",
                    output_dir=Path(output_dir) / ".." / "logs" / job_id,
                )
                carbon_tracker.start()
            except Exception as e:
                console.print(f"[yellow]⚠️  Failed to start carbon tracking: {e}[/yellow]")
                console.print("[yellow]Continuing training without carbon tracking...[/yellow]")
                carbon_tracker = None
        
        # Set evaluation strategy if validation dataset provided
        final_eval_strategy = eval_strategy if eval_dataset is not None else "no"
        eval_steps_value = eval_steps if eval_steps is not None else save_steps
        
        # Determine if we should load best model at end
        final_load_best = load_best_model_at_end and eval_dataset is not None
        final_metric = metric_for_best_model if eval_dataset is not None else None

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            fp16=not self.load_in_4bit,
            bf16=self.load_in_4bit,
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
            report_to="none",  # Disable wandb/tensorboard for now
            do_eval=eval_dataset is not None,
            eval_strategy=final_eval_strategy,
            eval_steps=eval_steps_value if eval_dataset else None,
            load_best_model_at_end=final_load_best,
            metric_for_best_model=final_metric,
        )

        # Ensure model is loaded
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,  # type: ignore
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            callbacks=callbacks if callbacks else [],
        )

        # Train
        trainer.train()
        console.print("[green]✓[/green] Training completed")
        
        # Stop carbon tracking
        if carbon_tracker is not None:
            try:
                emissions_data = carbon_tracker.stop()
                if emissions_data:
                    console.print(
                        f"[green]🌍 Carbon emissions: {emissions_data['emissions_kg_co2']:.6f} kg CO2[/green]"
                    )
            except Exception as e:
                console.print(f"[yellow]⚠️  Failed to stop carbon tracking: {e}[/yellow]")

        # Save final model
        console.print(f"[cyan]Saving model to: {output_dir}[/cyan]")
        trainer.save_model(output_dir)
        
        # Ensure tokenizer is available
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available.")
        self.tokenizer.save_pretrained(output_dir)
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
            console.print(f"[cyan]Memory settings: max_usage={maximum_memory_usage}, shard_size={max_shard_size}[/cyan]")
            self.model.save_pretrained_merged(
                str(output_path),
                self.tokenizer,
                save_method="merged_16bit",
                maximum_memory_usage=maximum_memory_usage,
                max_shard_size=max_shard_size,
            )
        elif save_method == "merged_4bit":
            # Merge and save in 4-bit
            console.print(f"[cyan]Memory settings: max_usage={maximum_memory_usage}, shard_size={max_shard_size}[/cyan]")
            console.print("[yellow]⚠️  Warning: 4-bit merge may reduce accuracy for GGUF conversion[/yellow]")
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
