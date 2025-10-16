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

# Suppress non-critical warnings
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Import unsloth first for optimal performance
from unsloth import FastLanguageModel

# Then import other ML libraries
from datasets import Dataset, load_dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from trl import SFTTrainer
from transformers import TrainingArguments

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
    ) -> None:
        """Prepare model for LoRA fine-tuning.

        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: Modules to apply LoRA to (None for auto)
            use_rslora: Whether to use rank-stabilized LoRA
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
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
            use_rslora=use_rslora,
            loftq_config=None,
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

        console.print(f"[green]✓[/green] Loaded {len(dataset)} examples")
        return dataset

    def load_dataset_from_hub(self, dataset_name: str, split: str = "train") -> Dataset:
        """Load dataset from HuggingFace Hub.

        Args:
            dataset_name: Dataset identifier on HuggingFace Hub
            split: Dataset split to load

        Returns:
            Loaded dataset
        """
        console.print(f"[cyan]Loading dataset from Hub: {dataset_name}[/cyan]")
        
        # Get HuggingFace token from environment for private datasets
        hf_token = os.getenv('HF_TOKEN')
        
        dataset = load_dataset(dataset_name, split=split, token=hf_token)
        console.print(f"[green]✓[/green] Loaded {len(dataset)} examples")
        return dataset

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
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 10,
        max_steps: int = -1,
        logging_steps: int = 10,
        save_steps: int = 100,
        optim: str = "adamw_8bit",
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
            optim: Optimizer to use
            callbacks: Optional list of TrainerCallback instances
            eval_dataset: Optional validation dataset for evaluation
            eval_steps: Optional number of steps between evaluations (defaults to save_steps)
        """
        console.print("[cyan]Starting training...[/cyan]")
        
        # Set evaluation strategy if validation dataset provided
        evaluation_strategy = "steps" if eval_dataset is not None else "no"
        eval_steps_value = eval_steps if eval_steps is not None else save_steps

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
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            save_steps=save_steps,
            save_total_limit=3,
            report_to="none",  # Disable wandb/tensorboard for now
            evaluation_strategy=evaluation_strategy,
            eval_steps=eval_steps_value if eval_dataset else None,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            callbacks=callbacks if callbacks else [],
        )

        # Train
        trainer.train()
        console.print("[green]✓[/green] Training completed")

        # Save final model
        console.print(f"[cyan]Saving model to: {output_dir}[/cyan]")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        console.print("[green]✓[/green] Model saved successfully")

    def save_model(
        self,
        output_dir: str,
        save_method: str = "merged_16bit",
    ) -> None:
        """Save the trained model.

        Args:
            output_dir: Directory to save the model
            save_method: How to save the model:
                - 'merged_16bit': Merge LoRA and save in 16-bit
                - 'merged_4bit': Merge LoRA and save in 4-bit
                - 'lora': Save only LoRA adapters
        """
        console.print(f"[cyan]Saving model with method: {save_method}[/cyan]")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if save_method == "lora":
            # Save only LoRA adapters
            self.model.save_pretrained(str(output_path))
            self.tokenizer.save_pretrained(str(output_path))
        elif save_method == "merged_16bit":
            # Merge and save in 16-bit
            self.model.save_pretrained_merged(
                str(output_path),
                self.tokenizer,
                save_method="merged_16bit",
            )
        elif save_method == "merged_4bit":
            # Merge and save in 4-bit
            self.model.save_pretrained_merged(
                str(output_path),
                self.tokenizer,
                save_method="merged_4bit",
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
