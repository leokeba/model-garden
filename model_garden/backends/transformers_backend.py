"""Transformers training backend for Model Garden.

This backend provides standard HuggingFace Transformers + PEFT training without Unsloth optimizations.
Use this for maximum compatibility or when Unsloth doesn't support your model.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast, Literal

# Configure HuggingFace cache from environment before importing HF libraries
from dotenv import load_dotenv
load_dotenv()

HF_HOME = os.getenv('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))
os.environ['HF_HOME'] = HF_HOME
os.environ['TRANSFORMERS_CACHE'] = str(Path(HF_HOME) / 'hub')
os.environ['HF_DATASETS_CACHE'] = str(Path(HF_HOME) / 'datasets')

# Configure PyTorch memory allocator
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from model_garden.backends.base import TrainingBackend, TextTrainer, VisionTrainer
from model_garden.carbon import CarbonTracker
from model_garden.training_utils import (
    detect_model_dtype,
    get_training_precision_config,
    MemoryMonitorCallback,
)

console = Console()


class TransformersTextTrainer(TextTrainer):
    """Text trainer using standard HuggingFace Transformers + PEFT."""

    def load_model(self) -> None:
        """Load the base model using Transformers."""
        console.print(f"[cyan]Loading base model: {self.base_model}[/cyan]")
        console.print(f"[cyan]Using HuggingFace Transformers (no Unsloth optimizations)[/cyan]")

        # Determine precision
        if self.load_in_8bit:
            precision = "8-bit (balanced quality/memory)"
        elif self.load_in_4bit:
            precision = "4-bit (memory efficient)"
        else:
            precision = "16-bit (full quality)"
        
        console.print(f"[cyan]Precision: {precision}[/cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(description="Loading model...", total=None)

            # Get HuggingFace token from environment
            hf_token = os.getenv('HF_TOKEN')
            
            # Determine torch dtype
            if not self.load_in_4bit and not self.load_in_8bit:
                # For 16-bit, use bfloat16 or float16 based on availability
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                # Let BitsAndBytes handle dtype for quantized models
                torch_dtype = None
            
            # Build model loading kwargs
            model_kwargs = {
                "pretrained_model_name_or_path": self.base_model,
                "torch_dtype": torch_dtype,
                "device_map": "auto",
                "token": hf_token,
                "trust_remote_code": True,
            }
            
            # Add quantization if requested
            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif self.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                token=hf_token,
                trust_remote_code=True,
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        console.print("[green]✓[/green] Model loaded successfully")

    def prepare_for_training(
        self,
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        use_rslora: bool = False,
        lora_bias: str = "none",
        task_type: str = "CAUSAL_LM",
        use_gradient_checkpointing: Union[str, bool] = "unsloth",
        random_state: int = 42,
        loftq_config: Optional[Dict] = None,
    ) -> None:
        """Prepare model for LoRA fine-tuning using PEFT."""
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

    def load_dataset_from_file(self, dataset_path: str) -> Dataset:
        """Load dataset from a local file."""
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

        dataset_len = len(dataset)  # type: ignore
        console.print(f"[green]✓[/green] Loaded {dataset_len} examples")
        return cast(Dataset, dataset)

    def load_dataset_from_hub(self, dataset_name: str, split: str = "train") -> Dataset:
        """Load dataset from HuggingFace Hub."""
        hf_token = os.getenv('HF_TOKEN')
        
        # Check if dataset_name includes a specific file
        if "::" in dataset_name:
            repo_name, file_name = dataset_name.split("::", 1)
            console.print(f"[cyan]Loading dataset from Hub: {repo_name} (file: {file_name})[/cyan]")
            dataset = load_dataset(repo_name, data_files=file_name, split="train", token=hf_token)
        else:
            console.print(f"[cyan]Loading dataset from Hub: {dataset_name} (split: {split})[/cyan]")
            dataset = load_dataset(dataset_name, split=split, token=hf_token)
        
        dataset_len = len(dataset)  # type: ignore
        console.print(f"[green]✓[/green] Loaded {dataset_len} examples")
        return cast(Dataset, dataset)

    def format_dataset(
        self,
        dataset: Dataset,
        instruction_field: str = "instruction",
        input_field: str = "input",
        output_field: str = "output",
        prompt_template: Optional[str] = None,
    ) -> Dataset:
        """Format dataset for instruction fine-tuning."""
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
        """Train the model using Transformers Trainer."""
        console.print("[cyan]Starting training with Transformers backend...[/cyan]")
        
        # Initialize carbon tracker
        carbon_tracker = None
        if enable_carbon_tracking:
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
                carbon_tracker = None
        
        # Set evaluation strategy
        final_eval_strategy = eval_strategy if eval_dataset is not None else "no"
        eval_steps_value = eval_steps if eval_steps is not None else save_steps
        final_load_best = load_best_model_at_end and eval_dataset is not None
        final_metric = metric_for_best_model if eval_dataset is not None else None

        # Detect model dtype and set precision
        model_dtype = detect_model_dtype(self.model, self.load_in_4bit, self.load_in_8bit)
        precision_config = get_training_precision_config(self.model, self.load_in_4bit, self.load_in_8bit)
        
        console.print(f"[cyan]🔍 Detected model dtype: {model_dtype}[/cyan]")
        console.print(f"[cyan]📊 Training precision: {'bf16' if precision_config['bf16'] else 'fp16'}[/cyan]")

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            fp16=precision_config['fp16'],
            bf16=precision_config['bf16'],
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
        )

        # Add memory monitoring callback
        memory_monitor = MemoryMonitorCallback()
        all_callbacks: List[Any] = [memory_monitor]
        if callbacks:
            all_callbacks.extend(callbacks)
        
        console.print("[cyan]💡 Memory monitoring enabled[/cyan]")
        
        # Tokenize datasets
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
            )
        
        tokenized_train = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        tokenized_eval = None
        if eval_dataset is not None:
            tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)
        
        # Create data collator for language modeling (adds labels automatically)
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            callbacks=all_callbacks,
            data_collator=data_collator,
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
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
        console.print("[green]✓[/green] Model saved successfully")

    def save_model(
        self,
        output_dir: str,
        save_method: str = "merged_16bit",
        maximum_memory_usage: float = 0.75,
        max_shard_size: str = "5GB",
    ) -> None:
        """Save the trained model."""
        console.print(f"[cyan]Saving model with method: {save_method}[/cyan]")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available.")
            
        if save_method == "lora":
            # Save only LoRA adapters
            self.model.save_pretrained(str(output_path))
            self.tokenizer.save_pretrained(str(output_path))
        elif save_method in ["merged_16bit", "merged_4bit"]:
            # Merge LoRA weights into base model
            console.print("[cyan]Merging LoRA weights...[/cyan]")
            merged_model = self.model.merge_and_unload()  # type: ignore
            merged_model.save_pretrained(  # type: ignore
                str(output_path),
                max_shard_size=max_shard_size,
            )
            self.tokenizer.save_pretrained(str(output_path))
        else:
            raise ValueError(f"Unknown save method: {save_method}")

        console.print(f"[green]✓[/green] Model saved to {output_path}")


class TransformersBackend(TrainingBackend):
    """Transformers training backend.
    
    Uses standard HuggingFace Transformers + PEFT without Unsloth optimizations.
    Provides maximum compatibility at the cost of training speed.
    """

    @property
    def name(self) -> str:
        return "transformers"

    @property
    def description(self) -> str:
        return "Standard HuggingFace Transformers + PEFT (maximum compatibility, slower than Unsloth)"

    def supports_text_training(self) -> bool:
        return True

    def supports_vision_training(self) -> bool:
        return False  # Vision support can be added later

    def create_text_trainer(
        self,
        base_model: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: Optional[str] = None,
    ) -> TextTrainer:
        """Create a Transformers text trainer."""
        return TransformersTextTrainer(
            base_model=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            dtype=dtype,
        )

    def create_vision_trainer(
        self,
        base_model: str,
        max_seq_length: int = 16384,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: Optional[Any] = None,
    ) -> VisionTrainer:
        """Create a Transformers vision trainer."""
        raise NotImplementedError(
            "Vision training not yet implemented for Transformers backend. Use Unsloth backend for vision models."
        )
