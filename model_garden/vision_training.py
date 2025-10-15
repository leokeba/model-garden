"""Vision-Language Model training using Unsloth.

Supports multimodal models like Qwen2.5-VL for fine-tuning on vision-language tasks.
"""

import json
import os
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

from datasets import Dataset, load_dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class VisionLanguageTrainer:
    """Handles vision-language model fine-tuning."""

    def __init__(
        self,
        base_model: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        dtype: Optional[str] = None,
    ):
        """Initialize the vision-language trainer.

        Args:
            base_model: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-VL-3B-Instruct")
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
        self.processor = None  # For vision models
        
        # Check if this is a vision model
        self.is_vision_model = "VL" in base_model or "vision" in base_model.lower()

    def load_model(self) -> None:
        """Load the vision-language model.
        
        Note: Qwen2.5-VL requires special handling as it's a multimodal model.
        """
        console.print(f"[cyan]Loading vision-language model: {self.base_model}[/cyan]")
        
        try:
            # Try loading with Unsloth first (if supported)
            from unsloth import FastLanguageModel
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description="Loading model...", total=None)
                
                try:
                    self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                        model_name=self.base_model,
                        max_seq_length=self.max_seq_length,
                        dtype=self.dtype,
                        load_in_4bit=self.load_in_4bit,
                    )
                    # Load processor for vision models
                    from transformers import AutoProcessor
                    self.processor = AutoProcessor.from_pretrained(self.base_model)
                    console.print("[green]✓[/green] Model loaded with Unsloth optimizations")
                except Exception as unsloth_error:
                    # Fall back to transformers for vision models
                    console.print(f"[yellow]⚠️  Unsloth not supported, using transformers[/yellow]")
                    from transformers import AutoModelForVision2Seq, AutoProcessor
                    
                    self.processor = AutoProcessor.from_pretrained(self.base_model)
                    self.tokenizer = self.processor.tokenizer
                    
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        self.base_model,
                        device_map="auto",
                        load_in_4bit=self.load_in_4bit if self.load_in_4bit else None,
                    )
                    console.print("[green]✓[/green] Model loaded with transformers")
                    
        except Exception as e:
            console.print(f"[red]❌ Failed to load model: {e}[/red]")
            raise

    def prepare_for_training(
        self,
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
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
        console.print("[cyan]Configuring LoRA adapters for vision-language model...[/cyan]")

        try:
            # Try Unsloth method first
            from unsloth import FastLanguageModel
            
            if target_modules is None:
                # Vision-language models often have different module names
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
            )
            console.print("[green]✓[/green] LoRA adapters configured (Unsloth)")
            
        except Exception:
            # Fall back to PEFT for vision models
            console.print("[yellow]Using PEFT for LoRA configuration[/yellow]")
            from peft import LoraConfig, get_peft_model
            
            if target_modules is None:
                target_modules = ["q_proj", "v_proj"]  # Minimal for vision models
            
            peft_config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            self.model = get_peft_model(self.model, peft_config)
            console.print("[green]✓[/green] LoRA adapters configured (PEFT)")

    def load_dataset_from_file(self, dataset_path: str) -> Dataset:
        """Load multimodal dataset from a local file.

        Args:
            dataset_path: Path to dataset file (JSONL with image paths or base64)

        Returns:
            Loaded dataset
        """
        console.print(f"[cyan]Loading vision-language dataset from: {dataset_path}[/cyan]")

        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        # Load dataset
        suffix = path.suffix.lower()
        if suffix in [".jsonl", ".json"]:
            dataset = load_dataset("json", data_files=str(path), split="train")
        elif suffix == ".parquet":
            dataset = load_dataset("parquet", data_files=str(path), split="train")
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        console.print(f"[green]✓[/green] Loaded {len(dataset)} examples")
        return dataset

    def format_dataset(
        self,
        dataset: Dataset,
        text_field: str = "text",
        image_field: str = "image",
        system_message: Optional[str] = None,
    ) -> list:
        """Format dataset for vision-language training using OpenAI message format.

        Expected input format:
        {
            "text": "Question about the image",
            "image": "/path/to/image.jpg",
            "response": "Answer text"
        }

        Returns list of message dictionaries for UnslothVisionDataCollator.

        Args:
            dataset: Input dataset
            text_field: Field name for text/questions
            image_field: Field name for images
            system_message: Optional system message

        Returns:
            List of formatted message dictionaries
        """
        from PIL import Image
        
        console.print("[cyan]Formatting vision-language dataset...[/cyan]")

        if system_message is None:
            system_message = "You are a helpful assistant that can analyze images."

        formatted_data = []
        for example in dataset:
            text = example.get(text_field, "")
            response = example.get("response", example.get("output", ""))
            image_path = example.get(image_field, "")
            
            # Load image as PIL Image
            if isinstance(image_path, str) and os.path.exists(image_path):
                image = Image.open(image_path)
            else:
                # Create blank image if path doesn't exist
                image = Image.new("RGB", (224, 224))
            
            # Format as OpenAI messages
            messages = {
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_message}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": text},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": response}],
                    },
                ],
            }
            formatted_data.append(messages)

        console.print(f"[green]✓[/green] Dataset formatted ({len(formatted_data)} examples)")
        return formatted_data

    def train(
        self,
        dataset: Dataset,
        output_dir: str,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 1,  # Smaller for vision models
        gradient_accumulation_steps: int = 8,  # Larger for vision models
        learning_rate: float = 2e-5,  # Lower for vision models
        warmup_steps: int = 10,
        max_steps: int = -1,
        logging_steps: int = 10,
        save_steps: int = 100,
        optim: str = "adamw_8bit",
    ) -> None:
        """Train the vision-language model.

        Args:
            dataset: Training dataset with 'text' and 'image' fields
            output_dir: Directory to save checkpoints
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device (use 1 for vision models)
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate (lower for vision models)
            warmup_steps: Number of warmup steps
            max_steps: Maximum training steps (-1 for full epochs)
            logging_steps: Log every N steps
            save_steps: Save checkpoint every N steps
            optim: Optimizer to use
        """
        console.print("[bold cyan]Starting vision-language model training...[/bold cyan]")

        from trl import SFTTrainer, SFTConfig
        from unsloth.trainer import UnslothVisionDataCollator
        
        # When using max_steps, still need to provide num_train_epochs
        use_max_steps = max_steps > 0
        
        training_args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps if use_max_steps else -1,
            num_train_epochs=1.0 if use_max_steps else num_train_epochs,
            learning_rate=learning_rate,
            fp16=not self.load_in_4bit,
            bf16=self.load_in_4bit,
            logging_steps=logging_steps,
            optim=optim,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            save_steps=save_steps,
            save_total_limit=3,
            report_to="none",
            # CRITICAL for vision models - Unsloth requirements:
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
        )

        console.print("[yellow]⚠️  Vision-language training uses UnslothVisionDataCollator[/yellow]")
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=dataset,
            data_collator=UnslothVisionDataCollator(self.model, self.processor),
        )

        console.print("[cyan]Training in progress...[/cyan]")
        trainer.train()
        console.print("[bold green]✨ Training completed![/bold green]")

    def save_model(
        self,
        output_dir: str,
        save_method: str = "lora",
    ) -> None:
        """Save the fine-tuned vision-language model.

        Args:
            output_dir: Directory to save the model
            save_method: How to save ('lora', 'merged_16bit', 'merged_4bit')
        """
        console.print(f"[cyan]Saving model to: {output_dir}[/cyan]")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if save_method == "lora":
            # Save only LoRA adapters
            self.model.save_pretrained(output_dir)
            if self.tokenizer:
                self.tokenizer.save_pretrained(output_dir)
            if self.processor:
                self.processor.save_pretrained(output_dir)
        else:
            # For vision models, merging is more complex
            console.print("[yellow]⚠️  Merged saving for vision models not yet implemented[/yellow]")
            console.print("[yellow]⚠️  Saving LoRA adapters only[/yellow]")
            self.model.save_pretrained(output_dir)
            if self.tokenizer:
                self.tokenizer.save_pretrained(output_dir)
            if self.processor:
                self.processor.save_pretrained(output_dir)

        console.print("[bold green]✓ Model saved successfully![/bold green]")


def create_vision_sample_dataset(output_path: str, num_examples: int = 10) -> None:
    """Create a sample vision-language dataset for testing.

    Args:
        output_path: Path to save the dataset
        num_examples: Number of examples to generate
    """
    console.print(f"[cyan]Creating sample vision-language dataset with {num_examples} examples...[/cyan]")

    examples = []
    for i in range(num_examples):
        examples.append({
            "text": f"What is shown in this image? Please describe it.",
            "image": f"/path/to/image_{i}.jpg",  # Placeholder - user should provide actual images
            "response": f"This is a sample response for image {i}. In a real dataset, this would describe the actual image content.",
        })

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    console.print(f"[green]✓[/green] Sample vision-language dataset created at {output_path}")
    console.print("[yellow]⚠️  Note: Replace placeholder image paths with actual image files[/yellow]")
