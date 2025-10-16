"""Vision-Language Model training using Unsloth.

Supports multimodal models like Qwen2.5-VL for fine-tuning on vision-language tasks.
"""

import base64
import io
import json
import os
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

from datasets import Dataset, load_dataset
from PIL import Image
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
        
        # Get HuggingFace token from environment for private models
        hf_token = os.getenv('HF_TOKEN')
        
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
                        token=hf_token,
                    )
                    # Load processor for vision models
                    from transformers import AutoProcessor
                    self.processor = AutoProcessor.from_pretrained(self.base_model, token=hf_token)
                    console.print("[green]✓[/green] Model loaded with Unsloth optimizations")
                except Exception as unsloth_error:
                    # Fall back to transformers for vision models
                    console.print(f"[yellow]⚠️  Unsloth not supported, using transformers[/yellow]")
                    from transformers import AutoModelForVision2Seq, AutoProcessor
                    
                    self.processor = AutoProcessor.from_pretrained(self.base_model, token=hf_token)
                    self.tokenizer = self.processor.tokenizer
                    
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        self.base_model,
                        device_map="auto",
                        load_in_4bit=self.load_in_4bit if self.load_in_4bit else None,
                        token=hf_token,
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

    def load_dataset_from_hub(
        self,
        dataset_name: str,
        split: str = "train",
        **kwargs
    ) -> Dataset:
        """Load multimodal dataset from HuggingFace Hub.

        Args:
            dataset_name: HuggingFace dataset identifier (e.g., "user/dataset-name")
            split: Dataset split to load (default: "train")
            **kwargs: Additional arguments passed to load_dataset

        Returns:
            Loaded dataset
        """
        console.print(f"[cyan]Loading dataset from HuggingFace Hub: {dataset_name}[/cyan]")
        
        # Get HuggingFace token from environment for private datasets
        hf_token = os.getenv('HF_TOKEN')
        
        try:
            dataset = load_dataset(dataset_name, split=split, token=hf_token, **kwargs)
            console.print(f"[green]✓[/green] Loaded {len(dataset)} examples from Hub")
            return dataset
        except Exception as e:
            console.print(f"[red]❌ Failed to load dataset from Hub: {e}[/red]")
            raise

    def load_dataset(
        self,
        dataset_path: str,
        from_hub: bool = False,
        split: str = "train",
        **kwargs
    ) -> Dataset:
        """Load multimodal dataset from file or HuggingFace Hub.

        Args:
            dataset_path: Path to local file or HuggingFace dataset identifier
            from_hub: If True, load from HuggingFace Hub; if False, load from local file
            split: Dataset split to load (for Hub datasets)
            **kwargs: Additional arguments passed to load_dataset

        Returns:
            Loaded dataset
        """
        if from_hub:
            return self.load_dataset_from_hub(dataset_path, split=split, **kwargs)
        else:
            return self.load_dataset_from_file(dataset_path)

    def _decode_base64_image(self, image_str: str) -> Image.Image:
        """Decode a base64-encoded image string to PIL Image.

        Args:
            image_str: Base64-encoded image string (with or without data URI prefix)

        Returns:
            PIL Image object
        """
        try:
            # Remove data URI prefix if present (e.g., "data:image/jpeg;base64,")
            if image_str.startswith("data:"):
                image_str = image_str.split(",", 1)[1]
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_str)
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            return image
        except Exception as e:
            console.print(f"[yellow]⚠️  Failed to decode base64 image: {e}[/yellow]")
            # Return blank image as fallback
            return Image.new("RGB", (224, 224))

    def _load_image(self, image_data: Any) -> Image.Image:
        """Load image from various sources (file path, base64, PIL Image, etc.).

        Args:
            image_data: Image data (file path, base64 string, PIL Image, etc.)

        Returns:
            PIL Image object
        """
        # Already a PIL Image
        if isinstance(image_data, Image.Image):
            return image_data
        
        # File path
        if isinstance(image_data, str):
            if image_data.startswith("data:image") or (len(image_data) > 100 and not os.path.exists(image_data)):
                # Looks like base64
                return self._decode_base64_image(image_data)
            elif os.path.exists(image_data):
                # File path
                return Image.open(image_data)
        
        # Fallback: create blank image
        console.print(f"[yellow]⚠️  Unknown image format, using blank image[/yellow]")
        return Image.new("RGB", (224, 224))

    def _convert_messages_to_simple_format(self, messages: List[Dict]) -> Dict[str, str]:
        """Convert OpenAI messages format to simple format.
        
        Extracts the first image and text from user message, and assistant's response.
        This ensures compatibility with UnslothVisionDataCollator.
        
        Args:
            messages: List of OpenAI-style messages
            
        Returns:
            Dict with 'text', 'image', and 'response' keys
        """
        result = {
            "text": "",
            "image": None,
            "response": ""
        }
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", [])
            
            if role == "user":
                # Extract text and image from user message
                for item in content:
                    item_type = item.get("type", "")
                    if item_type == "text" and not result["text"]:
                        result["text"] = item.get("text", "")
                    elif item_type in ("image", "image_url") and not result["image"]:
                        # Handle both old format (type: "image", image: "...") 
                        # and new format (type: "image_url", image_url: {url: "..."})
                        image_data = item.get("image", item.get("image_url", {}))
                        if isinstance(image_data, dict):
                            image_data = image_data.get("url", "")
                        result["image"] = image_data
            
            elif role == "assistant":
                # Extract response from assistant message
                for item in content:
                    if item.get("type") == "text" and not result["response"]:
                        result["response"] = item.get("text", "")
        
        return result

    def format_dataset(
        self,
        dataset: Dataset,
        text_field: str = "text",
        image_field: str = "image",
        system_message: Optional[str] = None,
        messages_field: Optional[str] = None,
    ) -> list:
        """Format dataset for vision-language training using OpenAI message format.

        Supports two input formats:
        
        1. Simple format (custom datasets):
           {
               "text": "Question about the image",
               "image": "/path/to/image.jpg" or "data:image/jpeg;base64,...",
               "response": "Answer text"
           }
        
        2. OpenAI messages format (HuggingFace datasets):
           {
               "messages": [
                   {"role": "system", "content": [{"type": "text", "text": "..."}]},
                   {"role": "user", "content": [
                       {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
                       {"type": "text", "text": "..."}
                   ]},
                   {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
               ]
           }
           
        Note: OpenAI messages format is automatically converted to simple format
        for compatibility with UnslothVisionDataCollator.
        Also supports older format with {"type": "image", "image": "..."}.

        Returns list of message dictionaries for UnslothVisionDataCollator.

        Args:
            dataset: Input dataset
            text_field: Field name for text/questions (for simple format)
            image_field: Field name for images (for simple format)
            system_message: Optional system message (for simple format)
            messages_field: Field name for messages (for OpenAI format, default: "messages")

        Returns:
            List of formatted message dictionaries
        """
        console.print("[cyan]Formatting vision-language dataset...[/cyan]")

        if system_message is None:
            system_message = "You are a helpful assistant that can analyze images."

        formatted_data = []
        
        # Check if dataset uses OpenAI messages format
        if messages_field is None:
            messages_field = "messages"
        
        has_messages_field = messages_field in dataset.column_names
        
        for example in dataset:
            if has_messages_field and messages_field in example:
                # OpenAI messages format - convert to simple format first
                console.print("[yellow]Converting OpenAI messages format to simple format for compatibility...[/yellow]") if len(formatted_data) == 0 else None
                
                messages = example[messages_field]
                simple = self._convert_messages_to_simple_format(messages)
                
                # Now process as simple format
                text = simple.get("text", "")
                response = simple.get("response", "")
                image_data = simple.get("image", "")
                
                # Load image (handles base64, file paths, etc.)
                pil_image = self._load_image(image_data)
                
                # Format as OpenAI messages (simple structure)
                formatted_messages = {
                    "messages": [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": system_message}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": pil_image},
                                {"type": "text", "text": text},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": response}],
                        },
                    ],
                }
                formatted_data.append(formatted_messages)
            else:
                # Simple format - build OpenAI messages structure
                text = example.get(text_field, "")
                response = example.get("response", example.get("output", ""))
                image_data = example.get(image_field, "")
                
                # Load image (handles file paths, base64, etc.)
                pil_image = self._load_image(image_data)
                
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
                                {"type": "image", "image": pil_image},
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
        dataset: Union[Dataset, List[Dict]],
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
        callbacks: Optional[List] = None,
    ) -> None:
        """Train the vision-language model.

        Args:
            dataset: Training dataset (Dataset object or list of formatted messages)
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
            callbacks: Optional list of TrainerCallback instances
        """
        console.print("[bold cyan]Starting vision-language model training...[/bold cyan]")

        from trl import SFTTrainer, SFTConfig
        from unsloth.trainer import UnslothVisionDataCollator
        
        # For vision models, keep data as list - don't convert to Dataset
        # The UnslothVisionDataCollator expects PIL Images which don't survive PyArrow serialization
        if isinstance(dataset, list):
            console.print(f"[cyan]Using formatted data directly ({len(dataset)} examples)[/cyan]")
            train_dataset = dataset
        else:
            train_dataset = dataset
        
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
            train_dataset=train_dataset,
            data_collator=UnslothVisionDataCollator(self.model, self.processor),
            callbacks=callbacks if callbacks else [],
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
