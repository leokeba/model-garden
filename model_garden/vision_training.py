"""Vision-Language Model training using Unsloth.

Supports multimodal models like Qwen2.5-VL for fine-tuning on vision-language tasks.
"""

import base64
import gc
import io
import json
import os
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, cast, Literal

# Configure HuggingFace cache from environment before importing HF libraries
from dotenv import load_dotenv
load_dotenv()

HF_HOME = os.getenv('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))
os.environ['HF_HOME'] = HF_HOME
os.environ['TRANSFORMERS_CACHE'] = str(Path(HF_HOME) / 'hub')
os.environ['HF_DATASETS_CACHE'] = str(Path(HF_HOME) / 'datasets')

# Configure PyTorch CUDA memory allocator for better performance
# max_split_size_mb limits memory fragmentation, expandable_segments allows dynamic growth
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

# CRITICAL: Import unsloth BEFORE any other ML libraries (datasets, transformers, trl, peft)
# This ensures Unsloth's PyTorch patches are applied correctly
from unsloth import FastLanguageModel

# Now import other ML libraries AFTER unsloth
import torch
from datasets import Dataset, load_dataset
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from transformers.trainer_callback import TrainerControl

# Import carbon tracking
from model_garden.carbon import CarbonTracker
# Import selective_loss at module level since unsloth is already imported
from model_garden.selective_loss import create_selective_loss_collator
# Import shared training utilities
from model_garden.training_utils import detect_model_dtype, get_training_precision_config, MemoryMonitorCallback

console = Console()


def _cleanup_memory_after_merge():
    """Clean up GPU and system memory after model merge.
    
    Performs:
    - Garbage collection (multiple passes)
    - GPU cache clearing
    - Memory synchronization
    """
    import gc
    import torch
    
    # Multiple passes of garbage collection to ensure all cycles are broken
    for _ in range(3):
        gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Reset peak memory stats to get accurate readings for next operation
        torch.cuda.reset_peak_memory_stats()


class VisionLanguageTrainer:
    """Handles vision-language model fine-tuning."""

    def __init__(
        self,
        base_model: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: Optional[str] = None,
    ):
        """Initialize the vision-language trainer.

        Args:
            base_model: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-VL-3B-Instruct")
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
        self.processor = None  # For vision models
        
        # Check if this is a vision model
        self.is_vision_model = "VL" in base_model or "vision" in base_model.lower()

    def load_model(self) -> None:
        """Load the vision-language model.
        
        Note: Qwen2.5-VL requires special handling as it's a multimodal model.
        Supports 4-bit, 8-bit, and 16-bit (full precision) loading.
        """
        # Determine precision for logging
        if self.load_in_8bit:
            precision = "8-bit (balanced quality/memory)"
        elif self.load_in_4bit:
            precision = "4-bit (memory efficient)"
        else:
            precision = "16-bit (full quality)"
        
        console.print(f"[cyan]Loading vision-language model: {self.base_model}[/cyan]")
        console.print(f"[cyan]Precision: {precision}[/cyan]")
        
        # Get HuggingFace token from environment for private models
        hf_token = os.getenv('HF_TOKEN')
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description="Loading model...", total=None)
                
                try:
                    # Unsloth supports both 4-bit and 8-bit quantization
                    # Note: For 16-bit, set both load_in_4bit and load_in_8bit to False
                    self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                        model_name=self.base_model,
                        max_seq_length=self.max_seq_length,
                        dtype=self.dtype,
                        load_in_4bit=self.load_in_4bit,
                        load_in_8bit=self.load_in_8bit,  # Add 8-bit support
                        token=hf_token,
                    )
                    # Load processor for vision models
                    from transformers import AutoProcessor
                    self.processor = AutoProcessor.from_pretrained(self.base_model, token=hf_token)
                    # Use processor's tokenizer for vision models
                    self.tokenizer = self.processor.tokenizer
                    console.print("[green]✓[/green] Model loaded with Unsloth optimizations")
                except Exception as unsloth_error:
                    # Fall back to transformers for vision models
                    console.print(f"[yellow]⚠️  Unsloth not supported, using transformers[/yellow]")
                    from transformers import AutoModelForVision2Seq, AutoProcessor
                    import torch
                    
                    self.processor = AutoProcessor.from_pretrained(self.base_model, token=hf_token)
                    self.tokenizer = self.processor.tokenizer
                    
                    # Determine torch_dtype based on quantization settings
                    # For 16-bit (no quantization), explicitly use bfloat16
                    if not self.load_in_4bit and not self.load_in_8bit:
                        # For 16-bit precision, explicitly use bfloat16 (Qwen2.5-VL's native precision)
                        # Using None doesn't always work - model can load as float32 then convert
                        torch_dtype = torch.bfloat16 if self.dtype is None else self.dtype
                    else:
                        # For quantized models, let BitsAndBytes handle dtype
                        torch_dtype = None
                    
                    # Transformers supports 4-bit and 8-bit via BitsAndBytes
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        self.base_model,
                        device_map="auto",
                        torch_dtype=torch_dtype,
                        load_in_4bit=self.load_in_4bit if self.load_in_4bit else None,
                        load_in_8bit=self.load_in_8bit if self.load_in_8bit else None,
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
        lora_bias: str = "none",
        task_type: str = "CAUSAL_LM",
        use_gradient_checkpointing: str | bool = "unsloth",
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
            use_gradient_checkpointing: Gradient checkpointing mode:
                - "unsloth": Most memory efficient (30% less VRAM), minor quality loss
                - True: Standard gradient checkpointing, better quality
                - False: No gradient checkpointing, best quality but most memory
            random_state: Random seed for reproducibility
            loftq_config: LoftQ quantization config (None to disable)
        """
        console.print("[cyan]Configuring LoRA adapters for vision-language model...[/cyan]")
        
        # Workaround: 8-bit quantization has compatibility issues with gradient checkpointing
        # due to torch compile + bitsandbytes interactions. Disable gradient checkpointing for 8-bit.
        if self.load_in_8bit and use_gradient_checkpointing not in [False, "false"]:
            console.print("[yellow]⚠️  8-bit quantization detected - disabling gradient checkpointing to avoid compatibility issues[/yellow]")
            console.print("[yellow]    (8-bit + gradient checkpointing causes torch compile errors)[/yellow]")
            use_gradient_checkpointing = False

        try:
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
                bias=lora_bias,
                use_gradient_checkpointing=use_gradient_checkpointing,  # type: ignore
                random_state=random_state,
                use_rslora=use_rslora,
                loftq_config=loftq_config,
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
                bias=cast(Literal["none", "all", "lora_only"], lora_bias if lora_bias in ["none", "all", "lora_only"] else "none"),
                task_type=task_type,
            )
            
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")
            self.model = get_peft_model(self.model, peft_config)  # type: ignore
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

        # Handle dataset types - cast to Dataset for type safety
        try:
            dataset_len = len(dataset)  # type: ignore
            console.print(f"[green]✓[/green] Loaded {dataset_len} examples")
        except (TypeError, AttributeError):
            console.print("[green]✓[/green] Loaded dataset (streaming)")
        return cast(Dataset, dataset)

    def load_dataset_from_hub(
        self,
        dataset_name: str,
        split: str = "train",
        **kwargs
    ) -> Dataset:
        """Load multimodal dataset from HuggingFace Hub.

        Args:
            dataset_name: HuggingFace dataset identifier (e.g., "user/dataset-name")
                         Can include specific file with '::' separator (e.g., 'user/repo::train.jsonl')
            split: Dataset split to load (default: "train", ignored if specific file is provided)
            **kwargs: Additional arguments passed to load_dataset

        Returns:
            Loaded dataset
        """
        # Get HuggingFace token from environment for private datasets
        hf_token = os.getenv('HF_TOKEN')
        
        try:
            # Check if dataset_name includes a specific file
            if "::" in dataset_name:
                repo_name, file_name = dataset_name.split("::", 1)
                console.print(f"[cyan]Loading dataset from HuggingFace Hub: {repo_name} (file: {file_name})[/cyan]")
                
                # Load specific file from the repo
                dataset = load_dataset(repo_name, data_files=file_name, split="train", token=hf_token, **kwargs)
            else:
                console.print(f"[cyan]Loading dataset from HuggingFace Hub: {dataset_name} (split: {split})[/cyan]")
                
                # Load standard split
                dataset = load_dataset(dataset_name, split=split, token=hf_token, **kwargs)
            
            # Handle dataset types - cast to Dataset for type safety
            try:
                dataset_len = len(dataset)  # type: ignore
                console.print(f"[green]✓[/green] Loaded {dataset_len} examples from Hub")
            except (TypeError, AttributeError):
                console.print("[green]✓[/green] Loaded dataset from Hub (streaming)")
            return cast(Dataset, dataset)
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
        
        Note:
            Images are loaded once and kept in RAM for efficiency. The conversion to RGB
            ensures consistent format and forces full loading (avoiding lazy loading issues).
        """
        # Already a PIL Image
        if isinstance(image_data, Image.Image):
            # Ensure RGB format for consistency
            if image_data.mode != "RGB":
                return image_data.convert("RGB")
            return image_data
        
        # File path
        if isinstance(image_data, str):
            if image_data.startswith("data:image") or (len(image_data) > 100 and not os.path.exists(image_data)):
                # Looks like base64
                img = self._decode_base64_image(image_data)
                # Ensure RGB format
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return img
            elif os.path.exists(image_data):
                # Load image from file
                img = Image.open(image_data)
                # Convert to RGB to ensure consistent format and fully load pixels
                if img.mode != "RGB":
                    img = img.convert("RGB")
                # Force load to ensure pixels are in memory (avoid lazy loading)
                img.load()
                return img
        
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
            # Ensure example is a dict-like object
            if isinstance(example, dict):
                example_dict = example
            else:
                # Handle list case (shouldn't happen with proper datasets)
                continue
                
            if has_messages_field and messages_field in example_dict:
                # OpenAI messages format - convert to simple format first
                console.print("[yellow]Converting OpenAI messages format to simple format for compatibility...[/yellow]") if len(formatted_data) == 0 else None
                
                messages = example_dict[messages_field]
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
                text = example_dict.get(text_field, "")
                response = example_dict.get("response", example_dict.get("output", ""))
                image_data = example_dict.get(image_field, "")
                
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
        job_id: Optional[str] = None,
        enable_carbon_tracking: bool = True,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 1,  # Smaller for vision models
        gradient_accumulation_steps: int = 8,  # Larger for vision models
        learning_rate: float = 2e-5,  # Lower for vision models
        warmup_steps: int = 10,
        max_steps: int = -1,
        logging_steps: int = 10,
        save_steps: int = 100,
        optim: str = "adamw_8bit",
        weight_decay: float = 0.01,
        lr_scheduler_type: str = "cosine",  # Cosine better for vision models
        max_grad_norm: float = 1.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        dataloader_num_workers: int = 0,
        dataloader_pin_memory: bool = False,  # CRITICAL: Disable to prevent RAM accumulation
        eval_strategy: str = "steps",
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "eval_loss",
        save_total_limit: int = 3,
        callbacks: Optional[List] = None,
        eval_dataset: Optional[Union[Dataset, List[Dict]]] = None,
        eval_steps: Optional[int] = None,
        selective_loss: bool = False,
        selective_loss_level: str = "conservative",
        selective_loss_schema_keys: Optional[List[str]] = None,
        selective_loss_masking_start_step: int = 0,
        selective_loss_verbose: bool = False,
    ) -> None:
        """Train the vision-language model.

        Args:
            dataset: Training dataset (Dataset object or list of formatted messages)
            output_dir: Directory to save checkpoints
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device (use 1 for vision models)
            gradient_accumulation_steps: Gradient accumulation steps (higher for vision models)
            learning_rate: Learning rate (lower for vision models, typically 2e-5)
            warmup_steps: Number of warmup steps
            max_steps: Maximum training steps (-1 for full epochs)
            logging_steps: Log every N steps
            save_steps: Save checkpoint every N steps
            optim: Optimizer to use (adamw_8bit, adamw_torch, adafactor, etc.)
            weight_decay: Weight decay for regularization
            lr_scheduler_type: Learning rate scheduler (cosine recommended for vision models)
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
            selective_loss: Enable selective loss masking for structured outputs
            selective_loss_level: Masking level ('conservative', 'moderate', 'aggressive')
            selective_loss_schema_keys: Schema keys to mask in aggressive mode
            selective_loss_masking_start_step: Delay masking until this step (0=immediate, 100=after 100 steps of structure learning)
            selective_loss_verbose: Print masking statistics during training
        """
        console.print("[bold cyan]Starting vision-language model training...[/bold cyan]")
        
        # Note: Using DataLoader workers with vision models can be tricky
        if dataloader_num_workers > 0:
            console.print(f"[yellow]⚠️  INFO: Using {dataloader_num_workers} DataLoader workers[/yellow]")
            console.print("[yellow]   Multiple workers can improve throughput but use more memory[/yellow]")
            console.print("[yellow]   If you experience issues, try setting dataloader_num_workers=0[/yellow]")
        
        # Initialize carbon tracker
        carbon_tracker = None
        emissions_data = None
        if enable_carbon_tracking:
            # Generate job_id if not provided
            if job_id is None:
                import time
                job_id = f"vision-training-{int(time.time())}"
            
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

        from trl.trainer.sft_trainer import SFTTrainer
        from trl.trainer.sft_config import SFTConfig
        from unsloth.trainer import UnslothVisionDataCollator
        from transformers import TrainerCallback
        
        # For vision models, keep data as list - don't convert to Dataset
        # The UnslothVisionDataCollator expects PIL Images which don't survive PyArrow serialization
        if isinstance(dataset, list):
            console.print(f"[cyan]Using formatted data directly ({len(dataset)} examples)[/cyan]")
            train_dataset = dataset
        else:
            train_dataset = dataset
        
        # Handle eval dataset similarly
        if isinstance(eval_dataset, list):
            console.print(f"[cyan]Using validation dataset ({len(eval_dataset)} examples)[/cyan]")
        
        # Set evaluation strategy if validation dataset provided
        do_eval = eval_dataset is not None
        eval_steps_value = eval_steps if eval_steps is not None else save_steps
        
        # When using max_steps, still need to provide num_train_epochs
        use_max_steps = max_steps > 0
        
        # Set evaluation strategy if validation dataset provided
        final_eval_strategy = eval_strategy if eval_dataset is not None else "no"
        eval_steps_value = eval_steps if eval_steps is not None else save_steps
        
        # Determine if we should load best model at end
        final_load_best = load_best_model_at_end and eval_dataset is not None
        final_metric = metric_for_best_model if eval_dataset is not None else None

        # Build training args - SFTConfig has different parameters than TrainingArguments
        # Detect model's actual dtype to set precision correctly
        model_dtype = detect_model_dtype(self.model, self.load_in_4bit, self.load_in_8bit)
        precision_config = get_training_precision_config(self.model, self.load_in_4bit, self.load_in_8bit)
        
        # Log detected dtype for debugging
        console.print(f"[cyan]🔍 Detected model dtype: {model_dtype}[/cyan]")
        console.print(f"[cyan]📊 Training precision: {'bf16' if precision_config['bf16'] else 'fp16'}[/cyan]")
        
        training_args_dict = {
            "output_dir": output_dir,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "warmup_steps": warmup_steps,
            "max_steps": max_steps if use_max_steps else -1,
            "num_train_epochs": 1.0 if use_max_steps else num_train_epochs,
            "learning_rate": learning_rate,
            # Precision settings: Match the model's actual dtype using shared utilities
            "fp16": precision_config['fp16'],
            "bf16": precision_config['bf16'],
            "logging_steps": logging_steps,
            "optim": optim,
            "weight_decay": weight_decay,
            "lr_scheduler_type": lr_scheduler_type,
            "max_grad_norm": max_grad_norm,
            "adam_beta1": adam_beta1,
            "adam_beta2": adam_beta2,
            "adam_epsilon": adam_epsilon,
            "dataloader_num_workers": dataloader_num_workers,
            "dataloader_pin_memory": dataloader_pin_memory,
            "seed": 42,
            "save_steps": save_steps,
            "save_total_limit": save_total_limit,
            "report_to": "none",
            # CRITICAL for vision models - Unsloth requirements:
            "remove_unused_columns": False,
            "dataset_text_field": "",
            "dataset_kwargs": {"skip_prepare_dataset": True},
            # IMPORTANT: Only train on responses/outputs, not inputs/instructions
            # This masks the input tokens so the model only learns to generate responses
            "completion_only_loss": True,
        }
        
        # Add evaluation settings if validation dataset provided
        if eval_dataset is not None:
            training_args_dict["per_device_eval_batch_size"] = per_device_train_batch_size
            training_args_dict["do_eval"] = True
            training_args_dict["eval_strategy"] = final_eval_strategy
            training_args_dict["eval_steps"] = eval_steps_value
            training_args_dict["load_best_model_at_end"] = final_load_best
            training_args_dict["metric_for_best_model"] = final_metric
        
        training_args = SFTConfig(**training_args_dict)

        console.print("[cyan]ℹ️  Vision training uses UnslothVisionDataCollator for efficient image processing[/cyan]")
        
        # Choose data collator based on selective_loss flag
        if selective_loss:
            # Lazy import to avoid spawning torch compile workers at module import time
            from model_garden.selective_loss import create_selective_loss_collator
            
            console.print(f"[cyan]🎯 Using selective loss masking (level: {selective_loss_level})[/cyan]")
            if selective_loss_masking_start_step > 0:
                console.print(f"[yellow]⏱️  Masking delayed until step {selective_loss_masking_start_step}[/yellow]")
            data_collator = create_selective_loss_collator(
                model=self.model,
                processor=self.processor,
                mask_level=selective_loss_level,
                schema_keys=selective_loss_schema_keys,
                dataset=train_dataset,  # Pass dataset for auto-detection
                masking_start_step=selective_loss_masking_start_step,
                verbose=selective_loss_verbose
            )
        else:
            # Use standard Unsloth collator (images stay in RAM for efficiency)
            data_collator = UnslothVisionDataCollator(self.model, self.processor)

        
        # Ensure model and tokenizer are loaded
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_model() first.")
        
        # Add memory monitoring callback (optional but useful for debugging)
        # Use shared implementation from training_utils to avoid duplication
        memory_monitor = MemoryMonitorCallback()
        all_callbacks = [memory_monitor]
        if callbacks:
            all_callbacks.extend(callbacks)

        console.print("[cyan]💡 Memory monitoring enabled: Tracking RAM usage every 10 steps[/cyan]")
            
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,  # type: ignore
            args=training_args,
            train_dataset=train_dataset,  # type: ignore
            eval_dataset=eval_dataset,  # type: ignore
            data_collator=data_collator,
            callbacks=all_callbacks,
        )

        console.print("[cyan]Training in progress...[/cyan]")
        trainer.train()
        console.print("[bold green]✨ Training completed![/bold green]")
        
        # Print selective loss statistics if enabled
        if selective_loss:
            from model_garden.selective_loss import SelectiveLossVisionCollator
            if isinstance(data_collator, SelectiveLossVisionCollator):
                data_collator.print_stats()
        
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
        
        # CRITICAL: Explicitly clear dataset references from trainer to enable garbage collection
        # Vision models keep PIL images in RAM which can accumulate across multiple training runs
        console.print("[cyan]🧹 Clearing dataset references from trainer...[/cyan]")
        try:
            if hasattr(trainer, 'train_dataset'):
                trainer.train_dataset = None
            if hasattr(trainer, 'eval_dataset'):
                trainer.eval_dataset = None
            if hasattr(trainer, 'data_collator'):
                trainer.data_collator = None
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Failed to clear trainer datasets: {e}[/yellow]")

    def _clean_merged_config(self, output_dir: str) -> None:
        """Remove quantization_config from merged model config for vLLM compatibility.
        
        Args:
            output_dir: Directory containing the model config
        """
        import json
        config_path = Path(output_dir) / "config.json"
        
        if not config_path.exists():
            console.print("[yellow]⚠️  config.json not found, skipping cleanup[/yellow]")
            return
        
        try:
            console.print("[cyan]Cleaning config.json for vLLM compatibility...[/cyan]")
            
            # Read config
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Backup original
            backup_path = Path(output_dir) / "config.json.backup"
            with open(backup_path, 'w') as f:
                json.dump(config, f, indent=2)
            console.print(f"[green]✓ Backed up original config to {backup_path.name}[/green]")
            
            # Remove quantization_config at all levels
            modified = False
            if "quantization_config" in config:
                del config["quantization_config"]
                modified = True
                console.print("[green]✓ Removed root-level quantization_config[/green]")
            
            if "text_config" in config and isinstance(config["text_config"], dict):
                if "quantization_config" in config["text_config"]:
                    del config["text_config"]["quantization_config"]
                    modified = True
                    console.print("[green]✓ Removed text_config quantization_config[/green]")
            
            # Also change torch_dtype to dtype if present
            if "torch_dtype" in config:
                config["dtype"] = config.pop("torch_dtype")
                modified = True
                console.print("[green]✓ Changed torch_dtype to dtype[/green]")
            
            if modified:
                # Write cleaned config
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                console.print("[green]✓ Config cleaned for vLLM compatibility[/green]")
            else:
                console.print("[yellow]⚠️  No modifications needed[/yellow]")
                
        except Exception as e:
            console.print(f"[red]❌ Failed to clean config: {e}[/red]")
            console.print("[yellow]   Model may not load properly in vLLM[/yellow]")

    def save_model(
        self,
        output_dir: str,
        save_method: str = "merged_16bit",
        maximum_memory_usage: float = 0.75,
        max_shard_size: str = "5GB",
    ) -> None:
        """Save the fine-tuned vision-language model.

        Args:
            output_dir: Directory to save the model
            save_method: How to save ('lora', 'merged_16bit', 'merged_4bit')
            maximum_memory_usage: Maximum RAM usage ratio (0.0-0.95, lower = less RAM, default: 0.75)
                                  Reduce this (e.g., 0.5) if you run out of memory during merge
            max_shard_size: Maximum size per shard file (e.g., "1GB", "2GB", "5GB")
                           Smaller values use less peak memory during save
        """
        console.print(f"[cyan]Saving model to: {output_dir}[/cyan]")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if save_method == "lora":
            # Save only LoRA adapters
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")
            self.model.save_pretrained(output_dir)
            if self.tokenizer:
                self.tokenizer.save_pretrained(output_dir)
            if self.processor:
                self.processor.save_pretrained(output_dir)
        elif save_method == "merged_16bit":
            # Merge LoRA weights and save in 16-bit
            console.print("[cyan]Merging LoRA weights and saving in 16-bit...[/cyan]")
            console.print(f"[cyan]Memory settings: max_usage={maximum_memory_usage}, shard_size={max_shard_size}[/cyan]")
            try:
                from unsloth import FastLanguageModel
                # Merge and save using Unsloth
                # Clear GPU cache before merging to free up memory
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    console.print(f"[cyan]💾 GPU memory cleared before merge[/cyan]")
                
                if self.model is None:
                    raise RuntimeError("Model not loaded. Call load_model() first.")
                self.model.save_pretrained_merged(  # type: ignore
                    output_dir,
                    self.tokenizer,
                    save_method="merged_16bit",
                    maximum_memory_usage=maximum_memory_usage,
                    max_shard_size=max_shard_size,
                )
                if self.processor:
                    self.processor.save_pretrained(output_dir)
                console.print("[green]✓ Merged model saved in 16-bit precision[/green]")
                
                # Clean config for vLLM compatibility
                self._clean_merged_config(output_dir)
                
                # Aggressively free memory after successful merge
                console.print("[cyan]🧹 Cleaning up memory after merge...[/cyan]")
                self.model = None
                self.tokenizer = None
                self.processor = None
                _cleanup_memory_after_merge()
                console.print("[green]✓ Memory cleaned up[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️  Unsloth merge failed: {e}[/yellow]")
                console.print("[cyan]Trying PEFT merge as fallback...[/cyan]")
                # Manual merge using PEFT
                try:
                    from peft import PeftModel
                    # Clear memory before merge attempt
                    _cleanup_memory_after_merge()
                    
                    # Merge adapters
                    if self.model is None:
                        raise RuntimeError("Model not loaded.")
                    merged_model = self.model.merge_and_unload()  # type: ignore
                    merged_model.save_pretrained(output_dir)  # type: ignore
                    if self.tokenizer:
                        self.tokenizer.save_pretrained(output_dir)
                    if self.processor:
                        self.processor.save_pretrained(output_dir)
                    console.print("[green]✓ Model merged and saved successfully (PEFT fallback)[/green]")
                    
                    # Clean config for vLLM compatibility
                    self._clean_merged_config(output_dir)
                    
                    # Aggressively free memory after successful merge
                    console.print("[cyan]🧹 Cleaning up memory after merge...[/cyan]")
                    self.model = None
                    self.tokenizer = None
                    self.processor = None
                    _cleanup_memory_after_merge()
                    console.print("[green]✓ Memory cleaned up[/green]")
                except Exception as merge_error:
                    console.print(f"[red]❌ Merge failed: {merge_error}[/red]")
                    console.print("[yellow]Falling back to saving LoRA adapters only[/yellow]")
                    if self.model is None:
                        raise RuntimeError("Model not loaded.")
                    self.model.save_pretrained(output_dir)
                    if self.tokenizer:
                        self.tokenizer.save_pretrained(output_dir)
                    if self.processor:
                        self.processor.save_pretrained(output_dir)
        elif save_method == "merged_4bit":
            # Merge LoRA weights and save in 4-bit
            console.print("[cyan]Merging LoRA weights and saving in 4-bit...[/cyan]")
            console.print(f"[cyan]Memory settings: max_usage={maximum_memory_usage}, shard_size={max_shard_size}[/cyan]")
            console.print("[yellow]⚠️  Warning: 4-bit merge may reduce accuracy for GGUF conversion[/yellow]")
            try:
                from unsloth import FastLanguageModel
                # Clear GPU cache before merging
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    console.print(f"[cyan]💾 GPU memory cleared before merge[/cyan]")
                
                # Merge and save
                if self.model is None:
                    raise RuntimeError("Model not loaded. Call load_model() first.")
                self.model.save_pretrained_merged(  # type: ignore
                    output_dir,
                    self.tokenizer,
                    save_method="merged_4bit_forced",
                    maximum_memory_usage=maximum_memory_usage,
                    max_shard_size=max_shard_size,
                )
                if self.processor:
                    self.processor.save_pretrained(output_dir)
                console.print("[green]✓ Merged model saved in 4-bit precision[/green]")
                
                # Clean config for vLLM compatibility
                self._clean_merged_config(output_dir)
                
                # Aggressively free memory after successful merge
                console.print("[cyan]🧹 Cleaning up memory after merge...[/cyan]")
                self.model = None
                self.tokenizer = None
                self.processor = None
                _cleanup_memory_after_merge()
                console.print("[green]✓ Memory cleaned up[/green]")
            except Exception as e:
                console.print(f"[red]❌ 4-bit merge not supported: {e}[/red]")
                console.print("[yellow]Falling back to 16-bit merge[/yellow]")
                # Fall back to 16-bit
                self.save_model(output_dir, save_method="merged_16bit")
                return
        else:
            # For vision models, merging is more complex
            console.print("[yellow]⚠️  Merged saving for vision models not yet implemented[/yellow]")
            console.print("[yellow]⚠️  Saving LoRA adapters only[/yellow]")
            if self.model is None:
                raise RuntimeError("Model not loaded.")
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
