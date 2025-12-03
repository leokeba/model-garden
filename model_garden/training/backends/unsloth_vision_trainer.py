"""Vision-Language Model training using Unsloth.

Supports multimodal models like Qwen2.5-VL for fine-tuning on vision-language tasks.
"""

import gc
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

# Configure HuggingFace cache BEFORE importing HF libraries
from model_garden.utils.hf_cache import (
    configure_hf_cache,
    configure_pytorch_memory,
    configure_unsloth_settings,
    get_hf_token,
)

configure_hf_cache()
configure_pytorch_memory()
configure_unsloth_settings()

# CRITICAL: Import unsloth BEFORE any other ML libraries (datasets, transformers, trl, peft)
# This ensures Unsloth's PyTorch patches are applied correctly
# Now import other ML libraries AFTER unsloth
import torch
from datasets import Dataset, load_dataset
from PIL import Image
from rich.progress import Progress, SpinnerColumn, TextColumn
from unsloth import FastVisionModel  # FastVisionModel for vision-language models

# Import backend base class
from model_garden.training.backends.base import VisionTrainer
from model_garden.training.callbacks import MemoryMonitorCallback
from model_garden.training.chat_template import ChatTemplateDetector

# Import configuration dataclasses
from model_garden.training.config import VisionTrainingConfig
from model_garden.training.dataset_formats import DatasetFormatConverter

# Import extracted modules
from model_garden.training.lazy_dataset import LazyVisionDataset

# Import shared training mixin and utilities (consolidated location)
from model_garden.training.mixins import (
    TrainerMixin,
    cleanup_memory,
    detect_model_dtype,
    get_training_precision_config,
)
from model_garden.training.sft_trainer import FixedSFTTrainer

# Import centralized utilities
from model_garden.utils.console import console
from model_garden.utils.image import decode_base64_image, load_image


def _cleanup_memory_after_merge():
    """Clean up GPU and system memory after model merge.

    This is a thin wrapper around cleanup_memory() for backwards compatibility.
    Consider using cleanup_memory() directly from model_garden.training.mixins.
    """
    cleanup_memory()


class VisionLanguageTrainer(TrainerMixin, VisionTrainer):
    """Handles vision-language model fine-tuning.

    This trainer uses Unsloth's FastVisionModel for optimized vision-language
    model training with selective loss masking support.

    Inherits shared functionality from TrainerMixin:
    - Carbon tracking (_start_carbon_tracking, _stop_carbon_tracking)
    - Memory management (cleanup_memory)
    - Precision detection (detect_model_dtype, get_training_precision_config)
    """

    def __init__(
        self,
        base_model: str,
        max_seq_length: int = 16384,  # Increased from 2048 to fit vision tokens + full responses
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the vision-language trainer.

        Args:
            base_model: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-VL-3B-Instruct")
            max_seq_length: Maximum sequence length - default 16384 for vision models with large images
            load_in_4bit: Whether to load model in 4-bit quantization (memory efficient, ~95 percent quality)
            load_in_8bit: Whether to load model in 8-bit quantization (balanced, ~98 percent quality, 2x memory vs 4-bit)
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

        # Warning callback for logging issues (e.g., image loading failures)
        # Set this to send warnings to WebSocket/UI when running via API
        self.warning_callback: Callable[[str], None] | None = None

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
        hf_token = get_hf_token()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(description="Loading model...", total=None)

            # Use FastVisionModel for vision-language models (optimized for VLMs)
            # Supports both 4-bit and 8-bit quantization
            # Note: For 16-bit, set both load_in_4bit and load_in_8bit to False
            # IMPORTANT: FastVisionModel returns (model, tokenizer) tuple, NOT (model, processor)
            # We need to load the processor separately from transformers
            from transformers import AutoProcessor

            self.model, self.tokenizer = FastVisionModel.from_pretrained(
                model_name=self.base_model,
                max_seq_length=self.max_seq_length,
                dtype=self.dtype,
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                token=hf_token,
            )
            # Load processor separately (vision models need both tokenizer and processor)
            self.processor = AutoProcessor.from_pretrained(self.base_model, token=hf_token)

        console.print("[green]✓[/green] Model loaded with Unsloth FastVisionModel optimizations")

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
        finetune_vision_layers: bool = True,
        finetune_language_layers: bool = True,
        finetune_attention_modules: bool = True,
        finetune_mlp_modules: bool = True,
    ) -> None:
        """Prepare model for LoRA fine-tuning with selective layer control.

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
            finetune_vision_layers: Whether to fine-tune vision encoder layers (for multimodal)
            finetune_language_layers: Whether to fine-tune language model layers
            finetune_attention_modules: Whether to fine-tune attention layers
            finetune_mlp_modules: Whether to fine-tune MLP layers
        """
        console.print("[cyan]Configuring LoRA adapters for vision-language model...[/cyan]")

        # Log selective fine-tuning choices
        layers_info = []
        if finetune_vision_layers:
            layers_info.append("vision")
        if finetune_language_layers:
            layers_info.append("language")
        if finetune_attention_modules:
            layers_info.append("attention")
        if finetune_mlp_modules:
            layers_info.append("MLP")
        console.print(
            f"[cyan]Fine-tuning layers: {', '.join(layers_info) if layers_info else 'none'}[/cyan]"
        )

        # Workaround: 8-bit quantization has compatibility issues with gradient checkpointing
        # due to torch compile + bitsandbytes interactions. Disable gradient checkpointing for 8-bit.
        if self.load_in_8bit and use_gradient_checkpointing not in [False, "false"]:
            console.print(
                "[yellow]⚠️  8-bit quantization detected - disabling gradient checkpointing to avoid compatibility issues[/yellow]"
            )
            console.print(
                "[yellow]    (8-bit + gradient checkpointing causes torch compile errors)[/yellow]"
            )
            use_gradient_checkpointing = False

        try:
            # Use FastVisionModel.get_peft_model for selective layer fine-tuning
            # This gives fine-grained control over which parts of the model to train
            # NOTE: When using selective layer flags, don't pass target_modules - FastVisionModel handles it
            self.model = FastVisionModel.get_peft_model(
                self.model,
                finetune_vision_layers=finetune_vision_layers,
                finetune_language_layers=finetune_language_layers,
                finetune_attention_modules=finetune_attention_modules,
                finetune_mlp_modules=finetune_mlp_modules,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=lora_bias,
                use_gradient_checkpointing=use_gradient_checkpointing,  # type: ignore
                random_state=random_state,
                use_rslora=use_rslora,
                loftq_config=loftq_config,
            )
            console.print("[green]✓[/green] LoRA adapters configured with FastVisionModel")

        except Exception as e:
            # Fall back to PEFT for vision models (without selective layer control)
            console.print(f"[yellow]⚠️  FastVisionModel.get_peft_model failed: {e}[/yellow]")
            console.print(
                "[yellow]Using PEFT for LoRA configuration (selective layer fine-tuning not available)[/yellow]"
            )
            from peft import LoraConfig, get_peft_model

            if target_modules is None:
                target_modules = ["q_proj", "v_proj"]  # Minimal for vision models

            peft_config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias=cast(
                    Literal["none", "all", "lora_only"],
                    lora_bias if lora_bias in ["none", "all", "lora_only"] else "none",
                ),
                task_type=task_type,
            )

            if self.model is None:
                raise RuntimeError("Model not loaded. Call load_model() first.") from None
            self.model = get_peft_model(self.model, peft_config)  # type: ignore
            console.print("[green]✓[/green] LoRA adapters configured (PEFT fallback)")

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

    def load_dataset_from_hub(self, dataset_name: str, split: str = "train", **kwargs) -> Dataset:
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
        hf_token = get_hf_token()

        try:
            # Check if dataset_name includes a specific file
            if "::" in dataset_name:
                repo_name, file_name = dataset_name.split("::", 1)
                console.print(
                    f"[cyan]Loading dataset from HuggingFace Hub: {repo_name} (file: {file_name})[/cyan]"
                )

                # Load specific file from the repo
                dataset = load_dataset(
                    repo_name, data_files=file_name, split="train", token=hf_token, **kwargs
                )
            else:
                console.print(
                    f"[cyan]Loading dataset from HuggingFace Hub: {dataset_name} (split: {split})[/cyan]"
                )

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
        self, dataset_path: str, from_hub: bool = False, split: str = "train", **kwargs
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
            return decode_base64_image(image_str)
        except ValueError as e:
            console.print(f"[yellow]⚠️  {e}[/yellow]")
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
        # Use centralized image loading with fallback warning
        result, success = load_image(image_data, fallback_size=(224, 224), convert_to_rgb=True)

        # Warn if loading failed
        if not success:
            # Show the actual path/value that couldn't be loaded
            display_value = (
                str(image_data)[:100] + "..." if len(str(image_data)) > 100 else str(image_data)
            )
            warning_msg = f"Could not load image: {display_value}"
            console.print(f"[yellow]⚠️  {warning_msg}[/yellow]")

            # Send warning to UI via callback if set
            if self.warning_callback is not None:
                try:
                    self.warning_callback(warning_msg)
                except Exception:
                    pass  # Don't let callback errors break training

        return result

    def _convert_messages_to_simple_format(self, messages: list[dict]) -> dict[str, str | None]:
        """Convert OpenAI messages format to simple format.

        Delegates to DatasetFormatConverter.convert_messages_to_simple().

        Args:
            messages: List of OpenAI-style messages

        Returns:
            Dict with 'text', 'image', 'response', and 'system' keys
        """
        return DatasetFormatConverter.convert_messages_to_simple(messages)

    def _detect_chat_markers(self, processor) -> tuple[str, str]:
        """Detect instruction and response markers from tokenizer's chat template.

        Delegates to the ChatTemplateDetector module for actual detection.
        Kept as a method for backwards compatibility and easy access from
        VisionLanguageTrainer instances.

        Args:
            processor: The model's processor (contains tokenizer)

        Returns:
            Tuple of (instruction_marker, response_marker)

        Example:
            >>> instruction, response = trainer._detect_chat_markers(processor)
            >>> print(f"User: {instruction}, Assistant: {response}")
            User: <|im_start|>user, Assistant: <|im_start|>assistant
        """
        detector = ChatTemplateDetector(verbose=True)
        return detector.detect(processor)

    def _detect_vqa_format(self, example: dict) -> bool:
        """Detect if example uses VQA format (question + answer/answers).

        Delegates to DatasetFormatConverter.detect_vqa_format().
        """
        return DatasetFormatConverter.detect_vqa_format(example)

    def _convert_vqa_to_simple(self, example: dict) -> dict[str, Any]:
        """Convert VQA-style formats to simple format.

        Delegates to DatasetFormatConverter.convert_vqa_to_simple().
        """
        return DatasetFormatConverter.convert_vqa_to_simple(example)

    def format_dataset(
        self,
        dataset: Dataset,
        text_field: str = "text",
        image_field: str = "image",
        system_message: str | None = None,
        messages_field: str | None = None,
        lazy_loading: bool = False,
    ) -> list | LazyVisionDataset:
        """Format dataset for vision-language training using OpenAI message format.

        Supports multiple input formats:

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

        3. VQA formats (auto-detected):
           - ScienceQA: {question, choices, answer (index), solution, image}
           - Generic VQA: {question, answers (list), image}
           - DocVQA: {question, answers, image}

        Note: OpenAI messages and VQA formats are automatically converted to simple format
        for compatibility with UnslothVisionDataCollator.

        Args:
            dataset: Input dataset
            text_field: Field name for text/questions (for simple format)
            image_field: Field name for images (for simple format)
            system_message: Optional system message (for simple format)
            messages_field: Field name for messages (for OpenAI format, default: "messages")
            lazy_loading: If True, return a LazyVisionDataset that loads images on-demand.
                         Recommended for large datasets (1000+ images) to prevent memory exhaustion.

        Returns:
            List of formatted message dictionaries (if lazy_loading=False) or
            LazyVisionDataset instance (if lazy_loading=True)
        """
        console.print("[cyan]Formatting vision-language dataset...[/cyan]")

        if system_message is None:
            system_message = "You are a helpful assistant that can analyze images."

        # Check if dataset uses OpenAI messages format
        if messages_field is None:
            messages_field = "messages"

        has_messages_field = messages_field in dataset.column_names

        # Check if dataset uses VQA format (check first example)
        first_example = dataset[0] if len(dataset) > 0 else {}
        is_vqa_format = self._detect_vqa_format(first_example) if first_example else False

        if is_vqa_format:
            console.print(
                "[yellow]✓ Detected VQA format - will auto-convert (question/answer/image)[/yellow]"
            )

        # For lazy loading, extract metadata without loading images
        if lazy_loading:
            console.print(
                "[cyan]📦 Using lazy loading - images will be loaded on-demand during training[/cyan]"
            )
            examples_metadata = []

            for example in dataset:
                if not isinstance(example, dict):
                    continue
                example_dict = example

                if is_vqa_format:
                    simple = self._convert_vqa_to_simple(example_dict)
                    examples_metadata.append(
                        {
                            "text": simple.get("text", ""),
                            "image": simple.get("image"),
                            "response": simple.get("response", ""),
                            "system": system_message,
                        }
                    )
                elif has_messages_field and messages_field in example_dict:
                    messages = example_dict[messages_field]
                    simple = self._convert_messages_to_simple_format(messages)
                    original_system = simple.get("system", "")
                    examples_metadata.append(
                        {
                            "text": simple.get("text", ""),
                            "image": simple.get("image"),
                            "response": simple.get("response", ""),
                            "system": original_system if original_system else system_message,
                        }
                    )
                else:
                    examples_metadata.append(
                        {
                            "text": example_dict.get(text_field, ""),
                            "image": example_dict.get(image_field, ""),
                            "response": example_dict.get(
                                "response", example_dict.get("output", "")
                            ),
                            "system": system_message,
                        }
                    )

            console.print(
                f"[green]✓[/green] Dataset prepared for lazy loading ({len(examples_metadata)} examples)"
            )
            console.print("[cyan]   Memory saved: Images will be loaded one batch at a time[/cyan]")

            return LazyVisionDataset(
                examples=examples_metadata,
                system_message=system_message,
                image_loader=self._load_image,
            )

        # Original eager loading behavior
        formatted_data = []

        for example in dataset:
            # Ensure example is a dict-like object
            if isinstance(example, dict):
                example_dict = example
            else:
                # Handle list case (shouldn't happen with proper datasets)
                continue

            if is_vqa_format:
                # VQA format - convert to simple format first
                simple = self._convert_vqa_to_simple(example_dict)
                text = simple.get("text", "")
                response = simple.get("response", "")
                pil_image = self._load_image(simple.get("image"))

                # Format as OpenAI messages
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

            elif has_messages_field and messages_field in example_dict:
                # OpenAI messages format - convert to simple format first
                console.print(
                    "[yellow]Converting OpenAI messages format to simple format for compatibility...[/yellow]"
                ) if len(formatted_data) == 0 else None

                messages = example_dict[messages_field]
                simple = self._convert_messages_to_simple_format(messages)

                # Now process as simple format
                text = simple.get("text", "")
                response = simple.get("response", "")
                image_data = simple.get("image", "")
                # Use the original system message from the dataset, fallback to default if none
                original_system = simple.get("system", "")
                effective_system_message = original_system if original_system else system_message

                # Load image (handles base64, file paths, etc.)
                pil_image = self._load_image(image_data)

                # Format as OpenAI messages (simple structure)
                formatted_messages = {
                    "messages": [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": effective_system_message}],
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
        dataset: Dataset | list[dict] | LazyVisionDataset,
        config: VisionTrainingConfig,
        job_id: str | None = None,
        enable_carbon_tracking: bool = True,
        callbacks: list | None = None,
        eval_dataset: Dataset | list[dict] | LazyVisionDataset | None = None,
    ) -> None:
        """Train the vision-language model.

        Args:
            dataset: Training dataset (Dataset, list of formatted messages, or LazyVisionDataset)
            config: Vision training configuration (hyperparameters, output directory,
                    selective loss settings, etc.)
            job_id: Optional job identifier for carbon tracking
            enable_carbon_tracking: Whether to track carbon emissions
            callbacks: Optional list of TrainerCallback instances
            eval_dataset: Optional validation dataset for evaluation

        Example:
            >>> config = VisionTrainingConfig(
            ...     output_dir="./models/vision-model",
            ...     num_epochs=3,
            ...     selective_loss=True,
            ...     selective_loss_level="aggressive",
            ...     lazy_loading=True  # Recommended for large datasets
            ... )
            >>> trainer.train(dataset, config)
        """
        console.print("[bold cyan]Starting vision-language model training...[/bold cyan]")

        # Log lazy loading status
        if isinstance(dataset, LazyVisionDataset):
            console.print(
                f"[cyan]📦 Using lazy-loaded dataset ({len(dataset)} examples) - memory efficient mode[/cyan]"
            )
        elif config.lazy_loading:
            console.print(
                "[yellow]⚠️  lazy_loading=True in config but dataset is not a LazyVisionDataset[/yellow]"
            )
            console.print(
                "[yellow]    Use format_dataset(..., lazy_loading=True) to enable lazy loading[/yellow]"
            )

        # Note: Using DataLoader workers with vision models can be tricky
        if config.dataloader_num_workers > 0:
            console.print(
                f"[yellow]⚠️  INFO: Using {config.dataloader_num_workers} DataLoader workers[/yellow]"
            )
            console.print(
                "[yellow]   Multiple workers can improve throughput but use more memory[/yellow]"
            )
            console.print(
                "[yellow]   If you experience issues, try setting dataloader_num_workers=0[/yellow]"
            )

        # Start carbon tracking (uses mixin helper)
        if enable_carbon_tracking:
            self._start_carbon_tracking(config.output_dir, job_id, "vision-training")

        from trl.trainer.sft_config import SFTConfig
        from unsloth.trainer import UnslothVisionDataCollator

        # For vision models, keep data as list or LazyVisionDataset - don't convert to HF Dataset
        # The UnslothVisionDataCollator expects PIL Images which don't survive PyArrow serialization
        if isinstance(dataset, (list, LazyVisionDataset)):
            console.print(f"[cyan]Using formatted data directly ({len(dataset)} examples)[/cyan]")
            train_dataset = dataset
        else:
            train_dataset = dataset

        # Handle eval dataset similarly
        if isinstance(eval_dataset, list):
            console.print(f"[cyan]Using validation dataset ({len(eval_dataset)} examples)[/cyan]")

        # Set evaluation steps
        eval_steps_value = config.eval_steps if config.eval_steps is not None else config.save_steps

        # When using max_steps, still need to provide num_train_epochs
        use_max_steps = config.max_steps > 0

        # Set evaluation strategy if validation dataset provided
        final_eval_strategy = config.eval_strategy if eval_dataset is not None else "no"

        # Determine if we should load best model at end
        final_load_best = config.load_best_model_at_end and eval_dataset is not None
        final_metric = config.metric_for_best_model if eval_dataset is not None else None

        # Build training args - SFTConfig has different parameters than TrainingArguments
        # Detect model's actual dtype to set precision correctly
        model_dtype = detect_model_dtype(self.model, self.load_in_4bit, self.load_in_8bit)
        precision_config = get_training_precision_config(
            self.model, self.load_in_4bit, self.load_in_8bit
        )

        # Log detected dtype for debugging
        console.print(f"[cyan]🔍 Detected model dtype: {model_dtype}[/cyan]")
        console.print(
            f"[cyan]📊 Training precision: {'bf16' if precision_config['bf16'] else 'fp16'}[/cyan]"
        )

        training_args_dict = {
            "output_dir": config.output_dir,
            "per_device_train_batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "warmup_steps": config.warmup_steps,
            "max_steps": config.max_steps if use_max_steps else -1,
            "num_train_epochs": 1.0 if use_max_steps else config.num_epochs,
            "learning_rate": config.learning_rate,
            # Precision settings: Match the model's actual dtype using shared utilities
            "fp16": precision_config["fp16"],
            "bf16": precision_config["bf16"],
            "logging_steps": config.logging_steps,
            "optim": config.optim,
            "weight_decay": config.weight_decay,
            "lr_scheduler_type": config.lr_scheduler_type,
            "max_grad_norm": config.max_grad_norm,
            "adam_beta1": config.adam_beta1,
            "adam_beta2": config.adam_beta2,
            "adam_epsilon": config.adam_epsilon,
            "dataloader_num_workers": config.dataloader_num_workers,
            "dataloader_pin_memory": config.dataloader_pin_memory,
            "seed": 42,
            "save_steps": config.save_steps,
            "save_total_limit": config.save_total_limit,
            "report_to": "none",
            # CRITICAL for vision models - Unsloth requirements:
            "remove_unused_columns": False,
            "dataset_text_field": "",
            "dataset_kwargs": {"skip_prepare_dataset": True},
            # NOTE: Prompt masking is handled by the data_collator (UnslothVisionDataCollator)
            # with train_on_responses_only=True, not by TrainingArguments.
            # See the collator initialization below for the actual masking configuration.
        }

        # Add evaluation settings if validation dataset provided
        if eval_dataset is not None:
            training_args_dict["per_device_eval_batch_size"] = config.batch_size
            training_args_dict["do_eval"] = True
            training_args_dict["eval_strategy"] = final_eval_strategy
            training_args_dict["eval_steps"] = eval_steps_value
            training_args_dict["load_best_model_at_end"] = final_load_best
            training_args_dict["metric_for_best_model"] = final_metric

        training_args = SFTConfig(**training_args_dict)

        console.print(
            "[cyan]ℹ️  Vision training uses UnslothVisionDataCollator for efficient image processing[/cyan]"
        )

        # Auto-detect chat markers from the model's tokenizer
        console.print("[cyan]🔍 Auto-detecting chat template markers...[/cyan]")
        instruction_marker, response_marker = self._detect_chat_markers(self.processor)

        # Choose data collator based on selective_loss flag
        if config.selective_loss:
            # Lazy import to avoid spawning torch compile workers at module import time
            from model_garden.training.selective_loss import create_selective_loss_collator

            console.print(
                f"[cyan]🎯 Using selective loss masking (level: {config.selective_loss_level})[/cyan]"
            )
            console.print(f"[cyan]   Strategy: {config.selective_loss_masking_strategy}[/cyan]")
            if (
                config.selective_loss_masking_strategy == "epoch_based"
                and config.selective_loss_masking_start_epoch > 0.0
            ):
                console.print(
                    f"[yellow]   ⏱️  Masking delayed until epoch {config.selective_loss_masking_start_epoch}[/yellow]"
                )
            elif config.selective_loss_masking_strategy == "alternating":
                console.print(
                    f"[yellow]   🔄 Alternating: ON for {config.selective_loss_mask_for_n_steps}/{config.selective_loss_mask_every_n_steps} steps per cycle[/yellow]"
                )
            elif config.selective_loss_masking_strategy == "weighted":
                console.print(
                    f"[yellow]   ⚖️  Weighted: structural tokens weight = {config.selective_loss_structural_weight}[/yellow]"
                )

            data_collator = create_selective_loss_collator(
                model=self.model,
                processor=self.processor,
                mask_level=config.selective_loss_level,
                schema_keys=config.selective_loss_schema_keys,
                dataset=train_dataset,  # Pass dataset for auto-detection
                masking_strategy=config.selective_loss_masking_strategy,
                masking_start_epoch=config.selective_loss_masking_start_epoch,
                mask_every_n_steps=config.selective_loss_mask_every_n_steps,
                mask_for_n_steps=config.selective_loss_mask_for_n_steps,
                structural_weight=config.selective_loss_structural_weight,
                verbose=config.selective_loss_verbose,
                train_on_responses_only=True,  # Enable prompt masking
                instruction_part=instruction_marker,  # Auto-detected from tokenizer
                response_part=response_marker,  # Auto-detected from tokenizer
            )
        else:
            # Use standard Unsloth collator with prompt masking enabled
            # Chat markers are automatically detected from the tokenizer's template
            #
            # CRITICAL: force_match=False is essential for vision models!
            # Vision tokens can interfere with marker detection. With force_match=True,
            # if markers aren't found, ALL tokens get masked (causing NaN loss).
            # With force_match=False, if markers aren't found, NO masking occurs (safer fallback).
            data_collator = UnslothVisionDataCollator(
                self.model,
                self.processor,
                train_on_responses_only=True,
                instruction_part=instruction_marker,  # Auto-detected from tokenizer
                response_part=response_marker,  # Auto-detected from tokenizer
                force_match=False,  # Don't mask everything if markers not found
            )

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

        console.print(
            "[cyan]💡 Memory monitoring enabled: Tracking RAM usage every 10 steps[/cyan]"
        )

        # CRITICAL: Warn about max_seq_length for vision models
        console.print(f"[yellow]📏 Max sequence length: {self.max_seq_length} tokens[/yellow]")
        if self.max_seq_length < 4096:
            console.print(
                f"[red]⚠️  WARNING: max_seq_length ({self.max_seq_length}) may be too small for vision models![/red]"
            )
            console.print(
                "[red]   Images can use 1500+ tokens, leaving little room for prompts/responses.[/red]"
            )
            console.print(
                "[red]   If you see 'ALL tokens masked' errors, increase max_seq_length to 16384+[/red]"
            )

        # Choose trainer based on masking strategy
        if config.selective_loss and config.selective_loss_masking_strategy == "weighted":
            # Use WeightedLossTrainer for weighted masking
            from model_garden.training.weighted_loss import WeightedLossTrainer

            console.print("[cyan]🎯 Using WeightedLossTrainer for weighted masking strategy[/cyan]")

            trainer = WeightedLossTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,  # type: ignore
                eval_dataset=eval_dataset,  # type: ignore
                data_collator=data_collator,
                callbacks=all_callbacks,
                tokenizer=self.tokenizer,  # type: ignore
                verbose_loss=config.selective_loss_verbose,
            )
        else:
            # Use our fixed SFTTrainer that passes num_items_in_batch during evaluation
            trainer = FixedSFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,  # type: ignore
                args=training_args,
                train_dataset=train_dataset,  # type: ignore
                eval_dataset=eval_dataset,  # type: ignore
                data_collator=data_collator,
                callbacks=all_callbacks,
            )

        # Link trainer to data collator for epoch-based masking
        if config.selective_loss:
            from model_garden.training.selective_loss import SelectiveLossVisionCollator

            if isinstance(data_collator, SelectiveLossVisionCollator):
                data_collator.set_trainer(trainer)

        console.print("[cyan]Training in progress...[/cyan]")
        trainer.train()
        console.print("[bold green]✨ Training completed![/bold green]")

        # Print selective loss statistics if enabled
        if config.selective_loss:
            from model_garden.training.selective_loss import SelectiveLossVisionCollator

            if isinstance(data_collator, SelectiveLossVisionCollator):
                data_collator.print_stats()

        # Stop carbon tracking (uses mixin helper)
        self._stop_carbon_tracking()

        # CRITICAL: Explicitly clear dataset references from trainer to enable garbage collection
        # Vision models keep PIL images in RAM which can accumulate across multiple training runs
        console.print("[cyan]🧹 Clearing dataset references from trainer...[/cyan]")
        try:
            if hasattr(trainer, "train_dataset"):
                trainer.train_dataset = None  # type: ignore
            if hasattr(trainer, "eval_dataset"):
                trainer.eval_dataset = None  # type: ignore
            if hasattr(trainer, "data_collator"):
                trainer.data_collator = None  # type: ignore

            # Clear lazy dataset caches if using LazyVisionDataset
            if isinstance(train_dataset, LazyVisionDataset):
                train_dataset.clear_cache()
            if isinstance(eval_dataset, LazyVisionDataset):
                eval_dataset.clear_cache()
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
            with open(config_path) as f:
                config = json.load(f)

            # Backup original
            backup_path = Path(output_dir) / "config.json.backup"
            with open(backup_path, "w") as f:
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
                with open(config_path, "w") as f:
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
            console.print(
                f"[cyan]Memory settings: max_usage={maximum_memory_usage}, shard_size={max_shard_size}[/cyan]"
            )
            try:
                # Use FastVisionModel for vision-language model merging
                # Clear GPU cache before merging to free up memory
                import gc

                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    console.print("[cyan]💾 GPU memory cleared before merge[/cyan]")

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
                # self._clean_merged_config(output_dir)

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
                    console.print(
                        "[green]✓ Model merged and saved successfully (PEFT fallback)[/green]"
                    )

                    # Clean config for vLLM compatibility
                    # self._clean_merged_config(output_dir)

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
                        raise RuntimeError("Model not loaded.") from None
                    self.model.save_pretrained(output_dir)
                    if self.tokenizer:
                        self.tokenizer.save_pretrained(output_dir)
                    if self.processor:
                        self.processor.save_pretrained(output_dir)
        elif save_method == "merged_4bit":
            # Merge LoRA weights and save in 4-bit
            console.print("[cyan]Merging LoRA weights and saving in 4-bit...[/cyan]")
            console.print(
                f"[cyan]Memory settings: max_usage={maximum_memory_usage}, shard_size={max_shard_size}[/cyan]"
            )
            console.print(
                "[yellow]⚠️  Warning: 4-bit merge may reduce accuracy for GGUF conversion[/yellow]"
            )
            try:
                # Use FastVisionModel for vision-language model merging
                # Clear GPU cache before merging
                import gc

                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    console.print("[cyan]💾 GPU memory cleared before merge[/cyan]")

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
                # self._clean_merged_config(output_dir)

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


def merge_vision_lora_adapter(
    adapter_path: str,
    output_dir: str,
    base_model: str | None = None,
    max_seq_length: int = 16384,
    load_in_4bit: bool = True,
) -> str:
    """Merge a vision LoRA adapter with its base model for inference.

    This function is specifically for preparing vision-language model adapters for vLLM inference,
    which doesn't support LoRA adapters on vision models. The adapter is loaded, merged with its
    base model, and saved as a complete model.

    Uses the standard transformers + PEFT approach for maximum compatibility.

    Args:
        adapter_path: Path to the LoRA adapter directory or HuggingFace model ID
        output_dir: Directory to save the merged model
        base_model: Optional base model path (auto-detected from adapter_config.json if not provided)
        max_seq_length: Maximum sequence length (unused, kept for API compatibility)
        load_in_4bit: Load base model in 4-bit for merging (reduces memory usage)

    Returns:
        Path to the merged model directory

    Raises:
        FileNotFoundError: If adapter or base model not found
        ValueError: If adapter_config.json doesn't contain base model info
    """
    console.print("[bold cyan]Merging vision LoRA adapter for inference...[/bold cyan]")
    console.print(f"[cyan]Adapter: {adapter_path}[/cyan]")

    # Check if adapter exists (local path)
    adapter_dir = Path(adapter_path)
    is_local = adapter_dir.exists()

    # Get base model from adapter config if not provided
    if base_model is None:
        console.print("[cyan]🔍 Detecting base model from adapter_config.json...[/cyan]")

        try:
            if is_local:
                adapter_config_file = adapter_dir / "adapter_config.json"
                if not adapter_config_file.exists():
                    raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")

                with open(adapter_config_file) as f:
                    adapter_config = json.load(f)
                    base_model = adapter_config.get("base_model_name_or_path")
            else:
                # HuggingFace model ID
                from huggingface_hub import hf_hub_download

                hf_token = get_hf_token()

                config_file = hf_hub_download(
                    repo_id=adapter_path, filename="adapter_config.json", token=hf_token
                )

                with open(config_file) as f:
                    adapter_config = json.load(f)
                    base_model = adapter_config.get("base_model_name_or_path")

            if not base_model:
                raise ValueError(
                    "Could not find base_model_name_or_path in adapter_config.json. "
                    "Please specify base_model explicitly."
                )

            console.print(f"[green]✓ Found base model: {base_model}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to detect base model: {e}[/red]")
            raise

    console.print(f"[cyan]Base model: {base_model}[/cyan]")
    console.print(f"[cyan]Output: {output_dir}[/cyan]")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load base model and adapter
        console.print("[cyan]Loading base model...[/cyan]")
        hf_token = get_hf_token()

        # Use transformers AutoModelForVision2Seq instead of FastVisionModel for merging
        # This is more reliable for vision-language models
        import shutil

        from huggingface_hub import snapshot_download
        from peft import PeftModel
        from transformers import AutoModelForVision2Seq

        console.print("[cyan]Using transformers AutoModelForVision2Seq for reliable merging[/cyan]")

        # Load base model with transformers
        base_torch_dtype = torch.bfloat16 if not load_in_4bit else None

        base_model_obj = AutoModelForVision2Seq.from_pretrained(
            base_model,
            torch_dtype=base_torch_dtype,
            load_in_4bit=load_in_4bit,
            device_map="auto",
            token=hf_token,
        )

        console.print("[green]✓ Base model loaded[/green]")
        console.print(f"[cyan]Loading LoRA adapter from {adapter_path}...[/cyan]")

        # Load LoRA adapter with PEFT
        from typing import Any

        peft_model: Any = PeftModel.from_pretrained(base_model_obj, adapter_path, token=hf_token)

        console.print("[green]✓ LoRA adapter loaded[/green]")

        # Merge and save
        console.print("[cyan]Merging adapter into base model...[/cyan]")

        # Clear GPU cache before merging
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Merge the adapter
        merged_model: Any = peft_model.merge_and_unload()
        console.print("[green]✓ Merge complete![/green]")

        # Save merged model. Prefer Unsloth's memory-aware merged save if available
        console.print(f"[cyan]Saving merged model to {output_dir}...[/cyan]")
        # Try calling Unsloth's save directly (use the library function) so we don't
        # rely on an instance method being monkey-patched onto the model returned
        # by PEFT's merge. This mirrors training where Unsloth's memory-aware
        # saver is used to produce `merged_16bit` artifacts.
        try:
            console.print(
                "[cyan]POST_MERGE: Attempting Unsloth unsloth_save_pretrained_merged(...) (memory-aware, full-rewrite requested)...[/cyan]"
            )
            try:
                # Import the helper directly from the installed unsloth package
                from unsloth.save import unsloth_save_pretrained_merged

                # Use a temporary location inside the output dir to ensure
                # the rewrite happens on the same filesystem (reduces risk of
                # fallback in-place behavior that can preserve quantization auxiliaries).
                temp_loc = str(Path(output_dir) / "_unsloth_temporary_saved_buffers")
                os.makedirs(temp_loc, exist_ok=True)

                # Decide save_method: if base was loaded in 4-bit we should
                # preserve/produce a 4-bit (bitsandbytes) serialized model so
                # vLLM uses its bitsandbytes loader and benefits from lower
                # GPU memory usage. If not, fall back to merged_16bit.
                chosen_save_method = "merged_4bit_forced" if load_in_4bit else "merged_16bit"

                unsloth_save_pretrained_merged(
                    merged_model,
                    save_directory=output_dir,
                    tokenizer=None,
                    save_method=chosen_save_method,
                    push_to_hub=False,
                    token=None,
                    is_main_process=True,
                    state_dict=None,
                    save_function=__import__("torch").save,
                    max_shard_size="5GB",
                    safe_serialization=True,
                    variant=None,
                    save_peft_format=True,
                    tags=[],
                    temporary_location=temp_loc,
                    maximum_memory_usage=0.95,
                )
                console.print(
                    f"[green]✓ Unsloth unsloth_save_pretrained_merged succeeded (full rewrite, method={chosen_save_method})[/green]"
                )
            except Exception as e:
                console.print(
                    f"[yellow]POST_MERGE: unsloth_save_pretrained_merged failed: {e}\nFalling back to regular save_pretrained()[/yellow]"
                )
                merged_model.save_pretrained(output_dir)
                console.print("[green]✓ Model saved (regular save_pretrained)[/green]")
        except Exception as outer_e:
            console.print(
                f"[red]❌ Unexpected error while attempting Unsloth save: {outer_e}[/red]"
            )
            merged_model.save_pretrained(output_dir)
            console.print("[green]✓ Model saved (regular save_pretrained)[/green]")

        # Decide whether to keep quantization metadata in config.json.
        # If we intentionally saved a 4-bit/bitsandbytes artifact above (load_in_4bit==True)
        # we SHOULD PRESERVE the quantization metadata so vLLM will select its
        # bitsandbytes loader and avoid doing large FP16 allocations. If we saved
        # a merged_16bit artifact, keep current behavior (no quant metadata).
        config_path = Path(output_dir) / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)

                if load_in_4bit:
                    console.print(
                        "[cyan]Keeping quantization_config in config.json so vLLM will use bitsandbytes loader[/cyan]"
                    )
                else:
                    # Remove quantization_config at root level for 16-bit merges
                    modified = False
                    if "quantization_config" in config:
                        del config["quantization_config"]
                        modified = True
                        console.print(
                            "[green]✓ Removed root-level quantization_config (16-bit save)[/green]"
                        )
                    if "text_config" in config and isinstance(config["text_config"], dict):
                        if "quantization_config" in config["text_config"]:
                            del config["text_config"]["quantization_config"]
                            modified = True
                            console.print(
                                "[green]✓ Removed text_config quantization_config (16-bit save)[/green]"
                            )

                    if modified:
                        with open(config_path, "w") as f:
                            json.dump(config, f, indent=2)
                        console.print(
                            "[green]✓ Config cleaned for vLLM compatibility (16-bit)[/green]"
                        )
            except Exception as e:
                console.print(
                    f"[yellow]⚠️ Failed to inspect/modify config.json: {e} - continuing[/yellow]"
                )
        else:
            console.print("[yellow]⚠️  config.json not found, skipping config adjustments[/yellow]")

        # Save tokenizer (from base model)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
        tokenizer.save_pretrained(output_dir)
        console.print("[green]✓ Tokenizer saved[/green]")

        # Copy vision processor configuration files from base model
        console.print("[cyan]Copying vision processor configuration files...[/cyan]")
        base_model_dir = snapshot_download(base_model, token=hf_token)

        files_to_copy = [
            "preprocessor_config.json",
            "processor_config.json",
            "video_preprocessor_config.json",  # For video support
        ]

        for file in files_to_copy:
            src = os.path.join(base_model_dir, file)
            if os.path.exists(src):
                shutil.copy(src, output_dir)
                console.print(f"[green]  ✓ Copied {file}[/green]")

        # Copy image_processor directory if it exists
        image_processor_dir = os.path.join(base_model_dir, "image_processor")
        if os.path.exists(image_processor_dir):
            shutil.copytree(
                image_processor_dir, os.path.join(output_dir, "image_processor"), dirs_exist_ok=True
            )
            console.print("[green]  ✓ Copied image_processor directory[/green]")

        console.print("[green]✓ Processor configuration files copied[/green]")

        # Validate that the merge actually produced a valid model
        config_file = Path(output_dir) / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(
                f"Model merge completed but config.json not found in {output_dir}. "
                "The merge may have failed to save model files."
            )

        # Check for model weight files
        has_weights = (
            list(Path(output_dir).glob("*.safetensors"))
            or list(Path(output_dir).glob("*.bin"))
            or list(Path(output_dir).glob("model-*.safetensors"))
        )
        if not has_weights:
            raise FileNotFoundError(
                f"Model merge completed but no weight files (.safetensors or .bin) found in {output_dir}. "
                "The model may not have been saved properly."
            )

        console.print(
            f"[green]✓ Validation passed: Found config.json and {len(has_weights)} weight file(s)[/green]"
        )

        # Clean up memory
        console.print("[cyan]🧹 Cleaning up memory...[/cyan]")
        del peft_model
        del merged_model
        del base_model_obj
        del tokenizer
        _cleanup_memory_after_merge()

        console.print("[bold green]✨ Vision LoRA adapter merged successfully![/bold green]")
        return str(output_path.absolute())

    except Exception as e:
        console.print(f"[red]❌ Failed to merge adapter: {e}[/red]")
        raise


def create_vision_sample_dataset(output_path: str, num_examples: int = 10) -> None:
    """Create a sample vision-language dataset for testing.

    Args:
        output_path: Path to save the dataset
        num_examples: Number of examples to generate
    """
    console.print(
        f"[cyan]Creating sample vision-language dataset with {num_examples} examples...[/cyan]"
    )

    examples = []
    for i in range(num_examples):
        examples.append(
            {
                "text": "What is shown in this image? Please describe it.",
                "image": f"/path/to/image_{i}.jpg",  # Placeholder - user should provide actual images
                "response": f"This is a sample response for image {i}. In a real dataset, this would describe the actual image content.",
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    console.print(f"[green]✓[/green] Sample vision-language dataset created at {output_path}")
    console.print(
        "[yellow]⚠️  Note: Replace placeholder image paths with actual image files[/yellow]"
    )


def create_vision_trainer(
    base_model: str,
    max_seq_length: int = 16384,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    dtype: Any | None = None,
    backend: str = "unsloth",
) -> VisionTrainer:
    """Create a vision trainer using the specified backend.

    This is a convenience function that creates a vision trainer through the backend system.
    It allows for backend selection while maintaining backward compatibility.

    Args:
        base_model: HuggingFace model identifier
        max_seq_length: Maximum sequence length (larger for vision models)
        load_in_4bit: Whether to load model in 4-bit quantization
        load_in_8bit: Whether to load model in 8-bit quantization
        dtype: Data type (None for auto-detection)
        backend: Backend to use ('unsloth', etc.)

    Returns:
        A vision trainer instance

    Example:
        >>> trainer = create_vision_trainer("Qwen/Qwen2.5-VL-3B-Instruct", backend="unsloth")
        >>> trainer.load_model()
    """
    from model_garden.training.backends import get_backend

    backend_instance = get_backend(backend)
    return backend_instance.create_vision_trainer(
        base_model=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        dtype=dtype,
    )
