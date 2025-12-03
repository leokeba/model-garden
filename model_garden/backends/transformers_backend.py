"""Transformers training backend for Model Garden.

This backend provides standard HuggingFace Transformers + PEFT training without Unsloth optimizations.
Use this for maximum compatibility or when Unsloth doesn't support your model.
"""

from typing import Any

# Configure HuggingFace cache BEFORE importing HF libraries
from model_garden.utils.hf_cache import configure_hf_cache, configure_pytorch_memory

configure_hf_cache()
configure_pytorch_memory()

from datasets import Dataset
from peft import PeftModel
from rich.progress import Progress, SpinnerColumn, TextColumn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
)

from model_garden.backends.base import TextTrainer, TrainingBackend, VisionTrainer
from model_garden.training.config import TrainingConfig, VisionTrainingConfig
from model_garden.training.mixins import TrainerMixin
from model_garden.utils.console import console


class TransformersVisionTrainer(TrainerMixin, VisionTrainer):
    """Vision trainer using standard HuggingFace Transformers + PEFT for vision-language models.

    Note: This is a basic implementation that may not support all features of the Unsloth vision trainer.
    For production use with vision models, consider using the Unsloth backend for better performance.
    """

    def load_model(self) -> None:
        """Load the vision-language model using Transformers."""
        hf_token = self._get_hf_token()

        console.print(f"[cyan]Loading vision model: {self.base_model}[/cyan]")
        console.print("[yellow]Using HuggingFace Transformers (basic vision support)[/yellow]")
        console.print(f"[cyan]Precision: {self._get_precision_description()}[/cyan]")

        # Prepare model kwargs
        torch_dtype = self._get_torch_dtype()

        model_kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": self.base_model,
            "torch_dtype": torch_dtype,
            "device_map": "auto",
            "token": hf_token,
            "trust_remote_code": True,
        }

        # Add quantization if requested
        quant_config = self._get_quantization_config()
        if quant_config:
            model_kwargs["quantization_config"] = quant_config
        elif self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(description="Loading model...", total=None)

            # Load model - use AutoModelForVision2Seq for vision-language models
            from transformers import AutoModelForVision2Seq

            try:
                self.model = AutoModelForVision2Seq.from_pretrained(**model_kwargs)
            except Exception:
                # Fallback to generic AutoModel if specific class not available
                console.print(
                    "[yellow]⚠️  AutoModelForVision2Seq not available, trying AutoModel[/yellow]"
                )
                from transformers import AutoModel

                self.model = AutoModel.from_pretrained(**model_kwargs)

            # Load processor (handles both text and image processing)
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                self.base_model,
                token=hf_token,
                trust_remote_code=True,
            )
            self.tokenizer = self.processor.tokenizer

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
        finetune_vision_layers: bool = True,
        finetune_language_layers: bool = True,
        finetune_attention_modules: bool = True,
        finetune_mlp_modules: bool = True,
    ) -> None:
        """Prepare model for LoRA fine-tuning using PEFT."""
        console.print("[cyan]Configuring LoRA adapters for vision model (PEFT)...[/cyan]")
        # Note: finetune_* parameters are Unsloth-specific, not used in standard PEFT
        self._configure_lora_peft(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            use_rslora=use_rslora,
            lora_bias=lora_bias,
            task_type=task_type,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

    def load_dataset_from_file(self, dataset_path: str) -> Dataset:
        """Load multimodal dataset from a local file."""
        # Call the parent mixin method which handles all file formats
        return TrainerMixin.load_dataset_from_file(self, dataset_path)

    def load_dataset_from_hub(self, dataset_name: str, split: str = "train", **kwargs) -> Dataset:
        """Load multimodal dataset from HuggingFace Hub."""
        return TrainerMixin.load_dataset_from_hub(self, dataset_name, split=split, **kwargs)

    def format_dataset(
        self,
        dataset: Dataset,
        text_field: str = "text",
        image_field: str = "image",
        system_message: str | None = None,
        messages_field: str | None = None,
    ) -> list[dict]:
        """Format dataset for vision-language training.

        Note: This is a simplified implementation. For advanced features like selective loss
        and multiple formats, use the Unsloth backend.
        """
        console.print("[cyan]Formatting vision-language dataset...[/cyan]")

        if system_message is None:
            system_message = "You are a helpful assistant that can analyze images."

        formatted_data = []

        for example in dataset:
            if not isinstance(example, dict):
                continue

            # Simple format: {text, image, response}
            text = example.get(text_field, example.get("question", ""))
            response = example.get("response", example.get("answer", ""))
            image_path = example.get(image_field, "")

            # Load image
            from PIL import Image

            if isinstance(image_path, str):
                if image_path.startswith("data:image"):
                    # Base64 encoded
                    import base64
                    import io

                    image_data = image_path.split(",")[1]
                    pil_image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                else:
                    # File path
                    pil_image = Image.open(image_path).convert("RGB")
            else:
                pil_image = image_path  # Already a PIL Image

            # Format as messages
            formatted_data.append(
                {
                    "messages": [
                        {"role": "system", "content": [{"type": "text", "text": system_message}]},
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
            )

        console.print(f"[green]✓[/green] Formatted {len(formatted_data)} examples")
        return formatted_data

    def train(
        self,
        dataset: Dataset | list[dict],
        config: VisionTrainingConfig,
        job_id: str | None = None,
        enable_carbon_tracking: bool = True,
        callbacks: list | None = None,
        eval_dataset: Dataset | list[dict] | None = None,
    ) -> None:
        """Train the vision-language model.

        Note: This is a basic implementation. Advanced features like selective loss
        are not supported. For full feature support, use the Unsloth backend.
        """
        console.print("[cyan]Starting vision training with Transformers backend...[/cyan]")

        if config.selective_loss:
            console.print(
                "[yellow]⚠️  Selective loss not supported in Transformers backend[/yellow]"
            )

        # Set up carbon tracking
        if enable_carbon_tracking:
            self._start_carbon_tracking(config.output_dir, job_id, "vision-training")

        # Create training arguments
        training_args = self._create_training_args(
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
            remove_unused_columns=False,  # Important for vision models
        )

        all_callbacks = self._get_default_callbacks(callbacks)

        # Create a simple data collator for vision-language models
        def collate_fn(examples):
            """Simple collator using standard Transformers processor API."""
            assert self.processor is not None, "Processor must be loaded before training"

            batch_messages = []
            for example in examples:
                messages = example.get("messages", [])
                batch_messages.append(messages)

            # Use the processor's apply_chat_template if available
            if hasattr(self.processor, "apply_chat_template"):
                texts = self.processor.apply_chat_template(
                    batch_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )

                # Extract images from messages
                batch_images = []
                for messages in batch_messages:
                    for message in messages:
                        for content_item in message.get("content", []):
                            if content_item.get("type") == "image":
                                batch_images.append(content_item["image"])

                # Process with images
                if batch_images:
                    inputs = self.processor(
                        text=texts,
                        images=batch_images,
                        return_tensors="pt",
                        padding=True,
                    )
                else:
                    inputs = self.processor(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                    )
            else:
                # Fallback: manual text construction
                texts = []
                batch_images = []

                for messages in batch_messages:
                    text_parts = []
                    for message in messages:
                        role = message.get("role", "")
                        for content_item in message.get("content", []):
                            if content_item.get("type") == "text":
                                text = content_item.get("text", "")
                                text_parts.append(f"{role}: {text}")
                            elif content_item.get("type") == "image":
                                batch_images.append(content_item["image"])

                    texts.append("\n".join(text_parts))

                if batch_images:
                    inputs = self.processor(
                        text=texts,
                        images=batch_images,
                        return_tensors="pt",
                        padding=True,
                    )
                else:
                    inputs = self.processor(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                    )

            # Add labels for causal language modeling
            if "input_ids" in inputs:
                inputs["labels"] = inputs["input_ids"].clone()

            return inputs

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,  # type: ignore
            eval_dataset=eval_dataset if isinstance(eval_dataset, Dataset) else None,
            data_collator=collate_fn,
            callbacks=all_callbacks,
        )

        # Train
        trainer.train()
        console.print("[green]✓[/green] Training completed")

        # Stop carbon tracking
        self._stop_carbon_tracking()

        # Save the model after training
        console.print(f"[cyan]Saving model to: {config.output_dir}[/cyan]")
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(config.output_dir)
        else:
            self.model.save_pretrained(config.output_dir)
        self.processor.save_pretrained(config.output_dir)
        console.print("[green]✓[/green] Model saved successfully")

    def save_model(
        self,
        output_dir: str,
        save_method: str = "merged_16bit",
        maximum_memory_usage: float = 0.75,
        max_shard_size: str = "5GB",
    ) -> None:
        """Save the fine-tuned vision-language model."""
        self._save_model_merged(output_dir, save_method, max_shard_size)


class TransformersTextTrainer(TrainerMixin, TextTrainer):
    """Text trainer using standard HuggingFace Transformers + PEFT."""

    def load_model(self) -> None:
        """Load the base model using Transformers."""
        console.print(f"[cyan]Loading base model: {self.base_model}[/cyan]")
        console.print("[cyan]Using HuggingFace Transformers (no Unsloth optimizations)[/cyan]")
        console.print(f"[cyan]Precision: {self._get_precision_description()}[/cyan]")

        hf_token = self._get_hf_token()

        # Determine torch dtype
        if not self.load_in_4bit and not self.load_in_8bit:
            torch_dtype = self._get_torch_dtype()
        else:
            torch_dtype = None  # Let BitsAndBytes handle dtype

        # Build model loading kwargs
        model_kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": self.base_model,
            "torch_dtype": torch_dtype,
            "device_map": "auto",
            "token": hf_token,
            "trust_remote_code": True,
        }

        # Add quantization if requested
        quant_config = self._get_quantization_config()
        if quant_config:
            model_kwargs["quantization_config"] = quant_config
        elif self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(description="Loading model...", total=None)

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
        target_modules: list[str] | None = None,
        use_rslora: bool = False,
        lora_bias: str = "none",
        task_type: str = "CAUSAL_LM",
        use_gradient_checkpointing: str | bool = "unsloth",
        random_state: int = 42,
        loftq_config: dict | None = None,
    ) -> None:
        """Prepare model for LoRA fine-tuning using PEFT."""
        self._configure_lora_peft(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            use_rslora=use_rslora,
            lora_bias=lora_bias,
            task_type=task_type,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

    def load_dataset_from_file(self, dataset_path: str) -> Dataset:
        """Load dataset from a local file."""
        # Call the parent mixin method which handles all file formats
        return TrainerMixin.load_dataset_from_file(self, dataset_path)

    def load_dataset_from_hub(self, dataset_name: str, split: str = "train") -> Dataset:
        """Load dataset from HuggingFace Hub."""
        return TrainerMixin.load_dataset_from_hub(self, dataset_name, split=split)

    def format_dataset(
        self,
        dataset: Dataset,
        instruction_field: str = "instruction",
        input_field: str = "input",
        output_field: str = "output",
        prompt_template: str | None = None,
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
        config: TrainingConfig,
        job_id: str | None = None,
        enable_carbon_tracking: bool = True,
        callbacks: list | None = None,
        eval_dataset: Dataset | None = None,
    ) -> None:
        """Train the model using Transformers Trainer."""
        console.print("[cyan]Starting training with Transformers backend...[/cyan]")

        # Set up carbon tracking
        if enable_carbon_tracking:
            self._start_carbon_tracking(config.output_dir, job_id, "training")

        # Create training arguments
        training_args = self._create_training_args(
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
        )

        all_callbacks = self._get_default_callbacks(callbacks)

        # Tokenize datasets
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
            )

        tokenized_train = dataset.map(
            tokenize_function, batched=True, remove_columns=dataset.column_names
        )
        tokenized_eval = None
        if eval_dataset is not None:
            tokenized_eval = eval_dataset.map(
                tokenize_function, batched=True, remove_columns=eval_dataset.column_names
            )

        # Create data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
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
        self._stop_carbon_tracking()

        # Save final model
        console.print(f"[cyan]Saving model to: {config.output_dir}[/cyan]")
        trainer.save_model(config.output_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(config.output_dir)
        console.print("[green]✓[/green] Model saved successfully")

    def save_model(
        self,
        output_dir: str,
        save_method: str = "merged_16bit",
        maximum_memory_usage: float = 0.75,
        max_shard_size: str = "5GB",
    ) -> None:
        """Save the trained model."""
        self._save_model_merged(output_dir, save_method, max_shard_size)


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
        return (
            "Standard HuggingFace Transformers + PEFT (maximum compatibility, slower than Unsloth)"
        )

    def supports_text_training(self) -> bool:
        return True

    def supports_vision_training(self) -> bool:
        return True  # Basic vision support available (limited features)

    def create_text_trainer(
        self,
        base_model: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: str | None = None,
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
        dtype: Any | None = None,
    ) -> VisionTrainer:
        """Create a Transformers vision trainer."""
        return TransformersVisionTrainer(
            base_model=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            dtype=dtype,
        )
