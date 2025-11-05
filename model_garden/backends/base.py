"""Base classes for training backends.

This module defines the abstract interfaces that all training backends must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset


class TextTrainer(ABC):
    """Abstract base class for text-only model training.
    
    This class defines the interface that all text training backends must implement.
    It handles model loading, dataset preparation, training, and saving for text-only models.
    """

    def __init__(
        self,
        base_model: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: Optional[str] = None,
    ):
        """Initialize the text trainer.

        Args:
            base_model: HuggingFace model identifier or local path
            max_seq_length: Maximum sequence length
            load_in_4bit: Whether to load model in 4-bit quantization
            load_in_8bit: Whether to load model in 8-bit quantization
            dtype: Data type (None for auto-detection)
        """
        self.base_model = base_model
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.dtype = dtype
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the base model with backend-specific optimizations."""
        pass

    @abstractmethod
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
        """Prepare model for LoRA fine-tuning."""
        pass

    @abstractmethod
    def load_dataset_from_file(self, dataset_path: str) -> Dataset:
        """Load dataset from a local file."""
        pass

    @abstractmethod
    def load_dataset_from_hub(
        self, dataset_name: str, split: str = "train"
    ) -> Dataset:
        """Load dataset from HuggingFace Hub."""
        pass

    @abstractmethod
    def format_dataset(
        self,
        dataset: Dataset,
        instruction_field: str = "instruction",
        input_field: str = "input",
        output_field: str = "output",
        prompt_template: Optional[str] = None,
    ) -> Dataset:
        """Format dataset for instruction fine-tuning."""
        pass

    @abstractmethod
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
        """Train the model."""
        pass

    @abstractmethod
    def save_model(
        self,
        output_dir: str,
        save_method: str = "merged_16bit",
        maximum_memory_usage: float = 0.75,
        max_shard_size: str = "5GB",
    ) -> None:
        """Save the trained model."""
        pass


class VisionTrainer(ABC):
    """Abstract base class for vision-language model training.
    
    This class defines the interface that all vision training backends must implement.
    It handles model loading, dataset preparation, training, and saving for vision-language models.
    """

    def __init__(
        self,
        base_model: str,
        max_seq_length: int = 16384,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: Optional[Any] = None,
    ):
        """Initialize the vision trainer.

        Args:
            base_model: HuggingFace model identifier
            max_seq_length: Maximum sequence length (larger for vision models)
            load_in_4bit: Whether to load model in 4-bit quantization
            load_in_8bit: Whether to load model in 8-bit quantization
            dtype: Data type (None for auto-detection)
        """
        self.base_model = base_model
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        self.processor = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the vision-language model."""
        pass

    @abstractmethod
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
        finetune_vision_layers: bool = True,
        finetune_language_layers: bool = True,
        finetune_attention_modules: bool = True,
        finetune_mlp_modules: bool = True,
    ) -> None:
        """Prepare model for LoRA fine-tuning with selective layer control."""
        pass

    @abstractmethod
    def load_dataset_from_file(self, dataset_path: str) -> Dataset:
        """Load multimodal dataset from a local file."""
        pass

    @abstractmethod
    def load_dataset_from_hub(
        self, dataset_name: str, split: str = "train", **kwargs
    ) -> Dataset:
        """Load multimodal dataset from HuggingFace Hub."""
        pass

    def load_dataset(
        self,
        dataset_path: str,
        from_hub: bool = False,
        split: str = "train",
        **kwargs
    ) -> Dataset:
        """Load multimodal dataset from file or HuggingFace Hub.
        
        This is a convenience method that delegates to either load_dataset_from_file
        or load_dataset_from_hub based on the from_hub parameter.
        
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

    @abstractmethod
    def format_dataset(
        self,
        dataset: Dataset,
        text_field: str = "text",
        image_field: str = "image",
        system_message: Optional[str] = None,
        messages_field: Optional[str] = None,
    ) -> List[Dict]:
        """Format dataset for vision-language training."""
        pass

    @abstractmethod
    def train(
        self,
        dataset: Union[Dataset, List[Dict]],
        output_dir: str,
        job_id: Optional[str] = None,
        enable_carbon_tracking: bool = True,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 2e-5,
        warmup_steps: int = 10,
        max_steps: int = -1,
        logging_steps: int = 10,
        save_steps: int = 100,
        optim: str = "adamw_8bit",
        weight_decay: float = 0.01,
        lr_scheduler_type: str = "cosine",
        max_grad_norm: float = 1.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        dataloader_num_workers: int = 0,
        dataloader_pin_memory: bool = False,
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
        selective_loss_masking_strategy: str = "epoch_based",
        selective_loss_masking_start_epoch: float = 0.0,
        selective_loss_mask_every_n_steps: int = 100,
        selective_loss_mask_for_n_steps: int = 50,
        selective_loss_structural_weight: float = 0.1,
        selective_loss_verbose: bool = False,
    ) -> None:
        """Train the vision-language model."""
        pass

    @abstractmethod
    def save_model(
        self,
        output_dir: str,
        save_method: str = "merged_16bit",
        maximum_memory_usage: float = 0.75,
        max_shard_size: str = "5GB",
    ) -> None:
        """Save the fine-tuned vision-language model."""
        pass


class TrainingBackend(ABC):
    """Abstract base class for training backends.
    
    A training backend provides both text and vision training capabilities.
    Backends can be registered and dynamically selected at runtime.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this backend (e.g., 'unsloth', 'transformers')."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """A description of this backend's capabilities and characteristics."""
        pass

    @abstractmethod
    def supports_text_training(self) -> bool:
        """Whether this backend supports text-only model training."""
        pass

    @abstractmethod
    def supports_vision_training(self) -> bool:
        """Whether this backend supports vision-language model training."""
        pass

    @abstractmethod
    def create_text_trainer(
        self,
        base_model: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: Optional[str] = None,
    ) -> TextTrainer:
        """Create a text trainer instance for this backend."""
        pass

    @abstractmethod
    def create_vision_trainer(
        self,
        base_model: str,
        max_seq_length: int = 16384,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: Optional[Any] = None,
    ) -> VisionTrainer:
        """Create a vision trainer instance for this backend."""
        pass
