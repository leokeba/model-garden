"""Base classes for training backends.

This module defines the abstract interfaces that all training backends must implement.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from datasets import Dataset

if TYPE_CHECKING:
    from model_garden.training.config import (
        TrainingConfig,
        VisionTrainingConfig,
    )


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
        dtype: str | None = None,
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
        self.model: Any = None
        self.tokenizer: Any = None

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
        target_modules: list[str] | None = None,
        use_rslora: bool = False,
        lora_bias: str = "none",
        task_type: str = "CAUSAL_LM",
        use_gradient_checkpointing: str | bool = "unsloth",
        random_state: int = 42,
        loftq_config: dict | None = None,
    ) -> None:
        """Prepare model for LoRA fine-tuning."""
        pass

    @abstractmethod
    def load_dataset_from_file(self, dataset_path: str) -> Dataset:
        """Load dataset from a local file."""
        pass

    @abstractmethod
    def load_dataset_from_hub(self, dataset_name: str, split: str = "train") -> Dataset:
        """Load dataset from HuggingFace Hub."""
        pass

    @abstractmethod
    def format_dataset(
        self,
        dataset: Dataset,
        instruction_field: str = "instruction",
        input_field: str = "input",
        output_field: str = "output",
        prompt_template: str | None = None,
    ) -> Dataset:
        """Format dataset for instruction fine-tuning."""
        pass

    @abstractmethod
    def train(
        self,
        dataset: Dataset,
        config: "TrainingConfig",
        job_id: str | None = None,
        enable_carbon_tracking: bool = True,
        callbacks: list | None = None,
        eval_dataset: Dataset | None = None,
    ) -> None:
        """Train the model.

        Args:
            dataset: Training dataset (should have 'text' field)
            config: Training configuration with all hyperparameters
            job_id: Optional job identifier for carbon tracking
            enable_carbon_tracking: Whether to track carbon emissions
            callbacks: Optional list of TrainerCallback instances
            eval_dataset: Optional validation dataset for evaluation
        """
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
        dtype: Any | None = None,
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
        self.model: Any = None
        self.tokenizer: Any = None
        self.processor: Any = None

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
        """Prepare model for LoRA fine-tuning with selective layer control."""
        pass

    @abstractmethod
    def load_dataset_from_file(self, dataset_path: str) -> Dataset:
        """Load multimodal dataset from a local file."""
        pass

    @abstractmethod
    def load_dataset_from_hub(self, dataset_name: str, split: str = "train", **kwargs) -> Dataset:
        """Load multimodal dataset from HuggingFace Hub."""
        pass

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

    @abstractmethod
    def format_dataset(
        self,
        dataset: Dataset,
        text_field: str = "text",
        image_field: str = "image",
        system_message: str | None = None,
        messages_field: str | None = None,
        lazy_loading: bool = False,
    ) -> list[dict] | Any:
        """Format dataset for vision-language training.

        Args:
            dataset: Input dataset
            text_field: Field name for text/questions
            image_field: Field name for images
            system_message: Optional system message
            messages_field: Field name for messages (for OpenAI format)
            lazy_loading: If True, return a lazy dataset that loads images on-demand

        Returns:
            List of formatted message dictionaries or a LazyVisionDataset
        """
        pass

    @abstractmethod
    def train(
        self,
        dataset: Dataset | list[dict],
        config: "VisionTrainingConfig",
        job_id: str | None = None,
        enable_carbon_tracking: bool = True,
        callbacks: list | None = None,
        eval_dataset: Dataset | list[dict] | None = None,
    ) -> None:
        """Train the vision-language model.

        Args:
            dataset: Training dataset (Dataset object or list of formatted messages)
            config: Vision training configuration with all hyperparameters
            job_id: Optional job identifier for carbon tracking
            enable_carbon_tracking: Whether to track carbon emissions
            callbacks: Optional list of TrainerCallback instances
            eval_dataset: Optional validation dataset for evaluation
        """
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
        dtype: str | None = None,
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
        dtype: Any | None = None,
    ) -> VisionTrainer:
        """Create a vision trainer instance for this backend."""
        pass
