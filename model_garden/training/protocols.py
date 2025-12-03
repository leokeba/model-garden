"""Protocol-based interfaces for training backends.

This module defines Protocol classes that provide structural subtyping (duck typing)
for training backends. Using Protocol instead of ABC has several advantages:

1. **Structural Subtyping**: Classes don't need to explicitly inherit from the protocol.
   Any class with the right methods/attributes automatically satisfies the protocol.

2. **Better Type Checking**: Works seamlessly with static type checkers like mypy/pyright.

3. **More Pythonic**: Embraces Python's "if it walks like a duck" philosophy.

4. **Less Coupling**: No need to import base classes to implement the interface.

Usage:
    # Type annotation using protocol
    def train_model(trainer: TextTrainerProtocol, dataset: Dataset) -> None:
        trainer.load_model()
        trainer.train(dataset)

    # Any class with the right methods works
    class MyTrainer:
        def load_model(self) -> None: ...
        def train(self, dataset, config, ...) -> None: ...

    train_model(MyTrainer(), dataset)  # Works! No inheritance needed.
"""

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from datasets import Dataset

if TYPE_CHECKING:
    from model_garden.training.config import (
        TrainingConfig,
        VisionTrainingConfig,
    )


@runtime_checkable
class TextTrainerProtocol(Protocol):
    """Protocol for text-only model trainers.

    Any class implementing these methods can be used as a text trainer,
    regardless of inheritance hierarchy.

    Attributes:
        base_model: HuggingFace model identifier or local path
        max_seq_length: Maximum sequence length
        load_in_4bit: Whether 4-bit quantization is used
        load_in_8bit: Whether 8-bit quantization is used
        model: The loaded model (set after load_model())
        tokenizer: The loaded tokenizer (set after load_model())
    """

    base_model: str
    max_seq_length: int
    load_in_4bit: bool
    load_in_8bit: bool
    model: Any
    tokenizer: Any

    def load_model(self) -> None:
        """Load the base model with backend-specific optimizations."""
        ...

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
        ...

    def load_dataset_from_file(self, dataset_path: str) -> Dataset:
        """Load dataset from a local file."""
        ...

    def load_dataset_from_hub(self, dataset_name: str, split: str = "train") -> Dataset:
        """Load dataset from HuggingFace Hub."""
        ...

    def format_dataset(
        self,
        dataset: Dataset,
        instruction_field: str = "instruction",
        input_field: str = "input",
        output_field: str = "output",
        prompt_template: str | None = None,
    ) -> Dataset:
        """Format dataset for instruction fine-tuning."""
        ...

    def train(
        self,
        dataset: Dataset,
        config: "TrainingConfig",
        job_id: str | None = None,
        enable_carbon_tracking: bool = True,
        callbacks: list | None = None,
        eval_dataset: Dataset | None = None,
    ) -> None:
        """Train the model."""
        ...

    def save_model(
        self,
        output_dir: str,
        save_method: str = "merged_16bit",
        maximum_memory_usage: float = 0.75,
        max_shard_size: str = "5GB",
    ) -> None:
        """Save the trained model."""
        ...


@runtime_checkable
class VisionTrainerProtocol(Protocol):
    """Protocol for vision-language model trainers.

    Any class implementing these methods can be used as a vision trainer,
    regardless of inheritance hierarchy.

    Attributes:
        base_model: HuggingFace model identifier
        max_seq_length: Maximum sequence length
        load_in_4bit: Whether 4-bit quantization is used
        load_in_8bit: Whether 8-bit quantization is used
        model: The loaded model
        tokenizer: The loaded tokenizer
        processor: Vision processor (for vision models)
    """

    base_model: str
    max_seq_length: int
    load_in_4bit: bool
    load_in_8bit: bool
    model: Any
    tokenizer: Any
    processor: Any

    def load_model(self) -> None:
        """Load the vision-language model."""
        ...

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
        ...

    def load_dataset_from_file(self, dataset_path: str) -> Dataset:
        """Load multimodal dataset from a local file."""
        ...

    def load_dataset_from_hub(self, dataset_name: str, split: str = "train", **kwargs) -> Dataset:
        """Load multimodal dataset from HuggingFace Hub."""
        ...

    def format_dataset(
        self,
        dataset: Dataset,
        text_field: str = "text",
        image_field: str = "image",
        system_message: str | None = None,
        messages_field: str | None = None,
    ) -> list[dict]:
        """Format dataset for vision-language training."""
        ...

    def train(
        self,
        dataset: Dataset | list[dict],
        config: "VisionTrainingConfig",
        job_id: str | None = None,
        enable_carbon_tracking: bool = True,
        callbacks: list | None = None,
        eval_dataset: Dataset | list[dict] | None = None,
    ) -> None:
        """Train the vision-language model."""
        ...

    def save_model(
        self,
        output_dir: str,
        save_method: str = "merged_16bit",
        maximum_memory_usage: float = 0.75,
        max_shard_size: str = "5GB",
    ) -> None:
        """Save the fine-tuned vision-language model."""
        ...


@runtime_checkable
class TrainingBackendProtocol(Protocol):
    """Protocol for training backends.

    A training backend provides both text and vision training capabilities.
    """

    @property
    def name(self) -> str:
        """The name of this backend (e.g., 'unsloth', 'transformers')."""
        ...

    @property
    def description(self) -> str:
        """A description of this backend's capabilities."""
        ...

    def supports_text_training(self) -> bool:
        """Whether this backend supports text-only model training."""
        ...

    def supports_vision_training(self) -> bool:
        """Whether this backend supports vision-language model training."""
        ...

    def create_text_trainer(
        self,
        base_model: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: str | None = None,
    ) -> TextTrainerProtocol:
        """Create a text trainer instance for this backend."""
        ...

    def create_vision_trainer(
        self,
        base_model: str,
        max_seq_length: int = 16384,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: Any | None = None,
    ) -> VisionTrainerProtocol:
        """Create a vision trainer instance for this backend."""
        ...


def is_text_trainer(obj: Any) -> bool:
    """Check if an object implements the TextTrainerProtocol.

    This uses runtime protocol checking to determine if the object
    has all required methods and attributes.

    Args:
        obj: Object to check

    Returns:
        True if object implements TextTrainerProtocol
    """
    return isinstance(obj, TextTrainerProtocol)


def is_vision_trainer(obj: Any) -> bool:
    """Check if an object implements the VisionTrainerProtocol.

    Args:
        obj: Object to check

    Returns:
        True if object implements VisionTrainerProtocol
    """
    return isinstance(obj, VisionTrainerProtocol)


def is_training_backend(obj: Any) -> bool:
    """Check if an object implements the TrainingBackendProtocol.

    Args:
        obj: Object to check

    Returns:
        True if object implements TrainingBackendProtocol
    """
    return isinstance(obj, TrainingBackendProtocol)
