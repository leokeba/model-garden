"""Unsloth training backend for Model Garden.

This backend provides Unsloth-optimized training for both text and vision-language models.
Unsloth offers significant speedups and memory savings through specialized optimizations.

Note: This backend requires the 'unsloth' package to be installed.
Install with: pip install 'model-garden[unsloth]' or pip install unsloth
"""

from typing import Any

from model_garden.training.backends.base import TextTrainer, TrainingBackend, VisionTrainer
from model_garden.utils.optional_deps import require_unsloth


class UnslothBackend(TrainingBackend):
    """Unsloth training backend.

    Unsloth provides optimized training with significant speedups and memory savings.
    It's the default backend for Model Garden when installed.

    Requires the 'unsloth' package to be installed.
    """

    @property
    def name(self) -> str:
        return "unsloth"

    @property
    def description(self) -> str:
        return "Unsloth-optimized training with 2x speedup and 60% memory savings"

    def supports_text_training(self) -> bool:
        return True

    def supports_vision_training(self) -> bool:
        return True

    def create_text_trainer(
        self,
        base_model: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: str | None = None,
    ) -> TextTrainer:
        """Create an Unsloth text trainer.

        Raises:
            ImportError: If Unsloth is not installed
        """
        require_unsloth("Unsloth text training")

        # Import from the backends folder where Unsloth-specific code lives
        from model_garden.training.backends.unsloth_text_trainer import (
            ModelTrainer as UnslothTextTrainer,
        )

        return UnslothTextTrainer(
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
        """Create an Unsloth vision trainer.

        Raises:
            ImportError: If Unsloth is not installed
        """
        require_unsloth("Unsloth vision training")

        # Import from the backends folder where Unsloth-specific code lives
        from model_garden.training.backends.unsloth_vision_trainer import (
            VisionLanguageTrainer as UnslothVisionTrainer,
        )

        return UnslothVisionTrainer(
            base_model=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            dtype=dtype,
        )
