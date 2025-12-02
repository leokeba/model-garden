"""Unsloth training backend for Model Garden.

This backend provides Unsloth-optimized training for both text and vision-language models.
Unsloth offers significant speedups and memory savings through specialized optimizations.
"""

import os
from pathlib import Path
from typing import Any

# Configure HuggingFace cache from environment before importing HF libraries
from dotenv import load_dotenv

load_dotenv()

HF_HOME = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = str(Path(HF_HOME) / "hub")
os.environ["HF_DATASETS_CACHE"] = str(Path(HF_HOME) / "datasets")

# Suppress non-critical warnings
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from model_garden.backends.base import TextTrainer, TrainingBackend, VisionTrainer

# DON'T import the trainer classes at module level - causes circular imports!
# They will be imported lazily when create_*_trainer() is called


class UnslothBackend(TrainingBackend):
    """Unsloth training backend.

    Unsloth provides optimized training with significant speedups and memory savings.
    It's the default backend for Model Garden.
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
        """Create an Unsloth text trainer."""
        # Lazy import to avoid circular dependencies
        from model_garden.training import ModelTrainer as UnslothTextTrainer

        # Return the existing Unsloth trainer - it already implements the interface
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
        """Create an Unsloth vision trainer."""
        # Lazy import to avoid circular dependencies
        from model_garden.training import VisionLanguageTrainer as UnslothVisionTrainer

        # Return the existing Unsloth vision trainer
        return UnslothVisionTrainer(
            base_model=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            dtype=dtype,
        )
