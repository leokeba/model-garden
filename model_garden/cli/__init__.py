"""Command-line interface for Model Garden.

This package provides the CLI commands organized into logical groups:
- train: Training commands (train, train-vision)
- inference: Inference commands (serve-model, inference-generate, inference-chat)
- dataset: Dataset commands (create-dataset, create-vision-dataset)
- serve: Server commands (serve)
"""

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option()
def main() -> None:
    """Model Garden - Fine-tune and serve LLMs."""
    pass


# Import and register command groups
# Commands are registered via decorators in their respective modules

# Training commands
from model_garden.cli.train import train, train_vision

main.add_command(train)
main.add_command(train_vision, name="train-vision")

# Inference commands
from model_garden.cli.inference import (
    inference_chat,
    inference_generate,
    serve_model,
)

main.add_command(serve_model, name="serve-model")
main.add_command(inference_generate, name="inference-generate")
main.add_command(inference_chat, name="inference-chat")

# Dataset commands
from model_garden.cli.dataset import create_dataset, create_vision_dataset

main.add_command(create_dataset, name="create-dataset")
main.add_command(create_vision_dataset, name="create-vision-dataset")

# Server commands
from model_garden.cli.server import serve

main.add_command(serve)

# Utility commands
from model_garden.cli.utils import generate, list_backends_cmd

main.add_command(generate)
main.add_command(list_backends_cmd, name="list-backends")

__all__ = ["main", "console"]
