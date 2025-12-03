"""Dataset commands for Model Garden CLI.

Contains:
- create-dataset: Create a sample text dataset for testing
- create-vision-dataset: Create a sample vision-language dataset for testing
"""

import click

from model_garden.utils.console import console


@click.command()
@click.option(
    "--output",
    "-o",
    default="./data/sample_dataset.jsonl",
    help="Output path for the sample dataset",
)
@click.option(
    "--num-examples",
    "-n",
    default=100,
    type=int,
    help="Number of examples to generate",
)
def create_dataset(output: str, num_examples: int) -> None:
    """Create a sample dataset for testing.

    Example:

        \b
        uv run model-garden create-dataset \\
            --output ./data/sample.jsonl \\
            --num-examples 100
    """
    try:
        # Lazy import to avoid loading unsloth for inference commands
        from model_garden.training import create_sample_dataset

        console.print("\n[bold cyan]🌱 Model Garden - Dataset Creation[/bold cyan]\n")
        create_sample_dataset(output, num_examples)
        console.print("\n[bold green]✨ Dataset created successfully![/bold green]\n")
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]\n")
        raise click.Abort()


@click.command()
@click.option(
    "--output",
    "-o",
    default="./data/vision_sample.jsonl",
    help="Output path for the sample vision dataset",
)
@click.option(
    "--num-examples",
    "-n",
    default=10,
    type=int,
    help="Number of examples to generate",
)
def create_vision_dataset(output: str, num_examples: int) -> None:
    """Create a sample vision-language dataset for testing.

    Example:

        \b
        uv run model-garden create-vision-dataset \\
            --output ./data/vision_sample.jsonl \\
            --num-examples 10
    """
    try:
        from model_garden.training import create_vision_sample_dataset

        console.print("\n[bold cyan]🌱 Model Garden - Vision Dataset Creation[/bold cyan]\n")
        create_vision_sample_dataset(output, num_examples)
        console.print("\n[bold green]✨ Dataset created successfully![/bold green]\n")
        console.print(
            "[yellow]⚠️  Remember to replace placeholder image paths with real images[/yellow]\n"
        )
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]\n")
        raise click.Abort()
