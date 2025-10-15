"""Command-line interface for Model Garden."""

import click
from rich.console import Console

from model_garden.training import ModelTrainer, create_sample_dataset

console = Console()


@click.group()
@click.version_option()
def main() -> None:
    """Model Garden - Fine-tune and serve LLMs."""
    pass


@main.command()
@click.option(
    "--base-model",
    "-m",
    required=True,
    help="Base model to fine-tune (HuggingFace ID or local path)",
)
@click.option(
    "--dataset",
    "-d",
    required=True,
    help="Path to dataset file (JSONL, JSON, CSV) or HuggingFace dataset ID",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    help="Directory to save the fine-tuned model",
)
@click.option(
    "--epochs",
    "-e",
    default=3,
    type=int,
    help="Number of training epochs",
)
@click.option(
    "--batch-size",
    "-b",
    default=2,
    type=int,
    help="Training batch size per device",
)
@click.option(
    "--learning-rate",
    "-lr",
    default=2e-4,
    type=float,
    help="Learning rate",
)
@click.option(
    "--max-seq-length",
    default=2048,
    type=int,
    help="Maximum sequence length",
)
@click.option(
    "--lora-r",
    default=16,
    type=int,
    help="LoRA rank",
)
@click.option(
    "--lora-alpha",
    default=16,
    type=int,
    help="LoRA alpha parameter",
)
@click.option(
    "--gradient-accumulation-steps",
    default=4,
    type=int,
    help="Gradient accumulation steps",
)
@click.option(
    "--max-steps",
    default=-1,
    type=int,
    help="Maximum training steps (-1 for full epochs)",
)
@click.option(
    "--logging-steps",
    default=10,
    type=int,
    help="Log every N steps",
)
@click.option(
    "--save-steps",
    default=100,
    type=int,
    help="Save checkpoint every N steps",
)
@click.option(
    "--save-method",
    type=click.Choice(["lora", "merged_16bit", "merged_4bit"]),
    default="merged_16bit",
    help="How to save the final model",
)
@click.option(
    "--instruction-field",
    default="instruction",
    help="Dataset field name for instructions",
)
@click.option(
    "--input-field",
    default="input",
    help="Dataset field name for inputs",
)
@click.option(
    "--output-field",
    default="output",
    help="Dataset field name for outputs",
)
@click.option(
    "--from-hub",
    is_flag=True,
    help="Load dataset from HuggingFace Hub instead of local file",
)
def train(
    base_model: str,
    dataset: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
    gradient_accumulation_steps: int,
    max_steps: int,
    logging_steps: int,
    save_steps: int,
    save_method: str,
    instruction_field: str,
    input_field: str,
    output_field: str,
    from_hub: bool,
) -> None:
    """Fine-tune a language model using Unsloth.

    Example:

        \b
        # Train with local dataset
        uv run model-garden train \\
            --base-model unsloth/tinyllama-bnb-4bit \\
            --dataset ./data/train.jsonl \\
            --output-dir ./models/my-model \\
            --epochs 3

        \b
        # Train with HuggingFace Hub dataset
        uv run model-garden train \\
            --base-model unsloth/tinyllama-bnb-4bit \\
            --dataset yahma/alpaca-cleaned \\
            --output-dir ./models/my-model \\
            --from-hub
    """
    try:
        console.print("\n[bold cyan]🌱 Model Garden - Fine-tuning[/bold cyan]\n")

        # Initialize trainer
        trainer = ModelTrainer(
            base_model=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )

        # Load model
        trainer.load_model()

        # Prepare for training with LoRA
        trainer.prepare_for_training(
            r=lora_r,
            lora_alpha=lora_alpha,
        )

        # Load dataset
        if from_hub:
            train_dataset = trainer.load_dataset_from_hub(dataset)
        else:
            train_dataset = trainer.load_dataset_from_file(dataset)

        # Format dataset
        train_dataset = trainer.format_dataset(
            train_dataset,
            instruction_field=instruction_field,
            input_field=input_field,
            output_field=output_field,
        )

        # Train
        trainer.train(
            dataset=train_dataset,
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            max_steps=max_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
        )

        # Save final model
        if save_method != "lora":
            trainer.save_model(output_dir, save_method=save_method)

        console.print("\n[bold green]✨ Training completed successfully![/bold green]\n")

    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]\n")
        raise click.Abort()


@main.command()
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
        console.print("\n[bold cyan]🌱 Model Garden - Dataset Creation[/bold cyan]\n")
        create_sample_dataset(output, num_examples)
        console.print("\n[bold green]✨ Dataset created successfully![/bold green]\n")
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]\n")
        raise click.Abort()


@main.command()
@click.argument("model_path")
@click.option(
    "--prompt",
    "-p",
    required=True,
    help="Prompt to generate from",
)
@click.option(
    "--max-tokens",
    default=256,
    type=int,
    help="Maximum tokens to generate",
)
@click.option(
    "--temperature",
    default=0.7,
    type=float,
    help="Sampling temperature",
)
def generate(model_path: str, prompt: str, max_tokens: int, temperature: float) -> None:
    """Generate text from a fine-tuned model.

    Example:

        \b
        uv run model-garden generate ./models/my-model \\
            --prompt "Explain quantum computing" \\
            --max-tokens 256
    """
    try:
        from unsloth import FastLanguageModel

        console.print("\n[bold cyan]🌱 Model Garden - Text Generation[/bold cyan]\n")
        console.print(f"[cyan]Loading model from: {model_path}[/cyan]")

        # Load model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        # Enable inference mode
        FastLanguageModel.for_inference(model)

        console.print("[green]✓[/green] Model loaded\n")

        # Format prompt
        formatted_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""

        # Generate
        console.print("[cyan]Generating...[/cyan]\n")
        inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            use_cache=True,
        )

        # Decode and display
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # Extract only the response part
        if "### Response:" in generated_text:
            response = generated_text.split("### Response:")[1].strip()
        else:
            response = generated_text

        console.print("[bold]Response:[/bold]")
        console.print(response)
        console.print()

    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]\n")
        raise click.Abort()


@main.command()
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind the server to",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port to bind the server to",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload for development",
)
def serve(host: str, port: int, reload: bool) -> None:
    """Start the FastAPI server.

    Example:

        \b
        # Start server on default host/port
        uv run model-garden serve

        \b
        # Start with auto-reload for development
        uv run model-garden serve --reload

        \b
        # Start on custom host/port
        uv run model-garden serve --host 127.0.0.1 --port 3000
    """
    try:
        import uvicorn
        from model_garden.api import app

        console.print("\n[bold cyan]🌱 Model Garden - API Server[/bold cyan]\n")
        console.print(f"[cyan]Starting server on http://{host}:{port}[/cyan]")
        
        if reload:
            console.print("[yellow]⚠️  Auto-reload enabled (development mode)[/yellow]")
        
        console.print("[green]✓[/green] Server starting...\n")

        uvicorn.run(
            "model_garden.api:app",
            host=host,
            port=port,
            reload=reload,
        )

    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]\n")
        raise click.Abort()


if __name__ == "__main__":
    main()
