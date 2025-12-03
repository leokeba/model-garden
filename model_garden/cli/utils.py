"""Utility commands for Model Garden CLI.

Contains:
- generate: Generate text from a fine-tuned model (simple Unsloth-based generation)
- list-backends: List available training backends
"""

import click

from model_garden.utils.console import console


@click.command()
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


@click.command()
def list_backends_cmd() -> None:
    """List available training backends.

    Shows all registered backends with their capabilities (text/vision support).

    Example:
        uv run model-garden list-backends
    """
    try:
        from model_garden.backends import list_backends

        console.print("\n[bold cyan]Available Training Backends[/bold cyan]\n")

        backends = list_backends()

        if not backends:
            console.print("[yellow]No backends registered[/yellow]\n")
            return

        from rich.table import Table

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Backend", style="green")
        table.add_column("Description")
        table.add_column("Text", justify="center")
        table.add_column("Vision", justify="center")

        for backend in backends:
            text_support = "✓" if backend["supports_text"] else "✗"
            vision_support = "✓" if backend["supports_vision"] else "✗"

            table.add_row(
                backend["name"],
                backend["description"],
                text_support,
                vision_support,
            )

        console.print(table)
        console.print()

    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]\n")
        raise click.Abort()
