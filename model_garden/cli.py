"""Command-line interface for Model Garden."""

import click
from typing import Optional
from rich.console import Console

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
    help="LoRA alpha parameter (scaling factor, typically equal to lora-r)",
)
@click.option(
    "--lora-dropout",
    default=0.0,
    type=float,
    help="LoRA dropout rate (0.0-0.3, higher = more regularization)",
)
@click.option(
    "--lora-bias",
    type=click.Choice(["none", "all", "lora_only"]),
    default="none",
    help="How to handle bias in LoRA layers",
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
    "--weight-decay",
    default=0.01,
    type=float,
    help="Weight decay for regularization (0.0-0.1)",
)
@click.option(
    "--lr-scheduler-type",
    type=click.Choice(["linear", "cosine", "constant", "constant_with_warmup", "polynomial"]),
    default="linear",
    help="Learning rate scheduler type",
)
@click.option(
    "--max-grad-norm",
    default=1.0,
    type=float,
    help="Maximum gradient norm for clipping",
)
@click.option(
    "--adam-beta1",
    default=0.9,
    type=float,
    help="Beta1 parameter for Adam optimizer",
)
@click.option(
    "--adam-beta2",
    default=0.999,
    type=float,
    help="Beta2 parameter for Adam optimizer",
)
@click.option(
    "--adam-epsilon",
    default=1e-8,
    type=float,
    help="Epsilon parameter for Adam optimizer",
)
@click.option(
    "--dataloader-num-workers",
    default=0,
    type=int,
    help="Number of dataloader workers (0 = main process only)",
)
@click.option(
    "--eval-strategy",
    type=click.Choice(["no", "steps", "epoch"]),
    default="steps",
    help="Evaluation strategy",
)
@click.option(
    "--save-total-limit",
    default=3,
    type=int,
    help="Maximum number of checkpoints to keep",
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
    lora_dropout: float,
    lora_bias: str,
    gradient_accumulation_steps: int,
    max_steps: int,
    logging_steps: int,
    save_steps: int,
    save_method: str,
    instruction_field: str,
    input_field: str,
    output_field: str,
    weight_decay: float,
    lr_scheduler_type: str,
    max_grad_norm: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float,
    dataloader_num_workers: int,
    eval_strategy: str,
    save_total_limit: int,
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
        # Lazy import to avoid loading unsloth for inference commands
        from model_garden.training import ModelTrainer
        
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
            lora_dropout=lora_dropout,
            lora_bias=lora_bias,
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
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            max_grad_norm=max_grad_norm,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            dataloader_num_workers=dataloader_num_workers,
            eval_strategy=eval_strategy,
            save_total_limit=save_total_limit,
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
        # Lazy import to avoid loading unsloth for inference commands
        from model_garden.training import create_sample_dataset
        
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


@main.command(name="train-vision")
@click.option(
    "--base-model",
    "-m",
    default="Qwen/Qwen2.5-VL-3B-Instruct",
    help="Vision-language model to fine-tune",
)
@click.option(
    "--dataset",
    "-d",
    required=True,
    help="Path to dataset file or HuggingFace dataset identifier",
)
@click.option(
    "--from-hub",
    is_flag=True,
    help="Load dataset from HuggingFace Hub instead of local file",
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
    default=1,
    type=int,
    help="Training batch size per device (use 1 for vision models)",
)
@click.option(
    "--learning-rate",
    "-lr",
    default=2e-5,
    type=float,
    help="Learning rate (lower for vision models)",
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
    help="LoRA alpha parameter (scaling factor, typically equal to lora-r)",
)
@click.option(
    "--lora-dropout",
    default=0.0,
    type=float,
    help="LoRA dropout rate (0.0-0.3, higher = more regularization)",
)
@click.option(
    "--lora-bias",
    type=click.Choice(["none", "all", "lora_only"]),
    default="none",
    help="How to handle bias in LoRA layers",
)
@click.option(
    "--gradient-accumulation-steps",
    default=8,
    type=int,
    help="Gradient accumulation steps (higher for vision models)",
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
    help="How to save the final model (default: merged_16bit)",
)
@click.option(
    "--text-field",
    default="text",
    help="Dataset field name for text/questions",
)
@click.option(
    "--image-field",
    default="image",
    help="Dataset field name for image paths",
)
@click.option(
    "--weight-decay",
    default=0.01,
    type=float,
    help="Weight decay for regularization (0.0-0.1)",
)
@click.option(
    "--lr-scheduler-type",
    type=click.Choice(["linear", "cosine", "constant", "constant_with_warmup", "polynomial"]),
    default="cosine",
    help="Learning rate scheduler type (cosine recommended for vision models)",
)
@click.option(
    "--max-grad-norm",
    default=1.0,
    type=float,
    help="Maximum gradient norm for clipping",
)
@click.option(
    "--adam-beta1",
    default=0.9,
    type=float,
    help="Beta1 parameter for Adam optimizer",
)
@click.option(
    "--adam-beta2",
    default=0.999,
    type=float,
    help="Beta2 parameter for Adam optimizer",
)
@click.option(
    "--adam-epsilon",
    default=1e-8,
    type=float,
    help="Epsilon parameter for Adam optimizer",
)
@click.option(
    "--dataloader-num-workers",
    default=0,
    type=int,
    help="Number of dataloader workers (0 = main process only)",
)
@click.option(
    "--eval-strategy",
    type=click.Choice(["no", "steps", "epoch"]),
    default="steps",
    help="Evaluation strategy",
)
@click.option(
    "--save-total-limit",
    default=3,
    type=int,
    help="Maximum number of checkpoints to keep",
)
@click.option(
    "--selective-loss/--no-selective-loss",
    default=False,
    help="Enable selective loss masking for structured outputs (masks JSON structure)",
)
@click.option(
    "--selective-loss-level",
    type=click.Choice(["conservative", "moderate", "aggressive"]),
    default="conservative",
    help="Selective loss masking level (conservative=structure only, moderate=+null, aggressive=+schema keys)",
)
@click.option(
    "--selective-loss-schema-keys",
    default=None,
    help="Comma-separated schema keys to mask (for aggressive mode, e.g., 'Marque,Modele,contents')",
)
@click.option(
    "--selective-loss-masking-start-step",
    type=int,
    default=0,
    help="Delay masking until this step (0=immediate, 100=learn structure first for 100 steps). Recommended: 50-200 for better JSON structure learning.",
)
@click.option(
    "--selective-loss-verbose/--no-selective-loss-verbose",
    default=False,
    help="Print selective loss masking statistics during training",
)
def train_vision(
    base_model: str,
    dataset: str,
    from_hub: bool,
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_bias: str,
    gradient_accumulation_steps: int,
    max_steps: int,
    logging_steps: int,
    save_steps: int,
    save_method: str,
    text_field: str,
    image_field: str,
    weight_decay: float,
    lr_scheduler_type: str,
    max_grad_norm: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_epsilon: float,
    dataloader_num_workers: int,
    eval_strategy: str,
    save_total_limit: int,
    selective_loss: bool,
    selective_loss_level: str,
    selective_loss_schema_keys: Optional[str],
    selective_loss_masking_start_step: int,
    selective_loss_verbose: bool,
) -> None:
    """Fine-tune a vision-language model (e.g., Qwen2.5-VL).

    Examples:

        \b
        # Train with local dataset
        uv run model-garden train-vision \\
            --base-model Qwen/Qwen2.5-VL-3B-Instruct \\
            --dataset ./data/vision_dataset.jsonl \\
            --output-dir ./models/my-vision-model \\
            --epochs 3 \\
            --batch-size 1

        \b
        # Train with HuggingFace Hub dataset
        uv run model-garden train-vision \\
            --base-model Qwen/Qwen2.5-VL-3B-Instruct \\
            --dataset Barth371/train_pop_valet_no_wrong_doc \\
            --from-hub \\
            --output-dir ./models/form-extraction-model \\
            --max-steps 100

    Dataset formats:
        
        Local JSONL:
            {"text": "What is in this image?", "image": "/path/to/img.jpg", "response": "A cat"}
        
        HuggingFace Hub (OpenAI messages format):
            {"messages": [{"role": "user", "content": [{"type": "image", "image": "data:image/jpeg;base64,..."}]}]}
    """
    try:
        from model_garden.vision_training import VisionLanguageTrainer
        
        console.print("\n[bold cyan]🌱 Model Garden - Vision-Language Fine-tuning[/bold cyan]\n")
        
        if from_hub:
            console.print(f"[cyan]Loading dataset from HuggingFace Hub: {dataset}[/cyan]")
        else:
            console.print(f"[cyan]Loading dataset from local file: {dataset}[/cyan]")
        
        console.print("[yellow]⚠️  Vision-language training is experimental[/yellow]\n")

        # Initialize trainer
        trainer = VisionLanguageTrainer(
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
            lora_dropout=lora_dropout,
            lora_bias=lora_bias,
        )

        # Load dataset (handles both local files and HuggingFace Hub)
        train_dataset = trainer.load_dataset(
            dataset_path=dataset,
            from_hub=from_hub,
        )

        # Format dataset
        train_dataset = trainer.format_dataset(
            train_dataset,
            text_field=text_field,
            image_field=image_field,
        )

        # Parse schema keys if provided
        schema_keys_list = None
        if selective_loss_schema_keys:
            schema_keys_list = [k.strip() for k in selective_loss_schema_keys.split(',')]
            console.print(f"[cyan]Schema keys to mask: {schema_keys_list}[/cyan]")
        
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
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            max_grad_norm=max_grad_norm,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            dataloader_num_workers=dataloader_num_workers,
            eval_strategy=eval_strategy,
            save_total_limit=save_total_limit,
            selective_loss=selective_loss,
            selective_loss_level=selective_loss_level,
            selective_loss_schema_keys=schema_keys_list,
            selective_loss_masking_start_step=selective_loss_masking_start_step,
            selective_loss_verbose=selective_loss_verbose,
        )

        # Save final model with specified method
        trainer.save_model(output_dir, save_method=save_method)

        console.print("\n[bold green]✨ Vision-language training completed successfully![/bold green]\n")
        console.print(f"[green]Model saved to: {output_dir}[/green]\n")

    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]\n")
        import traceback
        traceback.print_exc()
        raise click.Abort()


@main.command(name="create-vision-dataset")
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
        from model_garden.vision_training import create_vision_sample_dataset
        
        console.print("\n[bold cyan]🌱 Model Garden - Vision Dataset Creation[/bold cyan]\n")
        create_vision_sample_dataset(output, num_examples)
        console.print("\n[bold green]✨ Dataset created successfully![/bold green]\n")
        console.print("[yellow]⚠️  Remember to replace placeholder image paths with real images[/yellow]\n")
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]\n")
        raise click.Abort()


@main.command(name="serve-model")
@click.option("--model-path", required=True, help="Path to the model to serve")
@click.option("--port", default=8000, help="Port to run the inference server on")
@click.option("--host", default="0.0.0.0", help="Host to bind the server to")
@click.option("--tensor-parallel-size", default=1, help="Number of GPUs to use for tensor parallelism")
@click.option("--gpu-memory-utilization", default=0.0, type=float, help="GPU memory utilization (0.0-1.0, 0 = auto)")
@click.option("--quantization", type=click.Choice(["auto", "awq", "gptq", "squeezellm", "fp8", "bitsandbytes"]), default="auto", help="Quantization method (auto = detect from model)")
@click.option("--max-model-len", type=int, help="Maximum sequence length")
def serve_model(model_path, port, host, tensor_parallel_size, gpu_memory_utilization, quantization, max_model_len):
    """
    Start an inference server with vLLM for high-throughput model serving.
    
    This command loads a model using vLLM and starts a FastAPI server
    with OpenAI-compatible endpoints for text generation and chat completions.
    
    Examples:
    
        \b
        # Serve a model on default port 8000
        uv run model-garden serve-model --model-path ./models/my-model
        
        \b
        # Serve with custom port and GPU settings
        uv run model-garden serve-model \\
            --model-path ./models/my-model \\
            --port 8080 \\
            --tensor-parallel-size 2 \\
            --gpu-memory-utilization 0.8
        
        \b
        # Serve with quantization
        uv run model-garden serve-model \\
            --model-path ./models/my-model \\
            --quantization awq
    """
    try:
        import os
        import uvicorn
        
        console.print("\n[bold cyan]🚀 Model Garden - Inference Server[/bold cyan]\n")
        console.print(f"[cyan]Loading model:[/cyan] {model_path}")
        
        # Reduce torch compile workers to save memory (default is 24, we use 8)
        os.environ["TORCH_COMPILE_MAX_WORKERS"] = "8"
        
        # Set environment variables for the API to pick up during lifespan startup
        # This ensures the model is loaded in the same process as the API
        os.environ["MODEL_GARDEN_AUTOLOAD_MODEL"] = model_path
        
        if tensor_parallel_size > 1:
            os.environ["MODEL_GARDEN_TENSOR_PARALLEL_SIZE"] = str(tensor_parallel_size)
        # Always set GPU memory utilization (0 = auto mode)
        os.environ["MODEL_GARDEN_GPU_MEMORY_UTILIZATION"] = str(gpu_memory_utilization)
        if quantization:
            os.environ["MODEL_GARDEN_QUANTIZATION"] = quantization
        if max_model_len:
            os.environ["MODEL_GARDEN_MAX_MODEL_LEN"] = str(max_model_len)
        
        console.print(f"\n[cyan]Starting server on[/cyan] http://{host}:{port}")
        console.print(f"[cyan]API docs available at[/cyan] http://{host}:{port}/docs\n")
        console.print("[yellow]Press Ctrl+C to stop the server[/yellow]\n")
        
        # Start the server with minimal logging
        import logging
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        uvicorn.run(
            "model_garden.api:app", 
            host=host, 
            port=port, 
            reload=False,
            log_level="warning",
            access_log=False
        )
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]\n")
        import traceback
        traceback.print_exc()
        raise click.Abort()


@main.command(name="inference-generate")
@click.option("--model-path", required=True, help="Path to the model to use")
@click.option("--prompt", required=True, help="Prompt for text generation")
@click.option("--max-tokens", default=256, help="Maximum number of tokens to generate")
@click.option("--temperature", default=0.7, type=float, help="Sampling temperature")
@click.option("--top-p", default=0.9, type=float, help="Top-p (nucleus) sampling parameter")
@click.option("--stream/--no-stream", default=False, help="Enable streaming output")
@click.option("--tensor-parallel-size", default=1, type=int, help="Number of GPUs for tensor parallelism")
@click.option("--quantization", type=click.Choice(["auto", "awq", "gptq", "squeezellm", "fp8", "bitsandbytes"]), default="auto", help="Quantization method (auto = detect from model)")
def inference_generate(model_path, prompt, max_tokens, temperature, top_p, stream, tensor_parallel_size, quantization):
    """
    Generate text using vLLM inference engine (one-off generation).
    
    This command loads a model, generates a response, and exits.
    For persistent serving, use the 'serve-model' command instead.
    
    Examples:
    
        \b
        # Generate text with default settings
        uv run model-garden inference-generate \\
            --model-path ./models/my-model \\
            --prompt "Once upon a time"
        
        \b
        # Generate with custom parameters
        uv run model-garden inference-generate \\
            --model-path ./models/my-model \\
            --prompt "Explain quantum computing" \\
            --max-tokens 512 \\
            --temperature 0.8 \\
            --stream
        
        \b
        # Generate with quantization
        uv run model-garden inference-generate \\
            --model-path ./models/my-model \\
            --prompt "Write a poem" \\
            --quantization awq
    """
    try:
        from model_garden.inference import InferenceService
        import asyncio
        
        console.print("\n[bold cyan]🤖 Model Garden - Text Generation[/bold cyan]\n")
        console.print(f"[cyan]Loading model:[/cyan] {model_path}\n")
        
        # Create inference service
        service = InferenceService(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            quantization=quantization
        )
        
        async def generate():
            # Load model
            await service.load_model()
            console.print("[green]✅ Model loaded![/green]\n")
            console.print(f"[cyan]Prompt:[/cyan] {prompt}\n")
            console.print("[cyan]Generated text:[/cyan]\n")
            
            # Generate
            if stream:
                stream_result = await service.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True
                )
                # Handle streaming response - use try/except to handle type issues
                try:
                    async for chunk in stream_result:  # type: ignore
                        console.print(chunk, end="")
                except TypeError:
                    console.print("[red]Error: Stream response not iterable[/red]")
                console.print("\n")
            else:
                result = await service.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False
                )
                # Handle non-streaming response
                if isinstance(result, dict):
                    console.print(result.get("text", ""))
                    if "usage" in result:
                        console.print(f"\n[dim]Tokens: {result['usage'].get('total_tokens', 0)}[/dim]\n")
            
            # Cleanup
            await service.unload_model()
        
        asyncio.run(generate())
        console.print("[green]✨ Generation complete![/green]\n")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]\n")
        raise click.Abort()


@main.command(name="inference-chat")
@click.option("--model-path", required=True, help="Path to the model to use")
@click.option("--system-prompt", help="System prompt for the chat")
@click.option("--max-tokens", default=512, help="Maximum tokens per response")
@click.option("--temperature", default=0.7, type=float, help="Sampling temperature")
@click.option("--tensor-parallel-size", default=1, type=int, help="Number of GPUs for tensor parallelism")
@click.option("--quantization", type=click.Choice(["auto", "awq", "gptq", "squeezellm", "fp8", "bitsandbytes"]), default="auto", help="Quantization method (auto = detect from model)")
def inference_chat(model_path, system_prompt, max_tokens, temperature, tensor_parallel_size, quantization):
    """
    Interactive chat interface using vLLM inference engine.
    
    This command starts an interactive chat session with the model.
    Type your messages and press Enter. Type 'exit', 'quit', or press Ctrl+D to end.
    
    Examples:
    
        \b
        # Start a chat session
        uv run model-garden inference-chat --model-path ./models/my-model
        
        \b
        # Chat with system prompt
        uv run model-garden inference-chat \\
            --model-path ./models/my-model \\
            --system-prompt "You are a helpful AI assistant"
        
        \b
        # Chat with custom parameters
        uv run model-garden inference-chat \\
            --model-path ./models/my-model \\
            --temperature 0.8 \\
            --max-tokens 1024
    """
    try:
        from model_garden.inference import InferenceService
        import asyncio
        
        console.print("\n[bold cyan]💬 Model Garden - Interactive Chat[/bold cyan]\n")
        console.print(f"[cyan]Loading model:[/cyan] {model_path}\n")
        
        # Create inference service
        service = InferenceService(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            quantization=quantization
        )
        
        async def chat():
            # Load model
            await service.load_model()
            console.print("[green]✅ Model loaded![/green]\n")
            console.print("[yellow]Type your message and press Enter. Type 'exit' or 'quit' to end.[/yellow]\n")
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            try:
                while True:
                    # Get user input
                    try:
                        user_input = console.input("[bold blue]You:[/bold blue] ")
                    except EOFError:
                        break
                    
                    if user_input.strip().lower() in ["exit", "quit", ""]:
                        break
                    
                    # Add user message
                    messages.append({"role": "user", "content": user_input})
                    
                    # Generate response
                    console.print("\n[bold green]Assistant:[/bold green] ", end="")
                    
                    full_response = ""
                    stream_result = await service.chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True
                    )
                    
                    # Handle streaming response with type ignore
                    try:
                        async for chunk in stream_result:  # type: ignore
                            if isinstance(chunk, dict) and "choices" in chunk:
                                if chunk["choices"][0]["delta"].get("content"):
                                    content = chunk["choices"][0]["delta"]["content"]
                                    console.print(content, end="")
                                    full_response += content
                    except TypeError:
                        console.print("[red]Error: Stream response not iterable[/red]")
                    
                    console.print("\n")
                    
                    # Add assistant response to history
                    messages.append({"role": "assistant", "content": full_response})
            
            except KeyboardInterrupt:
                console.print("\n")
            
            # Cleanup
            console.print("\n[cyan]Cleaning up...[/cyan]")
            await service.unload_model()
        
        asyncio.run(chat())
        console.print("[green]✨ Chat session ended![/green]\n")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]\n")
        raise click.Abort()


if __name__ == "__main__":
    main()
