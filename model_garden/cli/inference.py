"""Inference commands for Model Garden CLI.

Contains:
- serve-model: Start an inference server with vLLM
- inference-generate: One-off text generation
- inference-chat: Interactive chat interface
"""

import click

from model_garden.utils.console import console


@click.command()
@click.option(
    "--model-path", required=True, help="Path to the model to serve (can be a LoRA adapter)"
)
@click.option(
    "--base-model",
    default=None,
    help="Base model path (only needed if adapter_config.json is missing)",
)
@click.option("--port", default=8000, help="Port to run the inference server on")
@click.option("--host", default="0.0.0.0", help="Host to bind the server to")
@click.option(
    "--tensor-parallel-size",
    default=None,
    type=int,
    help="Number of GPUs to use for tensor parallelism (default: from registry or 1)",
)
@click.option(
    "--gpu-memory-utilization",
    default=None,
    type=float,
    help="GPU memory utilization (0.0-1.0, default: from registry or auto)",
)
@click.option(
    "--quantization",
    type=click.Choice(["auto", "awq", "gptq", "squeezellm", "fp8", "bitsandbytes"]),
    default=None,
    help="Quantization method (default: from registry or auto)",
)
@click.option(
    "--max-model-len",
    type=int,
    default=None,
    help="Maximum sequence length (default: from registry)",
)
@click.option(
    "--dtype",
    type=click.Choice(["auto", "float16", "bfloat16", "float32"]),
    default=None,
    help="Data type (default: from registry or auto)",
)
@click.option("--enable-lora/--no-enable-lora", default=True, help="Enable LoRA adapter support")
@click.option(
    "--max-loras", default=1, type=int, help="Maximum number of LoRA adapters to load concurrently"
)
@click.option("--max-lora-rank", default=64, type=int, help="Maximum LoRA rank to support")
def serve_model(
    model_path,
    base_model,
    port,
    host,
    tensor_parallel_size,
    gpu_memory_utilization,
    quantization,
    max_model_len,
    dtype,
    enable_lora,
    max_loras,
    max_lora_rank,
):
    """
    Start an inference server with vLLM for high-throughput model serving.
    
    This command loads a model using vLLM and starts a FastAPI server
    with OpenAI-compatible endpoints for text generation and chat completions.
    
    Supports loading LoRA adapters directly from local paths or HuggingFace Hub.
    The adapter's base model is automatically detected from adapter_config.json.
    
    Parameters not specified will use defaults from the model registry if available.
    
    Examples:
    
        \b
        # Serve a merged model on default port 8000
        uv run model-garden serve-model --model-path ./models/my-model
        
        \b
        # Serve a LoRA adapter from HuggingFace Hub
        uv run model-garden serve-model \\
            --model-path Barth371/Qwen2.5-VL-72B-Instruct-bnb-4bit-2025-10-21_16-26_batch_size_4_cmr-block-2_adapters_4bit
        
        \b
        # Serve a local LoRA adapter with explicit base model
        uv run model-garden serve-model \\
            --model-path ./models/my-adapter \\
            --base-model Qwen/Qwen2.5-VL-72B-Instruct-bnb-4bit
        
        \b
        # Serve with custom GPU settings
        uv run model-garden serve-model \\
            --model-path ./models/my-model \\
            --port 8080 \\
            --tensor-parallel-size 2 \\
            --gpu-memory-utilization 0.8
    """
    try:
        import os

        import uvicorn

        from model_garden.model_registry import get_model

        console.print("\n[bold cyan]🚀 Model Garden - Inference Server[/bold cyan]\n")
        console.print(f"[cyan]Loading model:[/cyan] {model_path}")

        # Try to get model defaults from registry
        model_info = None
        try:
            model_info = get_model(model_path)
            if model_info:
                console.print(f"[green]📋 Found model in registry:[/green] {model_info.name}")
        except Exception:
            console.print("[yellow]ℹ️  Model not in registry, using defaults[/yellow]")

        # Apply defaults from registry if parameters not specified
        if model_info:
            if tensor_parallel_size is None:
                tensor_parallel_size = model_info.inference_defaults.tensor_parallel_size
                console.print(
                    f"  Using registry default tensor_parallel_size: {tensor_parallel_size}"
                )

            if gpu_memory_utilization is None:
                gpu_memory_utilization = model_info.inference_defaults.gpu_memory_utilization
                console.print(
                    f"  Using registry default gpu_memory_utilization: {gpu_memory_utilization}"
                )

            if quantization is None:
                quantization = model_info.inference_defaults.quantization or "auto"
                console.print(f"  Using registry default quantization: {quantization}")

            if max_model_len is None:
                max_model_len = model_info.inference_defaults.max_model_len
                console.print(f"  Using registry default max_model_len: {max_model_len}")

            if dtype is None:
                dtype = model_info.inference_defaults.dtype
                console.print(f"  Using registry default dtype: {dtype}")
        else:
            # Fallback defaults if not in registry
            if tensor_parallel_size is None:
                tensor_parallel_size = 1
            if gpu_memory_utilization is None:
                gpu_memory_utilization = 0.0  # auto mode
            if quantization is None:
                quantization = "auto"
            if dtype is None:
                dtype = "auto"

        # Reduce torch compile workers to save memory (default is 24, we use 8)
        os.environ["TORCH_COMPILE_MAX_WORKERS"] = "8"

        # Set environment variables for the API to pick up during lifespan startup
        # This ensures the model is loaded in the same process as the API
        os.environ["MODEL_GARDEN_AUTOLOAD_MODEL"] = model_path

        if base_model:
            os.environ["MODEL_GARDEN_BASE_MODEL"] = base_model
        if tensor_parallel_size > 1:
            os.environ["MODEL_GARDEN_TENSOR_PARALLEL_SIZE"] = str(tensor_parallel_size)
        # Always set GPU memory utilization
        os.environ["MODEL_GARDEN_GPU_MEMORY_UTILIZATION"] = str(gpu_memory_utilization)
        if quantization:
            os.environ["MODEL_GARDEN_QUANTIZATION"] = quantization
        if max_model_len:
            os.environ["MODEL_GARDEN_MAX_MODEL_LEN"] = str(max_model_len)
        if dtype:
            os.environ["MODEL_GARDEN_DTYPE"] = dtype

        # LoRA parameters
        os.environ["MODEL_GARDEN_ENABLE_LORA"] = str(enable_lora).lower()
        os.environ["MODEL_GARDEN_MAX_LORAS"] = str(max_loras)
        os.environ["MODEL_GARDEN_MAX_LORA_RANK"] = str(max_lora_rank)

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
            access_log=False,
        )

    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@click.command()
@click.option(
    "--model-path", required=True, help="Path to the model to use (can be a LoRA adapter)"
)
@click.option(
    "--base-model",
    default=None,
    help="Base model path (only needed if adapter_config.json is missing)",
)
@click.option("--prompt", required=True, help="Prompt for text generation")
@click.option("--max-tokens", default=256, help="Maximum number of tokens to generate")
@click.option("--temperature", default=0.7, type=float, help="Sampling temperature")
@click.option("--top-p", default=0.9, type=float, help="Top-p (nucleus) sampling parameter")
@click.option("--stream/--no-stream", default=False, help="Enable streaming output")
@click.option(
    "--tensor-parallel-size",
    default=None,
    type=int,
    help="Number of GPUs for tensor parallelism (default: from registry or 1)",
)
@click.option(
    "--gpu-memory-utilization",
    default=None,
    type=float,
    help="GPU memory utilization (0.0-1.0, default: from registry or auto)",
)
@click.option(
    "--quantization",
    type=click.Choice(["auto", "awq", "gptq", "squeezellm", "fp8", "bitsandbytes"]),
    default=None,
    help="Quantization method (default: from registry or auto)",
)
@click.option(
    "--max-model-len",
    type=int,
    default=None,
    help="Maximum sequence length (default: from registry)",
)
@click.option(
    "--dtype",
    type=click.Choice(["auto", "float16", "bfloat16", "float32"]),
    default=None,
    help="Data type (default: from registry or auto)",
)
@click.option("--enable-lora/--no-enable-lora", default=True, help="Enable LoRA adapter support")
@click.option("--max-loras", default=1, type=int, help="Maximum number of LoRA adapters")
@click.option("--max-lora-rank", default=64, type=int, help="Maximum LoRA rank")
def inference_generate(
    model_path,
    base_model,
    prompt,
    max_tokens,
    temperature,
    top_p,
    stream,
    tensor_parallel_size,
    gpu_memory_utilization,
    quantization,
    max_model_len,
    dtype,
    enable_lora,
    max_loras,
    max_lora_rank,
):
    """
    Generate text using vLLM inference engine (one-off generation).
    
    This command loads a model, generates a response, and exits.
    For persistent serving, use the 'serve-model' command instead.
    
    Supports loading LoRA adapters directly from local paths or HuggingFace Hub.
    
    Parameters not specified will use defaults from the model registry if available.
    
    Examples:
    
        \b
        # Generate with a merged model
        uv run model-garden inference-generate \\
            --model-path ./models/my-model \\
            --prompt "Once upon a time"
        
        \b
        # Generate with a LoRA adapter from HuggingFace Hub
        uv run model-garden inference-generate \\
            --model-path Barth371/Qwen2.5-VL-72B-Instruct-bnb-4bit-2025-10-21_16-26_batch_size_4_cmr-block-2_adapters_4bit \\
            --prompt "Extract information from this document"
        
        \b
        # Generate with streaming output
        uv run model-garden inference-generate \\
            --model-path ./models/my-model \\
            --prompt "Explain quantum computing" \\
            --max-tokens 512 \\
            --stream
    """
    try:
        import asyncio

        from model_garden.inference import InferenceService
        from model_garden.model_registry import get_model

        console.print("\n[bold cyan]🤖 Model Garden - Text Generation[/bold cyan]\n")
        console.print(f"[cyan]Loading model:[/cyan] {model_path}\n")

        # Try to get model defaults from registry
        model_info = None
        try:
            model_info = get_model(model_path)
            if model_info:
                console.print(f"[green]📋 Found model in registry:[/green] {model_info.name}\n")
        except Exception:
            pass

        # Apply defaults from registry if parameters not specified
        if model_info:
            if tensor_parallel_size is None:
                tensor_parallel_size = model_info.inference_defaults.tensor_parallel_size
            if gpu_memory_utilization is None:
                gpu_memory_utilization = model_info.inference_defaults.gpu_memory_utilization
            if quantization is None:
                quantization = model_info.inference_defaults.quantization or "auto"
            if max_model_len is None:
                max_model_len = model_info.inference_defaults.max_model_len
            if dtype is None:
                dtype = model_info.inference_defaults.dtype
        else:
            # Fallback defaults if not in registry
            if tensor_parallel_size is None:
                tensor_parallel_size = 1
            if gpu_memory_utilization is None:
                gpu_memory_utilization = 0.0  # auto mode
            if quantization is None:
                quantization = "auto"
            if dtype is None:
                dtype = "auto"

        # Create inference service
        service = InferenceService(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            quantization=quantization,
            enable_lora=enable_lora,
            max_loras=max_loras,
            max_lora_rank=max_lora_rank,
        )

        # If base_model is explicitly provided, override the auto-detection
        if base_model:
            console.print(f"[cyan]Using explicit base model: {base_model}[/cyan]")
            service.base_model_path = base_model
            service.is_adapter = True
            service.adapter_path = model_path

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
                    stream=True,
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
                    stream=False,
                )
                # Handle non-streaming response
                if isinstance(result, dict):
                    console.print(result.get("text", ""))
                    if "usage" in result:
                        console.print(
                            f"\n[dim]Tokens: {result['usage'].get('total_tokens', 0)}[/dim]\n"
                        )

            # Cleanup
            await service.unload_model()

        asyncio.run(generate())
        console.print("[green]✨ Generation complete![/green]\n")

    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]\n")
        raise click.Abort()


@click.command()
@click.option("--model-path", required=True, help="Path to the model to use")
@click.option("--system-prompt", help="System prompt for the chat")
@click.option("--max-tokens", default=512, help="Maximum tokens per response")
@click.option("--temperature", default=0.7, type=float, help="Sampling temperature")
@click.option(
    "--tensor-parallel-size", default=1, type=int, help="Number of GPUs for tensor parallelism"
)
@click.option(
    "--quantization",
    type=click.Choice(["auto", "awq", "gptq", "squeezellm", "fp8", "bitsandbytes"]),
    default="auto",
    help="Quantization method (auto = detect from model)",
)
def inference_chat(
    model_path, system_prompt, max_tokens, temperature, tensor_parallel_size, quantization
):
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
        import asyncio

        from model_garden.inference import InferenceService

        console.print("\n[bold cyan]💬 Model Garden - Interactive Chat[/bold cyan]\n")
        console.print(f"[cyan]Loading model:[/cyan] {model_path}\n")

        # Create inference service
        service = InferenceService(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            quantization=quantization,
        )

        async def chat():
            # Load model
            await service.load_model()
            console.print("[green]✅ Model loaded![/green]\n")
            console.print(
                "[yellow]Type your message and press Enter. Type 'exit' or 'quit' to end.[/yellow]\n"
            )

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
                        stream=True,
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
