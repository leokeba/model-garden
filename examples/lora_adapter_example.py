"""Example: Loading and using a LoRA adapter from HuggingFace Hub

This example demonstrates how to load and use the LoRA adapter:
Barth371/Qwen2.5-VL-72B-Instruct-bnb-4bit-2025-10-21_16-26_batch_size_4_cmr-block-2_adapters_4bit

The adapter will be automatically detected and applied to the base model.
"""

import asyncio
from model_garden.inference import InferenceService
from rich.console import Console

console = Console()


async def main():
    """Load and test a LoRA adapter from HuggingFace Hub."""
    
    # The adapter to load (this is a real adapter on HuggingFace Hub)
    adapter_path = "Barth371/Qwen2.5-VL-72B-Instruct-bnb-4bit-2025-10-21_16-26_batch_size_4_cmr-block-2_adapters_4bit"
    
    console.print("\n[bold cyan]🌱 Model Garden - LoRA Adapter Loading Example[/bold cyan]\n")
    console.print(f"[cyan]Adapter:[/cyan] {adapter_path}\n")
    
    # Create inference service
    # The base model will be automatically detected from adapter_config.json
    console.print("[cyan]Creating inference service...[/cyan]")
    service = InferenceService(
        model_path=adapter_path,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=64,
        gpu_memory_utilization=0.85,  # Adjust based on your GPU
        tensor_parallel_size=1  # Use 2+ if you have multiple GPUs
    )
    
    # Load the model (this will detect the adapter and load the base model)
    console.print("\n[cyan]Loading model (this may take a few minutes)...[/cyan]")
    await service.load_model()
    
    # Get model info to verify everything loaded correctly
    info = service.get_model_info()
    console.print("\n[bold green]✓ Model loaded successfully![/bold green]\n")
    console.print("[cyan]Model Information:[/cyan]")
    for key, value in info.items():
        console.print(f"  {key}: {value}")
    
    # Test generation with a simple prompt
    console.print("\n[cyan]Testing generation...[/cyan]\n")
    
    prompt = "Extract information from this document"
    
    console.print(f"[bold]Prompt:[/bold] {prompt}")
    console.print("\n[bold]Response:[/bold]\n")
    
    result = await service.generate(
        prompt=prompt,
        max_tokens=256,
        temperature=0.7,
        stream=False
    )
    
    if isinstance(result, dict):
        console.print(result.get("text", ""))
        console.print(f"\n[dim]Tokens: {result.get('usage', {}).get('total_tokens', 0)}[/dim]")
    
    # Cleanup
    console.print("\n[cyan]Cleaning up...[/cyan]")
    await service.unload_model()
    
    console.print("[bold green]✨ Example complete![/bold green]\n")


if __name__ == "__main__":
    asyncio.run(main())
