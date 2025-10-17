#!/usr/bin/env python3
"""
Example: Demonstrating Automatic GPU Memory Utilization

This example shows how the automatic GPU memory calculation works
with different model configurations.
"""

import asyncio
from model_garden.inference import InferenceService, calculate_gpu_memory_utilization
from rich.console import Console
from rich.table import Table

console = Console()


def demo_calculation():
    """Demonstrate GPU memory calculation for various models."""
    console.print("\n[bold cyan]🎯 GPU Memory Utilization Demo[/bold cyan]\n")
    
    # Different model configurations
    configs = [
        {
            "name": "Small Vision Model (3B)",
            "model_path": "Qwen/Qwen2.5-VL-3B-Instruct",
            "max_model_len": 4096,
            "tensor_parallel": 1,
        },
        {
            "name": "Medium Text Model (7B)",
            "model_path": "meta-llama/Llama-3.2-7B",
            "max_model_len": 4096,
            "tensor_parallel": 1,
        },
        {
            "name": "Large Context (7B, 8K)",
            "model_path": "meta-llama/Llama-3.2-7B",
            "max_model_len": 8192,
            "tensor_parallel": 1,
        },
        {
            "name": "Tiny Model (1B)",
            "model_path": "meta-llama/Llama-3.2-1B",
            "max_model_len": 2048,
            "tensor_parallel": 1,
        },
    ]
    
    # Create results table
    table = Table(title="Automatic GPU Memory Utilization Calculations")
    table.add_column("Configuration", style="cyan")
    table.add_column("Model Path", style="blue")
    table.add_column("Context", justify="right", style="yellow")
    table.add_column("Utilization", justify="right", style="green")
    table.add_column("Strategy", style="magenta")
    
    for config in configs:
        util = calculate_gpu_memory_utilization(
            model_path=config["model_path"],
            max_model_len=config["max_model_len"],
            tensor_parallel_size=config["tensor_parallel"],
        )
        
        # Determine strategy
        if util >= 0.95:
            strategy = "Aggressive"
        elif util >= 0.85:
            strategy = "Standard"
        else:
            strategy = "Conservative"
        
        console.print()  # Add spacing between calculations
        
        table.add_row(
            config["name"],
            config["model_path"].split("/")[-1],
            f"{config['max_model_len']:,}",
            f"{util:.2f}",
            strategy,
        )
    
    console.print("\n")
    console.print(table)


async def demo_service():
    """Demonstrate InferenceService with auto mode."""
    console.print("\n\n[bold cyan]🚀 InferenceService Auto Mode Demo[/bold cyan]\n")
    
    console.print("[yellow]Creating service with gpu_memory_utilization=0 (auto mode)...[/yellow]\n")
    
    service = InferenceService(
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        gpu_memory_utilization=0.0,  # Auto mode
        max_model_len=4096,
    )
    
    console.print(f"[cyan]Service created![/cyan]")
    console.print(f"  Model path: {service.model_path}")
    console.print(f"  GPU memory utilization: {service.gpu_memory_utilization} (auto)")
    console.print(f"  Max model length: {service.max_model_len}")
    
    console.print("\n[yellow]Note:[/yellow] When you call service.load_model(), the system will:")
    console.print("  1. Detect your GPU memory (e.g., 24GB)")
    console.print("  2. Estimate the model size (e.g., 6GB for 3B params)")
    console.print("  3. Calculate KV cache requirements (e.g., 2.6GB for 4K context)")
    console.print("  4. Choose optimal utilization (e.g., 0.57 for plenty of headroom)")
    
    console.print("\n[green]✨ Auto mode ensures optimal performance without manual tuning![/green]\n")


def main():
    """Run all demos."""
    try:
        # Demo 1: Show calculations for various models
        demo_calculation()
        
        # Demo 2: Show service creation with auto mode
        asyncio.run(demo_service())
        
        console.print("\n[bold green]✅ Demo complete![/bold green]")
        console.print("\n[dim]To actually load a model, use:[/dim]")
        console.print("[dim]  uv run model-garden serve-model --model-path <model-path>[/dim]\n")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
