"""Test loading the real LoRA adapter from HuggingFace Hub.

This script tests loading:
Barth371/Qwen2.5-VL-72B-Instruct-bnb-4bit-2025-10-21_16-26_batch_size_4_cmr-block-2_adapters_4bit

Expected behavior:
1. Detect it's a LoRA adapter
2. Extract base model: unsloth/qwen2.5-vl-72b-instruct-bnb-4bit
3. Load base model with LoRA support
4. Apply adapter to requests
"""

import asyncio
import os
from pathlib import Path
from rich.console import Console
from model_garden.inference import InferenceService, is_lora_adapter, get_base_model_from_adapter

console = Console()

# Load environment variables from .env file
def load_env():
    """Load environment variables from .env file."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        console.print(f"[cyan]Loading environment from {env_file}[/cyan]")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
                    if key in ['HF_HOME', 'HF_TOKEN', 'TRANSFORMERS_CACHE']:
                        console.print(f"  Set {key}={value}")
    else:
        console.print(f"[yellow]⚠️  No .env file found at {env_file}[/yellow]")

# Load env before importing anything else that might use HF
load_env()

# Set HuggingFace cache directories explicitly
if 'HF_HOME' in os.environ:
    hf_home = os.environ['HF_HOME']
    os.environ['TRANSFORMERS_CACHE'] = f"{hf_home}/transformers"
    os.environ['HF_DATASETS_CACHE'] = f"{hf_home}/datasets"
    console.print(f"[green]✓ HuggingFace cache set to: {hf_home}[/green]\n")


async def test_lora_loading():
    """Test loading the real LoRA adapter."""
    
    adapter_path = "Barth371/Qwen2.5-VL-72B-Instruct-bnb-4bit-2025-10-21_16-26_batch_size_4_cmr-block-2_adapters_4bit"
    
    console.print("\n[bold cyan]🧪 Testing Real LoRA Adapter Loading[/bold cyan]\n")
    console.print(f"[cyan]Adapter:[/cyan] {adapter_path}\n")
    
    # Step 1: Verify it's detected as an adapter
    console.print("[bold]Step 1: Adapter Detection[/bold]")
    is_adapter = is_lora_adapter(adapter_path)
    console.print(f"  Is adapter: {is_adapter}")
    
    if not is_adapter:
        console.print("[red]❌ Failed: Not detected as adapter[/red]")
        return False
    
    # Step 2: Get base model
    console.print("\n[bold]Step 2: Base Model Extraction[/bold]")
    base_model = get_base_model_from_adapter(adapter_path)
    console.print(f"  Base model: {base_model}")
    
    if not base_model:
        console.print("[red]❌ Failed: Could not determine base model[/red]")
        return False
    
    # Step 3: Create inference service
    console.print("\n[bold]Step 3: Creating Inference Service[/bold]")
    
    # Note: This is a 72B model, which requires significant VRAM
    # We'll configure it conservatively
    service = InferenceService(
        model_path=adapter_path,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=64,
        gpu_memory_utilization=0.95,  # Use most of available memory
        tensor_parallel_size=1,  # Change to 2+ if you have multiple GPUs
        max_model_len=4096,  # Conservative context length
        dtype="auto"
    )
    
    console.print("  ✓ Service created")
    
    # Step 4: Check service configuration
    console.print("\n[bold]Step 4: Service Configuration[/bold]")
    console.print(f"  Model path: {service.model_path}")
    console.print(f"  Base model: {service.base_model_path}")
    console.print(f"  Adapter path: {service.adapter_path}")
    console.print(f"  Is adapter: {service.is_adapter}")
    console.print(f"  LoRA enabled: {service.enable_lora}")
    console.print(f"  Max LoRAs: {service.max_loras}")
    console.print(f"  Max LoRA rank: {service.max_lora_rank}")
    
    # Step 5: Load the model
    console.print("\n[bold]Step 5: Loading Model[/bold]")
    console.print("[yellow]⚠️  This will download the 72B model (~50-70GB) if not cached[/yellow]")
    console.print("[yellow]⚠️  This may take 5-10 minutes depending on your connection[/yellow]")
    console.print("[yellow]⚠️  Ensure you have sufficient VRAM (recommended: 48GB+)[/yellow]\n")
    
    # Ask for confirmation
    try:
        response = input("Continue with model loading? (y/n): ")
        if response.lower() != 'y':
            console.print("\n[yellow]Skipping model loading (user cancelled)[/yellow]")
            console.print("[green]✓ Configuration tests passed![/green]\n")
            return True
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Skipping model loading (cancelled)[/yellow]")
        console.print("[green]✓ Configuration tests passed![/green]\n")
        return True
    
    try:
        await service.load_model()
        console.print("\n[green]✓ Model loaded successfully![/green]")
        
        # Step 6: Get model info
        console.print("\n[bold]Step 6: Model Information[/bold]")
        info = service.get_model_info()
        for key, value in info.items():
            console.print(f"  {key}: {value}")
        
        # Step 7: Test generation
        console.print("\n[bold]Step 7: Test Generation[/bold]")
        console.print("[cyan]Generating test response...[/cyan]\n")
        
        prompt = "What is machine learning?"
        result = await service.generate(
            prompt=prompt,
            max_tokens=128,
            temperature=0.7,
            stream=False
        )
        
        if isinstance(result, dict):
            console.print(f"[bold]Prompt:[/bold] {prompt}")
            console.print(f"\n[bold]Response:[/bold]\n{result.get('text', '')}")
            console.print(f"\n[dim]Tokens: {result.get('usage', {}).get('total_tokens', 0)}[/dim]")
        
        console.print("\n[green]✓ Generation successful![/green]")
        
        # Cleanup
        console.print("\n[bold]Step 8: Cleanup[/bold]")
        await service.unload_model()
        console.print("  ✓ Model unloaded")
        
        console.print("\n[bold green]✨ All tests passed![/bold green]\n")
        return True
        
    except Exception as e:
        console.print(f"\n[red]❌ Error during model loading: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(test_lora_loading())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Test cancelled by user[/yellow]\n")
        exit(0)
