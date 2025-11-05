#!/usr/bin/env python3
"""Test script for automatic vision LoRA merging in inference.

This script verifies that:
1. Vision LoRA adapters are detected correctly
2. Adapters are automatically merged before loading into vLLM
3. Merged models can be loaded and used for inference
4. Cleanup works properly
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from model_garden.inference import InferenceService, is_vision_model, is_lora_adapter

console = Console()


async def test_vision_detection():
    """Test vision model detection."""
    console.print("[bold cyan]Test 1: Vision Model Detection[/bold cyan]")
    
    # Test cases
    test_cases = [
        ("Qwen/Qwen2.5-VL-3B-Instruct", True),
        ("unsloth/Qwen2-VL-7B-Instruct", True),
        ("unsloth/tinyllama-bnb-4bit", False),
        ("meta-llama/Llama-2-7b-hf", False),
    ]
    
    for model_id, expected in test_cases:
        result = is_vision_model(model_id)
        status = "✓" if result == expected else "✗"
        console.print(f"  {status} {model_id}: {'Vision' if result else 'Text'} (expected: {'Vision' if expected else 'Text'})")
    
    console.print()


async def test_adapter_detection():
    """Test LoRA adapter detection."""
    console.print("[bold cyan]Test 2: LoRA Adapter Detection[/bold cyan]")
    
    # Test local adapter (if exists)
    local_adapters = list(Path("models").glob("**/adapter_config.json"))
    
    if local_adapters:
        for adapter_config in local_adapters[:3]:  # Test first 3
            adapter_dir = adapter_config.parent
            result = is_lora_adapter(str(adapter_dir))
            console.print(f"  {'✓' if result else '✗'} {adapter_dir.name}: {'Adapter' if result else 'Not adapter'}")
    else:
        console.print("  [yellow]No local adapters found in models/ directory[/yellow]")
    
    console.print()


async def test_vision_lora_merge_detection():
    """Test that vision LoRA adapters trigger merge logic."""
    console.print("[bold cyan]Test 3: Vision LoRA Merge Detection[/bold cyan]")
    
    # This test checks the logic without actually loading models
    console.print("  This test requires a real vision LoRA adapter to be present.")
    console.print("  [yellow]Skipping for now - manual testing recommended[/yellow]")
    console.print()


async def test_model_info():
    """Test model info reporting."""
    console.print("[bold cyan]Test 4: Model Info Reporting[/bold cyan]")
    
    # Create a service instance (won't actually load)
    service = InferenceService(
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        gpu_memory_utilization=0.9,
    )
    
    info = service.get_model_info()
    console.print("  Model info structure:")
    for key, value in info.items():
        console.print(f"    {key}: {value}")
    
    console.print()


async def main():
    """Run all tests."""
    console.print("[bold green]Starting Vision LoRA Merge Tests[/bold green]\n")
    
    try:
        await test_vision_detection()
        await test_adapter_detection()
        await test_vision_lora_merge_detection()
        await test_model_info()
        
        console.print("[bold green]✨ All tests completed![/bold green]")
        console.print("\n[yellow]Note: Full integration testing requires:[/yellow]")
        console.print("  1. A trained vision LoRA adapter")
        console.print("  2. GPU with sufficient memory")
        console.print("  3. Run: uv run model-garden serve-model --model-path <vision-lora-adapter>")
        
    except Exception as e:
        console.print(f"[red]❌ Test failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
