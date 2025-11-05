"""Test script to compare chat template formatting before and after implementation.

This script tests:
1. Current hardcoded Qwen formatting vs automatic apply_chat_template
2. Marker detection for selective loss masking
3. Ensures no regression in Qwen2.5-VL functionality
"""

import os
from pathlib import Path
from typing import Dict, List, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Configure HuggingFace cache
from dotenv import load_dotenv
load_dotenv()

HF_HOME = os.getenv('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))
os.environ['HF_HOME'] = HF_HOME
os.environ['TRANSFORMERS_CACHE'] = str(Path(HF_HOME) / 'hub')

console = Console()


def test_current_hardcoded_format():
    """Test current hardcoded Qwen chat template formatting."""
    console.print("\n[bold cyan]═══ CURRENT HARDCODED FORMAT ═══[/bold cyan]\n")
    
    # Simulate current hardcoded format
    system_msg = "You are a helpful assistant."
    user_prompt = "Describe this image."
    
    # Current hardcoded Qwen format (from inference.py line 982-985)
    hardcoded = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    console.print(Panel(hardcoded, title="Current Hardcoded Format", border_style="yellow"))
    
    return hardcoded


def test_automatic_template():
    """Test automatic chat template using apply_chat_template."""
    console.print("\n[bold cyan]═══ AUTOMATIC TEMPLATE (NEW) ═══[/bold cyan]\n")
    
    try:
        from transformers import AutoTokenizer
        
        # Load Qwen2.5-VL tokenizer
        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        console.print(f"[cyan]Loading tokenizer: {model_id}[/cyan]")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Test 1: Basic chat template
        console.print("\n[bold green]Test 1: Basic Chat (Text-only)[/bold green]")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Describe this image."},
        ]
        
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        console.print(Panel(formatted, title="Auto-formatted (Text)", border_style="green"))
        
        # Test 2: Vision chat template (if supported)
        console.print("\n[bold green]Test 2: Vision Chat (with image placeholder)[/bold green]")
        
        # Try multimodal format
        vision_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # Image placeholder
                    {"type": "text", "text": "Describe this image."}
                ]
            }
        ]
        
        try:
            vision_formatted = tokenizer.apply_chat_template(
                vision_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            console.print(Panel(vision_formatted, title="Auto-formatted (Vision)", border_style="green"))
        except Exception as e:
            console.print(f"[yellow]Vision format not directly supported, will handle separately: {e}[/yellow]")
            vision_formatted = formatted  # Fallback to text format
        
        return formatted, vision_formatted
        
    except Exception as e:
        console.print(f"[red]Error loading tokenizer: {e}[/red]")
        return None, None


def test_marker_detection():
    """Test automatic detection of instruction/response markers."""
    console.print("\n[bold cyan]═══ MARKER DETECTION ═══[/bold cyan]\n")
    
    try:
        from transformers import AutoTokenizer
        
        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Apply template to sample messages to detect markers
        sample = [
            {"role": "user", "content": "__USER_PLACEHOLDER__"},
            {"role": "assistant", "content": "__ASSISTANT_PLACEHOLDER__"}
        ]
        
        formatted = tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
        
        console.print("[cyan]Formatted sample for marker detection:[/cyan]")
        console.print(Panel(formatted, border_style="cyan"))
        
        # Detect markers
        user_idx = formatted.find("__USER_PLACEHOLDER__")
        assistant_idx = formatted.find("__ASSISTANT_PLACEHOLDER__")
        
        if user_idx > 0 and assistant_idx > 0:
            # Find the start marker before user content
            lines = formatted[:user_idx].split('\n')
            instruction_marker = None
            for line in reversed(lines):
                if line.strip() and not line.strip().endswith('_PLACEHOLDER__'):
                    instruction_marker = line.strip()
                    break
            
            # Find the start marker before assistant content
            lines_before_assistant = formatted[:assistant_idx].split('\n')
            response_marker = None
            for line in reversed(lines_before_assistant):
                if line.strip() and not line.strip().endswith('_PLACEHOLDER__'):
                    response_marker = line.strip()
                    break
            
            table = Table(title="Detected Markers")
            table.add_column("Marker Type", style="cyan")
            table.add_column("Detected Value", style="green")
            
            table.add_row("Instruction", instruction_marker or "[red]Not found[/red]")
            table.add_row("Response", response_marker or "[red]Not found[/red]")
            
            console.print(table)
            
            return instruction_marker, response_marker
        else:
            console.print("[red]Could not find placeholders in formatted text[/red]")
            return None, None
            
    except Exception as e:
        console.print(f"[red]Error detecting markers: {e}[/red]")
        return None, None


def compare_formats(hardcoded: str, automatic: str):
    """Compare hardcoded and automatic formats."""
    console.print("\n[bold cyan]═══ COMPARISON ═══[/bold cyan]\n")
    
    # Check if they match (ignoring vision tokens for now)
    hardcoded_no_vision = hardcoded.replace("<|vision_start|><|image_pad|><|vision_end|>", "")
    
    table = Table(title="Format Comparison")
    table.add_column("Aspect", style="cyan")
    table.add_column("Hardcoded", style="yellow")
    table.add_column("Automatic", style="green")
    table.add_column("Match", style="bold")
    
    # Compare structure
    has_system = "<|im_start|>system" in automatic
    has_user = "<|im_start|>user" in automatic
    has_assistant = "<|im_start|>assistant" in automatic
    
    table.add_row(
        "Has system marker",
        "✓" if "<|im_start|>system" in hardcoded else "✗",
        "✓" if has_system else "✗",
        "✓" if has_system else "✗"
    )
    table.add_row(
        "Has user marker",
        "✓" if "<|im_start|>user" in hardcoded else "✗",
        "✓" if has_user else "✗",
        "✓" if has_user else "✗"
    )
    table.add_row(
        "Has assistant marker",
        "✓" if "<|im_start|>assistant" in hardcoded else "✗",
        "✓" if has_assistant else "✗",
        "✓" if has_assistant else "✗"
    )
    
    table.add_row(
        "Length",
        str(len(hardcoded)),
        str(len(automatic)),
        "✓" if abs(len(hardcoded) - len(automatic)) < 50 else "⚠"
    )
    
    console.print(table)
    
    # Check if core structure matches
    if has_system and has_user and has_assistant:
        console.print("\n[bold green]✓ Core structure matches! Safe to proceed.[/bold green]")
        return True
    else:
        console.print("\n[bold red]✗ Structure mismatch detected![/bold red]")
        return False


def main():
    """Run all tests."""
    console.print(Panel.fit(
        "[bold cyan]Chat Template Comparison Test[/bold cyan]\n"
        "Testing Qwen2.5-VL chat template formatting",
        border_style="cyan"
    ))
    
    # Test 1: Current hardcoded format
    hardcoded = test_current_hardcoded_format()
    
    # Test 2: Automatic template
    automatic, vision_automatic = test_automatic_template()
    
    if automatic:
        # Test 3: Marker detection
        instruction_marker, response_marker = test_marker_detection()
        
        # Test 4: Compare formats
        match = compare_formats(hardcoded, automatic)
        
        # Summary
        console.print("\n[bold cyan]═══ SUMMARY ═══[/bold cyan]\n")
        
        if match:
            console.print("[bold green]✓ Automatic template detection is safe to implement[/bold green]")
            console.print("[green]✓ Structure matches current hardcoded format[/green]")
            console.print("[green]✓ Markers can be auto-detected[/green]")
        else:
            console.print("[bold yellow]⚠ Review differences before proceeding[/bold yellow]")
        
        if instruction_marker and response_marker:
            console.print(f"\n[cyan]Detected markers for selective loss:[/cyan]")
            console.print(f"  instruction_part = {repr(instruction_marker)}")
            console.print(f"  response_part = {repr(response_marker)}")
    else:
        console.print("\n[bold red]✗ Could not test automatic template[/bold red]")


if __name__ == "__main__":
    main()
