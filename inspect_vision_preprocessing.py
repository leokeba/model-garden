#!/usr/bin/env python3
"""
Inspect the vision dataset preprocessing pipeline.

This script mimics the production preprocessing flow:
1. Loads a vision dataset from JSONL (OpenAI messages format)
2. Converts to the format expected by UnslothVisionDataCollator
3. Applies the chat template via the processor (adds <|im_start|> markers)
4. Shows FULL contents of prompts and responses (except base64 images)
5. Analyzes token counts and masking behavior

Key Findings:
- Images are decoded from base64 to PIL Images
- System and user prompts consume ~90-99% of tokens but are MASKED
- Only assistant responses (1-10% of tokens) contribute to training loss
- Vision tokens (<|vision_start|><|image_pad|><|vision_end|>) are auto-inserted
- Chat markers (<|im_start|>user/assistant) delineate prompt vs response
- train_on_responses_only=True masks everything before <|im_start|>assistant

Usage:
    python inspect_vision_preprocessing.py
    
Output:
    - Full system messages, user prompts, and assistant responses
    - Token counts and percentages for masking analysis
    - Template snippets showing marker positions
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any
import base64
import io

# Configure HF cache before imports
from dotenv import load_dotenv
load_dotenv()

HF_HOME = os.getenv('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))
os.environ['HF_HOME'] = HF_HOME
os.environ['TRANSFORMERS_CACHE'] = str(Path(HF_HOME) / 'hub')
os.environ['HF_DATASETS_CACHE'] = str(Path(HF_HOME) / 'datasets')

# Import unsloth first!
import unsloth  # noqa
from transformers import AutoProcessor
from PIL import Image
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


def load_jsonl(filepath: str, max_examples: int = 5) -> List[Dict]:
    """Load JSONL file and return first N examples."""
    examples = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            examples.append(json.loads(line))
    return examples


def decode_base64_image(image_str: str) -> Image.Image:
    """Decode base64 image to PIL Image."""
    try:
        # Remove data URI prefix if present
        if image_str.startswith("data:"):
            image_str = image_str.split(",", 1)[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_str)
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        console.print(f"[yellow]⚠️  Failed to decode image: {e}[/yellow]")
        return Image.new("RGB", (224, 224))


def convert_messages_to_simple_format(messages: List[Dict]) -> Dict[str, Any]:
    """Convert OpenAI messages format to simple format."""
    result = {
        "text": "",
        "image": None,
        "response": "",
        "system": ""
    }
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", [])
        
        if role == "system":
            for item in content:
                if item.get("type") == "text" and not result["system"]:
                    result["system"] = item.get("text", "")
        
        elif role == "user":
            for item in content:
                item_type = item.get("type", "")
                if item_type == "text" and not result["text"]:
                    result["text"] = item.get("text", "")
                elif item_type in ("image", "image_url") and not result["image"]:
                    image_data = item.get("image", item.get("image_url", {}))
                    if isinstance(image_data, dict):
                        image_data = image_data.get("url", "")
                    result["image"] = image_data
        
        elif role == "assistant":
            for item in content:
                if item.get("type") == "text" and not result["response"]:
                    result["response"] = item.get("text", "")
    
    return result


def format_for_unsloth(example: Dict, system_message: str) -> Dict:
    """Format example into UnslothVisionDataCollator expected format."""
    # Check if this is OpenAI messages format
    if "messages" in example:
        simple = convert_messages_to_simple_format(example["messages"])
        text = simple.get("text", "")
        response = simple.get("response", "")
        image_data = simple.get("image", "")
        original_system = simple.get("system", "")
        effective_system = original_system if original_system else system_message
    else:
        # Simple format
        text = example.get("text", "")
        response = example.get("response", example.get("output", ""))
        image_data = example.get("image", "")
        effective_system = system_message
    
    # Load image (handles base64)
    if isinstance(image_data, str):
        if image_data.startswith("data:"):
            pil_image = decode_base64_image(image_data)
        elif os.path.exists(image_data):
            pil_image = Image.open(image_data)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            pil_image.load()
        else:
            pil_image = Image.new("RGB", (224, 224))
    else:
        pil_image = Image.new("RGB", (224, 224))
    
    # Return in UnslothVisionDataCollator format
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": effective_system}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            },
        ],
    }


def apply_chat_template_and_analyze(
    formatted_example: Dict,
    processor: Any,
    example_num: int
) -> None:
    """Apply chat template and analyze the result."""
    console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
    console.print(f"[bold cyan]Example #{example_num}[/bold cyan]")
    console.print(f"[bold cyan]{'='*80}[/bold cyan]")
    
    messages = formatted_example["messages"]
    
    # Extract components for display
    system_msg = messages[0]["content"][0]["text"]
    user_text = messages[1]["content"][1]["text"]
    user_image = messages[1]["content"][0]["image"]
    assistant_response = messages[2]["content"][0]["text"]
    
    # Display structured info
    console.print("\n[bold yellow]📋 Message Structure:[/bold yellow]")
    console.print(f"  System message length: {len(system_msg)} chars")
    console.print(f"  User text length: {len(user_text)} chars")
    console.print(f"  Image size: {user_image.size}")
    console.print(f"  Assistant response length: {len(assistant_response)} chars")
    
    # Show system message (FULL)
    console.print("\n[bold yellow]🤖 System Message (FULL):[/bold yellow]")
    console.print(Panel(system_msg, border_style="yellow", expand=False))
    
    # Show user prompt (FULL)
    console.print("\n[bold yellow]👤 User Prompt (FULL):[/bold yellow]")
    console.print(Panel(user_text, border_style="blue", expand=False))
    
    # Show assistant response (FULL)
    console.print("\n[bold yellow]🤖 Assistant Response (FULL):[/bold yellow]")
    console.print(Panel(assistant_response, border_style="green", expand=False))
    
    # Apply chat template (tokenize=False to see the text)
    console.print("\n[bold yellow]🔧 Applying Chat Template...[/bold yellow]")
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    
    # Find markers
    user_marker = "<|im_start|>user"
    assistant_marker = "<|im_start|>assistant"
    
    user_pos = text.find(user_marker)
    assistant_pos = text.find(assistant_marker)
    
    # Analyze the templated text
    console.print("\n[bold yellow]📊 Template Analysis:[/bold yellow]")
    console.print(f"  Total templated text length: {len(text)} chars")
    console.print(f"  User marker position: {user_pos}")
    console.print(f"  Assistant marker position: {assistant_pos}")
    
    if user_pos >= 0 and assistant_pos >= 0:
        prompt_section = text[:assistant_pos]
        response_section = text[assistant_pos:]
        
        console.print(f"  Prompt section (system + user): {len(prompt_section)} chars ({100*len(prompt_section)/len(text):.1f}%)")
        console.print(f"  Response section: {len(response_section)} chars ({100*len(response_section)/len(text):.1f}%)")
        
        # Check for vision tokens
        has_vision_tokens = "<|vision_start|>" in text or "<|image_pad|>" in text
        console.print(f"  Contains vision tokens: {has_vision_tokens}")
        
        # Show snippet with markers
        console.print("\n[bold yellow]📝 Templated Text Snippet (showing markers):[/bold yellow]")
        
        # Show around user marker
        snippet_start = max(0, user_pos - 50)
        snippet_end = min(len(text), user_pos + 200)
        snippet = text[snippet_start:snippet_end]
        console.print("\n  [dim]... around user marker ...[/dim]")
        console.print(Panel(snippet.replace(user_marker, f"[bold red]{user_marker}[/bold red]"), 
                           border_style="cyan"))
        
        # Show around assistant marker
        snippet_start = max(0, assistant_pos - 100)
        snippet_end = min(len(text), assistant_pos + 100)
        snippet = text[snippet_start:snippet_end]
        console.print("\n  [dim]... around assistant marker ...[/dim]")
        console.print(Panel(snippet.replace(assistant_marker, f"[bold red]{assistant_marker}[/bold red]"), 
                           border_style="cyan"))
    
    # Tokenize to count tokens
    console.print("\n[bold yellow]🔢 Tokenization Analysis:[/bold yellow]")
    tokenized = processor(
        text=text,
        images=user_image,
        return_tensors="pt",
        padding=False,
    )
    
    input_ids = tokenized["input_ids"][0]
    total_tokens = len(input_ids)
    
    console.print(f"  Total tokens: {total_tokens}")
    
    # Estimate tokens for each section
    # Note: The processor adds vision tokens automatically, so we can't accurately split
    # just by tokenizing the text sections separately. Instead, we'll find the assistant
    # marker in the token IDs.
    if user_pos >= 0 and assistant_pos > 0:
        # Find where the assistant marker appears in tokens
        assistant_marker_text = "<|im_start|>assistant"
        assistant_marker_ids = processor.tokenizer.encode(assistant_marker_text, add_special_tokens=False)
        
        # Search for the marker in the token sequence
        marker_pos = None
        for i in range(len(input_ids) - len(assistant_marker_ids) + 1):
            if input_ids[i:i+len(assistant_marker_ids)].tolist() == assistant_marker_ids:
                marker_pos = i
                break
        
        if marker_pos is not None:
            prompt_tokens = marker_pos
            response_tokens = total_tokens - marker_pos
            
            console.print(f"  Prompt tokens (to be MASKED): {prompt_tokens} ({100*prompt_tokens/total_tokens:.1f}%)")
            console.print(f"  Response tokens (to be TRAINED): {response_tokens} ({100*response_tokens/total_tokens:.1f}%)")
            console.print(f"  [dim]Note: Prompt includes system message, vision tokens, and user text[/dim]")
        else:
            console.print(f"  [yellow]⚠️  Could not locate assistant marker in tokens[/yellow]")
    
    # Show masking behavior
    console.print("\n[bold yellow]🎯 Masking Behavior with train_on_responses_only=True:[/bold yellow]")
    console.print(Panel(
        f"[green]✓[/green] All tokens before '{assistant_marker}' will be MASKED (loss ignored)\n"
        f"[green]✓[/green] All tokens after '{assistant_marker}' will be TRAINED (loss computed)\n"
        f"[yellow]ℹ️[/yellow]  This ensures the model learns to generate responses, not memorize prompts\n"
        f"[yellow]ℹ️[/yellow]  Vision tokens are automatically handled by the processor",
        border_style="green"
    ))


def main():
    """Main inspection function."""
    console.print("\n[bold cyan]🔍 Vision Dataset Preprocessing Inspector[/bold cyan]\n")
    
    # Dataset path
    dataset_path = "storage/datasets/openai_finetune_vision_inline_20251017_114023.jsonl"
    
    if not os.path.exists(dataset_path):
        console.print(f"[red]❌ Dataset not found: {dataset_path}[/red]")
        return
    
    console.print(f"[cyan]Loading dataset from: {dataset_path}[/cyan]")
    
    # Load examples
    max_examples = 3  # Reduced to 2 since we're showing full content now
    examples = load_jsonl(dataset_path, max_examples=max_examples)
    console.print(f"[green]✓ Loaded {len(examples)} examples[/green]")
    
    # Load processor
    console.print("\n[cyan]Loading Qwen2.5-VL processor...[/cyan]")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    console.print("[green]✓ Processor loaded[/green]")
    
    # Default system message
    system_message = "You are a helpful assistant that can analyze images."
    
    # Process each example
    for i, example in enumerate(examples, 1):
        # Format for Unsloth
        formatted = format_for_unsloth(example, system_message)
        
        # Analyze
        apply_chat_template_and_analyze(formatted, processor, i)
    
    # Summary
    console.print("\n[bold cyan]{'='*80}[/bold cyan]")
    console.print("[bold cyan]Summary[/bold cyan]")
    console.print("[bold cyan]{'='*80}[/bold cyan]\n")
    
    console.print("[bold yellow]Production Pipeline Steps:[/bold yellow]")
    console.print("  1. Load JSONL with OpenAI messages format")
    console.print("  2. Convert to simple format (system, user, assistant)")
    console.print("  3. Load images from base64 to PIL Images")
    console.print("  4. Format as messages dict for UnslothVisionDataCollator")
    console.print("  5. Apply chat template to add markers (<|im_start|>user, etc.)")
    console.print("  6. Tokenize with processor (handles both text and images)")
    console.print("  7. Apply masking (train_on_responses_only masks everything before assistant)")
    
    console.print("\n[bold yellow]Key Insights:[/bold yellow]")
    console.print("  • System and user prompts are MASKED (not trained on)")
    console.print("  • Only assistant responses contribute to loss")
    console.print("  • Vision tokens are automatically positioned by the processor")
    console.print("  • Chat markers (<|im_start|>, <|im_end|>) delineate sections")
    console.print("  • force_match=False prevents masking all tokens if markers not found")
    
    console.print("\n[bold green]✨ Inspection complete![/bold green]\n")


if __name__ == "__main__":
    main()
