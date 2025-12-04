"""Backend-agnostic LoRA adapter merging utilities.

This module provides functions for merging LoRA adapters with base models
using standard HuggingFace Transformers + PEFT, without requiring Unsloth.
"""

import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoTokenizer

from model_garden.utils.console import console
from model_garden.utils.hf_cache import get_hf_token


def _cleanup_memory() -> None:
    """Clean up GPU memory after operations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def merge_vision_lora_adapter(
    adapter_path: str,
    output_dir: str,
    base_model: str | None = None,
    max_seq_length: int = 16384,
    load_in_4bit: bool = True,
) -> str:
    """Merge a vision LoRA adapter with its base model for inference.

    This function is specifically for preparing vision-language model adapters for vLLM inference,
    which doesn't support LoRA adapters on vision models. The adapter is loaded, merged with its
    base model, and saved as a complete model.

    Uses standard HuggingFace Transformers + PEFT for maximum compatibility.

    Args:
        adapter_path: Path to the LoRA adapter directory or HuggingFace model ID
        output_dir: Directory to save the merged model
        base_model: Optional base model path (auto-detected from adapter_config.json if not provided)
        max_seq_length: Maximum sequence length (unused, kept for API compatibility)
        load_in_4bit: Load base model in 4-bit for merging (reduces memory usage)

    Returns:
        Path to the merged model directory

    Raises:
        FileNotFoundError: If adapter or base model not found
        ValueError: If adapter_config.json doesn't contain base model info
    """
    console.print("[bold cyan]Merging vision LoRA adapter for inference...[/bold cyan]")
    console.print(f"[cyan]Adapter: {adapter_path}[/cyan]")

    # Check if adapter exists (local path)
    adapter_dir = Path(adapter_path)
    is_local = adapter_dir.exists()

    # Get base model from adapter config if not provided
    if base_model is None:
        console.print("[cyan]🔍 Detecting base model from adapter_config.json...[/cyan]")

        try:
            if is_local:
                adapter_config_file = adapter_dir / "adapter_config.json"
                if not adapter_config_file.exists():
                    raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")

                with open(adapter_config_file) as f:
                    adapter_config = json.load(f)
                    base_model = adapter_config.get("base_model_name_or_path")
            else:
                # HuggingFace model ID
                hf_token = get_hf_token()

                config_file = hf_hub_download(
                    repo_id=adapter_path, filename="adapter_config.json", token=hf_token
                )

                with open(config_file) as f:
                    adapter_config = json.load(f)
                    base_model = adapter_config.get("base_model_name_or_path")

            if not base_model:
                raise ValueError(
                    "Could not find base_model_name_or_path in adapter_config.json. "
                    "Please specify base_model explicitly."
                )

            console.print(f"[green]✓ Found base model: {base_model}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to detect base model: {e}[/red]")
            raise

    console.print(f"[cyan]Base model: {base_model}[/cyan]")
    console.print(f"[cyan]Output: {output_dir}[/cyan]")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hf_token = get_hf_token()

    try:
        # Load base model
        console.print("[cyan]Loading base model...[/cyan]")
        base_torch_dtype = torch.bfloat16 if not load_in_4bit else None

        base_model_obj = AutoModelForVision2Seq.from_pretrained(
            base_model,
            torch_dtype=base_torch_dtype,
            load_in_4bit=load_in_4bit,
            device_map="auto",
            token=hf_token,
        )

        console.print("[green]✓ Base model loaded[/green]")
        console.print(f"[cyan]Loading LoRA adapter from {adapter_path}...[/cyan]")

        # Load LoRA adapter with PEFT
        peft_model: Any = PeftModel.from_pretrained(base_model_obj, adapter_path, token=hf_token)

        console.print("[green]✓ LoRA adapter loaded[/green]")

        # Merge and save
        console.print("[cyan]Merging adapter into base model...[/cyan]")

        # Clear GPU cache before merging
        _cleanup_memory()

        # Merge the adapter
        merged_model: Any = peft_model.merge_and_unload()
        console.print("[green]✓ Merge complete![/green]")

        # Save merged model
        console.print(f"[cyan]Saving merged model to {output_dir}...[/cyan]")

        # Try Unsloth's memory-aware save if available, otherwise use standard save
        saved_with_unsloth = False
        try:
            from unsloth.save import unsloth_save_pretrained_merged

            console.print("[cyan]Using Unsloth memory-aware save...[/cyan]")

            temp_loc = str(Path(output_dir) / "_unsloth_temporary_saved_buffers")
            os.makedirs(temp_loc, exist_ok=True)

            chosen_save_method = "merged_4bit_forced" if load_in_4bit else "merged_16bit"

            unsloth_save_pretrained_merged(
                merged_model,
                save_directory=output_dir,
                tokenizer=None,
                save_method=chosen_save_method,
                push_to_hub=False,
                token=None,
                is_main_process=True,
                state_dict=None,
                save_function=torch.save,
                max_shard_size="5GB",
                safe_serialization=True,
                variant=None,
                save_peft_format=True,
                tags=[],
                temporary_location=temp_loc,
                maximum_memory_usage=0.95,
            )
            console.print(f"[green]✓ Saved with Unsloth (method={chosen_save_method})[/green]")
            saved_with_unsloth = True
        except ImportError:
            console.print("[yellow]Unsloth not available, using standard save[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Unsloth save failed: {e}, using standard save[/yellow]")

        if not saved_with_unsloth:
            merged_model.save_pretrained(output_dir, safe_serialization=True)
            console.print("[green]✓ Model saved (standard save_pretrained)[/green]")

        # Handle config.json quantization metadata
        config_path = Path(output_dir) / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)

                if load_in_4bit:
                    console.print(
                        "[cyan]Keeping quantization_config for bitsandbytes loader[/cyan]"
                    )
                else:
                    # Remove quantization_config for 16-bit merges
                    modified = False
                    if "quantization_config" in config:
                        del config["quantization_config"]
                        modified = True
                        console.print("[green]✓ Removed quantization_config (16-bit save)[/green]")
                    if "text_config" in config and isinstance(config["text_config"], dict):
                        if "quantization_config" in config["text_config"]:
                            del config["text_config"]["quantization_config"]
                            modified = True
                            console.print(
                                "[green]✓ Removed text_config quantization_config[/green]"
                            )

                    if modified:
                        with open(config_path, "w") as f:
                            json.dump(config, f, indent=2)
                        console.print("[green]✓ Config cleaned for vLLM compatibility[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️ Config adjustment failed: {e}[/yellow]")
        else:
            console.print("[yellow]⚠️ config.json not found[/yellow]")

        # Save tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
        tokenizer.save_pretrained(output_dir)
        console.print("[green]✓ Tokenizer saved[/green]")

        # Copy vision processor configuration files from base model
        console.print("[cyan]Copying vision processor configuration files...[/cyan]")
        base_model_dir = snapshot_download(base_model, token=hf_token)

        files_to_copy = [
            "preprocessor_config.json",
            "processor_config.json",
            "video_preprocessor_config.json",
        ]

        for file in files_to_copy:
            src = os.path.join(base_model_dir, file)
            if os.path.exists(src):
                shutil.copy(src, output_dir)
                console.print(f"[green]  ✓ Copied {file}[/green]")

        # Copy image_processor directory if it exists
        image_processor_dir = os.path.join(base_model_dir, "image_processor")
        if os.path.exists(image_processor_dir):
            shutil.copytree(
                image_processor_dir,
                os.path.join(output_dir, "image_processor"),
                dirs_exist_ok=True,
            )
            console.print("[green]  ✓ Copied image_processor directory[/green]")

        console.print("[green]✓ Processor configuration files copied[/green]")

        # Validate the merge
        if not config_path.exists():
            raise FileNotFoundError(f"Merge completed but config.json not found in {output_dir}")

        has_weights = (
            list(Path(output_dir).glob("*.safetensors"))
            or list(Path(output_dir).glob("*.bin"))
            or list(Path(output_dir).glob("model-*.safetensors"))
        )
        if not has_weights:
            raise FileNotFoundError(f"Merge completed but no weight files found in {output_dir}")

        console.print(
            f"[green]✓ Validation passed: config.json and {len(has_weights)} weight file(s)[/green]"
        )

        # Clean up memory
        console.print("[cyan]🧹 Cleaning up memory...[/cyan]")
        del peft_model
        del merged_model
        del base_model_obj
        del tokenizer
        _cleanup_memory()

        console.print("[bold green]✨ Vision LoRA adapter merged successfully![/bold green]")
        return str(output_path.absolute())

    except Exception as e:
        console.print(f"[red]❌ Failed to merge adapter: {e}[/red]")
        raise


def merge_text_lora_adapter(
    adapter_path: str,
    output_dir: str,
    base_model: str | None = None,
    load_in_4bit: bool = False,
) -> str:
    """Merge a text LoRA adapter with its base model.

    Similar to merge_vision_lora_adapter but for text-only models.

    Args:
        adapter_path: Path to the LoRA adapter directory or HuggingFace model ID
        output_dir: Directory to save the merged model
        base_model: Optional base model path (auto-detected if not provided)
        load_in_4bit: Load base model in 4-bit for merging

    Returns:
        Path to the merged model directory
    """
    from transformers import AutoModelForCausalLM

    console.print("[bold cyan]Merging text LoRA adapter...[/bold cyan]")
    console.print(f"[cyan]Adapter: {adapter_path}[/cyan]")

    adapter_dir = Path(adapter_path)
    is_local = adapter_dir.exists()

    # Get base model from adapter config if not provided
    if base_model is None:
        console.print("[cyan]🔍 Detecting base model...[/cyan]")

        try:
            if is_local:
                adapter_config_file = adapter_dir / "adapter_config.json"
                if not adapter_config_file.exists():
                    raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")

                with open(adapter_config_file) as f:
                    adapter_config = json.load(f)
                    base_model = adapter_config.get("base_model_name_or_path")
            else:
                hf_token = get_hf_token()
                config_file = hf_hub_download(
                    repo_id=adapter_path, filename="adapter_config.json", token=hf_token
                )
                with open(config_file) as f:
                    adapter_config = json.load(f)
                    base_model = adapter_config.get("base_model_name_or_path")

            if not base_model:
                raise ValueError("Could not find base_model_name_or_path in adapter_config.json")

            console.print(f"[green]✓ Found base model: {base_model}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to detect base model: {e}[/red]")
            raise

    console.print(f"[cyan]Base model: {base_model}[/cyan]")
    console.print(f"[cyan]Output: {output_dir}[/cyan]")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hf_token = get_hf_token()

    try:
        console.print("[cyan]Loading base model...[/cyan]")
        base_torch_dtype = torch.bfloat16 if not load_in_4bit else None

        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=base_torch_dtype,
            load_in_4bit=load_in_4bit,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
        )

        console.print("[green]✓ Base model loaded[/green]")
        console.print("[cyan]Loading LoRA adapter...[/cyan]")

        peft_model: Any = PeftModel.from_pretrained(base_model_obj, adapter_path, token=hf_token)
        console.print("[green]✓ LoRA adapter loaded[/green]")

        console.print("[cyan]Merging...[/cyan]")
        _cleanup_memory()

        merged_model: Any = peft_model.merge_and_unload()
        console.print("[green]✓ Merge complete![/green]")

        console.print(f"[cyan]Saving to {output_dir}...[/cyan]")
        merged_model.save_pretrained(output_dir, safe_serialization=True)
        console.print("[green]✓ Model saved[/green]")

        tokenizer = AutoTokenizer.from_pretrained(
            base_model, token=hf_token, trust_remote_code=True
        )
        tokenizer.save_pretrained(output_dir)
        console.print("[green]✓ Tokenizer saved[/green]")

        # Clean up
        del peft_model
        del merged_model
        del base_model_obj
        del tokenizer
        _cleanup_memory()

        console.print("[bold green]✨ Text LoRA adapter merged successfully![/bold green]")
        return str(output_path.absolute())

    except Exception as e:
        console.print(f"[red]❌ Failed to merge adapter: {e}[/red]")
        raise
