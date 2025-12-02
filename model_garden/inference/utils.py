# Inference utilities
"""
Utility functions for model detection and GPU memory management:
- GPU memory detection and estimation
- Model type detection (LoRA adapter, vision model, quantization)
"""

import json
import os
from pathlib import Path

from rich.console import Console

# CRITICAL: Set HuggingFace cache directories BEFORE any HF imports
if "HF_HOME" in os.environ:
    hf_home = os.environ["HF_HOME"]
    os.environ["TRANSFORMERS_CACHE"] = os.environ.get(
        "TRANSFORMERS_CACHE", f"{hf_home}/transformers"
    )
    os.environ["HF_DATASETS_CACHE"] = os.environ.get("HF_DATASETS_CACHE", f"{hf_home}/datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ.get("HUGGINGFACE_HUB_CACHE", f"{hf_home}/hub")

console = Console()


def get_gpu_memory_gb() -> float:
    """Get total GPU memory in GB for the first available GPU.

    Returns:
        Total GPU memory in GB, or 0.0 if no GPU is available
    """
    try:
        import torch

        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            return total_memory / (1024**3)
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not detect GPU memory: {e}[/yellow]")
    return 0.0


def estimate_model_size_gb(model_path: str) -> float:
    """Estimate model size in GB by checking weight files or config.

    Args:
        model_path: Path to the model directory or HuggingFace model ID

    Returns:
        Estimated model size in GB
    """
    import re

    model_dir = Path(model_path)

    # If it's a HuggingFace model ID (contains slash and not a local path)
    if "/" in model_path and not model_dir.exists():
        size_match = re.search(r"[-_](\d+(?:\.\d+)?)[Bb](?!it)", model_path)
        if size_match:
            param_size = float(size_match.group(1))
            return max(1.0, param_size * 2)
        return 7.0

    # For local models, check actual file sizes
    if not model_dir.exists() or not model_dir.is_dir():
        return 7.0

    total_bytes = 0
    for pattern in ("*.safetensors", "*.bin"):
        for wf in model_dir.glob(pattern):
            try:
                total_bytes += wf.stat().st_size
            except Exception:
                continue

    total_gb = total_bytes / (1024**3) if total_bytes > 0 else 1.0
    return max(1.0, round(total_gb, 2))


def calculate_gpu_memory_utilization(
    model_path: str,
    max_model_len: int | None = None,
    tensor_parallel_size: int = 1,
) -> float:
    """Lightweight heuristic to calculate GPU memory utilization.

    This is a conservative estimate used when gpu_memory_utilization==0.0 (auto).
    """
    try:
        gpu_memory_gb = get_gpu_memory_gb() or 24.0
        model_size_gb = estimate_model_size_gb(model_path)

        if not max_model_len:
            max_model_len = 4096
        kv_cache_gb = (model_size_gb / 7.0) * (max_model_len / 1000) * 0.4

        total_needed = model_size_gb + kv_cache_gb
        total_with_margin = total_needed * 1.2

        if total_with_margin >= gpu_memory_gb:
            utilization = 0.88 if gpu_memory_gb >= 16 else 0.75
        elif total_with_margin >= gpu_memory_gb * 0.7:
            utilization = 0.80
        else:
            utilization = 0.60

        if tensor_parallel_size > 1:
            utilization = utilization * (1.0 - 0.05 * (tensor_parallel_size - 1))

        utilization = max(0.5, min(0.95, utilization))
        return round(utilization, 2)
    except Exception:
        return 0.88


def is_lora_adapter(model_path: str) -> bool:
    """Check if the model path is a LoRA adapter.

    Args:
        model_path: Path to the model directory or HuggingFace model ID

    Returns:
        True if it's a LoRA adapter, False otherwise
    """
    # For HuggingFace model IDs
    if "/" in model_path and not Path(model_path).exists():
        try:
            from huggingface_hub import HfFileSystem

            hf_token = os.getenv("HF_TOKEN")
            fs = HfFileSystem(token=hf_token)

            adapter_config_path = f"{model_path}/adapter_config.json"
            try:
                if fs.exists(adapter_config_path):
                    console.print(f"[cyan]📦 Detected LoRA adapter repository: {model_path}[/cyan]")
                    return True
            except Exception:
                pass

        except Exception as e:
            console.print(f"[yellow]⚠️  Could not check for adapter config on Hub: {e}[/yellow]")

    # For local paths
    model_dir = Path(model_path)
    if model_dir.exists() and model_dir.is_dir():
        if (model_dir / "adapter_config.json").exists():
            console.print(f"[cyan]📦 Detected LoRA adapter directory: {model_path}[/cyan]")
            return True

    return False


def get_base_model_from_adapter(adapter_path: str) -> str | None:
    """Get the base model name from a LoRA adapter configuration.

    Args:
        adapter_path: Path to the adapter directory or HuggingFace model ID

    Returns:
        Base model name/path, or None if not found
    """
    try:
        # For HuggingFace model IDs
        if "/" in adapter_path and not Path(adapter_path).exists():
            from huggingface_hub import hf_hub_download

            hf_token = os.getenv("HF_TOKEN")

            config_file = hf_hub_download(
                repo_id=adapter_path,
                filename="adapter_config.json",
                token=hf_token,
            )

            with open(config_file) as f:
                adapter_config = json.load(f)
                base_model = adapter_config.get("base_model_name_or_path")
                if base_model:
                    console.print(
                        f"[cyan]🔍 Found base model in adapter config: {base_model}[/cyan]"
                    )
                    return base_model
        else:
            # For local paths
            adapter_dir = Path(adapter_path)
            adapter_config_file = adapter_dir / "adapter_config.json"

            if adapter_config_file.exists():
                with open(adapter_config_file) as f:
                    adapter_config = json.load(f)
                    base_model = adapter_config.get("base_model_name_or_path")
                    if base_model:
                        console.print(
                            f"[cyan]🔍 Found base model in adapter config: {base_model}[/cyan]"
                        )
                        return base_model
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not read adapter config: {e}[/yellow]")

    return None


def is_vision_model(model_path: str) -> bool:
    """Check if a model is a vision-language model.

    Checks for various indicators:
    - "VL" or "vision" in the model name
    - Presence of processor_config.json
    - Vision-specific config fields

    Args:
        model_path: Path to the model directory or HuggingFace model ID

    Returns:
        True if it's a vision model, False otherwise
    """
    # Check model name for vision indicators
    model_name_lower = model_path.lower()
    if "vl" in model_name_lower or "vision" in model_name_lower:
        return True

    # For local models, check for processor_config.json
    model_dir = Path(model_path)
    if model_dir.exists() and model_dir.is_dir():
        if (model_dir / "processor_config.json").exists():
            console.print("[cyan]🔍 Detected vision model (found processor_config.json)[/cyan]")
            return True

        config_file = model_dir / "config.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                    model_type = config.get("model_type", "")
                    architectures = config.get("architectures", [])

                    vision_indicators = ["vision", "vl", "vlm", "multimodal", "qwen2_vl"]

                    if any(indicator in model_type.lower() for indicator in vision_indicators):
                        return True

                    if any(
                        any(indicator in arch.lower() for indicator in vision_indicators)
                        for arch in architectures
                    ):
                        return True

                    if "vision_config" in config or "visual_config" in config:
                        return True
            except Exception:
                pass

    # For HuggingFace model IDs
    if "/" in model_path and not Path(model_path).exists():
        try:
            from huggingface_hub import HfFileSystem

            hf_token = os.getenv("HF_TOKEN")
            fs = HfFileSystem(token=hf_token)

            processor_config_path = f"{model_path}/processor_config.json"
            if fs.exists(processor_config_path):
                console.print(
                    "[cyan]🔍 Detected vision model on Hub (found processor_config.json)[/cyan]"
                )
                return True
        except Exception:
            pass

    return False


def detect_quantization_method(model_path: str) -> str | None:
    """Auto-detect the appropriate quantization method for a model.

    Args:
        model_path: Path to the model directory

    Returns:
        Quantization method ('awq', 'gptq', or None)

    Note:
        - Merged fine-tuned models have quantization_config in config.json
          but weights are actually FP16 - should NOT use quantization.
        - True quantized models (AWQ, GPTQ) have special weight formats.
        - LoRA adapters should use None (load base model separately).
    """
    model_dir = Path(model_path)

    if not model_dir.exists() or not model_dir.is_dir():
        console.print(
            f"[yellow]⚠️  Model path {model_path} not found, skipping auto-detection[/yellow]"
        )
        return None

    # Check for adapter config (LoRA adapters only)
    if (model_dir / "adapter_config.json").exists():
        console.print("[cyan]📦 Detected LoRA adapters (no quantization)[/cyan]")
        return None

    # Check weight files
    has_safetensors = list(model_dir.glob("*.safetensors"))
    has_bin = list(model_dir.glob("*.bin"))

    if has_safetensors or has_bin:
        weight_file = has_safetensors[0] if has_safetensors else has_bin[0]
        file_name = weight_file.name.lower()

        if "-awq" in file_name or "awq" in file_name:
            console.print("[cyan]🔢 Detected AWQ quantized model[/cyan]")
            return "awq"
        elif "-gptq" in file_name or "gptq" in file_name:
            console.print("[cyan]🔢 Detected GPTQ quantized model[/cyan]")
            return "gptq"
        else:
            console.print("[cyan]💎 Detected merged/native format model (no quantization)[/cyan]")
            return None

    # Check config.json
    config_file = model_dir / "config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)

            quant_config = config.get("quantization_config", {})
            if quant_config:
                quant_method = quant_config.get("quant_method", "").lower()

                if "awq" in quant_method:
                    console.print("[cyan]🔢 Config indicates AWQ quantization[/cyan]")
                    return "awq"
                elif "gptq" in quant_method:
                    console.print("[cyan]🔢 Config indicates GPTQ quantization[/cyan]")
                    return "gptq"
                elif "bitsandbytes" in quant_method or "bnb" in quant_method:
                    console.print(
                        "[yellow]⚠️  BitsAndBytes config found but weights appear to be FP16[/yellow]"
                    )
                    console.print("[cyan]💎 Using native format (no quantization)[/cyan]")
                    return None
        except Exception as e:
            console.print(f"[yellow]⚠️  Failed to parse config.json: {e}[/yellow]")

    console.print("[cyan]ℹ️  No quantization detected, loading in native format[/cyan]")
    return None
