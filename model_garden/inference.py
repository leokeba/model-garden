"""vLLM-powered inference service for Model Garden."""

import os
import asyncio
import json
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Union

from rich.console import Console

# CRITICAL: Set HuggingFace cache directories BEFORE any HF imports
# This ensures models are downloaded to the correct location (e.g., /scratch instead of filling up root)
if 'HF_HOME' in os.environ:
    hf_home = os.environ['HF_HOME']
    os.environ['TRANSFORMERS_CACHE'] = os.environ.get('TRANSFORMERS_CACHE', f"{hf_home}/transformers")
    os.environ['HF_DATASETS_CACHE'] = os.environ.get('HF_DATASETS_CACHE', f"{hf_home}/datasets")
    os.environ['HUGGINGFACE_HUB_CACHE'] = os.environ.get('HUGGINGFACE_HUB_CACHE', f"{hf_home}/hub")

console = Console()


def get_gpu_memory_gb() -> float:
    """Get total GPU memory in GB for the first available GPU.
    
    Returns:
        Total GPU memory in GB, or 0.0 if no GPU is available
    """
    try:
        import torch
        if torch.cuda.is_available():
            # Get memory for the first GPU (device 0)
            total_memory = torch.cuda.get_device_properties(0).total_memory
            return total_memory / (1024 ** 3)  # Convert bytes to GB
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
    model_dir = Path(model_path)
    
    # If it's a HuggingFace model ID (contains slash and not a local path)
    if "/" in model_path and not model_dir.exists():
        # Try to extract size from model name (e.g., "7B", "13B", "3B", "1.1B")
        import re
        # Match patterns like "7B", "3B", "1.1B" but not "2.5" (version), "4bit", or "8bit"
        # We look for: hyphen/underscore/start of string, then number >= 1, then 'B'
        # This avoids matching decimal versions like "2.5" or "3.2"
        size_match = re.search(r'[-_](\d+(?:\.\d+)?)[Bb](?!it)', model_path)
        if size_match:
            param_size = float(size_match.group(1))
            # Rough estimate: FP16 = 2 bytes per parameter
            return param_size * 2
        # Default estimate for unknown HF models
        return 7.0  # Assume 7B model as default
    
    # For local models, check actual file sizes
    if not model_dir.exists() or not model_dir.is_dir():
        return 7.0  # Default estimate
    
    total_size = 0.0
    
    # Sum up all weight files (.safetensors and .bin)
    for pattern in ["*.safetensors", "*.bin"]:
        for weight_file in model_dir.glob(pattern):
            total_size += weight_file.stat().st_size
    
    if total_size > 0:
        return total_size / (1024 ** 3)  # Convert bytes to GB
    
    # If no weight files found, try to estimate from config
    config_file = model_dir / "config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
            
            # Try to estimate from model architecture params
            hidden_size = config.get("hidden_size", 4096)
            num_layers = config.get("num_hidden_layers", 32)
            vocab_size = config.get("vocab_size", 32000)
            
            # Rough estimate based on transformer architecture
            # Each layer has attention (4 * hidden_size^2) + FFN (varies)
            # Plus embeddings (vocab_size * hidden_size)
            params_estimate = (
                vocab_size * hidden_size  # Embeddings
                + num_layers * (4 * hidden_size * hidden_size * 2)  # Attention + FFN (rough)
            )
            
            # Assume FP16 (2 bytes per parameter)
            return (params_estimate * 2) / (1024 ** 3)
        except Exception:
            pass
    
    # Default fallback
    return 7.0


def get_free_gpu_memory_gb() -> float:
    """Get currently available (free) GPU memory in GB.
    
    Returns:
        Free GPU memory in GB, or 0.0 if no GPU is available
    """
    try:
        import torch
        if torch.cuda.is_available():
            # Force synchronize and clear cache to get accurate free memory
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            # Get free memory for the first GPU (device 0)
            free_memory, total_memory = torch.cuda.mem_get_info(0)
            return free_memory / (1024 ** 3)  # Convert bytes to GB
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not detect free GPU memory: {e}[/yellow]")
    return 0.0


def calculate_gpu_memory_utilization(
    model_path: str,
    max_model_len: Optional[int] = None,
    tensor_parallel_size: int = 1,
) -> float:
    """Calculate optimal GPU memory utilization based on model size and available VRAM.
    
    This function estimates the memory requirements for:
    - Model weights
    - KV cache (based on max_model_len)
    - Temporary buffers and overhead
    
    It also checks actual available memory to avoid OOM errors when switching models.
    
    Args:
        model_path: Path to the model or HuggingFace model ID
        max_model_len: Maximum sequence length (affects KV cache size)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        
    Returns:
        Recommended GPU memory utilization (0.0-1.0)
    """
    gpu_memory_gb = get_gpu_memory_gb()
    
    if gpu_memory_gb == 0:
        console.print("[yellow]⚠️  No GPU detected, using default utilization of 0.9[/yellow]")
        return 0.9
    
    # Check actual free memory (important when switching models)
    free_memory_gb = get_free_gpu_memory_gb()
    console.print(f"[cyan]💾 GPU memory: {free_memory_gb:.1f} GB free / {gpu_memory_gb:.1f} GB total[/cyan]")
    
    # Estimate model size
    model_size_gb = estimate_model_size_gb(model_path)
    console.print(f"[cyan]📊 Estimated model size: {model_size_gb:.1f} GB[/cyan]")
    
    # Divide by tensor parallel size (model is sharded across GPUs)
    model_size_per_gpu = model_size_gb / tensor_parallel_size
    
    # Estimate KV cache size (more conservative)
    # vLLM uses paged attention which is more memory efficient
    # Rule of thumb: ~0.3-0.5GB per 1K tokens for 7B models with reasonable batch sizes
    if max_model_len is None:
        max_model_len = 4096  # Default assumption
    
    # More conservative KV cache estimate: (model_size_gb / 7.0) * (max_model_len / 1000) * 0.4 GB
    kv_cache_estimate = (model_size_gb / 7.0) * (max_model_len / 1000) * 0.4
    console.print(f"[cyan]🗄️  Estimated KV cache: {kv_cache_estimate:.1f} GB (for max_model_len={max_model_len})[/cyan]")
    
    # Total memory needed per GPU
    total_needed = model_size_per_gpu + kv_cache_estimate
    
    # Add safety margin for temporary buffers and CUDA graphs (20%)
    total_with_margin = total_needed * 1.20
    
    # If free memory is significantly less than total memory, we need to be more conservative
    # This happens when switching models - old model may not be fully cleared yet
    memory_pressure_ratio = free_memory_gb / gpu_memory_gb
    if memory_pressure_ratio < 0.95:
        console.print(f"[yellow]⚠️  GPU memory not fully cleared ({memory_pressure_ratio*100:.1f}% free)[/yellow]")
        console.print(f"[yellow]   Adjusting target to use free memory instead of total memory[/yellow]")
        # Use free memory as the effective total, with extra safety margin
        effective_gpu_memory = free_memory_gb * 0.95  # 5% safety margin
    else:
        effective_gpu_memory = gpu_memory_gb
    
    # Check if KV cache is unreasonably large compared to available memory
    # If KV cache + model won't fit, warn but continue (vLLM will handle it)
    if total_needed > (effective_gpu_memory * 0.75):
        console.print(f"[yellow]⚠️  Requested max_model_len ({max_model_len}) may be too large for available memory[/yellow]")
        console.print(f"[yellow]   vLLM may automatically reduce max_model_len to fit[/yellow]")
    
    # Calculate utilization based on effective memory
    if total_with_margin >= effective_gpu_memory:
        # Model won't fit comfortably, need to use very high utilization
        # But cap at 0.95 to leave minimal room for overhead
        # Calculate how much we need: if we need 36GB but only have 23GB,
        # we need utilization of at least (36/23) * 0.90 = but cap at 0.95
        required_util = (total_needed / effective_gpu_memory) * 0.95
        utilization = min(0.95, max(0.85, required_util))
        console.print(f"[yellow]⚠️  Model memory requirements ({total_with_margin:.1f} GB) exceed available capacity ({effective_gpu_memory:.1f} GB)[/yellow]")
        console.print(f"[yellow]   Using high utilization: {utilization:.2f} (vLLM will adjust KV cache accordingly)[/yellow]")
    elif total_with_margin >= effective_gpu_memory * 0.7:
        # Model is a significant portion of memory
        utilization = 0.85
        console.print(f"[cyan]✓ Model requires ~{(total_with_margin/effective_gpu_memory)*100:.0f}% of available GPU memory[/cyan]")
        console.print(f"[cyan]  Using standard utilization: {utilization}[/cyan]")
    else:
        # Model fits comfortably, use conservative utilization
        # This leaves plenty of room for batching and multiple concurrent requests
        # Use at least 0.5 to ensure good throughput, cap at 0.75 to leave room for batching
        utilization = max(0.50, min(0.75, (total_with_margin / effective_gpu_memory) + 0.20))
        console.print(f"[cyan]✓ Model requires ~{(total_with_margin/effective_gpu_memory)*100:.0f}% of available GPU memory[/cyan]")
        console.print(f"[cyan]  Using conservative utilization: {utilization} (leaves room for batching)[/cyan]")
    
    return round(utilization, 2)


def is_lora_adapter(model_path: str) -> bool:
    """Check if the model path is a LoRA adapter.
    
    Args:
        model_path: Path to the model directory or HuggingFace model ID
        
    Returns:
        True if it's a LoRA adapter, False otherwise
    """
    # For HuggingFace model IDs, try to check if adapter_config.json exists
    if "/" in model_path and not Path(model_path).exists():
        try:
            from huggingface_hub import hf_hub_download, HfFileSystem
            import os
            
            hf_token = os.getenv('HF_TOKEN')
            fs = HfFileSystem(token=hf_token)
            
            # Check if adapter_config.json exists in the repo
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


def get_base_model_from_adapter(adapter_path: str) -> Optional[str]:
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
            import os
            
            hf_token = os.getenv('HF_TOKEN')
            
            # Download adapter_config.json
            config_file = hf_hub_download(
                repo_id=adapter_path,
                filename="adapter_config.json",
                token=hf_token
            )
            
            with open(config_file) as f:
                adapter_config = json.load(f)
                base_model = adapter_config.get("base_model_name_or_path")
                if base_model:
                    console.print(f"[cyan]🔍 Found base model in adapter config: {base_model}[/cyan]")
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
                        console.print(f"[cyan]🔍 Found base model in adapter config: {base_model}[/cyan]")
                        return base_model
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not read adapter config: {e}[/yellow]")
    
    return None


def is_vision_model(model_path: str) -> bool:
    """Check if a model is a vision-language model.
    
    Checks for various indicators:
    - "VL" or "vision" in the model name
    - Presence of processor_config.json (vision models use processors)
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
        # Check for processor config (vision models use processors, not just tokenizers)
        if (model_dir / "processor_config.json").exists():
            console.print(f"[cyan]🔍 Detected vision model (found processor_config.json)[/cyan]")
            return True
        
        # Check config.json for vision-specific fields
        config_file = model_dir / "config.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                    # Check for vision-specific architecture types
                    model_type = config.get("model_type", "")
                    architectures = config.get("architectures", [])
                    
                    vision_indicators = ["vision", "vl", "vlm", "multimodal", "qwen2_vl"]
                    
                    if any(indicator in model_type.lower() for indicator in vision_indicators):
                        return True
                    
                    if any(any(indicator in arch.lower() for indicator in vision_indicators) 
                           for arch in architectures):
                        return True
                    
                    # Check for vision_config or visual_config keys
                    if "vision_config" in config or "visual_config" in config:
                        return True
            except Exception:
                pass
    
    # For HuggingFace model IDs, check if processor_config.json exists
    if "/" in model_path and not Path(model_path).exists():
        try:
            from huggingface_hub import HfFileSystem
            
            hf_token = os.getenv('HF_TOKEN')
            fs = HfFileSystem(token=hf_token)
            
            # Check if processor_config.json exists
            processor_config_path = f"{model_path}/processor_config.json"
            if fs.exists(processor_config_path):
                console.print(f"[cyan]🔍 Detected vision model on Hub (found processor_config.json)[/cyan]")
                return True
        except Exception:
            pass
    
    return False


def detect_quantization_method(model_path: str) -> Optional[str]:
    """Auto-detect the appropriate quantization method for a model.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Quantization method ('awq', 'gptq', or None)
        
    Note:
        - Merged fine-tuned models (from Unsloth with save_method="merged_16bit") 
          have quantization_config in config.json but weights are actually FP16.
          These should NOT use quantization in vLLM.
        - True quantized models (AWQ, GPTQ) have special weight formats.
        - LoRA adapters should use None (load base model separately).
    """
    model_dir = Path(model_path)
    
    # Check if directory exists
    if not model_dir.exists() or not model_dir.is_dir():
        console.print(f"[yellow]⚠️  Model path {model_path} not found, skipping auto-detection[/yellow]")
        return None
    
    # Check for adapter config (LoRA adapters only)
    if (model_dir / "adapter_config.json").exists():
        console.print("[cyan]📦 Detected LoRA adapters (no quantization)[/cyan]")
        return None
    
    # Check if it's a merged fine-tuned model with standard weights
    # Unsloth's merged_16bit models have regular safetensors/bin files
    has_safetensors = list(model_dir.glob("*.safetensors"))
    has_bin = list(model_dir.glob("*.bin"))
    
    if has_safetensors or has_bin:
        # Check the first weight file to see if it's standard format
        weight_file = has_safetensors[0] if has_safetensors else has_bin[0]
        
        # Merged models have standard weight files (not quantized tensors)
        # AWQ/GPTQ models have special file names like *-awq.safetensors or contain qweight/qzeros
        file_name = weight_file.name.lower()
        
        if "-awq" in file_name or "awq" in file_name:
            console.print("[cyan]🔢 Detected AWQ quantized model[/cyan]")
            return "awq"
        elif "-gptq" in file_name or "gptq" in file_name:
            console.print("[cyan]🔢 Detected GPTQ quantized model[/cyan]")
            return "gptq"
        else:
            # Standard weight file - this is a merged/native format model
            console.print("[cyan]💎 Detected merged/native format model (no quantization)[/cyan]")
            return None
    
    # Check for quantization config in config.json (less reliable due to Unsloth)
    config_file = model_dir / "config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
            
            # Check quantization config
            quant_config = config.get("quantization_config", {})
            if quant_config:
                quant_method = quant_config.get("quant_method", "").lower()
                
                # Only trust AWQ/GPTQ configs, not BitsAndBytes
                # (Unsloth leaves BnB config even after merging to FP16)
                if "awq" in quant_method:
                    console.print("[cyan]🔢 Config indicates AWQ quantization[/cyan]")
                    return "awq"
                elif "gptq" in quant_method:
                    console.print("[cyan]🔢 Config indicates GPTQ quantization[/cyan]")
                    return "gptq"
                elif "bitsandbytes" in quant_method or "bnb" in quant_method:
                    # Don't trust BnB config - check if weights are actually quantized
                    console.print("[yellow]⚠️  BitsAndBytes config found but weights appear to be FP16 (Unsloth merged model)[/yellow]")
                    console.print("[cyan]💎 Using native format (no quantization)[/cyan]")
                    return None
        except Exception as e:
            console.print(f"[yellow]⚠️  Failed to parse config.json: {e}[/yellow]")
    
    # Default: no quantization
    console.print("[cyan]ℹ️  No quantization detected, loading in native format[/cyan]")
    return None


class InferenceService:
    """Manages model inference using vLLM."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.0,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
        quantization: Optional[str] = "auto",
        trust_remote_code: bool = False,
        enable_lora: bool = True,
        max_loras: int = 1,
        max_lora_rank: int = 64,
    ):
        """Initialize the inference service.

        Args:
            model_path: Path to the model or HuggingFace model ID (can be LoRA adapter)
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0, 0 = auto)
            max_model_len: Maximum sequence length
            dtype: Data type (auto, float16, bfloat16, float32)
            quantization: Quantization method (auto, awq, gptq, squeezellm, fp8, bitsandbytes, or None)
            trust_remote_code: Whether to trust remote code
            enable_lora: Enable LoRA adapter support (auto-enabled if model_path is an adapter)
            max_loras: Maximum number of LoRA adapters to load concurrently
            max_lora_rank: Maximum LoRA rank to support
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.quantization = quantization
        self.trust_remote_code = trust_remote_code
        self.enable_lora = enable_lora
        self.max_loras = max_loras
        self.max_lora_rank = max_lora_rank
        
        self.engine = None
        self.is_loaded = False
        
        # LoRA adapter tracking
        self.is_adapter = False
        self.base_model_path: Optional[str] = None
        self.adapter_path: Optional[str] = None
        
        # Vision model tracking
        self.is_vision_lora_adapter = False
        self.merged_vision_model_path: Optional[str] = None  # Temp merged model path

    async def load_model(self) -> None:
        """Load the model into vLLM engine.
        
        Automatically detects and handles LoRA adapters by:
        1. Checking if model_path is a LoRA adapter
        2. For text models: Loading the base model and applying LoRA on top
        3. For vision models: Merging the LoRA with base model first (vLLM doesn't support vision LoRAs)
        """
        if self.is_loaded:
            console.print("[yellow]Model already loaded[/yellow]")
            return

        console.print(f"[cyan]Loading model: {self.model_path}[/cyan]")
        
        # Check if this is a LoRA adapter
        if is_lora_adapter(self.model_path):
            self.is_adapter = True
            self.adapter_path = self.model_path
            
            # Get base model from adapter config
            base_model = get_base_model_from_adapter(self.model_path)
            if not base_model:
                raise ValueError(
                    f"Could not determine base model for adapter {self.model_path}. "
                    "Please specify the base model explicitly or ensure adapter_config.json contains 'base_model_name_or_path'."
                )
            
            self.base_model_path = base_model
            
            # Check if this is a vision model adapter
            # We check both the adapter path itself and the base model
            adapter_is_vision = is_vision_model(self.adapter_path)
            base_is_vision = is_vision_model(base_model)
            
            if adapter_is_vision or base_is_vision:
                console.print("[yellow]⚠️  Detected vision-language model adapter[/yellow]")
                console.print("[yellow]   vLLM doesn't support LoRA on vision models - merging adapter with base model first[/yellow]")
                
                self.is_vision_lora_adapter = True
                
                # Create temporary directory for merged model in HF_HOME (not /tmp/)
                # This avoids filling up the main drive
                import time
                hf_home = os.getenv('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))
                temp_base = Path(hf_home) / 'temp_merges'
                temp_base.mkdir(parents=True, exist_ok=True)
                
                temp_dir = temp_base / f"model-garden-merged-{int(time.time())}"
                temp_dir.mkdir(parents=True, exist_ok=True)
                self.merged_vision_model_path = str(temp_dir)
                
                console.print(f"[cyan]🔧 Merging vision LoRA adapter...[/cyan]")
                console.print(f"[cyan]   Adapter: {self.adapter_path}[/cyan]")
                console.print(f"[cyan]   Base model: {base_model}[/cyan]")
                console.print(f"[cyan]   Output: {self.merged_vision_model_path}[/cyan]")
                
                try:
                    # Import the merge function from vision_training
                    from model_garden.vision_training import merge_vision_lora_adapter
                    
                    # Merge the adapter - this will save the merged model to temp directory
                    merged_path = merge_vision_lora_adapter(
                        adapter_path=self.adapter_path,
                        output_dir=self.merged_vision_model_path,
                        base_model=base_model,
                        load_in_4bit=True,  # Use 4-bit for memory efficiency during merge
                    )
                    
                    console.print(f"[green]✓ Vision LoRA merged successfully[/green]")
                    
                    # Verify that the merge actually produced a valid model directory
                    merged_config = Path(merged_path) / "config.json"
                    if not merged_config.exists():
                        raise FileNotFoundError(
                            f"Merge completed but config.json not found in {merged_path}. "
                            "The merge may have failed silently."
                        )
                    
                    # Update model_path to point to merged model
                    self.base_model_path = merged_path
                    
                    # Disable LoRA support since we've merged
                    self.enable_lora = False
                    console.print("[cyan]📦 Loading merged vision model into vLLM...[/cyan]")
                    
                except Exception as e:
                    console.print(f"[red]❌ Failed to merge vision LoRA adapter: {e}[/red]")
                    import traceback
                    console.print(f"[red]Full error:[/red]")
                    console.print(traceback.format_exc())
                    # Clean up temp directory on failure
                    if self.merged_vision_model_path and Path(self.merged_vision_model_path).exists():
                        import shutil
                        shutil.rmtree(self.merged_vision_model_path, ignore_errors=True)
                    self.merged_vision_model_path = None
                    raise
            else:
                # Text model adapter - can use vLLM's LoRA support
                console.print(f"[cyan]📦 Loading base model: {base_model}[/cyan]")
                console.print(f"[cyan]🔧 Will apply LoRA adapter: {self.adapter_path}[/cyan]")
                
                # Enable LoRA support
                self.enable_lora = True
        else:
            self.base_model_path = self.model_path
        
        # Force aggressive GPU cleanup before loading to ensure clean state
        # This is important when switching models to avoid OOM errors
        try:
            import torch
            import gc
            
            # Multiple GC passes to handle circular references
            for _ in range(3):
                gc.collect()
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            console.print("[cyan]✓ Pre-load cleanup completed[/cyan]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Pre-load cleanup warning: {e}[/yellow]")
        
        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine
            
            # Auto-calculate GPU memory utilization if set to 0
            gpu_memory_utilization = self.gpu_memory_utilization
            if gpu_memory_utilization == 0.0:
                console.print("[cyan]🔧 Auto mode enabled for GPU memory utilization[/cyan]")
                gpu_memory_utilization = calculate_gpu_memory_utilization(
                    model_path=self.model_path,
                    max_model_len=self.max_model_len,
                    tensor_parallel_size=self.tensor_parallel_size,
                )
                console.print(f"[green]✓[/green] Calculated GPU memory utilization: {gpu_memory_utilization}")
            else:
                console.print(f"[cyan]💾 Using manual GPU memory utilization: {gpu_memory_utilization}[/cyan]")
            
            # Auto-detect quantization if not specified
            quantization = self.quantization
            load_format = "auto"  # Default to auto-detection
            
            # Check if this is a HuggingFace model ID (contains slash and doesn't exist as local path)
            is_hf_model = "/" in self.model_path and not Path(self.model_path).exists()
            
            if quantization == "auto" or quantization is None:
                if is_hf_model:
                    # For HuggingFace models, use auto-detection from vLLM
                    quantization = None  # Let vLLM auto-detect
                    load_format = "auto"
                    console.print(f"[cyan]🤗 Loading HuggingFace model: {self.model_path}[/cyan]")
                    console.print("[cyan]   Using auto-detection for quantization[/cyan]")
                else:
                    # For local models, use our custom detection
                    detected = detect_quantization_method(self.model_path)
                    if detected:
                        quantization = detected
                        console.print(f"[green]✓[/green] Auto-detected quantization: {quantization}")
                    else:
                        quantization = None
                        load_format = "safetensors"  # Force standard format, ignore config.json quantization
                        console.print("[green]✓[/green] No quantization needed (merged or native format)")
                        console.print("[cyan]   Using load_format=safetensors to ignore quantization_config in model[/cyan]")
            
            # For HuggingFace models, enable trust_remote_code by default if not explicitly set
            trust_remote_code = self.trust_remote_code
            if is_hf_model and not trust_remote_code:
                trust_remote_code = True
                console.print("[cyan]   Enabling trust_remote_code for HuggingFace model[/cyan]")
            
            # Configure engine arguments
            # Ensure dtype is properly typed
            valid_dtypes = ["auto", "half", "float16", "bfloat16", "float", "float32"]
            dtype_param = self.dtype if self.dtype in valid_dtypes else "auto"
            
            # Ensure quantization is properly typed
            valid_quantization = ["awq", "deepspeedfp", "tpu_int8", "fp8", "ptpc_fp8", "marlin", "ggml", "gptq", "squeezellm", "compressed-tensors", "bitsandbytes", "qqq", "experts_int8", "fbgemm_fp8", "modelopt"]
            quantization_param = quantization if quantization in valid_quantization else None
            
            # Prepare engine args
            engine_args_dict = {
                "model": self.base_model_path,  # Use base model path (same as model_path if not adapter)
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": self.max_model_len,
                "dtype": dtype_param,  # type: ignore
                "quantization": quantization_param,  # type: ignore
                "load_format": load_format,
                "trust_remote_code": trust_remote_code,
                "enforce_eager": False,  # Use CUDA graphs for better performance
                "disable_log_stats": False,
            }
            
            # Add LoRA support if enabled
            if self.enable_lora:
                console.print(f"[cyan]🔧 Enabling LoRA support (max_loras={self.max_loras}, max_rank={self.max_lora_rank})[/cyan]")
                engine_args_dict["enable_lora"] = True
                engine_args_dict["max_loras"] = self.max_loras
                engine_args_dict["max_lora_rank"] = self.max_lora_rank
            
            engine_args = AsyncEngineArgs(**engine_args_dict)
            
            # Create async engine
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.is_loaded = True
            
            console.print("[green]✓[/green] Base model loaded successfully")
            
            # If we have an adapter, load it now
            if self.is_adapter and self.adapter_path:
                console.print(f"[cyan]🔧 Loading LoRA adapter: {self.adapter_path}[/cyan]")
                try:
                    # For vLLM, adapters are loaded per-request via lora_request parameter
                    # We just need to verify the adapter exists
                    console.print("[green]✓[/green] LoRA adapter ready (will be applied per-request)")
                except Exception as e:
                    console.print(f"[yellow]⚠️  LoRA adapter preparation warning: {e}[/yellow]")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to load model: {e}[/red]")
            raise

    async def unload_model(self) -> None:
        """Unload the model from memory."""
        if not self.is_loaded:
            console.print("[yellow]No model loaded[/yellow]")
            return

        console.print("[cyan]Unloading model...[/cyan]")
        
        # Delete the vLLM engine first
        if self.engine:
            del self.engine
            self.engine = None
        
        # Force garbage collection to release Python references
        import gc
        gc.collect()
        console.print("[green]✓[/green] Garbage collection completed")
        
        # Clear CUDA cache to free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                console.print("[green]✓[/green] GPU cache cleared")
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not clear GPU cache: {e}[/yellow]")
        
        # Clean up temporary merged vision model if it exists
        if self.merged_vision_model_path and Path(self.merged_vision_model_path).exists():
            console.print(f"[cyan]🧹 Cleaning up temporary merged model: {self.merged_vision_model_path}[/cyan]")
            try:
                import shutil
                shutil.rmtree(self.merged_vision_model_path, ignore_errors=True)
                console.print("[green]✓[/green] Temporary merged model deleted")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not delete temporary model: {e}[/yellow]")
            self.merged_vision_model_path = None
        
        self.is_loaded = False
        console.print("[green]✓[/green] Model unloaded successfully")

    async def close(self) -> None:
        """Close the inference service and clean up resources."""
        await self.unload_model()

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = -1,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        images: Optional[List[str]] = None,
        structured_outputs: Optional[Dict] = None,
    ) -> Union[Dict, AsyncIterator[str]]:
        """Generate text from a prompt with optional multimodal inputs.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate (None = auto: 16384 for structured outputs, 512 otherwise)
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling probability
            top_k: Top-k sampling (-1 to disable)
            frequency_penalty: Frequency penalty (-2.0 to 2.0, None = auto: 0.5 for structured outputs, 0.0 otherwise)
            presence_penalty: Presence penalty (-2.0 to 2.0, None = auto: 0.3 for structured outputs, 0.0 otherwise)
            repetition_penalty: Repetition penalty (>1.0 = penalty, None = auto: 1.1 for structured outputs, 1.0 otherwise)
            stop: List of stop sequences
            stream: Whether to stream the response
            images: List of image URLs or file paths (for vision models)
            structured_outputs: Optional structured output parameters (json, regex, choice, grammar, structural_tag)
            
        Note:
            When structured_outputs is provided, anti-repetition penalties are automatically applied
            unless explicitly overridden. This prevents degeneration like "BEUG/BEUG/BEUG/BEUG".
            Client can override any parameter by passing explicit values.

        Returns:
            Dict with text and usage, or async iterator of text chunks if streaming
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from vllm import SamplingParams
        
        # Set default max_tokens if not provided
        if max_tokens is None:
            if structured_outputs:
                max_tokens = 16384  # High default for complex documents (CMRs can have 10k+ tokens)
                # Note: Qwen2.5-VL has 32k context, most prompts are 5-15k, so 16k output is safe
            else:
                max_tokens = 512  # Standard default
        
        # Set sensible defaults for penalties if not provided by client
        # For structured outputs, use STRONG anti-repetition penalties to combat model degeneration
        if frequency_penalty is None:
            if structured_outputs:
                frequency_penalty = 1.0  # Increased from 0.5 - stronger penalty for repeated tokens
            else:
                frequency_penalty = 0.0  # Standard default
        
        if presence_penalty is None:
            if structured_outputs:
                presence_penalty = 0.6  # Increased from 0.3 - encourage more diversity
            else:
                presence_penalty = 0.0  # Standard default
        
        if repetition_penalty is None:
            if structured_outputs:
                repetition_penalty = 1.2  # Increased from 1.1 - stronger n-gram penalty
            else:
                repetition_penalty = 1.0  # Standard default (no penalty)
        
        # Create structured outputs params if provided
        structured_outputs_params = None
        if structured_outputs:
            try:
                from vllm.sampling_params import StructuredOutputsParams
                structured_outputs_params = StructuredOutputsParams(**structured_outputs)
            except ImportError:
                console.print("[yellow]Warning: StructuredOutputsParams not available in this vLLM version[/yellow]")
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            stop=stop,
            structured_outputs=structured_outputs_params,
        )
        
        # Prepare inputs (text + optional images)
        inputs = self._prepare_inputs(prompt, images)
        
        # Generate unique request ID
        request_id = f"req-{id(prompt)}-{asyncio.get_event_loop().time()}"
        
        if stream:
            return self._generate_streaming(inputs, sampling_params, request_id)
        else:
            return await self._generate_complete(inputs, sampling_params, request_id)  # type: ignore

    def _prepare_inputs(self, prompt: str, images: Optional[List[str]] = None):
        """Prepare inputs for generation, handling multimodal data if images are provided."""
        if images is None or len(images) == 0:
            return prompt
        
        try:
            from vllm.inputs import TextPrompt
            import requests
            from io import BytesIO
            from PIL import Image
            import base64
            import tempfile
            
            # Load images from URLs, file paths, or base64 data
            # For vLLM multiprocessing compatibility, we'll convert base64 to temp files
            loaded_images = []
            for img_data in images:
                if img_data.startswith('data:image/'):
                    # It's a data URL with base64 - this shouldn't happen as we extract it in the API
                    # but handle it just in case
                    import re
                    match = re.match(r"data:image/[^;]+;base64,(.+)", img_data)
                    if match:
                        img_data = match.group(1)
                    # Fall through to base64 handling
                
                if img_data.startswith(('http://', 'https://')):
                    # Download the image from URL and save to temp file
                    # vLLM's Qwen2.5-VL processor doesn't handle URL downloading
                    try:
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                        }
                        response = requests.get(img_data, timeout=10, headers=headers)
                        response.raise_for_status()
                        img = Image.open(BytesIO(response.content))
                        # Ensure image is in RGB mode for vLLM compatibility
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Save to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png', mode='wb') as tmp_file:
                            img.save(tmp_file, format='PNG')
                            img_path = tmp_file.name
                            print(f"✅ Downloaded image from URL: {img.size} {img.mode}, saved to {img_path}")
                            loaded_images.append(img_path)
                    except Exception as e:
                        print(f"❌ Failed to download image from URL {img_data}: {e}")
                        raise
                elif '/' not in img_data or len(img_data) > 200:
                    # Likely base64 data (no path separators, or long string)
                    # For vLLM multiprocessing, we need to save to a temp file instead of passing PIL objects
                    try:
                        # Decode base64 to image
                        img_bytes = base64.b64decode(img_data)
                        img = Image.open(BytesIO(img_bytes))
                        # Ensure image is in RGB mode for vLLM compatibility
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Save to temporary file and pass the path instead of PIL object
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png', mode='wb') as tmp_file:
                            img.save(tmp_file, format='PNG')
                            img_path = tmp_file.name
                            print(f"✅ Loaded image from base64 data: {img.size} {img.mode}, saved to {img_path}")
                            loaded_images.append(img_path)
                    except Exception as e:
                        print(f"❌ Failed to decode base64 image: {e}")
                        raise
                else:
                    # File path - verify it exists and pass the path
                    img_file = Path(img_data)
                    if not img_file.exists():
                        raise FileNotFoundError(f"Image file not found: {img_data}")
                    loaded_images.append(str(img_file))
            
            # For Qwen2.5-VL and similar models, ensure prompt includes vision tokens
            # If the prompt doesn't have vision tokens, add them
            if '<|vision_start|>' not in prompt and '<|image_pad|>' not in prompt:
                # Wrap the prompt with Qwen2.5-VL vision tokens
                # Format: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n
                formatted_prompt = (
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                    f"{prompt}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                prompt = formatted_prompt
            
            # Create multimodal input
            # For Qwen2-VL models, vLLM expects "image" (singular) key with a LIST of images
            multi_modal_data = {"image": loaded_images}  # Always pass as list
            
            return TextPrompt(
                prompt=prompt,
                multi_modal_data=multi_modal_data
            )
        except ImportError:
            # Fall back to text-only if multimodal not available
            console.print("[yellow]⚠️  Multimodal imports not available, falling back to text-only mode[/yellow]")
            return prompt

    def _sanitize_json_output(self, json_text: str) -> str:
        """Sanitize JSON output to fix common generation issues.
        
        Fixes:
        - Invalid Unicode escape sequences (lone surrogates)
        - Malformed escape sequences
        - Invalid control characters
        
        Args:
            json_text: Generated JSON text that may contain errors
            
        Returns:
            Sanitized JSON text that should be valid
        """
        import re
        
        # Fix 1: Remove or fix invalid Unicode escape sequences
        # Pattern matches \uXXXX where XXXX is a hex number
        def fix_unicode_escape(match):
            hex_code = match.group(1)
            try:
                code_point = int(hex_code, 16)
                # Check if it's a lone surrogate (0xD800-0xDFFF)
                if 0xD800 <= code_point <= 0xDFFF:
                    # Replace with a safe placeholder or remove
                    return ''  # Remove invalid surrogates
                return match.group(0)  # Keep valid escapes
            except (ValueError, OverflowError):
                return ''  # Remove invalid hex codes
        
        json_text = re.sub(r'\\u([0-9a-fA-F]{4})', fix_unicode_escape, json_text)
        
        # Fix 2: Remove invalid escape sequences (backslash followed by invalid char)
        # Valid escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
        # Remove any \x where x is not one of these
        def fix_invalid_escape(match):
            char = match.group(1)
            valid_escapes = {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'}
            if char in valid_escapes:
                return match.group(0)  # Keep valid escape
            # Invalid escape - either remove backslash or escape it
            return char  # Just keep the character without backslash
        
        json_text = re.sub(r'\\(.)', fix_invalid_escape, json_text)
        
        # Fix 3: Remove invalid control characters (except valid whitespace)
        # JSON only allows tab (\t), newline (\n), carriage return (\r)
        json_text = ''.join(char for char in json_text if ord(char) >= 32 or char in '\t\n\r')
        
        return json_text

    async def _generate_complete(
        self,
        inputs: str,
        sampling_params,
        request_id: str,
    ) -> Dict:
        """Generate complete response (non-streaming)."""
        if self.engine is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Prepare lora_request if we have an adapter (only for text models, not vision)
        lora_request = None
        if self.is_adapter and self.adapter_path and not self.is_vision_lora_adapter:
            try:
                from vllm.lora.request import LoRARequest
                # Create LoRA request with adapter path
                # The lora_int_id must be unique per adapter (use 1 for single adapter)
                lora_request = LoRARequest(
                    lora_name=f"adapter_{Path(self.adapter_path).name}",
                    lora_int_id=1,
                    lora_local_path=self.adapter_path
                )
            except ImportError:
                console.print("[yellow]⚠️  LoRA support not available in this vLLM version[/yellow]")
        
        results_generator = self.engine.generate(inputs, sampling_params, request_id, lora_request=lora_request)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        if final_output is None:
            return {"text": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        
        # Return the generated text with usage stats
        generated_text = final_output.outputs[0].text
        finish_reason = final_output.outputs[0].finish_reason
        
        # Get token counts
        prompt_tokens = len(final_output.prompt_token_ids) if final_output.prompt_token_ids else 0
        completion_tokens = len(final_output.outputs[0].token_ids)
        total_tokens = prompt_tokens + completion_tokens
        
        # Warn if we hit max_tokens (generation was truncated)
        if finish_reason == "length":
            console.print(f"[red]⚠️  Output truncated: Hit max_tokens limit ({sampling_params.max_tokens})[/red]")
            console.print(f"[red]   Prompt: {prompt_tokens} tokens, Output: {completion_tokens} tokens[/red]")
        
        # Warn if prompt is very long and might cause truncation issues
        if prompt_tokens > 20000:
            console.print(f"[yellow]⚠️  Very long prompt: {prompt_tokens} tokens. "
                         f"Total with output: {total_tokens} tokens[/yellow]")
        
        # Only log abnormal stops (not "stop" which is natural completion, not "length" which we already warned about)
        abnormal_reasons = {"abort", "error"}  # Add other abnormal reasons as needed
        if finish_reason in abnormal_reasons:
            console.print(f"[red]⚠️  Abnormal stop: finish_reason={finish_reason}, "
                         f"completion_tokens={completion_tokens}/{sampling_params.max_tokens}[/red]")
        
        # # Post-process structured outputs to fix common JSON issues
        # if hasattr(sampling_params, 'structured_outputs') and sampling_params.structured_outputs:
        #     generated_text = self._sanitize_json_output(generated_text)
        
        return {
            "text": generated_text,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "finish_reason": finish_reason  # Include finish_reason in response
        }

    async def _generate_streaming(
        self,
        inputs,  # Can be str or TextPrompt
        sampling_params,
        request_id: str,
    ) -> AsyncIterator[str]:
        """Generate streaming response."""
        if self.engine is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Prepare lora_request if we have an adapter (only for text models, not vision)
        lora_request = None
        if self.is_adapter and self.adapter_path and not self.is_vision_lora_adapter:
            try:
                from vllm.lora.request import LoRARequest
                lora_request = LoRARequest(
                    lora_name=f"adapter_{Path(self.adapter_path).name}",
                    lora_int_id=1,
                    lora_local_path=self.adapter_path
                )
            except ImportError:
                console.print("[yellow]⚠️  LoRA support not available in this vLLM version[/yellow]")
        
        results_generator = self.engine.generate(inputs, sampling_params, request_id, lora_request=lora_request)
        
        previous_text = ""
        async for request_output in results_generator:
            text = request_output.outputs[0].text
            # Yield only the new tokens
            new_text = text[len(previous_text):]
            if new_text:
                yield new_text
            previous_text = text

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = False,
        image: Optional[str] = None,
        structured_outputs: Optional[Dict] = None,
        **kwargs
    ) -> Union[Dict, AsyncIterator[Dict]]:
        """OpenAI-compatible chat completion with vision support.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate (None = auto: 16384 for structured outputs, 512 otherwise)
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            stream: Whether to stream the response
            image: Optional image URL or base64 data for vision models
            structured_outputs: Optional structured output parameters
            **kwargs: Additional generation parameters

        Returns:
            Chat completion response in OpenAI format
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
                # Set default max_tokens if not provided
        if max_tokens is None:
            if structured_outputs:
                max_tokens = 6144  # Reduced from 8192 - most CMR docs are 2k-4k tokens
            else:
                max_tokens = 512  # Standard default

        # Format messages into a single prompt
        # This is a simple implementation - you may need to customize for specific models
        prompt = self._format_chat_messages(messages)
        
        if stream:
            return self._chat_completion_stream(messages, prompt, max_tokens, temperature, top_p, image=image, structured_outputs=structured_outputs, **kwargs)
        else:
            return await self._chat_completion_complete(messages, prompt, max_tokens, temperature, top_p, image=image, structured_outputs=structured_outputs, **kwargs)

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a prompt string.
        
        This is a simple implementation. For specific models, you should use their
        official chat template from the tokenizer.
        """
        formatted_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        
        # Add final "Assistant:" to prompt the model to respond
        formatted_parts.append("Assistant:")
        
        return "\n".join(formatted_parts)

    async def _chat_completion_complete(
        self,
        messages: List[Dict[str, str]],
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        image: Optional[str] = None,
        structured_outputs: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """Generate complete chat completion response."""
        # Convert single image to list format
        images = [image] if image else None
        
        # Keep original prompt string for token counting
        prompt_str = prompt
        
        result = await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
            images=images,
            structured_outputs=structured_outputs,
            **kwargs
        )
        
        # Extract text from result
        response_text = result.get('text', '') if isinstance(result, dict) else str(result)
        usage_info = result.get('usage', {}) if isinstance(result, dict) else {}
        
        # Format as OpenAI-compatible response
        return {
            "id": f"chatcmpl-{id(result)}",
            "object": "chat.completion",
            "created": int(asyncio.get_event_loop().time()),
            "model": self.model_path,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": usage_info if usage_info else {
                "prompt_tokens": len(prompt_str.split()),  # Rough estimate
                "completion_tokens": len(response_text.split()),  # Rough estimate
                "total_tokens": len(prompt_str.split()) + len(response_text.split()),
            },
        }

    async def _chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        image: Optional[str] = None,
        structured_outputs: Optional[Dict] = None,
        **kwargs
    ) -> AsyncIterator[Dict]:
        """Generate streaming chat completion response."""
        # Convert single image to list format
        images = [image] if image else None
        
        stream_generator = await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            images=images,
            structured_outputs=structured_outputs,
            **kwargs
        )
        
        async for chunk in stream_generator:  # type: ignore
            # Format as OpenAI-compatible streaming response
            yield {
                "id": f"chatcmpl-{id(chunk)}",
                "object": "chat.completion.chunk",
                "created": int(asyncio.get_event_loop().time()),
                "model": self.model_path,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": chunk,
                        },
                        "finish_reason": None,
                    }
                ],
            }
        
        # Send final chunk with finish_reason
        yield {
            "id": f"chatcmpl-final",
            "object": "chat.completion.chunk",
            "created": int(asyncio.get_event_loop().time()),
            "model": self.model_path,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        # For local models, return just the model name instead of full path
        model_display_path = self.model_path
        if Path(self.model_path).is_absolute():
            # Extract just the model name from the path
            model_display_path = Path(self.model_path).name
        
        info = {
            "model_path": model_display_path,
            "is_loaded": self.is_loaded,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "dtype": self.dtype,
            "quantization": self.quantization,
        }
        
        # Add LoRA adapter information if applicable
        if self.is_adapter:
            info["is_lora_adapter"] = True
            info["base_model"] = self.base_model_path
            info["adapter_path"] = self.adapter_path
            info["lora_enabled"] = self.enable_lora
            
            # Add vision-specific info
            if self.is_vision_lora_adapter:
                info["is_vision_adapter"] = True
                info["merged_automatically"] = True
                info["note"] = "Vision LoRA was automatically merged (vLLM doesn't support LoRA on vision models)"
        
        return info


# Global inference service instance (will be managed by FastAPI lifespan)
_inference_service: Optional[InferenceService] = None


def get_inference_service() -> Optional[InferenceService]:
    """Get the global inference service instance."""
    return _inference_service


def set_inference_service(service: Optional[InferenceService]) -> None:
    """Set the global inference service instance."""
    global _inference_service
    _inference_service = service
