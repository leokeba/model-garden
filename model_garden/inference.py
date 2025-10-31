"""vLLM-powered inference service for Model Garden."""

import os
import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

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
        import re
        size_match = re.search(r'[-_](\d+(?:\.\d+)?)[Bb](?!it)', model_path)
        if size_match:
            param_size = float(size_match.group(1))
            # Rough estimate: FP16 = 2 bytes per parameter -> size in GB
            return max(1.0, param_size * 2)
        return 7.0

    # For local models, check actual file sizes
    if not model_dir.exists() or not model_dir.is_dir():
        return 7.0  # Default estimate

    total_bytes = 0
    for pattern in ("*.safetensors", "*.bin"):
        for wf in model_dir.glob(pattern):
            try:
                total_bytes += wf.stat().st_size
            except Exception:
                continue

    # Convert bytes to GB and ensure a sensible minimum
    total_gb = total_bytes / (1024 ** 3) if total_bytes > 0 else 1.0
    return max(1.0, round(total_gb, 2))


def calculate_gpu_memory_utilization(
    model_path: str,
    max_model_len: Optional[int] = None,
    tensor_parallel_size: int = 1,
) -> float:
    """Lightweight heuristic to calculate GPU memory utilization.

    This is a conservative estimate used when gpu_memory_utilization==0.0 (auto).
    It uses detected GPU memory, an estimate of model size, and a rough KV cache
    estimation to return a utilization fraction between 0.5 and 0.95.
    """
    try:
        gpu_memory_gb = get_gpu_memory_gb() or 24.0
        model_size_gb = estimate_model_size_gb(model_path)

        # KV cache estimate: proportional to model size and sequence length
        if not max_model_len:
            max_model_len = 4096
        kv_cache_gb = (model_size_gb / 7.0) * (max_model_len / 1000) * 0.4

        total_needed = model_size_gb + kv_cache_gb

        # Safety margin
        total_with_margin = total_needed * 1.2

        # Simple rules to pick utilization
        if total_with_margin >= gpu_memory_gb:
            utilization = 0.88 if gpu_memory_gb >= 16 else 0.75
        elif total_with_margin >= gpu_memory_gb * 0.7:
            utilization = 0.80
        else:
            utilization = 0.60

        # Adjust for tensor parallelism (reduce per-GPU utilization)
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
        
        # Tokenizer for chat template formatting (loaded separately from vLLM)
        self.tokenizer = None
        
        # LoRA adapter tracking
        self.is_adapter = False
        self.base_model_path: Optional[str] = None
        self.adapter_path: Optional[str] = None
        
        # Vision model tracking
        self.is_vision_lora_adapter = False
        self.merged_vision_model_path: Optional[str] = None  # Temp merged model path
        self.original_base_model: Optional[str] = None  # Original base model for tokenizer (for merged vision models)
        
        # Request serialization for vision models (prevents vLLM deadlocks with concurrent multimodal requests)
        self._vision_request_semaphore = asyncio.Semaphore(1)  # Only 1 concurrent vision request

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
                # Store the original base model for tokenizer loading
                self.original_base_model = base_model
                
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
                    # Option A: Run merge in main process for debugging
                    # If MODEL_GARDEN_DEBUG_RUN_MERGE_IN_MAIN is set to 1/true, perform the merge
                    # inline so tracebacks and prints appear in the main logs for easier debugging.
                    # Always run merge in-process (subprocess merge support removed)
                    try:
                        from model_garden.vision_training import merge_vision_lora_adapter
                        merged_path = merge_vision_lora_adapter(
                            adapter_path=self.adapter_path,
                            output_dir=self.merged_vision_model_path,
                            base_model=base_model,
                            load_in_4bit=True,
                        )

                        console.print("[cyan]POST_MERGE: Merge handler completed by vision_training.merge_vision_lora_adapter().[/cyan]")

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
                # Enable vLLM optimizations that are on by default in vLLM CLI
                "enable_prefix_caching": True,  # Enables prefix caching for better performance
                "enable_chunked_prefill": True,  # Enables chunked prefill (auto-sized)
            }
            
            # For vision models (Qwen2.5-VL, LLaVA, etc), use the base model tokenizer
            # This is critical because fine-tuned vision models may have incomplete tokenizers
            is_vision = is_vision_model(self.model_path)
            
            if is_vision:
                # Determine the correct base tokenizer for vision models
                if self.is_vision_lora_adapter and self.original_base_model:
                    # For merged adapters, we stored the original base model
                    tokenizer_path = self.original_base_model
                    console.print(f"[cyan]📝 Vision adapter: using original base tokenizer: {tokenizer_path}[/cyan]")
                elif "qwen2.5-vl" in self.model_path.lower() or "qwen2-vl" in self.model_path.lower():
                    # For Qwen2.5-VL models, use the official Qwen tokenizer
                    # Extract size (72B, 7B, etc) from model name
                    model_name_lower = self.model_path.lower()
                    if "72b" in model_name_lower:
                        tokenizer_path = "unsloth/Qwen2.5-VL-72B-Instruct"
                    elif "7b" in model_name_lower:
                        tokenizer_path = "unsloth/Qwen2.5-VL-7B-Instruct"  
                    elif "3b" in model_name_lower:
                        tokenizer_path = "Qwen/Qwen2.5-VL-3B-Instruct"
                    else:
                        # Default to 7B if size not detected
                        tokenizer_path = "unsloth/Qwen2.5-VL-7B-Instruct"
                    console.print(f"[cyan]📝 Qwen2.5-VL model: using base tokenizer: {tokenizer_path}[/cyan]")
                else:
                    # For other vision models, use the model itself as tokenizer
                    tokenizer_path = self.base_model_path
                    console.print(f"[cyan]📝 Vision model: using model's own tokenizer: {tokenizer_path}[/cyan]")
                
                engine_args_dict["tokenizer"] = tokenizer_path

            # Debug: print engine args we will pass to vLLM so we can compare with
            # the vllm CLI behavior when troubleshooting timeouts.
            try:
                console.print("[magenta]🔍 vLLM engine args preview:[/magenta]")
                # Pretty-print keys we explicitly set
                for k, v in engine_args_dict.items():
                    console.print(f"  {k}: {v}")
            except Exception:
                pass
            
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
            
            # Load tokenizer separately for chat template formatting
            try:
                from transformers import AutoTokenizer
                console.print(f"[cyan]📝 Loading tokenizer for chat template support...[/cyan]")
                # For vision models with merged adapters, use the original base model tokenizer
                # Otherwise use the actual loaded model path
                if self.is_vision_lora_adapter and self.original_base_model:
                    tokenizer_path = self.original_base_model
                    console.print(f"[cyan]   Using original base model tokenizer: {tokenizer_path}[/cyan]")
                else:
                    tokenizer_path = self.base_model_path if self.is_adapter else self.model_path
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    trust_remote_code=self.trust_remote_code
                )
                console.print("[green]✓[/green] Tokenizer loaded successfully")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not load tokenizer: {e}[/yellow]")
                console.print("[yellow]   Chat formatting will use simple fallback[/yellow]")
                self.tokenizer = None
            
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

        # Log request details for debugging
        console.print(f"[magenta]🎯 generate() called:[/magenta]")
        console.print(f"  prompt length: {len(prompt)} chars")
        console.print(f"  max_tokens: {max_tokens}")
        console.print(f"  temperature: {temperature}")
        console.print(f"  images: {len(images) if images else 0}")
        console.print(f"  structured_outputs: {bool(structured_outputs)}")
        console.print(f"  stream: {stream}")

        from vllm import SamplingParams
        
        # Set default max_tokens if not provided
        if max_tokens is None:
            if structured_outputs:
                max_tokens = 16384  # High default for complex documents (CMRs can have 10k+ tokens)
                # Note: Qwen2.5-VL has 32k context, most prompts are 5-15k, so 16k output is safe
            else:
                max_tokens = 512  # Standard default
        
        # Create structured outputs params if provided
        structured_outputs_params = None
        if structured_outputs:
            try:
                from vllm.sampling_params import StructuredOutputsParams
                structured_outputs_params = StructuredOutputsParams(**structured_outputs)
            except ImportError:
                console.print("[yellow]Warning: StructuredOutputsParams not available in this vLLM version[/yellow]")
        
        # Create sampling parameters - use vLLM defaults for any None values
        # Only pass parameters that are explicitly provided by the client
        sampling_params_dict = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": 0,  # Deterministic generation like vLLM CLI default
        }
        
        # Add optional parameters only if provided (let vLLM use defaults otherwise)
        if top_p is not None:
            sampling_params_dict["top_p"] = top_p
        if top_k is not None:
            sampling_params_dict["top_k"] = top_k
        if frequency_penalty is not None:
            sampling_params_dict["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            sampling_params_dict["presence_penalty"] = presence_penalty
        if repetition_penalty is not None:
            sampling_params_dict["repetition_penalty"] = repetition_penalty
        if stop is not None:
            sampling_params_dict["stop"] = stop
        if structured_outputs_params is not None:
            sampling_params_dict["structured_outputs"] = structured_outputs_params
        
        sampling_params = SamplingParams(**sampling_params_dict)
        
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
            
            # Note: The prompt should already be formatted with proper chat template
            # including vision tokens if needed (done by _format_chat_messages with apply_chat_template)
            # Vision tokens like <|vision_start|><|image_pad|><|vision_end|> are automatically
            # added by the tokenizer's chat template when formatting multimodal messages
            
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
        """Generate complete response (non-streaming).
        
        Uses a semaphore to serialize vision model requests to prevent vLLM deadlocks.
        """
        if self.engine is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Check if this is a vision request (inputs is TextPrompt with multi_modal_data)
        is_vision_request = not isinstance(inputs, str)
        
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
        
        # For vision requests, use semaphore to serialize (prevents vLLM deadlocks)
        if is_vision_request:
            async with self._vision_request_semaphore:
                console.print(f"[cyan]🔒 Acquired vision request lock for {request_id}[/cyan]")
                console.print(f"[cyan]📊 Calling engine.generate with sampling_params: max_tokens={sampling_params.max_tokens}, temp={sampling_params.temperature}[/cyan]")
                console.print(f"[cyan]📊 Input type: {type(inputs)}, is TextPrompt: {hasattr(inputs, 'prompt')}[/cyan]")
                
                import time
                start_time = time.time()
                results_generator = self.engine.generate(inputs, sampling_params, request_id, lora_request=lora_request)
                console.print(f"[green]✓ engine.generate() returned generator in {time.time()-start_time:.2f}s[/green]")
                
                final_output = None
                iteration_count = 0
                async for request_output in results_generator:
                    iteration_count += 1
                    if iteration_count % 10 == 0:
                        console.print(f"[cyan]📊 Generator iteration {iteration_count}, outputs: {len(request_output.outputs)}[/cyan]")
                    final_output = request_output
                    
                console.print(f"[green]✓ Generation completed after {iteration_count} iterations in {time.time()-start_time:.2f}s[/green]")
                console.print(f"[cyan]🔓 Released vision request lock for {request_id}[/cyan]")
        else:
            console.print(f"[cyan]📊 Non-vision request: calling engine.generate[/cyan]")
            import time
            start_time = time.time()
            results_generator = self.engine.generate(inputs, sampling_params, request_id, lora_request=lora_request)
            console.print(f"[green]✓ engine.generate() returned generator in {time.time()-start_time:.2f}s[/green]")
            
            final_output = None
            iteration_count = 0
            async for request_output in results_generator:
                iteration_count += 1
                if iteration_count % 10 == 0:
                    console.print(f"[cyan]📊 Generator iteration {iteration_count}[/cyan]")
                final_output = request_output
            console.print(f"[green]✓ Generation completed after {iteration_count} iterations in {time.time()-start_time:.2f}s[/green]")
        
        if final_output is None:
            return {"text": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        
        # Return the generated text with usage stats
        generated_text = final_output.outputs[0].text
        finish_reason = final_output.outputs[0].finish_reason
        
        # Log a sample of the output for debugging repetition issues
        text_preview = generated_text[:500] if len(generated_text) > 500 else generated_text
        console.print(f"[cyan]📝 Generated text preview (first 500 chars): {text_preview}[/cyan]")
        console.print(f"[cyan]📝 Total generated length: {len(generated_text)} chars, finish_reason: {finish_reason}[/cyan]")
        
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
        """Generate streaming response.
        
        Uses a semaphore to serialize vision model requests to prevent vLLM deadlocks.
        """
        if self.engine is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Check if this is a vision request (inputs is TextPrompt with multi_modal_data)
        is_vision_request = not isinstance(inputs, str)
        
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
        
        # For vision requests, use semaphore to serialize (prevents vLLM deadlocks)
        if is_vision_request:
            async with self._vision_request_semaphore:
                console.print(f"[cyan]🔒 Acquired vision request lock for {request_id}[/cyan]")
                results_generator = self.engine.generate(inputs, sampling_params, request_id, lora_request=lora_request)
                
                previous_text = ""
                async for request_output in results_generator:
                    text = request_output.outputs[0].text
                    # Yield only the new tokens
                    new_text = text[len(previous_text):]
                    if new_text:
                        yield new_text
                    previous_text = text
                console.print(f"[cyan]🔓 Released vision request lock for {request_id}[/cyan]")
        else:
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
        # For vision models, convert to multimodal format before applying template
        if image and self.tokenizer:
            # Convert messages to multimodal format with image placeholder
            messages = self._inject_image_into_messages(messages, image)
        
        prompt = self._format_chat_messages(messages)
        
        if stream:
            return self._chat_completion_stream(messages, prompt, max_tokens, temperature, top_p, image=image, structured_outputs=structured_outputs, **kwargs)
        else:
            return await self._chat_completion_complete(messages, prompt, max_tokens, temperature, top_p, image=image, structured_outputs=structured_outputs, **kwargs)

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages using the model's native chat template.
        
        Automatically uses the tokenizer's apply_chat_template() method, which supports
        any chat model (Qwen, Llama, Phi, Mistral, etc.) without hardcoding templates.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            
        Returns:
            Formatted prompt string ready for the model
        """
        if not self.tokenizer:
            console.print("[yellow]⚠️  No tokenizer available, using simple format[/yellow]")
            return self._format_simple(messages)
        
        try:
            # Use the tokenizer's built-in chat template (works for any model!)
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not apply chat template: {e}[/yellow]")
            console.print("[yellow]    Falling back to simple format[/yellow]")
            return self._format_simple(messages)
    
    def _inject_image_into_messages(self, messages: List[Dict[str, Any]], image: str) -> List[Dict[str, Any]]:
        """Convert messages to multimodal format by injecting image placeholder.
        
        Transforms the last user message to include an image placeholder in the
        OpenAI multimodal format, which the chat template will process correctly.
        
        Args:
            messages: Original text-only messages
            image: Image data (will be passed separately to vLLM)
            
        Returns:
            Messages with image placeholder injected into last user message
        """
        # Deep copy to avoid modifying original
        import copy
        modified_messages = copy.deepcopy(messages)
        
        # Find the last user message
        for i in range(len(modified_messages) - 1, -1, -1):
            if modified_messages[i].get('role') == 'user':
                content = modified_messages[i].get('content', '')
                
                # Convert to multimodal format if not already
                if isinstance(content, str):
                    modified_messages[i]['content'] = [
                        {"type": "image"},  # Placeholder for vision tokens
                        {"type": "text", "text": content}
                    ]
                break
        
        return modified_messages
    def _format_simple(self, messages: List[Dict[str, str]]) -> str:
        """Simple fallback formatting for models without chat templates.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            
        Returns:
            Simple formatted prompt string
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
