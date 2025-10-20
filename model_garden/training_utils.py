"""Shared utilities for training configuration.

This module contains common helper functions used by both training.py and vision_training.py.
"""

import gc
import torch
import psutil
from typing import Optional
from transformers.trainer_callback import TrainerCallback
from rich.console import Console

console = Console()


def detect_model_dtype(
    model,
    load_in_4bit: bool,
    load_in_8bit: bool,
) -> torch.dtype:
    """Detect the actual dtype of a model's parameters.
    
    This function reliably detects the precision of model parameters by checking
    the actual parameter tensors, not just model attributes which can be misleading.
    
    Why this matters:
    - Many modern models (e.g., Qwen2.5-VL) use bfloat16 natively
    - The model's .dtype attribute may return float32 (default) even when parameters are bfloat16
    - Unsloth's SFTTrainer validates precision and throws an error on mismatch
    - We need to match training precision (fp16/bf16) to actual model precision
    
    Detection strategy:
    1. For quantized models (4-bit/8-bit): Always return bfloat16 (standard practice)
    2. For 16-bit models: Check actual parameter dtype (most reliable)
    3. Fallback: Check model attributes if parameters not accessible
    
    Args:
        model: The model to check (can be wrapped in PeftModel, etc.)
        load_in_4bit: Whether model was loaded with 4-bit quantization
        load_in_8bit: Whether model was loaded with 8-bit quantization
        
    Returns:
        torch.dtype: The detected dtype (e.g., torch.bfloat16, torch.float16, torch.float32)
        
    Examples:
        >>> model_dtype = detect_model_dtype(model, load_in_4bit=False, load_in_8bit=False)
        >>> is_bf16 = model_dtype == torch.bfloat16
        >>> training_args = {"fp16": not is_bf16, "bf16": is_bf16}
    """
    model_dtype = None
    
    # For quantized models, always use bfloat16 for training
    # This is standard practice for 4-bit/8-bit quantized models
    if load_in_4bit or load_in_8bit:
        return torch.bfloat16
    
    # For 16-bit (non-quantized) models, detect the actual dtype
    if model is not None:
        # Method 1: Check actual parameter dtypes (MOST RELIABLE)
        # This directly inspects the tensor dtype, which is the ground truth
        try:
            first_param = next(model.parameters())
            model_dtype = first_param.dtype
            return model_dtype
        except (StopIteration, AttributeError):
            pass
        
        # Method 2: Check model.dtype attribute
        if hasattr(model, 'dtype'):
            model_dtype = model.dtype
            if model_dtype is not None:
                return model_dtype
        
        # Method 3: For wrapped models (PeftModel, etc), check the base model
        if hasattr(model, 'model') and hasattr(model.model, 'dtype'):
            model_dtype = model.model.dtype
            if model_dtype is not None:
                return model_dtype
        
        # Method 4: Check config as last resort
        if hasattr(model, 'config') and hasattr(model.config, 'torch_dtype'):
            model_dtype = model.config.torch_dtype
            if model_dtype is not None:
                return model_dtype
    
    # Default to float32 if we couldn't detect anything
    # This is a safe fallback that won't cause precision mismatches
    return torch.float32


def get_training_precision_config(
    model,
    load_in_4bit: bool,
    load_in_8bit: bool,
) -> dict:
    """Get fp16/bf16 configuration for training based on model dtype.
    
    This is a convenience wrapper around detect_model_dtype() that returns
    a dictionary suitable for passing to TrainingArguments or SFTConfig.
    
    Args:
        model: The model to check
        load_in_4bit: Whether model was loaded with 4-bit quantization
        load_in_8bit: Whether model was loaded with 8-bit quantization
        
    Returns:
        dict: Dictionary with 'fp16' and 'bf16' keys set appropriately
        
    Examples:
        >>> precision_config = get_training_precision_config(model, False, False)
        >>> # precision_config = {"fp16": False, "bf16": True}  # for bfloat16 model
        >>> training_args = SFTConfig(
        ...     output_dir="./output",
        ...     **precision_config,
        ...     # ... other args
        ... )
    """
    model_dtype = detect_model_dtype(model, load_in_4bit, load_in_8bit)
    is_bfloat16 = model_dtype == torch.bfloat16
    
    return {
        "fp16": not is_bfloat16,
        "bf16": is_bfloat16,
    }


class MemoryMonitorCallback(TrainerCallback):
    """Monitor memory usage and tensor count during training.
    
    This callback provides visibility into memory usage patterns during training.
    Memory grows during the first ~80-100 steps (warmup phase) as PyTorch
    allocates memory pools, then stabilizes for the rest of training.
    
    The callback handles weak references gracefully to prevent crashes during
    garbage collection cycles.
    """
    
    def on_step_end(self, args, state, control, **kwargs):
        """Log memory stats every 10 steps."""
        if state.global_step % 10 == 0:
            try:
                # Count tensor objects for debugging
                # Note: Wrap in try-except to handle weak references that may have been collected
                tensors = []
                for obj in gc.get_objects():
                    try:
                        if isinstance(obj, torch.Tensor):
                            tensors.append(obj)
                    except ReferenceError:
                        # Object was weakly referenced and has been collected
                        continue
                
                cpu_tensors = [t for t in tensors if t.device.type == 'cpu']
                cuda_tensors = [t for t in tensors if t.device.type == 'cuda']
                
                # Get process memory usage
                process = psutil.Process()
                mem_mb = process.memory_info().rss / (1024 * 1024)
                
                console.print(f"[cyan]Step {state.global_step}: {len(tensors)} tensors "
                              f"(CPU: {len(cpu_tensors)}, GPU: {len(cuda_tensors)}), RAM: {int(mem_mb)} MB[/cyan]")
            except Exception as e:
                # If memory monitoring fails, log but don't crash training
                console.print(f"[yellow]⚠️  Memory monitoring error at step {state.global_step}: {e}[/yellow]")
        # Return None to match base class signature (control is passed by reference and modified in place)
        return None
