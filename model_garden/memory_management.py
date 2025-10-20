"""Memory management utilities for training cleanup."""

import gc
import os
from typing import Any, Optional


def clear_trainer_internals(trainer: Any) -> None:
    """Clear internal references in a Trainer object to enable garbage collection.
    
    The Trainer holds references to many large objects. Clearing these
    explicitly helps Python's garbage collector free memory faster.
    
    Args:
        trainer: A Trainer or SFTTrainer instance
    """
    if trainer is None:
        return
    
    try:
        # Clear model references
        if hasattr(trainer, 'model'):
            trainer.model = None
        if hasattr(trainer, 'model_wrapped'):
            trainer.model_wrapped = None
        
        # Clear optimizer and scheduler
        if hasattr(trainer, 'optimizer'):
            trainer.optimizer = None
        if hasattr(trainer, 'lr_scheduler'):
            trainer.lr_scheduler = None
        
        # Clear dataloaders
        if hasattr(trainer, 'train_dataloader'):
            trainer.train_dataloader = None
        if hasattr(trainer, 'eval_dataloader'):
            trainer.eval_dataloader = None
        
        # Clear tokenizer and processor
        if hasattr(trainer, 'tokenizer'):
            trainer.tokenizer = None
        if hasattr(trainer, 'processor'):
            trainer.processor = None
        
        # Clear dataset references
        if hasattr(trainer, 'train_dataset'):
            trainer.train_dataset = None
        if hasattr(trainer, 'eval_dataset'):
            trainer.eval_dataset = None
        
        # Clear callbacks
        if hasattr(trainer, 'callback_handler'):
            trainer.callback_handler = None
        if hasattr(trainer, 'callbacks'):
            trainer.callbacks = []
        
        # Clear state and control
        if hasattr(trainer, 'state'):
            trainer.state = None
        if hasattr(trainer, 'control'):
            trainer.control = None
        
    except Exception as e:
        print(f"⚠️  Warning: Error clearing trainer internals: {e}")


def cleanup_training_resources(*objects_to_delete: Any) -> None:
    """Clean up training resources and free memory.
    
    Performs essential cleanup steps:
    1. Clear trainer internal references
    2. Clear dataset image caches
    3. Move models from GPU to CPU
    4. Delete objects
    5. Garbage collection (multiple passes)
    6. Clear GPU cache
    7. Clear HuggingFace caches
    8. Return memory to OS
    
    Args:
        *objects_to_delete: Variable number of objects to delete
        
    Example:
        cleanup_training_resources(
            trainer.model, trainer.tokenizer, trainer,
            dataset, callback
        )
    """
    print("🧹 Cleaning up training resources...")
    
    import torch
    
    # Step 1: Clear trainer internals first
    for obj in objects_to_delete:
        if obj is not None and hasattr(obj, '__class__'):
            class_name = obj.__class__.__name__
            if 'Trainer' in class_name:
                clear_trainer_internals(obj)
    
    # Step 2: Clear dataset image caches for vision models
    for obj in objects_to_delete:
        if obj is not None:
            try:
                # Clear list-based datasets (vision models store PIL images)
                if isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, dict):
                            # Clear PIL images
                            if 'image' in item:
                                try:
                                    if hasattr(item['image'], 'close'):
                                        item['image'].close()
                                    del item['image']
                                except Exception:
                                    pass
                            # Clear any other large objects
                            item.clear()
                    obj.clear()
                # Clear HuggingFace Dataset objects
                elif hasattr(obj, 'cleanup_cache_files'):
                    try:
                        obj.cleanup_cache_files()
                    except Exception:
                        pass
            except Exception:
                pass
    
    # Step 3: Move models to CPU to free GPU memory
    for obj in objects_to_delete:
        if obj is not None:
            try:
                if hasattr(obj, 'cpu') and callable(getattr(obj, 'cpu')):
                    obj.cpu()
                elif hasattr(obj, 'model') and hasattr(obj.model, 'cpu'):
                    obj.model.cpu()
            except Exception:
                pass
    
    # Step 4: Delete objects
    for obj in objects_to_delete:
        if obj is not None:
            try:
                del obj
            except Exception:
                pass
    
    # Step 5: Aggressive garbage collection (multiple passes)
    total_collected = 0
    for i in range(3):  # 3 passes to catch circular references
        collected = gc.collect()
        total_collected += collected
    if total_collected > 0:
        print(f"  ✓ Collected {total_collected} objects")
    
    # Step 6: Clear GPU cache
    if torch.cuda.is_available():
        try:
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            reserved_before = torch.cuda.memory_reserved() / 1024**3
            
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Try to clear IPC memory (shared memory between processes)
            try:
                torch.cuda.ipc_collect()
            except (AttributeError, RuntimeError):
                pass
            
            torch.cuda.empty_cache()
            
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            reserved_after = torch.cuda.memory_reserved() / 1024**3
            freed_gb = reserved_before - reserved_after
            
            if freed_gb > 0.1:  # Only print if significant
                print(f"  ✓ Freed {freed_gb:.2f} GB GPU memory")
            
        except Exception as e:
            print(f"  ⚠️  Warning: GPU cleanup error: {e}")
    
    # Step 7: Clear HuggingFace caches
    try:
        # Clear transformers cache
        from transformers.utils import is_torch_available
        if is_torch_available():
            import torch
            # Clear torch hub cache
            if hasattr(torch.hub, '_get_torch_home'):
                pass  # No specific clear method available
    except Exception:
        pass
    
    # Step 8: Return memory to OS (Linux only)
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        result = libc.malloc_trim(0)
        if result:
            print(f"  ✓ Returned unused memory to OS")
    except Exception:
        pass  # Not critical if this fails
    
    # Step 9: Final garbage collection
    gc.collect()
    
    # Report final memory state
    rss_mb = get_process_memory_mb()
    if rss_mb:
        print(f"  📊 Current RAM usage: {rss_mb:.0f} MB")
    
    print("✅ Training resources cleaned up")


def get_process_memory_mb() -> Optional[float]:
    """Get current process RSS memory in MB.
    
    Returns:
        Memory in MB, or None if unable to read
    """
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024  # Convert KB to MB
    except Exception:
        return None
    
    return None


def report_memory_usage(label: str = "") -> None:
    """Print current memory usage for debugging.
    
    Args:
        label: Optional label to identify the measurement point
    """
    try:
        import torch
        
        prefix = f"[{label}] " if label else ""
        
        # Process memory
        rss_mb = get_process_memory_mb()
        if rss_mb:
            print(f"{prefix}RAM: {rss_mb:.0f} MB")
        
        # GPU memory
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / 1024**3
            reserved_gb = torch.cuda.memory_reserved() / 1024**3
            print(f"{prefix}GPU: {allocated_gb:.2f} GB allocated, {reserved_gb:.2f} GB reserved")
    
    except Exception as e:
        print(f"⚠️  Warning: Could not report memory usage: {e}")
