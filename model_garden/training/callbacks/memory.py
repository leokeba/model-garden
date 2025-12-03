"""Memory monitoring callback for training.

This module provides the MemoryMonitorCallback for tracking GPU and
RAM memory usage during training.
"""

import psutil
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from model_garden.utils.console import console


class MemoryMonitorCallback(TrainerCallback):
    """Monitor memory usage during training.

    This callback provides visibility into memory usage patterns during training.
    Memory grows during the first ~80-100 steps (warmup phase) as PyTorch
    allocates memory pools, then stabilizes for the rest of training.

    Uses efficient CUDA memory APIs instead of iterating over all Python objects,
    which was causing significant overhead (1M+ objects to iterate).

    Features:
    - Tracks GPU memory (allocated and reserved)
    - Tracks process RAM usage
    - Reports peak memory at end of training
    - Configurable logging frequency

    Example:
        >>> callback = MemoryMonitorCallback(log_every_n_steps=10)
        >>> trainer = SFTTrainer(..., callbacks=[callback])
        >>> trainer.train()
        >>> print(f"Peak GPU: {callback.peak_gpu_mb}MB")
    """

    def __init__(self, log_every_n_steps: int = 10):
        """Initialize the memory monitor.

        Args:
            log_every_n_steps: How often to log memory stats (default: every 10 steps)
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._peak_gpu_mb = 0.0
        self._peak_ram_mb = 0.0

    @property
    def peak_gpu_mb(self) -> float:
        """Get peak GPU memory usage in MB."""
        return self._peak_gpu_mb

    @property
    def peak_ram_mb(self) -> float:
        """Get peak RAM usage in MB."""
        return self._peak_ram_mb

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log memory stats periodically."""
        if state.global_step % self.log_every_n_steps == 0:
            try:
                # Get process RAM usage (fast - single syscall)
                process = psutil.Process()
                ram_mb = process.memory_info().rss / (1024 * 1024)
                self._peak_ram_mb = max(self._peak_ram_mb, ram_mb)

                # Get GPU memory usage (fast - uses CUDA APIs directly)
                if torch.cuda.is_available():
                    gpu_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    gpu_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                    self._peak_gpu_mb = max(self._peak_gpu_mb, gpu_allocated_mb)

                    console.print(
                        f"[cyan]Step {state.global_step}: "
                        f"GPU {gpu_allocated_mb:.0f}MB allocated / {gpu_reserved_mb:.0f}MB reserved, "
                        f"RAM {ram_mb:.0f}MB[/cyan]"
                    )
                else:
                    console.print(f"[cyan]Step {state.global_step}: RAM {ram_mb:.0f}MB[/cyan]")
            except Exception as e:
                # If memory monitoring fails, log but don't crash training
                console.print(
                    f"[yellow]⚠️  Memory monitoring error at step {state.global_step}: {e}[/yellow]"
                )
        # Return None to match base class signature
        return None

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log peak memory usage at end of training."""
        try:
            console.print(
                f"[cyan]📊 Peak memory usage: RAM {self._peak_ram_mb:.0f}MB"
                + (f", GPU {self._peak_gpu_mb:.0f}MB" if self._peak_gpu_mb > 0 else "")
                + "[/cyan]"
            )
        except Exception:
            pass
        return None
