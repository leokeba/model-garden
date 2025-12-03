"""Training metrics callback for real-time monitoring.

This module provides the TrainingMetricsCallback for tracking training
metrics including loss, learning rate, GPU memory, and throughput.
"""

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import psutil
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from model_garden.utils.console import console


@dataclass
class TrainingMetrics:
    """Container for training metrics at a point in time.

    Attributes:
        step: Current training step
        epoch: Current epoch (float for fractional epochs)
        loss: Current training loss
        learning_rate: Current learning rate
        gpu_memory_allocated_mb: GPU memory allocated (MB)
        gpu_memory_reserved_mb: GPU memory reserved (MB)
        ram_usage_mb: Process RAM usage (MB)
        tokens_per_second: Training throughput (tokens/sec)
        samples_per_second: Training throughput (samples/sec)
        grad_norm: Gradient norm (if available)
        timestamp: Unix timestamp when metrics were collected
    """

    step: int = 0
    epoch: float = 0.0
    loss: float | None = None
    learning_rate: float | None = None
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_reserved_mb: float = 0.0
    ram_usage_mb: float = 0.0
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0
    grad_norm: float | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "gpu_memory_allocated_mb": self.gpu_memory_allocated_mb,
            "gpu_memory_reserved_mb": self.gpu_memory_reserved_mb,
            "ram_usage_mb": self.ram_usage_mb,
            "tokens_per_second": self.tokens_per_second,
            "samples_per_second": self.samples_per_second,
            "grad_norm": self.grad_norm,
            "timestamp": self.timestamp,
        }


class TrainingMetricsCallback(TrainerCallback):
    """Callback that tracks and exposes training metrics in real-time.

    This callback collects comprehensive training metrics at each logging step
    and makes them available for external consumers (e.g., WebSocket updates,
    monitoring dashboards, API endpoints).

    Features:
    - Tracks loss, learning rate, GPU memory, RAM usage
    - Calculates throughput (tokens/sec, samples/sec)
    - Maintains rolling history for trend analysis
    - Supports external metric consumers via callbacks

    Example:
        >>> def on_metrics(metrics: TrainingMetrics):
        ...     print(f"Step {metrics.step}: loss={metrics.loss:.4f}")
        ...
        >>> callback = TrainingMetricsCallback(
        ...     on_metrics_callback=on_metrics,
        ...     history_size=100
        ... )
        >>> trainer = SFTTrainer(..., callbacks=[callback])
        >>> trainer.train()
        >>> # Access metrics after training
        >>> print(callback.current_metrics)
        >>> print(callback.metrics_history)
    """

    def __init__(
        self,
        on_metrics_callback: Callable[[TrainingMetrics], None] | None = None,
        history_size: int = 100,
        log_to_console: bool = True,
    ):
        """Initialize the metrics callback.

        Args:
            on_metrics_callback: Optional callback function invoked with metrics
                                at each logging step. Use this to stream metrics
                                to WebSocket, save to database, etc.
            history_size: Number of metrics entries to keep in history (default: 100).
                         Older entries are discarded to limit memory usage.
            log_to_console: Whether to print metrics to console (default: True).
        """
        super().__init__()
        self.on_metrics_callback = on_metrics_callback
        self.history_size = history_size
        self.log_to_console = log_to_console

        # Current and historical metrics
        self._current_metrics: TrainingMetrics | None = None
        self._metrics_history: deque[TrainingMetrics] = deque(maxlen=history_size)

        # Peak values for summary
        self._peak_gpu_memory_mb: float = 0.0
        self._peak_ram_mb: float = 0.0
        self._min_loss: float | None = None

        # Timing for throughput calculation
        self._start_time: float | None = None
        self._last_step_time: float | None = None
        self._total_samples: int = 0

    @property
    def current_metrics(self) -> TrainingMetrics | None:
        """Get the most recent metrics snapshot."""
        return self._current_metrics

    @property
    def metrics_history(self) -> list[TrainingMetrics]:
        """Get the metrics history as a list."""
        return list(self._metrics_history)

    @property
    def peak_gpu_memory_mb(self) -> float:
        """Get peak GPU memory usage in MB."""
        return self._peak_gpu_memory_mb

    @property
    def peak_ram_mb(self) -> float:
        """Get peak RAM usage in MB."""
        return self._peak_ram_mb

    @property
    def min_loss(self) -> float | None:
        """Get minimum training loss observed."""
        return self._min_loss

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Initialize timing at start of training."""
        self._start_time = time.time()
        self._last_step_time = time.time()
        self._total_samples = 0
        return None

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Collect metrics when trainer logs.

        This is called at each logging_steps interval.
        """
        if logs is None:
            logs = {}

        current_time = time.time()

        # Calculate throughput
        samples_per_second = 0.0
        if self._last_step_time is not None:
            elapsed = current_time - self._last_step_time
            if elapsed > 0:
                # Samples since last log
                batch_size = args.per_device_train_batch_size
                grad_accum = args.gradient_accumulation_steps
                logging_steps = args.logging_steps
                samples = batch_size * grad_accum * logging_steps
                samples_per_second = samples / elapsed

        self._last_step_time = current_time

        # Get memory usage
        gpu_allocated_mb = 0.0
        gpu_reserved_mb = 0.0
        if torch.cuda.is_available():
            gpu_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            self._peak_gpu_memory_mb = max(self._peak_gpu_memory_mb, gpu_allocated_mb)

        # Get RAM usage
        try:
            process = psutil.Process()
            ram_mb = process.memory_info().rss / (1024 * 1024)
            self._peak_ram_mb = max(self._peak_ram_mb, ram_mb)
        except Exception:
            ram_mb = 0.0

        # Extract metrics from logs
        loss = logs.get("loss")
        learning_rate = logs.get("learning_rate")
        grad_norm = logs.get("grad_norm")

        # Track minimum loss
        if loss is not None:
            if self._min_loss is None or loss < self._min_loss:
                self._min_loss = loss

        # Create metrics object
        metrics = TrainingMetrics(
            step=state.global_step,
            epoch=state.epoch or 0.0,
            loss=loss,
            learning_rate=learning_rate,
            gpu_memory_allocated_mb=gpu_allocated_mb,
            gpu_memory_reserved_mb=gpu_reserved_mb,
            ram_usage_mb=ram_mb,
            samples_per_second=samples_per_second,
            tokens_per_second=0.0,  # Would need tokenizer to calculate
            grad_norm=grad_norm,
            timestamp=current_time,
        )

        # Store metrics
        self._current_metrics = metrics
        self._metrics_history.append(metrics)

        # Log to console if enabled
        if self.log_to_console:
            loss_str = f"loss={loss:.4f}" if loss is not None else "loss=N/A"
            lr_str = f"lr={learning_rate:.2e}" if learning_rate is not None else ""
            gpu_str = f"GPU={gpu_allocated_mb:.0f}MB" if gpu_allocated_mb > 0 else ""

            parts = [f"Step {state.global_step}", loss_str]
            if lr_str:
                parts.append(lr_str)
            if gpu_str:
                parts.append(gpu_str)
            if samples_per_second > 0:
                parts.append(f"{samples_per_second:.1f} samples/s")

            console.print(f"[cyan]📊 {' | '.join(parts)}[/cyan]")

        # Invoke external callback if provided
        if self.on_metrics_callback is not None:
            try:
                self.on_metrics_callback(metrics)
            except Exception as e:
                console.print(f"[yellow]⚠️  Metrics callback error: {e}[/yellow]")

        return None

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Print summary at end of training."""
        if self._start_time is not None:
            total_time = time.time() - self._start_time

            console.print("\n[bold cyan]📊 Training Metrics Summary[/bold cyan]")
            console.print(f"  Total time: {total_time / 60:.1f} minutes")
            console.print(f"  Total steps: {state.global_step}")

            if self._min_loss is not None:
                console.print(f"  Min loss: {self._min_loss:.4f}")

            if self._peak_gpu_memory_mb > 0:
                console.print(f"  Peak GPU memory: {self._peak_gpu_memory_mb:.0f} MB")

            console.print(f"  Peak RAM: {self._peak_ram_mb:.0f} MB")

        return None
