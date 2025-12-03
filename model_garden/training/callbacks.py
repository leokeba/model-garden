"""Training callbacks for Model Garden.

This module provides callbacks for monitoring and controlling training:
- TrainingMetricsCallback: Real-time metrics tracking (loss, LR, GPU memory)
- ProgressEstimationCallback: ETA and progress percentage estimation
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

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


@dataclass
class ProgressEstimate:
    """Training progress estimation.

    Attributes:
        current_step: Current training step
        total_steps: Total training steps
        progress_percent: Progress as percentage (0-100)
        elapsed_seconds: Time elapsed since training started
        eta_seconds: Estimated time remaining (seconds)
        eta_formatted: Human-readable ETA string
        steps_per_second: Average steps per second
        current_epoch: Current epoch (float)
        total_epochs: Total epochs
    """

    current_step: int = 0
    total_steps: int = 0
    progress_percent: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0
    eta_formatted: str = "calculating..."
    steps_per_second: float = 0.0
    current_epoch: float = 0.0
    total_epochs: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress_percent": self.progress_percent,
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
            "eta_formatted": self.eta_formatted,
            "steps_per_second": self.steps_per_second,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
        }


class ProgressEstimationCallback(TrainerCallback):
    """Callback that provides training progress estimation with ETA.

    This callback calculates and reports:
    - Progress percentage based on steps or epochs
    - Estimated time remaining (ETA)
    - Steps per second throughput
    - Formatted progress messages

    Uses exponential moving average for smoother ETA estimates.

    Example:
        >>> def on_progress(estimate: ProgressEstimate):
        ...     print(f"{estimate.progress_percent:.1f}% - ETA: {estimate.eta_formatted}")
        ...
        >>> callback = ProgressEstimationCallback(
        ...     on_progress_callback=on_progress,
        ...     update_interval_steps=10
        ... )
        >>> trainer = SFTTrainer(..., callbacks=[callback])
    """

    def __init__(
        self,
        on_progress_callback: Callable[[ProgressEstimate], None] | None = None,
        update_interval_steps: int = 10,
        ema_alpha: float = 0.1,
        log_to_console: bool = True,
    ):
        """Initialize the progress estimation callback.

        Args:
            on_progress_callback: Optional callback invoked with progress updates.
            update_interval_steps: How often to update progress (default: every 10 steps).
            ema_alpha: Exponential moving average alpha for smoothing speed estimates.
                      Lower values = smoother but slower to adapt (default: 0.1).
            log_to_console: Whether to print progress to console (default: True).
        """
        super().__init__()
        self.on_progress_callback = on_progress_callback
        self.update_interval_steps = update_interval_steps
        self.ema_alpha = ema_alpha
        self.log_to_console = log_to_console

        # Timing state
        self._start_time: float | None = None
        self._last_update_time: float | None = None
        self._last_update_step: int = 0

        # Speed estimation (EMA)
        self._ema_steps_per_second: float | None = None

        # Current estimate
        self._current_estimate: ProgressEstimate | None = None

    @property
    def current_estimate(self) -> ProgressEstimate | None:
        """Get the most recent progress estimate."""
        return self._current_estimate

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Initialize timing at start of training."""
        self._start_time = time.time()
        self._last_update_time = time.time()
        self._last_update_step = 0
        self._ema_steps_per_second = None

        # Log initial progress
        total_steps = state.max_steps if state.max_steps > 0 else self._estimate_total_steps(args, state)
        console.print(f"[cyan]🚀 Training started: {total_steps} total steps[/cyan]")

        return None

    def _estimate_total_steps(self, args: TrainingArguments, state: TrainerState) -> int:
        """Estimate total steps from epochs and dataset size."""
        if state.max_steps > 0:
            return state.max_steps

        # Estimate from epochs
        # state.num_train_samples gives dataset size
        if hasattr(state, "num_train_samples") and state.num_train_samples:
            samples_per_epoch = state.num_train_samples
            effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
            steps_per_epoch = samples_per_epoch // effective_batch
            return int(steps_per_epoch * args.num_train_epochs)

        # Fallback: can't estimate
        return 0

    def _format_eta(self, seconds: float) -> str:
        """Format ETA as human-readable string."""
        if seconds < 0 or seconds > 86400 * 7:  # More than 7 days
            return "calculating..."

        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            if hours < 24:
                return f"{hours:.1f}h"
            else:
                days = hours / 24
                return f"{days:.1f}d"

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Update progress estimation periodically."""
        # Only update at specified interval
        if state.global_step % self.update_interval_steps != 0:
            return None

        if self._start_time is None:
            return None

        current_time = time.time()
        elapsed = current_time - self._start_time

        # Calculate steps per second
        if self._last_update_time is not None and state.global_step > self._last_update_step:
            interval_elapsed = current_time - self._last_update_time
            interval_steps = state.global_step - self._last_update_step
            current_speed = interval_steps / interval_elapsed if interval_elapsed > 0 else 0

            # Update EMA
            if self._ema_steps_per_second is None:
                self._ema_steps_per_second = current_speed
            else:
                self._ema_steps_per_second = (
                    self.ema_alpha * current_speed
                    + (1 - self.ema_alpha) * self._ema_steps_per_second
                )

        self._last_update_time = current_time
        self._last_update_step = state.global_step

        # Calculate progress and ETA
        total_steps = state.max_steps if state.max_steps > 0 else self._estimate_total_steps(args, state)
        progress_percent = (state.global_step / total_steps * 100) if total_steps > 0 else 0

        # Calculate ETA
        eta_seconds = 0.0
        if self._ema_steps_per_second and self._ema_steps_per_second > 0:
            remaining_steps = max(0, total_steps - state.global_step)
            eta_seconds = remaining_steps / self._ema_steps_per_second

        # Create estimate
        estimate = ProgressEstimate(
            current_step=state.global_step,
            total_steps=total_steps,
            progress_percent=progress_percent,
            elapsed_seconds=elapsed,
            eta_seconds=eta_seconds,
            eta_formatted=self._format_eta(eta_seconds),
            steps_per_second=self._ema_steps_per_second or 0.0,
            current_epoch=state.epoch or 0.0,
            total_epochs=args.num_train_epochs,
        )

        self._current_estimate = estimate

        # Log to console
        if self.log_to_console:
            console.print(
                f"[cyan]⏱️  Progress: {progress_percent:.1f}% "
                f"({state.global_step}/{total_steps}) | "
                f"ETA: {estimate.eta_formatted} | "
                f"{estimate.steps_per_second:.2f} steps/s[/cyan]"
            )

        # Invoke callback
        if self.on_progress_callback is not None:
            try:
                self.on_progress_callback(estimate)
            except Exception as e:
                console.print(f"[yellow]⚠️  Progress callback error: {e}[/yellow]")

        return None

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Print completion message."""
        if self._start_time is not None:
            total_time = time.time() - self._start_time
            console.print(
                f"[bold green]✅ Training complete! "
                f"Total time: {self._format_eta(total_time)}[/bold green]"
            )

        return None
