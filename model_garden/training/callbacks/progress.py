"""Progress estimation callback with ETA calculation.

This module provides the ProgressEstimationCallback for tracking training
progress and estimating time remaining using exponential moving averages.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from model_garden.utils.console import console


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
        total_steps = (
            state.max_steps if state.max_steps > 0 else self._estimate_total_steps(args, state)
        )
        console.print(f"[cyan]🚀 Training started: {total_steps} total steps[/cyan]")

        return None

    def _estimate_total_steps(self, args: TrainingArguments, state: TrainerState) -> int:
        """Estimate total steps from epochs and dataset size."""
        if state.max_steps > 0:
            return state.max_steps

        # Estimate from epochs
        # state.num_train_samples gives dataset size
        num_train_samples = getattr(state, "num_train_samples", None)
        if num_train_samples:
            samples_per_epoch = num_train_samples
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
        total_steps = (
            state.max_steps if state.max_steps > 0 else self._estimate_total_steps(args, state)
        )
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
