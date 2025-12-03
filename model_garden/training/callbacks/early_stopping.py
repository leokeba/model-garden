"""Early stopping callback for training.

This module provides the EarlyStoppingCallback that stops training
when validation loss stops improving.
"""

from typing import Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from model_garden.utils.console import console


class EarlyStoppingCallback(TrainerCallback):
    """Stop training when validation loss stops improving.

    This callback monitors a metric (default: eval_loss) during training
    and stops training if no improvement is seen for a specified number
    of evaluations (patience).

    Features:
    - Configurable patience (number of evaluations without improvement)
    - Configurable threshold for what counts as improvement
    - Supports both "lower is better" and "higher is better" metrics
    - Logs progress and final status

    Example:
        >>> callback = EarlyStoppingCallback(
        ...     patience=3,
        ...     threshold=0.001,
        ...     metric="eval_loss",
        ...     greater_is_better=False,
        ... )
        >>> trainer = SFTTrainer(..., callbacks=[callback])
        >>> trainer.train()

    Args:
        patience: Number of evaluations with no improvement before stopping
        threshold: Minimum change to qualify as improvement
        metric: Metric to monitor (default: "eval_loss", lower is better)
        greater_is_better: Whether higher metric values are better
    """

    def __init__(
        self,
        patience: int = 3,
        threshold: float = 0.0,
        metric: str = "eval_loss",
        greater_is_better: bool = False,
    ):
        """Initialize the early stopping callback.

        Args:
            patience: Number of evaluations without improvement to wait
            threshold: Minimum change to count as improvement
            metric: Name of the metric to monitor
            greater_is_better: Whether higher values are better
        """
        self.patience = patience
        self.threshold = threshold
        self.metric = metric
        self.greater_is_better = greater_is_better

        self.best_metric: Optional[float] = None
        self.patience_counter = 0
        self.should_stop = False

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Check if we should stop after evaluation."""
        metrics = kwargs.get("metrics", {})

        if self.metric not in metrics:
            return

        current_metric = metrics[self.metric]

        if self.best_metric is None:
            # First evaluation
            self.best_metric = current_metric
            console.print(
                f"[cyan]🎯 Early stopping: Initial {self.metric} = {current_metric:.4f}[/cyan]"
            )
        else:
            # Check for improvement
            if self.greater_is_better:
                improved = current_metric > (self.best_metric + self.threshold)
            else:
                improved = current_metric < (self.best_metric - self.threshold)

            if improved:
                # Improvement found
                improvement = abs(current_metric - self.best_metric)
                self.best_metric = current_metric
                self.patience_counter = 0
                console.print(
                    f"[green]✅ Early stopping: {self.metric} improved by "
                    f"{improvement:.4f} to {current_metric:.4f}[/green]"
                )
            else:
                # No improvement
                self.patience_counter += 1
                console.print(
                    f"[yellow]⏳ Early stopping: No improvement in {self.metric} "
                    f"({self.patience_counter}/{self.patience})[/yellow]"
                )

                if self.patience_counter >= self.patience:
                    console.print(
                        f"[red]🛑 Early stopping: Stopping training (patience reached)[/red]"
                    )
                    control.should_training_stop = True
                    self.should_stop = True

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Print final early stopping status."""
        if self.should_stop:
            console.print(
                f"[bold cyan]🏁 Training stopped early. "
                f"Best {self.metric}: {self.best_metric:.4f}[/bold cyan]"
            )
        elif self.best_metric is not None:
            console.print(
                f"[bold cyan]🏁 Training completed. "
                f"Best {self.metric}: {self.best_metric:.4f}[/bold cyan]"
            )
