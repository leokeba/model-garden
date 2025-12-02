"""Early stopping callback for training."""

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from typing import Optional


class EarlyStoppingCallback(TrainerCallback):
    """Stop training when validation loss stops improving.
    
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
        **kwargs
    ):
        """Check if we should stop after evaluation."""
        metrics = kwargs.get("metrics", {})
        
        if self.metric not in metrics:
            return
        
        current_metric = metrics[self.metric]
        
        if self.best_metric is None:
            # First evaluation
            self.best_metric = current_metric
            print(f"🎯 Early stopping: Initial {self.metric} = {current_metric:.4f}")
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
                print(f"✅ Early stopping: {self.metric} improved by {improvement:.4f} to {current_metric:.4f}")
            else:
                # No improvement
                self.patience_counter += 1
                print(
                    f"⏳ Early stopping: No improvement in {self.metric} "
                    f"({self.patience_counter}/{self.patience})"
                )
                
                if self.patience_counter >= self.patience:
                    print(f"🛑 Early stopping: Stopping training (patience reached)")
                    control.should_training_stop = True
                    self.should_stop = True
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Print final early stopping status."""
        if self.should_stop:
            print(f"🏁 Training stopped early. Best {self.metric}: {self.best_metric:.4f}")
        elif self.best_metric is not None:
            print(f"🏁 Training completed. Best {self.metric}: {self.best_metric:.4f}")
