"""Training callbacks package.

This package consolidates all training callbacks into focused modules:
- metrics.py: TrainingMetricsCallback for tracking training progress
- progress.py: ProgressEstimationCallback for ETA calculations
- early_stopping.py: EarlyStoppingCallback for loss monitoring
- memory.py: MemoryMonitorCallback for GPU memory tracking

Example:
    from model_garden.training.callbacks import (
        TrainingMetricsCallback,
        ProgressEstimationCallback,
        EarlyStoppingCallback,
        MemoryMonitorCallback,
        TrainingMetrics,
        ProgressEstimate,
    )
"""

from model_garden.training.callbacks.metrics import (
    TrainingMetrics,
    TrainingMetricsCallback,
)
from model_garden.training.callbacks.progress import (
    ProgressEstimate,
    ProgressEstimationCallback,
)
from model_garden.training.callbacks.early_stopping import EarlyStoppingCallback
from model_garden.training.callbacks.memory import MemoryMonitorCallback

__all__ = [
    # Callback classes
    "TrainingMetricsCallback",
    "ProgressEstimationCallback",
    "EarlyStoppingCallback",
    "MemoryMonitorCallback",
    # Data classes
    "TrainingMetrics",
    "ProgressEstimate",
]
