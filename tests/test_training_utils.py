"""Tests for training utilities and callbacks.

These tests verify training utility functions like dtype detection,
early stopping, and memory monitoring work correctly.

Note: These tests require real torch imports, so they're marked as requires_gpu
to bypass the mock_heavy_imports fixture in conftest.py.
"""

from unittest.mock import MagicMock

import pytest
import torch

# Mark all tests in this module to use real torch (not mocked)
pytestmark = pytest.mark.requires_gpu


class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback."""

    def test_init_default_values(self):
        """Test default initialization values."""
        from model_garden.training.callbacks import EarlyStoppingCallback

        callback = EarlyStoppingCallback()
        assert callback.patience == 3
        assert callback.threshold == 0.0
        assert callback.metric == "eval_loss"
        assert callback.greater_is_better is False
        assert callback.best_metric is None
        assert callback.patience_counter == 0
        assert callback.should_stop is False

    def test_init_custom_values(self):
        """Test custom initialization values."""
        from model_garden.training.callbacks import EarlyStoppingCallback

        callback = EarlyStoppingCallback(
            patience=5,
            threshold=0.01,
            metric="eval_accuracy",
            greater_is_better=True,
        )
        assert callback.patience == 5
        assert callback.threshold == 0.01
        assert callback.metric == "eval_accuracy"
        assert callback.greater_is_better is True

    def test_on_evaluate_first_eval(self):
        """Test first evaluation sets best_metric."""
        from model_garden.training.callbacks import EarlyStoppingCallback

        callback = EarlyStoppingCallback()
        control = MagicMock()
        control.should_training_stop = False

        callback.on_evaluate(
            args=MagicMock(), state=MagicMock(), control=control, metrics={"eval_loss": 0.5}
        )

        assert callback.best_metric == 0.5
        assert callback.patience_counter == 0
        assert control.should_training_stop is False

    def test_on_evaluate_improvement_lower_is_better(self):
        """Test improvement detection when lower is better."""
        from model_garden.training.callbacks import EarlyStoppingCallback

        callback = EarlyStoppingCallback(greater_is_better=False)
        callback.best_metric = 0.5
        control = MagicMock()

        # Loss improved (decreased)
        callback.on_evaluate(
            args=MagicMock(), state=MagicMock(), control=control, metrics={"eval_loss": 0.4}
        )

        assert callback.best_metric == 0.4
        assert callback.patience_counter == 0

    def test_on_evaluate_improvement_higher_is_better(self):
        """Test improvement detection when higher is better."""
        from model_garden.training.callbacks import EarlyStoppingCallback

        callback = EarlyStoppingCallback(metric="eval_accuracy", greater_is_better=True)
        callback.best_metric = 0.8
        control = MagicMock()

        # Accuracy improved (increased)
        callback.on_evaluate(
            args=MagicMock(), state=MagicMock(), control=control, metrics={"eval_accuracy": 0.85}
        )

        assert callback.best_metric == 0.85
        assert callback.patience_counter == 0

    def test_on_evaluate_no_improvement(self):
        """Test patience counter increments on no improvement."""
        from model_garden.training.callbacks import EarlyStoppingCallback

        callback = EarlyStoppingCallback(patience=3)
        callback.best_metric = 0.5
        control = MagicMock()
        control.should_training_stop = False

        # Loss didn't improve
        callback.on_evaluate(
            args=MagicMock(), state=MagicMock(), control=control, metrics={"eval_loss": 0.6}
        )

        assert callback.best_metric == 0.5
        assert callback.patience_counter == 1
        assert control.should_training_stop is False

    def test_on_evaluate_patience_reached(self):
        """Test training stops when patience is reached."""
        from model_garden.training.callbacks import EarlyStoppingCallback

        callback = EarlyStoppingCallback(patience=2)
        callback.best_metric = 0.5
        callback.patience_counter = 1  # Already had 1 bad eval
        control = MagicMock()

        # Another bad eval - should trigger stop
        callback.on_evaluate(
            args=MagicMock(), state=MagicMock(), control=control, metrics={"eval_loss": 0.6}
        )

        assert callback.patience_counter == 2
        assert control.should_training_stop is True
        assert callback.should_stop is True

    def test_on_evaluate_missing_metric(self):
        """Test that missing metric doesn't crash."""
        from model_garden.training.callbacks import EarlyStoppingCallback

        callback = EarlyStoppingCallback(metric="eval_loss")
        control = MagicMock()

        # Metrics don't contain the monitored metric
        callback.on_evaluate(
            args=MagicMock(), state=MagicMock(), control=control, metrics={"other_metric": 0.5}
        )

        # Should not change state
        assert callback.best_metric is None
        assert callback.patience_counter == 0

    def test_on_evaluate_with_threshold(self):
        """Test that threshold is applied to improvements."""
        from model_garden.training.callbacks import EarlyStoppingCallback

        callback = EarlyStoppingCallback(threshold=0.1)
        callback.best_metric = 0.5
        control = MagicMock()

        # Small improvement - not enough to exceed threshold
        callback.on_evaluate(
            args=MagicMock(),
            state=MagicMock(),
            control=control,
            metrics={"eval_loss": 0.45},  # Only 0.05 improvement
        )

        # Should not count as improvement
        assert callback.best_metric == 0.5
        assert callback.patience_counter == 1


class TestDetectModelDtype:
    """Tests for detect_model_dtype function."""

    def test_quantized_4bit_returns_bfloat16(self):
        """Test that 4-bit quantized models return bfloat16."""
        from model_garden.training.mixins import detect_model_dtype

        model = MagicMock()
        dtype = detect_model_dtype(model, load_in_4bit=True, load_in_8bit=False)
        assert dtype == torch.bfloat16

    def test_quantized_8bit_returns_bfloat16(self):
        """Test that 8-bit quantized models return bfloat16."""
        from model_garden.training.mixins import detect_model_dtype

        model = MagicMock()
        dtype = detect_model_dtype(model, load_in_4bit=False, load_in_8bit=True)
        assert dtype == torch.bfloat16

    def test_detect_from_parameter_dtype(self):
        """Test detecting dtype from model parameters."""
        from model_garden.training.mixins import detect_model_dtype

        # Create a real tensor with bfloat16 dtype (isinstance check needs real tensor)
        real_param = torch.zeros(1, dtype=torch.bfloat16)

        model = MagicMock()
        model.parameters.return_value = iter([real_param])

        dtype = detect_model_dtype(model, load_in_4bit=False, load_in_8bit=False)
        assert dtype == torch.bfloat16

    def test_detect_from_parameter_dtype_float16(self):
        """Test detecting float16 dtype from parameters."""
        from model_garden.training.mixins import detect_model_dtype

        # Create a real tensor with float16 dtype
        real_param = torch.zeros(1, dtype=torch.float16)

        model = MagicMock()
        model.parameters.return_value = iter([real_param])

        dtype = detect_model_dtype(model, load_in_4bit=False, load_in_8bit=False)
        assert dtype == torch.float16

    def test_detect_fallback_to_float32(self):
        """Test fallback to float32 when detection fails."""
        from model_garden.training.mixins import detect_model_dtype

        # Model with no accessible parameters or config
        model = MagicMock()
        model.parameters.return_value = iter([])  # No parameters
        model.dtype = None
        model.config = MagicMock()
        model.config.torch_dtype = None
        model.config.model_type = "unknown"

        # Remove fallback attributes - MagicMock auto-creates attributes
        del model.base_model  # Remove base_model attribute

        # Also need to handle model.model.dtype - set to None to prevent match
        model.model.dtype = None

        dtype = detect_model_dtype(model, load_in_4bit=False, load_in_8bit=False)
        assert dtype == torch.float32


class TestGetTrainingPrecisionConfig:
    """Tests for get_training_precision_config function."""

    def test_bfloat16_config(self):
        """Test config for bfloat16 model."""
        from model_garden.training.mixins import get_training_precision_config

        # Use real tensor for proper dtype detection
        real_param = torch.zeros(1, dtype=torch.bfloat16)

        model = MagicMock()
        model.parameters.return_value = iter([real_param])

        config = get_training_precision_config(model, False, False)
        assert config == {"fp16": False, "bf16": True}

    def test_float16_config(self):
        """Test config for float16 model."""
        from model_garden.training.mixins import get_training_precision_config

        # Use real tensor for proper dtype detection
        real_param = torch.zeros(1, dtype=torch.float16)

        model = MagicMock()
        model.parameters.return_value = iter([real_param])

        config = get_training_precision_config(model, False, False)
        assert config == {"fp16": True, "bf16": False}

    def test_quantized_config(self):
        """Test config for quantized model."""
        from model_garden.training.mixins import get_training_precision_config

        model = MagicMock()
        config = get_training_precision_config(model, load_in_4bit=True, load_in_8bit=False)
        assert config == {"fp16": False, "bf16": True}


class TestMemoryMonitorCallback:
    """Tests for MemoryMonitorCallback."""

    def test_on_step_end_logs_at_interval(self):
        """Test that callback logs at correct intervals."""
        from model_garden.training.callbacks import MemoryMonitorCallback

        callback = MemoryMonitorCallback()
        state = MagicMock()

        # Step 10 should log
        state.global_step = 10
        result = callback.on_step_end(args=MagicMock(), state=state, control=MagicMock())
        assert result is None  # Should return None

    def test_on_step_end_skips_non_interval(self):
        """Test that callback skips non-interval steps."""
        from model_garden.training.callbacks import MemoryMonitorCallback

        callback = MemoryMonitorCallback()
        state = MagicMock()

        # Step 5 should not log (not divisible by 10)
        state.global_step = 5
        result = callback.on_step_end(args=MagicMock(), state=state, control=MagicMock())
        assert result is None


class TestClearTrainerInternals:
    """Tests for clear_trainer_internals function."""

    def test_clears_trainer_attributes(self):
        """Test that trainer attributes are cleared."""
        from model_garden.utils.memory import clear_trainer_internals

        trainer = MagicMock()
        trainer.model = MagicMock()
        trainer.optimizer = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.train_dataset = MagicMock()

        clear_trainer_internals(trainer)

        assert trainer.model is None
        assert trainer.optimizer is None
        assert trainer.tokenizer is None
        assert trainer.train_dataset is None

    def test_handles_none_trainer(self):
        """Test that None trainer doesn't crash."""
        from model_garden.utils.memory import clear_trainer_internals

        # Should not raise
        clear_trainer_internals(None)

    def test_handles_missing_attributes(self):
        """Test handling of trainer without all attributes."""
        from model_garden.utils.memory import clear_trainer_internals

        # Create a trainer with only some attributes
        trainer = MagicMock(spec=["model", "optimizer"])
        trainer.model = MagicMock()
        trainer.optimizer = MagicMock()

        # Should not raise
        clear_trainer_internals(trainer)

        assert trainer.model is None
        assert trainer.optimizer is None


class TestCleanupTrainingResources:
    """Tests for cleanup_training_resources function."""

    @pytest.mark.requires_gpu
    def test_cleanup_with_trainer(self):
        """Test cleanup with a trainer object."""
        from model_garden.utils.memory import cleanup_training_resources

        trainer = MagicMock()
        trainer.__class__.__name__ = "SFTTrainer"
        trainer.model = MagicMock()

        # Should not raise
        cleanup_training_resources(trainer)

    def test_cleanup_with_none(self):
        """Test cleanup with None objects."""
        from model_garden.utils.memory import cleanup_training_resources

        # Should not raise
        cleanup_training_resources(None, None, None)

    def test_cleanup_clears_gpu_cache(self):
        """Test that GPU cache is cleared if torch is available."""
        from model_garden.utils.memory import cleanup_training_resources

        # This should work even without GPU - it handles the case gracefully
        cleanup_training_resources()
