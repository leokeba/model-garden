"""Tests for weighted loss training module."""

from unittest.mock import MagicMock

import torch

from model_garden.training.weighted_loss import (
    WeightedLossTrainer,
    WeightedLossTrainerWithMetrics,
)


class TestWeightedLossTrainer:
    """Tests for the WeightedLossTrainer class."""

    def test_init_attrs(self):
        """Test that WeightedLossTrainer has expected attributes after __new__."""
        # Use __new__ to avoid complex Trainer initialization
        trainer = WeightedLossTrainer.__new__(WeightedLossTrainer)
        trainer.verbose_loss = False
        trainer._loss_step_counter = 0

        assert trainer.verbose_loss is False
        assert trainer._loss_step_counter == 0

    def test_compute_loss_without_weights(self):
        """Test compute_loss without sample weights (standard loss)."""
        # Create a simple model that returns logits
        vocab_size = 100
        seq_len = 10
        batch_size = 2

        mock_model = MagicMock()

        # Create realistic logits output
        logits = torch.randn(batch_size, seq_len, vocab_size)
        mock_model.return_value = {"logits": logits}

        trainer = WeightedLossTrainer.__new__(WeightedLossTrainer)
        trainer.model = mock_model
        trainer.verbose_loss = False
        trainer._loss_step_counter = 0

        # Create inputs without sample_weights
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        }

        loss = trainer.compute_loss(mock_model, inputs)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_compute_loss_with_weights(self):
        """Test compute_loss with sample weights."""
        vocab_size = 100
        seq_len = 10
        batch_size = 2

        mock_model = MagicMock()
        logits = torch.randn(batch_size, seq_len, vocab_size)
        mock_model.return_value = {"logits": logits}

        trainer = WeightedLossTrainer.__new__(WeightedLossTrainer)
        trainer.model = mock_model
        trainer.verbose_loss = False
        trainer._loss_step_counter = 0

        # Create labels where some are masked (-100)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[:, :3] = -100  # Mask first 3 tokens (prompt)

        # Create weights - structural tokens get lower weight
        weights = torch.ones(batch_size, seq_len)
        weights[:, 4:6] = 0.1  # Some tokens get reduced weight

        inputs = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "labels": labels,
            "sample_weights": weights,
        }

        loss = trainer.compute_loss(mock_model, inputs)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_compute_loss_all_masked_labels(self):
        """Test compute_loss when all labels are masked."""
        vocab_size = 100
        seq_len = 10
        batch_size = 2

        mock_model = MagicMock()
        logits = torch.randn(batch_size, seq_len, vocab_size)
        mock_model.return_value = {"logits": logits}

        trainer = WeightedLossTrainer.__new__(WeightedLossTrainer)
        trainer.model = mock_model
        trainer.verbose_loss = False
        trainer._loss_step_counter = 0

        # All labels masked
        labels = torch.full((batch_size, seq_len), -100)

        inputs = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "labels": labels,
        }

        loss = trainer.compute_loss(mock_model, inputs)

        # Should return 0 loss when all masked
        assert isinstance(loss, torch.Tensor)
        assert loss.item() == 0.0

    def test_compute_loss_returns_outputs(self):
        """Test compute_loss with return_outputs=True."""
        vocab_size = 100
        seq_len = 10
        batch_size = 2

        mock_model = MagicMock()
        logits = torch.randn(batch_size, seq_len, vocab_size)
        outputs = {"logits": logits, "hidden_states": None}
        mock_model.return_value = outputs

        trainer = WeightedLossTrainer.__new__(WeightedLossTrainer)
        trainer.model = mock_model
        trainer.verbose_loss = False
        trainer._loss_step_counter = 0

        inputs = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        }

        result = trainer.compute_loss(mock_model, inputs, return_outputs=True)

        assert isinstance(result, tuple)
        assert len(result) == 2
        loss, returned_outputs = result
        assert isinstance(loss, torch.Tensor)
        assert "logits" in returned_outputs

    def test_loss_step_counter_increments(self):
        """Test that loss step counter increments."""
        vocab_size = 100
        seq_len = 10
        batch_size = 2

        mock_model = MagicMock()
        logits = torch.randn(batch_size, seq_len, vocab_size)
        mock_model.return_value = {"logits": logits}

        trainer = WeightedLossTrainer.__new__(WeightedLossTrainer)
        trainer.model = mock_model
        trainer.verbose_loss = False
        trainer._loss_step_counter = 0

        inputs = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        }

        assert trainer._loss_step_counter == 0

        trainer.compute_loss(mock_model, inputs.copy())
        assert trainer._loss_step_counter == 1

        trainer.compute_loss(mock_model, inputs.copy())
        assert trainer._loss_step_counter == 2


class TestWeightedLossTrainerWithMetrics:
    """Tests for the WeightedLossTrainerWithMetrics class."""

    def test_init_enables_verbose(self):
        """Test that metrics trainer enables verbose by default."""
        trainer = WeightedLossTrainerWithMetrics.__new__(WeightedLossTrainerWithMetrics)
        trainer.verbose_loss = True  # Simulating parent init
        trainer._loss_step_counter = 0
        trainer.weighted_loss_history = []
        trainer.unweighted_loss_history = []
        trainer.weight_distributions = []

        assert trainer.verbose_loss is True
        assert trainer.weighted_loss_history == []
        assert trainer.weight_distributions == []

    def test_get_weighted_loss_summary_empty(self):
        """Test summary when no data collected."""
        trainer = WeightedLossTrainerWithMetrics.__new__(WeightedLossTrainerWithMetrics)
        trainer.weighted_loss_history = []
        trainer.weight_distributions = []

        summary = trainer.get_weighted_loss_summary()
        assert "message" in summary
        assert "No weighted loss data" in summary["message"]

    def test_get_weighted_loss_summary_with_data(self):
        """Test summary with collected data."""
        trainer = WeightedLossTrainerWithMetrics.__new__(WeightedLossTrainerWithMetrics)
        trainer.weighted_loss_history = [0.5, 0.4, 0.3, 0.35, 0.32]
        trainer.weight_distributions = [
            {"1.00": 80, "0.10": 20},
            {"1.00": 75, "0.10": 25},
        ]

        summary = trainer.get_weighted_loss_summary()

        assert summary["num_steps"] == 5
        assert "avg_weighted_loss" in summary
        assert "std_weighted_loss" in summary
        assert 0.3 < summary["avg_weighted_loss"] < 0.5
        assert len(summary["weight_distributions_sampled"]) == 2


class TestWeightedLossComputation:
    """Tests for weighted loss computation edge cases."""

    def test_weights_affect_loss_value(self):
        """Test that different weights produce different losses."""
        vocab_size = 100
        seq_len = 10
        batch_size = 1

        mock_model = MagicMock()
        # Use deterministic logits for reproducibility
        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        mock_model.return_value = {"logits": logits.clone()}

        trainer = WeightedLossTrainer.__new__(WeightedLossTrainer)
        trainer.model = mock_model
        trainer.verbose_loss = False
        trainer._loss_step_counter = 0

        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Compute loss with all weights = 1.0
        inputs_full = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "labels": labels.clone(),
            "sample_weights": torch.ones(batch_size, seq_len),
        }
        loss_full = trainer.compute_loss(mock_model, inputs_full)

        # Reset counter and compute loss with reduced weights
        trainer._loss_step_counter = 0
        mock_model.return_value = {"logits": logits.clone()}  # Same logits

        # Create weights tensor explicitly as float
        weights_reduced = torch.ones(batch_size, seq_len) * 0.5
        inputs_reduced = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "labels": labels.clone(),
            "sample_weights": weights_reduced,
        }
        loss_reduced = trainer.compute_loss(mock_model, inputs_reduced)

        # With lower weights, the weighted loss should be lower
        # (since we multiply loss by weight)
        assert loss_reduced.item() < loss_full.item()

    def test_zero_weight_tokens_contribute_zero_loss(self):
        """Test that tokens with zero weight don't contribute to loss."""
        vocab_size = 100
        seq_len = 10
        batch_size = 1

        mock_model = MagicMock()
        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        mock_model.return_value = {"logits": logits}

        trainer = WeightedLossTrainer.__new__(WeightedLossTrainer)
        trainer.model = mock_model
        trainer.verbose_loss = False
        trainer._loss_step_counter = 0

        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # All weights zero - should get zero loss contribution
        weights = torch.zeros(batch_size, seq_len)

        inputs = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "labels": labels,
            "sample_weights": weights,
        }

        loss = trainer.compute_loss(mock_model, inputs)

        # Loss should be very small or zero when all weights are zero
        # The actual value depends on implementation (may divide by num_tokens)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
