"""Tests for FixedSFTTrainer.

These tests verify that FixedSFTTrainer correctly overrides compute_loss
to ensure consistent loss computation between training and evaluation.
"""

from unittest.mock import MagicMock, patch

import torch
from accelerate import PartialState
from transformers import PreTrainedTokenizerBase

from model_garden.training.sft_trainer import FixedSFTTrainer


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MagicMock()
        self.config._name_or_path = "model_name"
        self.tp_size = 1
        # Add dummy parameters to make it look like a real model if needed
        self.layer = torch.nn.Linear(1, 1)

    def forward(self, *args, **kwargs):
        return MagicMock()


class TestFixedSFTTrainer:
    """Tests for FixedSFTTrainer."""

    def setup_method(self):
        """Initialize accelerate state."""
        PartialState()

    @patch("trl.trainer.sft_trainer.SFTTrainer.compute_loss")
    def test_compute_loss_forces_num_items_none(self, mock_super_compute_loss):
        """Test that compute_loss forces num_items_in_batch to None."""
        # Setup
        tokenizer = MagicMock()
        tokenizer.convert_tokens_to_ids.return_value = 1
        # Mock isinstance check
        tokenizer.__class__ = PreTrainedTokenizerBase

        train_dataset = MagicMock()
        train_dataset.__iter__.return_value = iter([{"input_ids": [1]}])

        args = MagicMock()
        args.assistant_only_loss = False
        args.packing = False
        args.padding_free = False
        args.loss_type = "nll"
        args.eval_strategy = "no"
        args.seed = 42
        args.deepspeed_plugin = None
        args.get_process_log_level.return_value = 20
        args.use_liger_kernel = False
        args.max_steps = -1
        args.num_train_epochs = 1
        args.label_smoothing_factor = 0
        args.fsdp_config = {"xla": False, "xla_fsdp_v2": False}
        args.fsdp = []
        args.device = torch.device("cpu")

        model = SimpleModel()

        trainer = FixedSFTTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )

        inputs = {"input_ids": torch.tensor([[1, 2, 3]])}

        # Call compute_loss with a value for num_items_in_batch
        trainer.compute_loss(model, inputs, num_items_in_batch=10)

        # Verify super().compute_loss was called with num_items_in_batch=None
        mock_super_compute_loss.assert_called_once()
        call_args = mock_super_compute_loss.call_args
        assert call_args.kwargs["num_items_in_batch"] is None
        assert call_args.kwargs["return_outputs"] is False

    @patch("trl.trainer.sft_trainer.SFTTrainer.compute_loss")
    def test_compute_loss_passes_return_outputs(self, mock_super_compute_loss):
        """Test that compute_loss passes return_outputs correctly."""
        # Setup
        tokenizer = MagicMock()
        tokenizer.convert_tokens_to_ids.return_value = 1
        # Mock isinstance check
        tokenizer.__class__ = PreTrainedTokenizerBase

        train_dataset = MagicMock()
        train_dataset.__iter__.return_value = iter([{"input_ids": [1]}])

        args = MagicMock()
        args.assistant_only_loss = False
        args.packing = False
        args.padding_free = False
        args.loss_type = "nll"
        args.eval_strategy = "no"
        args.seed = 42
        args.deepspeed_plugin = None
        args.get_process_log_level.return_value = 20
        args.use_liger_kernel = False
        args.max_steps = -1
        args.num_train_epochs = 1
        args.label_smoothing_factor = 0
        args.fsdp_config = {"xla": False, "xla_fsdp_v2": False}
        args.fsdp = []
        args.device = torch.device("cpu")

        model = SimpleModel()

        trainer = FixedSFTTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )

        inputs = {"input_ids": torch.tensor([[1, 2, 3]])}

        # Call compute_loss with return_outputs=True
        trainer.compute_loss(model, inputs, return_outputs=True)

        # Verify super().compute_loss was called with return_outputs=True
        mock_super_compute_loss.assert_called_once()
        call_args = mock_super_compute_loss.call_args
        assert call_args.kwargs["return_outputs"] is True
