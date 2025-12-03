"""Custom SFT trainer with fixes for eval loss computation.

This module provides FixedSFTTrainer which fixes a bug in the TRL SFTTrainer
where eval loss is computed inconsistently with training loss when using
selective loss masking.
"""

from trl.trainer.sft_trainer import SFTTrainer


class FixedSFTTrainer(SFTTrainer):
    """Custom SFTTrainer that fixes the eval loss computation bug.

    The bug: Trainer.prediction_step() doesn't pass num_items_in_batch to compute_loss,
    causing incorrect loss normalization during evaluation when using masked tokens.

    Training path: compute_loss(model, inputs, num_items_in_batch=...) → correct
    Eval path: compute_loss(model, inputs, return_outputs=True) → MISSING num_items_in_batch!

    Problem explained:
    - Training sums tokens across gradient_accumulation_steps batches (~1700 tokens)
      and computes loss = sum / 1700
    - Eval uses a single batch (~425 tokens) with loss = sum / 425
    - Even though both are "per-token averages", they differ due to batch composition
    - This makes train/eval loss comparison unreliable

    Solution:
    Force num_items_in_batch=None for both train and eval to use consistent
    reduction='mean' behavior across all tokens in each batch independently.

    Note:
    This fix ensures train and eval losses are computed using the same method,
    making them directly comparable. The tradeoff is slightly different gradient
    scaling, but in practice this has minimal impact on training quality.

    Example:
        >>> from model_garden.training.sft_trainer import FixedSFTTrainer
        >>> trainer = FixedSFTTrainer(
        ...     model=model,
        ...     args=training_args,
        ...     train_dataset=train_dataset,
        ...     eval_dataset=eval_dataset,
        ... )
        >>> trainer.train()  # Train and eval losses now comparable
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override to disable num_items_in_batch entirely.

        This makes both training and evaluation use reduction='mean' (default behavior),
        ensuring consistent loss computation across both phases.

        Args:
            model: The model to compute loss for
            inputs: The input batch
            return_outputs: Whether to return model outputs along with loss
            num_items_in_batch: Ignored - always set to None for consistency

        Returns:
            Loss tensor, or (loss, outputs) if return_outputs=True
        """
        # Force num_items_in_batch=None for both training and eval
        # This makes both use reduction='mean' (default behavior)
        num_items_in_batch = None

        # Call parent with num_items_in_batch=None
        return super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )


class ConsistentLossSFTTrainer(FixedSFTTrainer):
    """Alias for FixedSFTTrainer with a more descriptive name.

    This class is identical to FixedSFTTrainer but uses a name that better
    describes its purpose: ensuring consistent loss computation between
    training and evaluation.
    """

    pass
