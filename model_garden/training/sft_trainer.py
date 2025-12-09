"""Custom SFT trainer with fixes for eval loss computation.

This module provides FixedSFTTrainer which fixes a bug in the TRL SFTTrainer
where eval loss is computed inconsistently with training loss when using
selective loss masking.
"""

from trl.trainer.sft_trainer import SFTTrainer


class FixedSFTTrainer(SFTTrainer):
    """Custom SFTTrainer that keeps loss scaling consistent.

    HF/TRL recently added token-count-aware loss scaling via ``num_items_in_batch``.
    For models that *ignore* that kwarg (like our Unsloth vision/text setups), the
    Trainer will **skip** dividing by ``gradient_accumulation_steps`` whenever
    ``num_items_in_batch`` is provided, inflating the reported/used train loss by
    exactly that factor. Eval never receives ``num_items_in_batch``, so train vs eval
    losses become incomparable (train is N× larger where N = grad_accum steps).

    We opt out of that path entirely by:
    1) Forcing ``num_items_in_batch`` to ``None`` in ``compute_loss``
    2) Disabling token counting via ``_get_num_items_in_batch``

    This restores the classic behavior: per-batch mean loss with explicit division
    by ``gradient_accumulation_steps`` handled by the Trainer, matching eval.
    """

    def __init__(self, *args, **kwargs):
        """Force classic loss scaling (ignore token-count kwargs)."""
        super().__init__(*args, **kwargs)
        # Disable loss kwargs so Trainer always applies GA scaling
        self.model_accepts_loss_kwargs = False

    def _get_num_items_in_batch(self, batch_samples, device):  # type: ignore[override]
        """Disable token counting so GA scaling remains active."""
        return None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Ignore ``num_items_in_batch`` to keep loss reduction uniform."""
        num_items_in_batch = None
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
