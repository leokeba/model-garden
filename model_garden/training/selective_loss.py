"""Selective loss computation for structured output training.

This module provides data collators that mask structural JSON tokens during training,
allowing the model to focus on learning semantic content rather than trivial structure.

Architecture:
- SelectiveLossMixin: Backend-agnostic mixin with all masking logic
- SelectiveLossCollator: Generic wrapper that works with any base collator
- SelectiveLossUnslothCollator: Unsloth-optimized collator (when Unsloth is available)

Key Features:
- Masks JSON structural characters ({, }, [, ], :, ,) and whitespace
- Optional masking of schema keys and null keyword
- Works with any backend (Unsloth, Transformers, etc.)
- Supports multiple masking strategies:
  * epoch_based: Enable masking after a certain epoch threshold
  * alternating: Cycle between masking ON/OFF to learn structure and semantics throughout training
  * weighted: Apply soft masking with reduced loss weights for structural tokens (always active)

Usage:
    from model_garden.training.selective_loss import create_selective_loss_collator

    # Alternating strategy (recommended for balanced learning)
    collator = create_selective_loss_collator(
        model=model,
        processor=processor,
        mask_level="conservative",
        masking_strategy="alternating",
        mask_every_n_steps=100,  # Full cycle every 100 steps
        mask_for_n_steps=50,     # Mask ON for 50 steps, OFF for 50 steps
        verbose=True
    )

    # Epoch-based strategy (good for initial exploration)
    collator = create_selective_loss_collator(
        model=model,
        processor=processor,
        mask_level="aggressive",
        masking_strategy="epoch_based",
        masking_start_epoch=0.5,  # Start masking halfway through first epoch
        verbose=True
    )

    # Weighted strategy (soft constraints, experimental)
    collator = create_selective_loss_collator(
        model=model,
        processor=processor,
        mask_level="aggressive",
        masking_strategy="weighted",
        structural_weight=0.1,  # Structural tokens get 10% loss weight
        verbose=True
    )
"""

from __future__ import annotations

import re
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

import torch

from model_garden.utils.console import console

if TYPE_CHECKING:
    pass


# =============================================================================
# Protocol for tokenizers (allows duck typing)
# =============================================================================


class TokenizerProtocol(Protocol):
    """Protocol for tokenizers that can be used with selective loss."""

    def decode(self, token_ids: list[int] | torch.Tensor, skip_special_tokens: bool = False) -> str:
        """Decode token IDs to text."""
        ...


class ProcessorProtocol(Protocol):
    """Protocol for processors that contain a tokenizer."""

    tokenizer: TokenizerProtocol


# =============================================================================
# Selective Loss Mixin (Backend-Agnostic)
# =============================================================================


class SelectiveLossMixin:
    """Mixin providing selective loss masking logic for any data collator.

    This mixin contains all the backend-agnostic logic for masking structural
    JSON tokens during training. It can be combined with any base collator.

    The mixin expects the following attributes to be set by the subclass:
    - processor: A processor with a tokenizer attribute
    - mask_structural: Whether to mask structural tokens
    - mask_keys: Whether to mask schema keys
    - schema_keys: Set of schema key names to mask
    - mask_keywords: Whether to mask JSON keywords (null)
    - masking_strategy: The masking strategy ("epoch_based", "alternating", "weighted")
    - masking_start_epoch: Epoch at which to start masking
    - mask_every_n_steps: Cycle length for alternating strategy
    - mask_for_n_steps: Steps with masking ON per cycle
    - structural_weight: Weight for structural tokens in weighted strategy
    - verbose: Whether to print debug info

    Class Constants:
    - STRUCTURAL_CHARS: Characters considered structural
    - JSON_KEYWORDS: JSON keywords to potentially mask
    - JSON_TYPE_KEYWORDS: JSON type keywords to mask
    - SCHEMA_KEYWORDS: JSON Schema keywords to mask
    """

    # JSON structural characters to mask (NOT including < and > which are for XML/HTML tags)
    # Quotes are structural in JSON - they delimit strings but carry no semantic meaning
    STRUCTURAL_CHARS: set[str] = {"{", "}", "[", "]", ":", ",", '"', " ", "\n", "\t", "\r"}

    # Only null is truly structural - true/false can be semantic content
    JSON_KEYWORDS: set[str] = {"null"}

    # JSON type keywords (these appear as values but are not semantic for form extraction)
    JSON_TYPE_KEYWORDS: set[str] = {
        "object",
        "array",
        "string",
        "number",
        "integer",
        "boolean",
        "null",
    }

    # JSON Schema keywords that should be masked (not semantic content for form extraction)
    SCHEMA_KEYWORDS: set[str] = {
        # Schema structure
        "$schema",
        "$id",
        "$ref",
        "$defs",
        "definitions",
        # Type keywords
        "type",
        "properties",
        "items",
        "required",
        "additionalProperties",
        "enum",
        "const",
        "anyOf",
        "oneOf",
        "allOf",
        "not",
        # Validation keywords
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "format",
        # Metadata (keep 'title' as it might be semantic)
        "description",
        "default",
        "examples",
    }

    # Attributes expected from the subclass (for type checking)
    processor: Any
    mask_structural: bool
    mask_keys: bool
    schema_keys: set[str]
    mask_keywords: bool
    masking_strategy: str
    masking_start_epoch: float
    mask_every_n_steps: int
    mask_for_n_steps: int
    structural_weight: float
    verbose: bool

    def _init_selective_loss(
        self,
        mask_structural_tokens: bool = True,
        mask_schema_keys: bool = False,
        schema_keys: list[str] | None = None,
        mask_json_keywords: bool = False,
        masking_start_epoch: float = 0.0,
        masking_strategy: str = "epoch_based",
        mask_every_n_steps: int = 100,
        mask_for_n_steps: int = 50,
        structural_weight: float = 0.1,
        verbose: bool = False,
    ) -> None:
        """Initialize selective loss configuration.

        Call this from the subclass's __init__ method.
        """
        self.mask_structural = mask_structural_tokens
        self.mask_keys = mask_schema_keys
        self.schema_keys = set(schema_keys) if schema_keys else set()
        self.mask_keywords = mask_json_keywords

        # Masking strategy configuration
        self.masking_strategy = masking_strategy
        self.masking_start_epoch = masking_start_epoch
        self.mask_every_n_steps = mask_every_n_steps
        self.mask_for_n_steps = mask_for_n_steps
        self.structural_weight = structural_weight
        self.verbose = verbose

        # Validate strategy
        if self.masking_strategy not in ["epoch_based", "alternating", "weighted"]:
            raise ValueError(
                f"Invalid masking_strategy: {masking_strategy}. "
                f"Choose from: 'epoch_based', 'alternating', 'weighted'"
            )

        # Validate weighted masking parameters
        if self.masking_strategy == "weighted" and not (0.0 <= self.structural_weight <= 1.0):
            raise ValueError(
                f"structural_weight must be between 0.0 and 1.0, got {structural_weight}"
            )

        # Statistics for debugging
        self.total_tokens = 0
        self.masked_tokens = 0
        self.batch_count = 0
        self.current_step = 0

        # For epoch-based masking, we need to track the trainer state
        self._trainer: weakref.ref | None = None
        self._masking_enabled = False

        if self.verbose:
            self._print_init_info()

    def _print_init_info(self) -> None:
        """Print initialization information."""
        console.print("[cyan]Initialized SelectiveLossCollator[/cyan]")
        console.print(f"  Mask structural tokens: {self.mask_structural}")
        console.print(f"  Mask schema keys: {self.mask_keys}")
        if self.schema_keys:
            console.print(
                f"  Schema keys to mask ({len(self.schema_keys)}): {list(self.schema_keys)[:10]}"
            )

        console.print(f"  Masking strategy: [yellow]{self.masking_strategy}[/yellow]")

        if self.masking_strategy == "epoch_based":
            if self.masking_start_epoch > 0.0:
                console.print(
                    f"  [yellow]Masking delayed until epoch {self.masking_start_epoch}[/yellow]"
                )
                console.print(
                    "  [yellow]Model will learn JSON structure first, then apply selective loss[/yellow]"
                )
        elif self.masking_strategy == "alternating":
            console.print("  [yellow]Alternating masking pattern:[/yellow]")
            console.print(f"  [yellow]  - Mask ON for {self.mask_for_n_steps} steps[/yellow]")
            console.print(
                f"  [yellow]  - Mask OFF for {self.mask_every_n_steps - self.mask_for_n_steps} steps[/yellow]"
            )
            console.print(
                f"  [yellow]  - Cycle repeats every {self.mask_every_n_steps} steps[/yellow]"
            )
            console.print(
                "  [yellow]This ensures learning of both structure and semantics throughout training[/yellow]"
            )
        elif self.masking_strategy == "weighted":
            console.print("  [yellow]Weighted masking enabled:[/yellow]")
            console.print(
                f"  [yellow]  - Structural token weight: {self.structural_weight:.2f}[/yellow]"
            )
            console.print("  [yellow]  - Semantic token weight: 1.00[/yellow]")
            console.print(
                "  [yellow]This applies soft constraints - structural tokens still contribute to loss[/yellow]"
            )

    def _get_tokenizer(self) -> TokenizerProtocol:
        """Get the tokenizer from the processor.

        Override this method if your processor has a different structure.
        """
        if hasattr(self.processor, "tokenizer"):
            return self.processor.tokenizer
        return self.processor  # Assume processor is the tokenizer

    def _apply_selective_loss_to_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Apply selective loss masking to a batch.

        This is the main entry point for selective loss logic. Call this
        from the subclass's __call__ method after the base collator has
        processed the batch.

        Args:
            batch: Batch dictionary with 'labels' key

        Returns:
            Modified batch with selective loss masking applied
        """
        # Show initial batch info only once for debugging
        if self.current_step == 0 and self.verbose:
            console.print("[cyan]SelectiveLossCollator processing batch[/cyan]")

        # Show batch shape info only on first call
        if self.current_step == 0 and self.verbose:
            console.print("[cyan]After base collator:[/cyan]")
            if "labels" in batch:
                labels_sample = batch["labels"][0]
                total = len(labels_sample)
                masked = (labels_sample == -100).sum().item()
                console.print(
                    f"[cyan]  Labels: {total} tokens, {masked} masked, {total - masked} unmasked[/cyan]"
                )
            if "input_ids" in batch:
                console.print(f"[cyan]  Input IDs shape: {batch['input_ids'].shape}[/cyan]")

        # CRITICAL: Always check labels to diagnose NaN eval_loss
        self._check_for_all_masked_tokens(batch)

        # Only increment step counter during training (not evaluation)
        is_training = torch.is_grad_enabled()
        if is_training:
            self.current_step += 1

        # Determine if masking should be enabled based on the chosen strategy
        masking_should_be_enabled = self._should_enable_masking()

        # For weighted masking strategy, we always apply weights (no on/off switching)
        if self.masking_strategy == "weighted" and self.mask_structural:
            if "labels" in batch:
                batch = self._apply_weighted_masking(batch)
            return batch

        # Check if we should apply masking yet (for epoch_based and alternating)
        if not self.mask_structural:
            return batch  # No masking, use standard behavior

        if not masking_should_be_enabled:
            self._log_masking_disabled(is_training)
            return batch  # No masking yet

        # Log when masking starts
        self._log_masking_enabled(masking_should_be_enabled, is_training)

        # Apply selective loss masking
        if "labels" in batch:
            batch = self._apply_selective_masking_to_batch(batch, is_training)

        return batch

    def _check_for_all_masked_tokens(self, batch: dict[str, Any]) -> None:
        """Check if all tokens are masked and warn."""
        if "labels" not in batch or len(batch["labels"]) == 0:
            return

        first_labels = batch["labels"][0]
        total_tokens = len(first_labels)
        masked_tokens = (first_labels == -100).sum().item()
        unmasked_tokens = total_tokens - masked_tokens

        if unmasked_tokens == 0:
            console.print("[red]❌ CRITICAL: ALL tokens masked in batch![/red]")
            console.print(f"[red]   Total tokens: {total_tokens}, Step: {self.current_step}[/red]")
            console.print("[red]   This will cause NaN loss![/red]")

            # Check for sequence truncation
            if "input_ids" in batch and len(batch["input_ids"]) > 0:
                try:
                    input_ids = batch["input_ids"][0]
                    tokenizer = self._get_tokenizer()
                    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
                    user_count = full_text.count("<|im_start|>user")
                    assistant_count = full_text.count("<|im_start|>assistant")

                    if assistant_count == 0 and user_count > 0:
                        console.print("[red]⚠️  SEQUENCE TRUNCATION DETECTED![/red]")
                        console.print(
                            "[red]   Assistant response was CUT OFF by max_seq_length![/red]"
                        )
                        console.print(
                            f"[red]   Current sequence length: {len(input_ids)} tokens[/red]"
                        )
                        console.print(
                            "[red]   → SOLUTION: Increase max_seq_length to 8192+ for vision models[/red]"
                        )
                    else:
                        console.print(
                            f"[red]   Markers found: user={user_count}, assistant={assistant_count}[/red]"
                        )
                        console.print("[red]   Check train_on_responses_only configuration[/red]")
                except Exception as e:
                    console.print(f"[red]   Could not analyze sequence: {e}[/red]")

    def _log_masking_disabled(self, is_training: bool) -> None:
        """Log when masking is disabled."""
        if not (self.verbose and self.current_step % 10 == 0 and is_training):
            return

        if self.masking_strategy == "epoch_based":
            current_epoch = self._get_current_epoch()
            console.print(
                f"[dim]Epoch {current_epoch:.2f}/{self.masking_start_epoch}: "
                f"Learning structure (masking disabled)[/dim]"
            )
        elif self.masking_strategy == "alternating":
            cycle_pos = self.current_step % self.mask_every_n_steps
            steps_until_on = self.mask_every_n_steps - cycle_pos
            console.print(
                f"[dim]Step {self.current_step}: Masking OFF - Learning structure "
                f"({steps_until_on} steps until masking ON)[/dim]"
            )

    def _log_masking_enabled(self, masking_should_be_enabled: bool, is_training: bool) -> None:
        """Log when masking is enabled."""
        if self.masking_strategy == "epoch_based":
            if (
                not self._masking_enabled
                and masking_should_be_enabled
                and self.verbose
                and is_training
            ):
                current_epoch = self._get_current_epoch()
                console.print(
                    f"[green]✓ Epoch {current_epoch:.2f}: Masking activated! "
                    f"(after {self.masking_start_epoch} epochs of structure learning)[/green]"
                )
                self._masking_enabled = True
        elif self.masking_strategy == "alternating":
            if self.verbose and is_training:
                cycle_pos = self.current_step % self.mask_every_n_steps
                if cycle_pos == 0 and self.current_step > 0:
                    console.print(
                        f"[green]🔄 Step {self.current_step}: Masking ON for next "
                        f"{self.mask_for_n_steps} steps[/green]"
                    )
                elif cycle_pos == self.mask_for_n_steps and self.current_step > 0:
                    off_steps = self.mask_every_n_steps - self.mask_for_n_steps
                    console.print(
                        f"[yellow]🔄 Step {self.current_step}: Masking OFF for next "
                        f"{off_steps} steps[/yellow]"
                    )

    def _apply_selective_masking_to_batch(
        self, batch: dict[str, Any], is_training: bool
    ) -> dict[str, Any]:
        """Apply selective masking to the labels in a batch."""
        # MEMORY FIX: Only clone for statistics if verbose, and detach immediately
        original_masked_count = 0
        if self.verbose and is_training:
            original_masked_count = (batch["labels"] == -100).sum().item()

        # DEBUG: On first masked batch, check structure
        if is_training and self.batch_count == 0 and self.verbose:
            first_seq = batch["labels"][0]
            total_tokens = len(first_seq)
            prompt_masked = (first_seq == -100).sum().item()
            console.print("[cyan]🔍 Batch structure check (first sequence):[/cyan]")
            console.print(f"   Total tokens: {total_tokens}")
            console.print(
                f"   Prompt tokens (masked to -100): {prompt_masked} "
                f"({prompt_masked / total_tokens * 100:.1f}%)"
            )
            console.print(
                f"   Assistant tokens (not -100): {total_tokens - prompt_masked} "
                f"({(total_tokens - prompt_masked) / total_tokens * 100:.1f}%)"
            )

        if is_training:
            self.batch_count += 1

        # Store original for logging only if needed
        original_labels = None
        should_print_sample = self.verbose and is_training and self.batch_count % 10 == 0
        if should_print_sample:
            original_labels = batch["labels"][0].clone().detach()

        batch["labels"] = self._apply_selective_masking(
            batch["labels"], batch.get("input_ids", None)
        )

        # Update statistics
        if self.verbose and is_training:
            newly_masked_count = (batch["labels"] == -100).sum().item() - original_masked_count
            assistant_tokens_count = batch["labels"].numel() - original_masked_count
            self.total_tokens += assistant_tokens_count
            self.masked_tokens += newly_masked_count

            if should_print_sample:
                mask_pct = (
                    (self.masked_tokens / self.total_tokens) * 100 if self.total_tokens > 0 else 0
                )
                current_epoch = self._get_current_epoch()
                progress_info = f"Epoch {current_epoch:.2f}"

                console.print(
                    f"[dim]Batch {self.batch_count}: Masked {mask_pct:.1f}% of tokens "
                    f"({self.masked_tokens}/{self.total_tokens}) [{progress_info}][/dim]"
                )

                if original_labels is not None:
                    self._print_unmasked_sample(batch["labels"][0], original_labels)
                    del original_labels

        return batch

    def _should_enable_masking(self) -> bool:
        """Determine if masking should be enabled based on current training progress."""
        if self.masking_strategy == "epoch_based":
            current_epoch = self._get_current_epoch()
            return current_epoch >= self.masking_start_epoch
        elif self.masking_strategy == "alternating":
            cycle_position = self.current_step % self.mask_every_n_steps
            return cycle_position < self.mask_for_n_steps
        return False

    def _get_current_epoch(self) -> float:
        """Get the current epoch from the trainer state."""
        # Primary method: get trainer from our stored reference
        trainer = self._get_trainer()
        if trainer is not None and hasattr(trainer, "state") and trainer.state is not None:
            return trainer.state.epoch

        # Fallback: try to access trainer through transformers
        try:
            import transformers

            for attr_name in ["_current_trainer", "current_trainer", "_trainer"]:
                if hasattr(transformers, attr_name):
                    trainer = getattr(transformers, attr_name)
                    if (
                        trainer is not None
                        and hasattr(trainer, "state")
                        and trainer.state is not None
                    ):
                        return trainer.state.epoch

            if hasattr(transformers, "trainer"):
                trainer_module = transformers.trainer
                for attr_name in ["_current_trainer", "current_trainer", "_trainer"]:
                    if hasattr(trainer_module, attr_name):
                        trainer = getattr(trainer_module, attr_name)
                        if (
                            trainer is not None
                            and hasattr(trainer, "state")
                            and trainer.state is not None
                        ):
                            return trainer.state.epoch
        except Exception:
            pass

        # Final fallback: estimate from step count
        if self.current_step > 0:
            estimated_epoch = self.current_step / 100.0
            if self.verbose and self.current_step % 50 == 0:
                console.print(
                    f"[yellow]Warning: Using estimated epoch {estimated_epoch:.2f} "
                    f"(trainer state unavailable)[/yellow]"
                )
            return estimated_epoch

        return 0.0

    def set_trainer(self, trainer: Any) -> None:
        """Set the trainer reference for epoch-based masking."""
        self._trainer = weakref.ref(trainer) if trainer is not None else None

        if self.verbose and self.masking_start_epoch > 0.0:
            console.print(
                f"[cyan]✓ Trainer set for epoch-based masking "
                f"(start: {self.masking_start_epoch})[/cyan]"
            )

    def _get_trainer(self) -> Any | None:
        """Get the trainer reference if available."""
        if self._trainer is not None:
            return self._trainer()
        return None

    def _apply_selective_masking(
        self, labels: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Mask structural tokens in labels tensor."""
        for i in range(labels.size(0)):
            label_tokens = labels[i]
            mask_indices = self._find_structural_indices(label_tokens)
            if mask_indices:
                labels[i, mask_indices] = -100
        return labels

    def _apply_weighted_masking(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Apply weighted masking to batch - creates weight tensor instead of binary mask."""
        labels = batch["labels"]
        weights = torch.ones_like(labels, dtype=torch.float32)

        total_structural = 0
        total_semantic = 0

        for i in range(labels.size(0)):
            label_tokens = labels[i]
            structural_indices = self._find_structural_indices(label_tokens)

            if structural_indices:
                weights[i, structural_indices] = self.structural_weight
                total_structural += len(structural_indices)

            valid_mask = label_tokens != -100
            total_semantic += valid_mask.sum().item() - len(structural_indices)

        batch["sample_weights"] = weights

        if self.verbose and torch.is_grad_enabled():
            self.total_tokens += total_semantic + total_structural
            self.masked_tokens += total_structural

        return batch

    def _find_structural_indices(self, token_ids: torch.Tensor) -> list[int]:
        """Identify which token positions are structural (not semantic)."""
        valid_mask = token_ids != -100
        if not valid_mask.any():
            return []

        valid_tokens = token_ids[valid_mask]
        tokenizer = self._get_tokenizer()

        try:
            full_text = tokenizer.decode(valid_tokens, skip_special_tokens=False)
        except Exception as e:
            if self.verbose:
                console.print(f"[yellow]Warning: Failed to decode tokens: {e}[/yellow]")
            return []

        token_indices_to_mask: set[int] = set()

        # First pass: mask individual structural tokens
        for i, token_id in enumerate(valid_tokens):
            try:
                token_text = tokenizer.decode([int(token_id.item())], skip_special_tokens=False)
            except Exception:
                continue

            if self._is_structural_token(token_text, full_text, check_schema_keys=False):
                original_idx = int(torch.where(valid_mask)[0][i].item())
                token_indices_to_mask.add(original_idx)

        # Second pass: mask schema keys using sliding window
        if self.mask_keys and self.schema_keys:
            token_indices_to_mask.update(self._find_schema_key_spans(valid_tokens, valid_mask))

        return list(token_indices_to_mask)

    def _find_schema_key_spans(
        self, valid_tokens: torch.Tensor, valid_mask: torch.Tensor
    ) -> set[int]:
        """Find token spans that correspond to schema keys using sliding window."""
        indices_to_mask: set[int] = set()
        tokenizer = self._get_tokenizer()

        for key in self.schema_keys:
            for window_size in range(1, min(11, len(valid_tokens) + 1)):
                for start_idx in range(len(valid_tokens) - window_size + 1):
                    window_tokens = valid_tokens[start_idx : start_idx + window_size]
                    try:
                        window_text = tokenizer.decode(
                            window_tokens, skip_special_tokens=False
                        ).strip()
                    except Exception:
                        continue

                    if window_text == key or window_text == f'"{key}"' or window_text == f"'{key}'":
                        for i in range(start_idx, start_idx + window_size):
                            original_idx = int(torch.where(valid_mask)[0][i].item())
                            indices_to_mask.add(original_idx)

        return indices_to_mask

    def _is_structural_token(
        self, token_text: str, full_context: str, check_schema_keys: bool = True
    ) -> bool:
        """Determine if a token is structural (should be masked)."""
        stripped = token_text.strip()

        # 1. Mask pure whitespace tokens
        if not stripped:
            return True

        # 2. Mask pure structural character tokens
        if all(c in self.STRUCTURAL_CHARS for c in token_text):
            return True

        # 3. Mask JSON Schema keywords
        for keyword in self.SCHEMA_KEYWORDS:
            if re.search(r"\b" + re.escape(keyword) + r"\b", stripped):
                return True

        # 4. Mask JSON type keywords
        for type_keyword in self.JSON_TYPE_KEYWORDS:
            if stripped.lower() == type_keyword:
                return True

        # 5. Mask JSON null keyword
        if self.mask_keywords and stripped == "null":
            return True

        # 6. Mask schema field names if enabled
        if check_schema_keys and self.mask_keys and self.schema_keys:
            for key in self.schema_keys:
                if stripped == key or stripped == f'"{key}"':
                    return True

        return False

    def get_masking_stats(self) -> dict[str, Any]:
        """Get statistics about token masking."""
        if self.total_tokens == 0:
            return {"total_tokens": 0, "masked_tokens": 0, "mask_percentage": 0.0, "batch_count": 0}

        return {
            "total_tokens": self.total_tokens,
            "masked_tokens": self.masked_tokens,
            "mask_percentage": (self.masked_tokens / self.total_tokens) * 100,
            "batch_count": self.batch_count,
        }

    def _print_unmasked_sample(
        self, masked_labels: torch.Tensor, original_labels: torch.Tensor
    ) -> None:
        """Print a sample of the unmasked (semantic) content being learned."""
        try:
            unmasked_after = masked_labels != -100
            unmasked_before = original_labels != -100

            if not unmasked_after.any():
                console.print("[dim]  └─ No unmasked tokens in this example[/dim]")
                return

            valid_original_count = unmasked_before.sum().item()
            if valid_original_count == 0:
                console.print("[dim]  └─ No valid tokens in original (all prompt)[/dim]")
                return

            tokenizer = self._get_tokenizer()

            # DEBUG: Show structure on first verbose batch
            if self.batch_count == 10:
                console.print(
                    "[yellow]  🔍 DEBUG: Checking original labels (what base collator gave us):[/yellow]"
                )
                total_tokens = len(original_labels)
                original_masked = (original_labels == -100).sum().item()
                original_unmasked = total_tokens - original_masked
                console.print(f"[yellow]     Total tokens in sequence: {total_tokens}[/yellow]")
                console.print(
                    f"[yellow]     Masked (prompt): {original_masked} "
                    f"({original_masked / total_tokens * 100:.1f}%)[/yellow]"
                )
                console.print(
                    f"[yellow]     Unmasked (assistant): {original_unmasked} "
                    f"({original_unmasked / total_tokens * 100:.1f}%)[/yellow]"
                )

                if unmasked_before.any():
                    original_valid_tokens = original_labels[unmasked_before]
                    original_decoded = tokenizer.decode(
                        original_valid_tokens[:100], skip_special_tokens=False
                    )
                    console.print(
                        f"[yellow]     First 100 unmasked tokens: {repr(original_decoded[:300])}[/yellow]"
                    )

            unmasked_tokens = masked_labels[unmasked_after]
            unmasked_text = tokenizer.decode(unmasked_tokens, skip_special_tokens=True)

            original_tokens = original_labels[unmasked_before]
            original_text = tokenizer.decode(original_tokens, skip_special_tokens=True)

            kept_tokens = unmasked_after.sum().item()
            kept_pct = (kept_tokens / valid_original_count * 100) if valid_original_count > 0 else 0

            console.print(
                f"[dim]  └─ Unmasked content ({kept_pct:.1f}% kept, "
                f"{kept_tokens}/{valid_original_count} tokens):[/dim]"
            )

            max_display = 500
            display_text = (
                unmasked_text[:max_display] + "..."
                if len(unmasked_text) > max_display
                else unmasked_text
            )
            console.print(f"[green]     {repr(display_text)}[/green]")

            if len(original_text) <= 400:
                console.print(
                    f"[dim]     Original ({len(original_text)} chars): "
                    f"{repr(original_text[:200])}{'...' if len(original_text) > 200 else ''}[/dim]"
                )

        except Exception as e:
            console.print(f"[yellow]  └─ Could not decode unmasked tokens: {e}[/yellow]")
            if self.verbose:
                import traceback

                traceback.print_exc()

    def print_stats(self) -> None:
        """Print masking statistics."""
        stats = self.get_masking_stats()
        console.print("\n[bold cyan]Selective Loss Masking Statistics:[/bold cyan]")
        console.print(f"  Total tokens processed: {stats['total_tokens']:,}")
        console.print(f"  Tokens masked: {stats['masked_tokens']:,}")
        console.print(f"  Mask percentage: {stats['mask_percentage']:.2f}%")
        console.print(f"  Batches processed: {stats['batch_count']}")


# =============================================================================
# Generic Selective Loss Collator (Backend-Agnostic)
# =============================================================================


class SelectiveLossCollator(SelectiveLossMixin):
    """Generic selective loss collator that wraps any base data collator.

    This collator can be used with any backend (Unsloth, Transformers, etc.)
    by wrapping the backend's data collator.

    Example:
        >>> from transformers import DataCollatorForLanguageModeling
        >>> base_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        >>> selective_collator = SelectiveLossCollator(
        ...     base_collator=base_collator,
        ...     processor=tokenizer,  # Can be tokenizer or processor
        ...     mask_structural_tokens=True,
        ...     verbose=True
        ... )
    """

    def __init__(
        self,
        base_collator: Callable[[list[dict]], dict[str, Any]],
        processor: Any,
        mask_structural_tokens: bool = True,
        mask_schema_keys: bool = False,
        schema_keys: list[str] | None = None,
        mask_json_keywords: bool = False,
        masking_start_epoch: float = 0.0,
        masking_strategy: str = "epoch_based",
        mask_every_n_steps: int = 100,
        mask_for_n_steps: int = 50,
        structural_weight: float = 0.1,
        verbose: bool = False,
    ):
        """Initialize the selective loss collator.

        Args:
            base_collator: The underlying data collator to wrap
            processor: Processor with tokenizer (or tokenizer directly)
            mask_structural_tokens: Whether to mask JSON structure
            mask_schema_keys: Whether to mask field names
            schema_keys: List of field names to mask
            mask_json_keywords: Whether to mask 'null' keyword
            masking_start_epoch: Epoch at which to start masking
            masking_strategy: "epoch_based", "alternating", or "weighted"
            mask_every_n_steps: Cycle length for alternating strategy
            mask_for_n_steps: Steps with masking ON per cycle
            structural_weight: Weight for structural tokens (0.0-1.0)
            verbose: Whether to print debug info
        """
        self.base_collator = base_collator
        self.processor = processor

        self._init_selective_loss(
            mask_structural_tokens=mask_structural_tokens,
            mask_schema_keys=mask_schema_keys,
            schema_keys=schema_keys,
            mask_json_keywords=mask_json_keywords,
            masking_start_epoch=masking_start_epoch,
            masking_strategy=masking_strategy,
            mask_every_n_steps=mask_every_n_steps,
            mask_for_n_steps=mask_for_n_steps,
            structural_weight=structural_weight,
            verbose=verbose,
        )

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        """Process batch through base collator then apply selective loss."""
        # First, use the base collator
        batch = self.base_collator(features)

        # Then apply selective loss masking
        return self._apply_selective_loss_to_batch(batch)


# =============================================================================
# Unsloth-Specific Collator (for when Unsloth is available)
# =============================================================================


# Store reference to avoid repeated imports
_UnslothVisionDataCollator: type | None = None


def _get_unsloth_collator_class() -> type | None:
    """Lazily import UnslothVisionDataCollator."""
    global _UnslothVisionDataCollator
    if _UnslothVisionDataCollator is None:
        try:
            from unsloth.trainer import UnslothVisionDataCollator

            _UnslothVisionDataCollator = UnslothVisionDataCollator
        except ImportError:
            pass
    return _UnslothVisionDataCollator


def is_unsloth_available() -> bool:
    """Check if Unsloth is available."""
    return _get_unsloth_collator_class() is not None


class SelectiveLossUnslothCollator(SelectiveLossMixin):
    """Selective loss collator optimized for Unsloth.

    This collator extends Unsloth's UnslothVisionDataCollator with selective
    loss masking. It's automatically used when Unsloth is available.

    Note: This class dynamically inherits from UnslothVisionDataCollator at
    instantiation time to avoid import issues when Unsloth is not installed.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        mask_structural_tokens: bool = True,
        mask_schema_keys: bool = False,
        schema_keys: list[str] | None = None,
        mask_json_keywords: bool = False,
        masking_start_epoch: float = 0.0,
        masking_strategy: str = "epoch_based",
        mask_every_n_steps: int = 100,
        mask_for_n_steps: int = 50,
        structural_weight: float = 0.1,
        verbose: bool = False,
        **kwargs,
    ):
        """Initialize the Unsloth selective loss collator.

        Args:
            model: The model being trained
            processor: Vision processor
            mask_structural_tokens: Whether to mask JSON structure
            mask_schema_keys: Whether to mask field names
            schema_keys: List of field names to mask
            mask_json_keywords: Whether to mask 'null' keyword
            masking_start_epoch: Epoch at which to start masking
            masking_strategy: "epoch_based", "alternating", or "weighted"
            mask_every_n_steps: Cycle length for alternating strategy
            mask_for_n_steps: Steps with masking ON per cycle
            structural_weight: Weight for structural tokens (0.0-1.0)
            verbose: Whether to print debug info
            **kwargs: Additional args passed to UnslothVisionDataCollator
        """
        UnslothCollatorClass = _get_unsloth_collator_class()
        if UnslothCollatorClass is None:
            raise ImportError(
                "Unsloth is not installed. Use SelectiveLossCollator instead, "
                "or install Unsloth with: pip install unsloth"
            )

        # Store processor before calling parent init
        self.processor = processor
        self._model = model
        self._kwargs = kwargs

        # Create the base Unsloth collator
        self._base_collator = UnslothCollatorClass(model, processor, **kwargs)

        # Initialize selective loss
        self._init_selective_loss(
            mask_structural_tokens=mask_structural_tokens,
            mask_schema_keys=mask_schema_keys,
            schema_keys=schema_keys,
            mask_json_keywords=mask_json_keywords,
            masking_start_epoch=masking_start_epoch,
            masking_strategy=masking_strategy,
            mask_every_n_steps=mask_every_n_steps,
            mask_for_n_steps=mask_for_n_steps,
            structural_weight=structural_weight,
            verbose=verbose,
        )

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        """Process batch through Unsloth collator then apply selective loss."""
        # Show initial batch info only once for debugging
        if self.current_step == 0 and self.verbose:
            if len(features) > 0 and "messages" in features[0]:
                msgs = features[0]["messages"]
                console.print(f"[cyan]SelectiveLossUnslothCollator: {len(msgs)} messages[/cyan]")  # type: ignore[arg-type]

        # First, use Unsloth's collator to handle vision data properly
        batch = self._base_collator(features)

        # Then apply selective loss masking
        return self._apply_selective_loss_to_batch(batch)


# =============================================================================
# Backwards Compatibility Alias
# =============================================================================

# Alias for backwards compatibility
SelectiveLossVisionCollator = SelectiveLossUnslothCollator


# =============================================================================
# Helper Functions
# =============================================================================


def detect_schema_keys_from_dataset(
    dataset: Any,
    processor: Any,
    num_samples: int = 50,
    threshold: float = 0.3,
    verbose: bool = False,
) -> set[str]:
    """Pre-analyze dataset to detect schema keys before training starts.

    Args:
        dataset: Training dataset with formatted messages
        processor: Vision processor with tokenizer
        num_samples: Number of samples to analyze (default: 50)
        threshold: Minimum frequency (0-1) for a key to be included (default: 0.3)
        verbose: Whether to print detection progress

    Returns:
        Set of detected schema keys
    """
    import json

    if verbose:
        console.print(
            f"[cyan]🔍 Pre-analyzing {num_samples} samples to detect schema keys...[/cyan]"
        )

    detected_keys_counter: dict[str, int] = {}
    num_samples = min(num_samples, len(dataset))

    for idx in range(num_samples):
        sample = dataset[idx]

        # Extract assistant response from messages
        if "messages" in sample:
            messages = sample["messages"]
            assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)

            if assistant_msg:
                # Get text content from assistant message
                content = assistant_msg.get("content", [])
                if isinstance(content, list):
                    text_content = next(
                        (c.get("text") for c in content if c.get("type") == "text"), None
                    )
                else:
                    text_content = content

                if text_content:
                    # Try to parse as JSON
                    try:
                        json_data = json.loads(text_content)

                        # Extract all field names recursively
                        def extract_keys(obj: Any, keys_found: set[str]) -> None:
                            if isinstance(obj, dict):
                                for key in obj.keys():
                                    keys_found.add(key)
                                    extract_keys(obj[key], keys_found)
                            elif isinstance(obj, list):
                                for item in obj:
                                    extract_keys(item, keys_found)

                        keys_found: set[str] = set()
                        extract_keys(json_data, keys_found)

                        # Update counter
                        for key in keys_found:
                            detected_keys_counter[key] = detected_keys_counter.get(key, 0) + 1

                    except Exception:
                        # Not valid JSON, skip
                        pass

    # Select keys that appear in at least threshold% of samples
    min_count = int(num_samples * threshold)
    detected_keys = {key for key, count in detected_keys_counter.items() if count >= min_count}

    if verbose:
        if detected_keys:
            console.print(
                f"[green]✅ Detected {len(detected_keys)} schema keys from {num_samples} samples:[/green]"
            )
            sorted_keys = sorted(detected_keys_counter.items(), key=lambda x: x[1], reverse=True)[
                :20
            ]
            for key, count in sorted_keys:
                if key in detected_keys:
                    pct = (count / num_samples) * 100
                    console.print(f"  ✓ {key} ({pct:.1f}% of samples)")
        else:
            console.print("[yellow]⚠️  No schema keys detected! Check dataset format.[/yellow]")
            if detected_keys_counter:
                console.print(f"[dim]   Keys seen (below {threshold * 100}% threshold):[/dim]")
                for key, count in list(detected_keys_counter.items())[:10]:
                    pct = (count / num_samples) * 100
                    console.print(f"[dim]   - {key}: {pct:.1f}%[/dim]")

    return detected_keys


def create_selective_loss_collator(
    model: Any,
    processor: Any,
    mask_level: str = "conservative",
    schema_keys: list[str] | None = None,
    dataset: Any = None,
    masking_strategy: str = "epoch_based",
    masking_start_epoch: float = 0.0,
    mask_every_n_steps: int = 100,
    mask_for_n_steps: int = 50,
    structural_weight: float = 0.1,
    verbose: bool = False,
    train_on_responses_only: bool = False,
    instruction_part: str | None = None,
    response_part: str | None = None,
    backend: str = "auto",
) -> SelectiveLossMixin:
    """Create a selective loss collator with preset masking levels.

    This factory function automatically selects the best collator implementation
    based on the available backend (Unsloth or Transformers).

    Args:
        model: The model being trained
        processor: Vision processor
        mask_level: Masking aggressiveness:
            - "none": No masking (standard training)
            - "conservative": Mask only structural characters ({, }, [, ], :, ,, ")
            - "moderate": Conservative + 'null' keyword
            - "aggressive": Moderate + schema keys (auto-detected if not specified)
        schema_keys: Optional list of field names to mask (auto-detected if None)
        dataset: Training dataset (required for auto-detection in aggressive mode)
        masking_strategy: How to apply masking during training:
            - "epoch_based": Enable masking after masking_start_epoch (default)
            - "alternating": Alternate between masking ON/OFF every n steps
            - "weighted": Apply soft masking with reduced weights (always active)
        masking_start_epoch: For epoch_based: delay masking until this epoch (0.0 = immediate)
        mask_every_n_steps: For alternating: cycle length in steps (default: 100)
        mask_for_n_steps: For alternating: steps with masking ON per cycle (default: 50)
        structural_weight: For weighted: weight for structural tokens, 0.0-1.0 (default: 0.1)
        verbose: Whether to print statistics
        train_on_responses_only: Whether to mask prompts (train only on assistant responses)
        instruction_part: Chat template marker for user messages (e.g., "<|im_start|>user")
        response_part: Chat template marker for assistant messages (e.g., "<|im_start|>assistant")
        backend: Backend to use: "auto" (default), "unsloth", or "transformers"

    Returns:
        Configured selective loss collator (either SelectiveLossUnslothCollator or
        SelectiveLossCollator wrapped around a standard collator)

    Example:
        >>> # Auto-detect backend
        >>> collator = create_selective_loss_collator(
        ...     model, processor,
        ...     mask_level="aggressive",
        ...     dataset=train_dataset,
        ...     verbose=True
        ... )
    """
    # Validate parameters
    if train_on_responses_only:
        if instruction_part is None or response_part is None:
            raise ValueError(
                "train_on_responses_only=True requires instruction_part and response_part. "
                "For Qwen models, use: instruction_part='<|im_start|>user', "
                "response_part='<|im_start|>assistant'"
            )

    # Determine which backend to use
    use_unsloth = False
    if backend == "auto":
        use_unsloth = is_unsloth_available()
    elif backend == "unsloth":
        if not is_unsloth_available():
            raise ImportError(
                "Unsloth backend requested but Unsloth is not installed. "
                "Install with: pip install unsloth"
            )
        use_unsloth = True
    elif backend == "transformers":
        use_unsloth = False
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Choose from: 'auto', 'unsloth', 'transformers'"
        )

    # For "none" mask level, return the appropriate base collator
    if mask_level == "none":
        if use_unsloth:
            UnslothCollatorClass = _get_unsloth_collator_class()
            assert UnslothCollatorClass is not None
            kwargs: dict[str, Any] = {}
            if train_on_responses_only:
                kwargs["train_on_responses_only"] = True
                kwargs["instruction_part"] = instruction_part
                kwargs["response_part"] = response_part
                kwargs["force_match"] = False
            return UnslothCollatorClass(model, processor, **kwargs)  # type: ignore
        else:
            # Return a simple pass-through for non-Unsloth
            from transformers import DataCollatorForLanguageModeling

            tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)  # type: ignore

    # Determine masking configuration based on level
    mask_structural_tokens = True
    mask_schema_keys = False
    mask_json_keywords = False

    if mask_level == "conservative":
        pass  # defaults are fine
    elif mask_level == "moderate":
        mask_json_keywords = True
    elif mask_level == "aggressive":
        mask_json_keywords = True
        mask_schema_keys = True

        # Auto-detect schema keys from dataset if not provided
        if schema_keys is None:
            if dataset is None:
                raise ValueError(
                    "For aggressive mode with auto-detection, you must provide the 'dataset' parameter. "
                    "Either pass dataset=train_dataset or specify schema_keys manually."
                )
            detected_keys = detect_schema_keys_from_dataset(
                dataset=dataset,
                processor=processor,
                num_samples=min(50, len(dataset)),
                threshold=0.3,
                verbose=verbose,
            )
            schema_keys = list(detected_keys)
    else:
        raise ValueError(
            f"Unknown mask_level: {mask_level}. "
            f"Choose from: 'none', 'conservative', 'moderate', 'aggressive'"
        )

    # Create the appropriate collator
    if use_unsloth:
        kwargs = {}
        if train_on_responses_only:
            kwargs["train_on_responses_only"] = True
            kwargs["instruction_part"] = instruction_part
            kwargs["response_part"] = response_part
            kwargs["force_match"] = False

        return SelectiveLossUnslothCollator(
            model=model,
            processor=processor,
            mask_structural_tokens=mask_structural_tokens,
            mask_schema_keys=mask_schema_keys,
            schema_keys=schema_keys,
            mask_json_keywords=mask_json_keywords,
            masking_strategy=masking_strategy,
            masking_start_epoch=masking_start_epoch,
            mask_every_n_steps=mask_every_n_steps,
            mask_for_n_steps=mask_for_n_steps,
            structural_weight=structural_weight,
            verbose=verbose,
            **kwargs,
        )
    else:
        # Create a base collator for Transformers
        from transformers import DataCollatorForLanguageModeling

        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        return SelectiveLossCollator(
            base_collator=base_collator,
            processor=processor,
            mask_structural_tokens=mask_structural_tokens,
            mask_schema_keys=mask_schema_keys,
            schema_keys=schema_keys,
            mask_json_keywords=mask_json_keywords,
            masking_strategy=masking_strategy,
            masking_start_epoch=masking_start_epoch,
            mask_every_n_steps=mask_every_n_steps,
            mask_for_n_steps=mask_for_n_steps,
            structural_weight=structural_weight,
            verbose=verbose,
        )
