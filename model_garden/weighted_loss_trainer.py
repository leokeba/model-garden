"""Custom Trainer for weighted masking strategy.

This module provides a Trainer subclass that supports per-token loss weighting,
required for the weighted masking strategy in selective loss training.

Usage:
    from model_garden.weighted_loss_trainer import WeightedLossTrainer
    from model_garden.selective_loss import create_selective_loss_collator
    
    # Create weighted collator
    collator = create_selective_loss_collator(
        model=model,
        processor=processor,
        mask_level="aggressive",
        masking_strategy="weighted",
        structural_weight=0.1,
        dataset=train_dataset,
        verbose=True
    )
    
    # Use custom trainer
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    
    trainer.train()
"""

import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, Optional, Union, Any
from rich.console import Console

console = Console()


class WeightedLossTrainer(Trainer):
    """Custom Trainer that supports per-token loss weighting.
    
    This trainer overrides compute_loss to handle the 'sample_weights' key
    added by SelectiveLossVisionCollator when using masking_strategy="weighted".
    
    Key Features:
    - Computes per-token loss with reduction='none'
    - Applies per-token weights from 'sample_weights' if present
    - Properly averages weighted loss (sum(loss * weight) / sum(weight))
    - Falls back to standard loss computation if no weights provided
    - Compatible with all other Trainer features (mixed precision, gradient accumulation, etc.)
    
    Args:
        Same as transformers.Trainer
        
    Example:
        >>> trainer = WeightedLossTrainer(
        ...     model=model,
        ...     args=training_args,
        ...     train_dataset=train_dataset,
        ...     data_collator=weighted_collator,  # Must use weighted masking collator
        ... )
        >>> trainer.train()
    """
    
    def __init__(self, *args, verbose_loss: bool = False, **kwargs):
        """Initialize WeightedLossTrainer.
        
        Args:
            verbose_loss: If True, print loss statistics every N steps (default: False)
            *args, **kwargs: Passed to parent Trainer class
        """
        super().__init__(*args, **kwargs)
        self.verbose_loss = verbose_loss
        self._loss_step_counter = 0
        
        # Print info on initialization
        if self.verbose_loss:
            console.print("[cyan]Initialized WeightedLossTrainer with verbose loss logging[/cyan]")
    
    def compute_loss(
        self, 
        model, 
        inputs: Dict[str, Union[torch.Tensor, Any]], 
        return_outputs: bool = False,
        num_items_in_batch: Optional[Union[int, torch.Tensor]] = None
    ):
        """Compute loss with per-token weighting support.
        
        This method:
        1. Extracts 'sample_weights' from inputs if present
        2. Computes per-token cross-entropy loss
        3. Applies per-token weights if available
        4. Returns properly averaged loss
        
        Args:
            model: The model to compute loss for
            inputs: Dictionary with 'input_ids', 'labels', and optionally 'sample_weights'
            return_outputs: Whether to return model outputs along with loss
            num_items_in_batch: Number of items in batch (for proper scaling)
            
        Returns:
            loss (torch.Tensor): Scalar loss value
            OR
            (loss, outputs) if return_outputs=True
        """
        # Extract labels and weights
        labels = inputs.pop("labels")
        sample_weights = inputs.pop("sample_weights", None)
        
        # Check if we have weights
        has_weights = sample_weights is not None
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Compute per-token loss (no reduction)
        # ignore_index=-100 handles prompt tokens that are already masked
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        
        # Reshape for cross-entropy: [batch * seq_len, vocab_size] and [batch * seq_len]
        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Apply per-token weights if provided
        if has_weights:
            weights_flat = sample_weights.view(-1)
            
            # Multiply loss by weights (element-wise)
            weighted_loss = loss * weights_flat
            
            # Compute weighted average: sum(loss * weight) / sum(weight)
            # Only consider valid tokens (not -100)
            valid_mask = (labels != -100).view(-1)
            
            if valid_mask.any():
                # Sum of weighted losses for valid tokens
                total_weighted_loss = weighted_loss[valid_mask].sum()
                # Sum of weights for valid tokens (for proper averaging)
                total_weights = weights_flat[valid_mask].sum()
                
                # Avoid division by zero
                if total_weights > 0:
                    final_loss = total_weighted_loss / total_weights
                else:
                    final_loss = weighted_loss[valid_mask].mean()
            else:
                final_loss = torch.tensor(0.0, device=loss.device)
            
            # Verbose logging
            if self.verbose_loss and self._loss_step_counter % 10 == 0:
                self._log_weighted_loss_stats(
                    loss=loss,
                    weights=weights_flat,
                    valid_mask=valid_mask,
                    final_loss=final_loss
                )
        else:
            # Standard averaging over valid tokens (no weights)
            valid_mask = (labels != -100).view(-1)
            
            if valid_mask.any():
                final_loss = loss[valid_mask].mean()
            else:
                final_loss = torch.tensor(0.0, device=loss.device)
        
        # Increment step counter
        self._loss_step_counter += 1
        
        return (final_loss, outputs) if return_outputs else final_loss
    
    def _log_weighted_loss_stats(
        self,
        loss: torch.Tensor,
        weights: torch.Tensor,
        valid_mask: torch.Tensor,
        final_loss: torch.Tensor
    ):
        """Log statistics about weighted loss computation.
        
        Args:
            loss: Per-token losses
            weights: Per-token weights
            valid_mask: Boolean mask of valid tokens
            final_loss: Final averaged loss
        """
        with torch.no_grad():
            # Calculate statistics
            valid_loss = loss[valid_mask]
            valid_weights = weights[valid_mask]
            
            # Get unique weights to see distribution
            unique_weights = torch.unique(valid_weights).cpu().tolist()
            
            # Count tokens by weight
            weight_counts = {}
            for w in unique_weights:
                count = (valid_weights == w).sum().item()
                weight_counts[f"{w:.2f}"] = count
            
            console.print(f"[dim]Step {self._loss_step_counter}: Weighted Loss Stats[/dim]")
            console.print(f"  Final loss: {final_loss.item():.4f}")
            console.print(f"  Valid tokens: {valid_mask.sum().item()}")
            console.print(f"  Weight distribution: {weight_counts}")
            console.print(f"  Avg loss (unweighted): {valid_loss.mean().item():.4f}")
            console.print(f"  Avg weight: {valid_weights.mean().item():.4f}")


class WeightedLossTrainerWithMetrics(WeightedLossTrainer):
    """Extended WeightedLossTrainer that tracks additional metrics.
    
    This variant logs extra statistics about weighted masking during training:
    - Percentage of tokens at each weight level
    - Unweighted vs weighted average loss
    - Distribution of structural vs semantic tokens
    
    Useful for debugging and understanding weighted masking behavior.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, verbose_loss=True, **kwargs)
        
        # Track cumulative statistics
        self.weighted_loss_history = []
        self.unweighted_loss_history = []
        self.weight_distributions = []
        
        console.print("[cyan]Initialized WeightedLossTrainerWithMetrics[/cyan]")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with additional metric tracking."""
        # Save weights before parent removes them
        sample_weights = inputs.get("sample_weights", None)
        labels = inputs.get("labels", None)
        
        # Call parent compute_loss
        result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        
        # Extract loss and outputs
        if return_outputs:
            loss, outputs = result
        else:
            loss = result
            outputs = None
        
        # Track statistics (only during training, not eval)
        model_training = getattr(self.model, 'training', False) if self.model is not None else False
        if model_training and sample_weights is not None:
            with torch.no_grad():
                if labels is not None:
                    valid_mask = (labels != -100)
                    if valid_mask.any():
                        valid_weights = sample_weights[valid_mask]
                        
                        # Track weight distribution
                        unique_weights, counts = torch.unique(valid_weights, return_counts=True)
                        distribution = {
                            f"{w.item():.2f}": c.item() 
                            for w, c in zip(unique_weights, counts)
                        }
                        self.weight_distributions.append(distribution)
                        
                        # Track weighted loss
                        if isinstance(loss, torch.Tensor):
                            self.weighted_loss_history.append(loss.item())
        
        return result if return_outputs else loss
    
    def get_weighted_loss_summary(self) -> Dict[str, Any]:
        """Get summary of weighted loss statistics.
        
        Returns:
            Dictionary with aggregate statistics
        """
        if not self.weighted_loss_history:
            return {"message": "No weighted loss data collected yet"}
        
        import numpy as np
        
        return {
            "num_steps": len(self.weighted_loss_history),
            "avg_weighted_loss": np.mean(self.weighted_loss_history),
            "std_weighted_loss": np.std(self.weighted_loss_history),
            "weight_distributions_sampled": self.weight_distributions[:5],  # First 5 distributions
        }
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging - print weighted loss summary periodically."""
        # Call parent if it has on_log
        if hasattr(super(), 'on_log'):
            super().on_log(args, state, control, logs, **kwargs)  # type: ignore[attr-defined]
        
        # Every 100 steps, print summary
        if state.global_step % 100 == 0 and self.weighted_loss_history:
            summary = self.get_weighted_loss_summary()
            console.print(f"\n[bold cyan]Weighted Loss Summary (Step {state.global_step}):[/bold cyan]")
            console.print(f"  Avg weighted loss: {summary['avg_weighted_loss']:.4f}")
            console.print(f"  Std weighted loss: {summary['std_weighted_loss']:.4f}")
            console.print()
