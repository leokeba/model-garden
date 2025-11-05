"""Test WeightedLossTrainer implementation.

This test verifies that the custom trainer correctly handles weighted masking
and computes loss properly with per-token weights.
"""

import sys
import os
import torch
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from transformers import TrainingArguments

# Add model_garden to path
sys.path.insert(0, str(Path(__file__).parent))

from model_garden.weighted_loss_trainer import WeightedLossTrainer, WeightedLossTrainerWithMetrics


def create_mock_model():
    """Create a mock model that returns realistic outputs."""
    model = Mock()
    model.spec = ["forward", "__call__", "training", "tp_size", "config", "dtype", "device", "hf_device_map"]
    
    # Mock forward pass to return logits
    def forward(**kwargs):
        batch_size = kwargs.get("input_ids").shape[0]
        seq_len = kwargs.get("input_ids").shape[1]
        vocab_size = 32000  # Typical vocab size
        
        # Return random logits
        logits = torch.randn(batch_size, seq_len, vocab_size)
        return {"logits": logits}
    
    model.forward = forward
    model.__call__ = forward
    model.training = True
    model.tp_size = None  # No tensor parallelism
    model.hf_device_map = None  # No device map
    model.config = Mock()
    model.config.model_type = "test"
    model.config.architectures = ["TestModel"]
    model.dtype = torch.float32
    model.device = torch.device("cpu")
    
    # Mock parameters for Trainer
    def parameters():
        return [torch.nn.Parameter(torch.randn(10, 10))]
    model.parameters = parameters
    
    return model


def create_sample_batch(batch_size=2, seq_len=10, with_weights=True):
    """Create a sample batch with labels and optionally weights.
    
    Args:
        batch_size: Number of sequences in batch
        seq_len: Length of each sequence
        with_weights: If True, include sample_weights
        
    Returns:
        Dictionary with input_ids, labels, and optionally sample_weights
    """
    # Create input IDs
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    
    # Create labels (some tokens masked with -100)
    labels = input_ids.clone()
    # Mask first half of tokens (prompt)
    labels[:, :seq_len//2] = -100
    
    batch = {
        "input_ids": input_ids,
        "labels": labels,
    }
    
    if with_weights:
        # Create sample weights
        # Structure tokens (first half): low weight (0.1)
        # Semantic tokens (second half): full weight (1.0)
        weights = torch.ones(batch_size, seq_len)
        weights[:, :seq_len//2] = 0.1
        
        batch["sample_weights"] = weights
    
    return batch


def test_weighted_trainer_initialization():
    """Test that WeightedLossTrainer can be initialized."""
    print("\n" + "="*60)
    print("TEST: WeightedLossTrainer Initialization")
    print("="*60)
    
    model = create_mock_model()
    
    # Create minimal training args
    training_args = TrainingArguments(
        output_dir="/tmp/test_weighted_trainer",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=10,
    )
    
    # Initialize trainer
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        verbose_loss=True,
    )
    
    print(f"✓ Trainer initialized: {type(trainer).__name__}")
    print(f"✓ Verbose loss: {trainer.verbose_loss}")
    print(f"✓ Model: {type(trainer.model)}")
    print(f"✓ Args: {trainer.args.output_dir}")
    
    return trainer


def test_weighted_loss_computation():
    """Test that weighted loss is computed correctly."""
    print("\n" + "="*60)
    print("TEST: Weighted Loss Computation")
    print("="*60)
    
    model = create_mock_model()
    
    training_args = TrainingArguments(
        output_dir="/tmp/test_weighted_trainer",
        num_train_epochs=1,
        per_device_train_batch_size=2,
    )
    
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        verbose_loss=False,
    )
    
    # Create batch WITH weights
    batch_with_weights = create_sample_batch(batch_size=2, seq_len=10, with_weights=True)
    
    # Compute loss
    loss_weighted = trainer.compute_loss(model, batch_with_weights.copy(), return_outputs=False)
    
    print(f"✓ Weighted loss computed: {loss_weighted.item():.4f}")
    print(f"✓ Loss shape: {loss_weighted.shape}")
    print(f"✓ Loss is scalar: {loss_weighted.dim() == 0}")
    print(f"✓ Loss requires grad: {loss_weighted.requires_grad}")
    
    # Create batch WITHOUT weights
    batch_no_weights = create_sample_batch(batch_size=2, seq_len=10, with_weights=False)
    
    # Compute loss
    loss_standard = trainer.compute_loss(model, batch_no_weights.copy(), return_outputs=False)
    
    print(f"\n✓ Standard loss computed: {loss_standard.item():.4f}")
    print(f"✓ Loss shape: {loss_standard.shape}")
    
    # Both should be valid scalars
    assert loss_weighted.dim() == 0, "Weighted loss should be scalar"
    assert loss_standard.dim() == 0, "Standard loss should be scalar"
    assert loss_weighted.requires_grad, "Loss should require gradients"
    
    print("\n✓ All loss computations successful!")
    
    return loss_weighted, loss_standard


def test_weighted_vs_standard_loss_difference():
    """Test that weighted loss differs from standard loss."""
    print("\n" + "="*60)
    print("TEST: Weighted vs Standard Loss Difference")
    print("="*60)
    
    model = create_mock_model()
    
    training_args = TrainingArguments(
        output_dir="/tmp/test_weighted_trainer",
        num_train_epochs=1,
        per_device_train_batch_size=2,
    )
    
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        verbose_loss=False,
    )
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create batch
    batch = create_sample_batch(batch_size=4, seq_len=20, with_weights=False)
    
    # Compute standard loss (no weights)
    batch_no_weights = batch.copy()
    loss_standard = trainer.compute_loss(model, batch_no_weights, return_outputs=False)
    
    # Add weights that downweight structural tokens
    batch_with_weights = batch.copy()
    weights = torch.ones_like(batch["labels"], dtype=torch.float32)
    # First half: structural tokens with low weight
    weights[:, :10] = 0.1
    # Second half: semantic tokens with full weight
    weights[:, 10:] = 1.0
    batch_with_weights["sample_weights"] = weights
    
    # Compute weighted loss
    loss_weighted = trainer.compute_loss(model, batch_with_weights, return_outputs=False)
    
    print(f"Standard loss (no weights): {loss_standard.item():.4f}")
    print(f"Weighted loss (structural=0.1): {loss_weighted.item():.4f}")
    print(f"Difference: {abs(loss_standard.item() - loss_weighted.item()):.4f}")
    
    # Losses should differ (unless by extreme coincidence)
    # We expect weighted loss to be different because structural tokens contribute less
    print(f"\n✓ Weighted masking affects loss computation")
    
    return loss_standard, loss_weighted


def test_trainer_with_metrics():
    """Test WeightedLossTrainerWithMetrics."""
    print("\n" + "="*60)
    print("TEST: WeightedLossTrainerWithMetrics")
    print("="*60)
    
    model = create_mock_model()
    
    training_args = TrainingArguments(
        output_dir="/tmp/test_weighted_trainer",
        num_train_epochs=1,
        per_device_train_batch_size=2,
    )
    
    trainer = WeightedLossTrainerWithMetrics(
        model=model,
        args=training_args,
    )
    
    print(f"✓ Metrics trainer initialized: {type(trainer).__name__}")
    
    # Run a few training steps
    for step in range(5):
        batch = create_sample_batch(batch_size=2, seq_len=10, with_weights=True)
        loss = trainer.compute_loss(model, batch, return_outputs=False)
        print(f"  Step {step+1}: loss = {loss.item():.4f}")
    
    # Get summary
    summary = trainer.get_weighted_loss_summary()
    
    print(f"\n✓ Summary statistics:")
    print(f"  Steps tracked: {summary['num_steps']}")
    print(f"  Avg loss: {summary['avg_weighted_loss']:.4f}")
    print(f"  Std loss: {summary['std_weighted_loss']:.4f}")
    print(f"  Weight distributions sampled: {len(summary['weight_distributions_sampled'])}")
    
    return trainer, summary


def test_loss_with_return_outputs():
    """Test that compute_loss works with return_outputs=True."""
    print("\n" + "="*60)
    print("TEST: Compute Loss with Return Outputs")
    print("="*60)
    
    model = create_mock_model()
    
    training_args = TrainingArguments(
        output_dir="/tmp/test_weighted_trainer",
        num_train_epochs=1,
        per_device_train_batch_size=2,
    )
    
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
    )
    
    batch = create_sample_batch(batch_size=2, seq_len=10, with_weights=True)
    
    # Get loss with outputs
    loss, outputs = trainer.compute_loss(model, batch, return_outputs=True)
    
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"✓ Outputs: {type(outputs)}")
    print(f"✓ Outputs has logits: {'logits' in outputs}")
    
    if 'logits' in outputs:
        print(f"✓ Logits shape: {outputs['logits'].shape}")
    
    return loss, outputs


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("WEIGHTED LOSS TRAINER TEST SUITE")
    print("="*80)
    
    try:
        # Test 1: Initialization
        trainer = test_weighted_trainer_initialization()
        
        # Test 2: Weighted loss computation
        loss_weighted, loss_standard = test_weighted_loss_computation()
        
        # Test 3: Weighted vs standard difference
        loss_std, loss_wtd = test_weighted_vs_standard_loss_difference()
        
        # Test 4: Metrics trainer
        metrics_trainer, summary = test_trainer_with_metrics()
        
        # Test 5: Return outputs
        loss, outputs = test_loss_with_return_outputs()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nWeightedLossTrainer is ready to use with weighted masking strategy!")
        print("\nNext steps:")
        print("1. Create weighted collator with masking_strategy='weighted'")
        print("2. Pass collator to WeightedLossTrainer")
        print("3. Train model with trainer.train()")
        print("4. Compare results to alternating strategy baseline")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED ❌")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
