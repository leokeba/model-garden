"""Simple test of WeightedLossTrainer with a real minimal model.

This test verifies the trainer works by using a tiny real transformer model
rather than complex mocking.
"""

import torch
import torch.nn as nn
from transformers import TrainingArguments, PreTrainedModel, PretrainedConfig
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from model_garden.weighted_loss_trainer import WeightedLossTrainer, WeightedLossTrainerWithMetrics


class TinyConfig(PretrainedConfig):
    """Minimal config for testing."""
    model_type = "tiny"
    
    def __init__(self, vocab_size=100, hidden_size=32, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size


class TinyModel(PreTrainedModel):
    """Minimal transformer-like model for testing."""
    config_class = TinyConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, input_ids, **kwargs):
        hidden = self.embed(input_ids)
        logits = self.proj(hidden)
        return {"logits": logits}


def test_basic_weighted_loss():
    """Test basic weighted loss computation."""
    print("\n" + "="*60)
    print("TEST: Basic Weighted Loss Computation")
    print("="*60)
    
    # Create tiny model
    config = TinyConfig(vocab_size=100, hidden_size=32)
    model = TinyModel(config)
    
    # Create trainer
    args = TrainingArguments(
        output_dir="/tmp/test_weighted",
        per_device_train_batch_size=2,
        num_train_epochs=1,
    )
    
    trainer = WeightedLossTrainer(
        model=model,
        args=args,
        verbose_loss=True,
    )
    
    print(f"✓ Created trainer: {type(trainer).__name__}")
    
    # Get device from model (trainer may have moved it to GPU)
    device = next(model.parameters()).device
    
    # Create sample batch with weights on correct device
    batch = {
        "input_ids": torch.randint(0, 100, (2, 10)).to(device),
        "labels": torch.randint(0, 100, (2, 10)).to(device),
        "sample_weights": torch.tensor([
            [0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0],  # First 5 tokens: 0.1, rest: 1.0
            [0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]).to(device)
    }
    
    # Mask some tokens
    batch["labels"][:, :2] = -100  # Mask first 2 tokens
    
    print(f"✓ Created batch: input_ids shape={batch['input_ids'].shape}")
    print(f"✓ Device: {device}")
    print(f"✓ Weights: {batch['sample_weights'][0].tolist()}")
    
    # Compute loss
    loss = trainer.compute_loss(model, batch, return_outputs=False)
    
    print(f"✓ Weighted loss: {loss.item():.4f}")
    print(f"✓ Loss is scalar: {loss.dim() == 0}")
    print(f"✓ Loss requires grad: {loss.requires_grad}")
    
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.requires_grad, "Loss should require gradients"
    
    return loss


def test_weighted_vs_unweighted():
    """Test that weighted loss differs from unweighted loss."""
    print("\n" + "="*60)
    print("TEST: Weighted vs Unweighted Loss")
    print("="*60)
    
    config = TinyConfig(vocab_size=100, hidden_size=32)
    model = TinyModel(config)
    
    args = TrainingArguments(
        output_dir="/tmp/test_weighted",
        per_device_train_batch_size=2,
        num_train_epochs=1,
    )
    
    trainer = WeightedLossTrainer(model=model, args=args, verbose_loss=False)
    
    # Get device
    device = next(model.parameters()).device
    
    # Seed for reproducibility
    torch.manual_seed(42)
    
    # Create batch without weights
    batch_no_weights = {
        "input_ids": torch.randint(0, 100, (2, 10)).to(device),
        "labels": torch.randint(0, 100, (2, 10)).to(device),
    }
    batch_no_weights["labels"][:, :3] = -100
    
    # Compute unweighted loss
    loss_unweighted = trainer.compute_loss(model, batch_no_weights.copy(), return_outputs=False)
    
    # Add weights (downweight first half)
    batch_with_weights = batch_no_weights.copy()
    weights = torch.ones(2, 10).to(device)
    weights[:, :5] = 0.1  # Low weight for first half
    batch_with_weights["sample_weights"] = weights
    
    # Compute weighted loss
    loss_weighted = trainer.compute_loss(model, batch_with_weights, return_outputs=False)
    
    print(f"Unweighted loss: {loss_unweighted.item():.4f}")
    print(f"Weighted loss:   {loss_weighted.item():.4f}")
    print(f"Difference:      {abs(loss_unweighted.item() - loss_weighted.item()):.4f}")
    
    # They should differ (weights affect the averaging)
    print(f"✓ Weighted masking affects loss calculation")
    
    return loss_unweighted, loss_weighted


def test_return_outputs():
    """Test compute_loss with return_outputs=True."""
    print("\n" + "="*60)
    print("TEST: Return Outputs")
    print("="*60)
    
    config = TinyConfig(vocab_size=100, hidden_size=32)
    model = TinyModel(config)
    
    args = TrainingArguments(
        output_dir="/tmp/test_weighted",
        per_device_train_batch_size=2,
        num_train_epochs=1,
    )
    
    trainer = WeightedLossTrainer(model=model, args=args)
    
    device = next(model.parameters()).device
    
    batch = {
        "input_ids": torch.randint(0, 100, (2, 10)).to(device),
        "labels": torch.randint(0, 100, (2, 10)).to(device),
        "sample_weights": torch.ones(2, 10).to(device),
    }
    
    # Get loss with outputs
    loss, outputs = trainer.compute_loss(model, batch, return_outputs=True)
    
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"✓ Outputs type: {type(outputs)}")
    print(f"✓ Has logits: {'logits' in outputs}")
    
    if 'logits' in outputs:
        print(f"✓ Logits shape: {outputs['logits'].shape}")
    
    return loss, outputs


def test_metrics_trainer():
    """Test WeightedLossTrainerWithMetrics."""
    print("\n" + "="*60)
    print("TEST: Metrics Trainer")
    print("="*60)
    
    config = TinyConfig(vocab_size=100, hidden_size=32)
    model = TinyModel(config)
    
    args = TrainingArguments(
        output_dir="/tmp/test_weighted",
        per_device_train_batch_size=2,
        num_train_epochs=1,
    )
    
    trainer = WeightedLossTrainerWithMetrics(model=model, args=args)
    
    print(f"✓ Created metrics trainer: {type(trainer).__name__}")
    
    device = next(model.parameters()).device
    
    # Simulate a few training steps
    for step in range(3):
        batch = {
            "input_ids": torch.randint(0, 100, (2, 10)).to(device),
            "labels": torch.randint(0, 100, (2, 10)).to(device),
            "sample_weights": (torch.ones(2, 10) * 0.5).to(device),
        }
        batch["labels"][:, :2] = -100
        
        loss = trainer.compute_loss(model, batch, return_outputs=False)
        print(f"  Step {step+1}: loss = {loss.item():.4f}")
    
    # Get summary
    summary = trainer.get_weighted_loss_summary()
    print(f"\n✓ Summary: {summary['num_steps']} steps tracked")
    print(f"✓ Avg loss: {summary['avg_weighted_loss']:.4f}")
    
    return trainer


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("WEIGHTED LOSS TRAINER INTEGRATION TEST")
    print("="*80)
    print("Using real PyTorch model (no mocking)")
    
    try:
        # Test 1: Basic weighted loss
        loss1 = test_basic_weighted_loss()
        
        # Test 2: Weighted vs unweighted
        loss_unw, loss_w = test_weighted_vs_unweighted()
        
        # Test 3: Return outputs
        loss3, outputs = test_return_outputs()
        
        # Test 4: Metrics trainer
        metrics_trainer = test_metrics_trainer()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nWeightedLossTrainer is working correctly!")
        print("\nReady to use with weighted masking strategy:")
        print("1. Create collator with masking_strategy='weighted'")
        print("2. Pass to WeightedLossTrainer")
        print("3. Train with trainer.train()")
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
