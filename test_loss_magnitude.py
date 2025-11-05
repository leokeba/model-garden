"""Test that weighted loss produces values in the correct range.

This test verifies that the fix for weighted loss normalization works correctly.
The weighted loss should be in the same magnitude range as standard loss (0.01-10.0),
not inflated to 25+ due to incorrect normalization.
"""

import torch
from model_garden.weighted_loss_trainer import WeightedLossTrainer


class TinyModel(torch.nn.Module):
    """Minimal model for testing."""
    def __init__(self, vocab_size=100, hidden_size=32):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids, **kwargs):
        hidden = self.embedding(input_ids)
        logits = self.lm_head(hidden)
        return {"logits": logits}


def test_weighted_loss_magnitude():
    """Test that weighted loss is in correct range (not inflated)."""
    print("\n" + "="*80)
    print("TEST: Weighted Loss Magnitude")
    print("="*80)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    # Create a simple model
    model = TinyModel(vocab_size=100).to(device)
    
    # Create trainer (no args needed for this test)
    trainer = WeightedLossTrainer(
        model=model,
        args=None,
        verbose_loss=True
    )
    
    # Create test batch with realistic weight distribution
    batch_size = 4
    seq_len = 128
    
    # Simulate realistic scenario:
    # - 70% structural tokens (weight=0.1)
    # - 30% semantic tokens (weight=1.0)
    
    input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 100, (batch_size, seq_len), device=device)
    
    # Create weights: 70% structural (0.1), 30% semantic (1.0)
    sample_weights = torch.zeros((batch_size, seq_len), device=device)
    for i in range(batch_size):
        # First 90 tokens are structural (weight=0.1)
        sample_weights[i, :90] = 0.1
        # Last 38 tokens are semantic (weight=1.0)
        sample_weights[i, 90:] = 1.0
    
    inputs = {
        "input_ids": input_ids,
        "labels": labels,
        "sample_weights": sample_weights
    }
    
    # Compute weighted loss (returns tensor when return_outputs=False)
    weighted_loss = trainer.compute_loss(model, inputs, return_outputs=False)
    assert isinstance(weighted_loss, torch.Tensor), "compute_loss should return tensor"
    
    print(f"\n📊 Results:")
    print(f"   Weighted Loss: {weighted_loss.item():.4f}")
    print(f"   Expected Range: 0.01 - 10.0 (typical for cross-entropy)")
    
    # Test 1: Loss should be in reasonable range
    assert 0.01 <= weighted_loss.item() <= 10.0, (
        f"Weighted loss {weighted_loss.item():.4f} is outside expected range [0.01, 10.0]! "
        f"This indicates the normalization is still broken."
    )
    
    # Test 2: Compare to unweighted loss (should be in same ballpark)
    # Remove weights and compute standard loss
    inputs_unweighted = {
        "input_ids": input_ids.clone(),
        "labels": labels.clone()
    }
    unweighted_loss = trainer.compute_loss(model, inputs_unweighted, return_outputs=False)
    assert isinstance(unweighted_loss, torch.Tensor), "compute_loss should return tensor"
    
    print(f"   Unweighted Loss: {unweighted_loss.item():.4f}")
    print(f"   Ratio (weighted/unweighted): {weighted_loss.item() / unweighted_loss.item():.2f}x")
    
    # Weighted loss should be lower than unweighted (we're down-weighting 70% of tokens)
    # But it should not be orders of magnitude different
    ratio = weighted_loss.item() / unweighted_loss.item()
    assert 0.1 <= ratio <= 2.0, (
        f"Ratio {ratio:.2f}x is unreasonable! Expected 0.1-2.0x. "
        f"This suggests normalization is still incorrect."
    )
    
    print(f"\n✅ Test passed! Weighted loss is in correct range.")
    print(f"   The fix successfully keeps loss magnitude comparable to standard training.")
    
    # Test 3: Edge case with all structural tokens (weight=0.1)
    all_structural_weights = torch.full((batch_size, seq_len), 0.1, device=device)
    inputs_all_structural = {
        "input_ids": input_ids.clone(),
        "labels": labels.clone(),
        "sample_weights": all_structural_weights
    }
    all_structural_loss = trainer.compute_loss(model, inputs_all_structural, return_outputs=False)
    assert isinstance(all_structural_loss, torch.Tensor), "compute_loss should return tensor"
    
    print(f"\n🧪 Edge Case: All structural tokens (weight=0.1)")
    print(f"   Loss: {all_structural_loss.item():.4f}")
    print(f"   Expected: ~{unweighted_loss.item() * 0.1:.4f} (10% of unweighted)")
    
    # With all weights at 0.1, weighted loss should be ~10% of unweighted
    expected_structural_loss = unweighted_loss.item() * 0.1
    assert abs(all_structural_loss.item() - expected_structural_loss) / expected_structural_loss < 0.5, (
        f"All-structural loss {all_structural_loss.item():.4f} doesn't match expected "
        f"{expected_structural_loss:.4f} (10% of unweighted). Normalization may be incorrect."
    )
    
    print(f"   ✅ Edge case passed!")
    
    print("\n" + "="*80)
    print("All tests passed! Weighted loss normalization is correct.")
    print("="*80 + "\n")


def test_gradient_flow():
    """Test that gradients flow correctly with weighted loss."""
    print("\n" + "="*80)
    print("TEST: Gradient Flow with Weighted Loss")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyModel(vocab_size=100).to(device)
    trainer = WeightedLossTrainer(model=model, args=None)
    
    # Create batch
    batch_size = 2
    seq_len = 64
    
    input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 100, (batch_size, seq_len), device=device)
    sample_weights = torch.rand((batch_size, seq_len), device=device) * 0.9 + 0.1  # Weights between 0.1 and 1.0
    
    inputs = {
        "input_ids": input_ids,
        "labels": labels,
        "sample_weights": sample_weights
    }
    
    # Compute loss and backward
    loss = trainer.compute_loss(model, inputs, return_outputs=False)
    assert isinstance(loss, torch.Tensor), "compute_loss should return tensor"
    loss.backward()
    
    # Check that gradients exist
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            print(f"   ✓ {name}: grad magnitude = {param.grad.abs().mean().item():.6f}")
    
    assert has_gradients, "No gradients found! Weighted loss is not flowing gradients correctly."
    
    print(f"\n✅ Gradients flow correctly through weighted loss!")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_weighted_loss_magnitude()
    test_gradient_flow()
    print("\n🎉 All weighted loss magnitude tests passed!")
    print("   Loss values are now in the correct range (0.01 - 10.0)")
    print("   Training should proceed normally with comparable loss magnitudes.\n")
