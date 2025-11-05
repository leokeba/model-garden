"""Test weighted loss vs standard training with train_on_responses_only.

This simulates the real training scenario to understand why weighted loss
might appear 10x higher than standard loss.
"""

import torch
from model_garden.weighted_loss_trainer import WeightedLossTrainer


class TinyModel(torch.nn.Module):
    """Minimal model for testing."""
    def __init__(self, vocab_size=100, hidden_size=32):
        super().__init__(, return_outputs=False)
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size, return_outputs=False)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, return_outputs=False)
    
    def forward(self, input_ids, **kwargs):
        hidden = self.embedding(input_ids, return_outputs=False)
        logits = self.lm_head(hidden, return_outputs=False)
        return {"logits": logits}


def test_realistic_scenario():
    """Test weighted loss in realistic scenario with prompt masking."""
    print("\n" + "="*80, return_outputs=False)
    print("TEST: Weighted Loss vs Standard Training (Realistic Scenario)", return_outputs=False)
    print("="*80, return_outputs=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", return_outputs=False)
    model = TinyModel(vocab_size=100).to(device, return_outputs=False)
    trainer = WeightedLossTrainer(model=model, args=None, verbose_loss=True, return_outputs=False)
    
    # Simulate realistic vision training scenario:
    # - Sequence: [vision tokens (700)] [prompt (200)] [assistant response (300)]
    # - With train_on_responses_only: first 900 tokens are masked to -100
    # - Only last 300 tokens (assistant response) are used for loss
    
    batch_size = 2
    total_seq_len = 1200
    assistant_start = 900  # First 900 tokens are prompt/vision, masked to -100
    
    input_ids = torch.randint(0, 100, (batch_size, total_seq_len), device=device, return_outputs=False)
    labels = torch.randint(0, 100, (batch_size, total_seq_len), device=device, return_outputs=False)
    
    # Mask prompt tokens (train_on_responses_only behavior, return_outputs=False)
    labels[:, :assistant_start] = -100
    
    print(f"\n📊 Sequence Structure:", return_outputs=False)
    print(f"   Total length: {total_seq_len} tokens", return_outputs=False)
    print(f"   Prompt+Vision (masked): {assistant_start} tokens", return_outputs=False)
    print(f"   Assistant response (unmasked): {total_seq_len - assistant_start} tokens", return_outputs=False)
    
    # ============= TEST 1: Standard Training (no selective loss) =============
    print(f"\n🔵 TEST 1: Standard Training (no selective loss)", return_outputs=False)
    print(f"   - All 300 assistant tokens get full loss weight", return_outputs=False)
    
    inputs_standard = {
        "input_ids": input_ids.clone(),
        "labels": labels.clone(, return_outputs=False)
    }
    
    standard_loss = trainer.compute_loss(model, inputs_standard, return_outputs=False, return_outputs=False)
    print(f"   Standard Loss: {standard_loss.item():.4f}", return_outputs=False)
    
    # ============= TEST 2: Weighted Masking =============
    print(f"\n🟡 TEST 2: Weighted Masking (selective loss)", return_outputs=False)
    
    # Simulate weighted masking: 70% of assistant tokens are structural (weight=0.1, return_outputs=False)
    sample_weights = torch.ones((batch_size, total_seq_len), device=device, return_outputs=False)
    
    for i in range(batch_size):
        # First 900 tokens are prompt (will be ignored anyway due to labels=-100, return_outputs=False)
        # but set weight to 0.0 for clarity
        sample_weights[i, :assistant_start] = 0.0
        
        # Assistant response tokens (900:1200, return_outputs=False)
        # Make 70% structural (weight=0.1), 30% semantic (weight=1.0, return_outputs=False)
        assistant_len = total_seq_len - assistant_start
        structural_count = int(assistant_len * 0.7, return_outputs=False)
        
        # First 70% of assistant tokens are structural
        sample_weights[i, assistant_start:assistant_start + structural_count] = 0.1
        # Last 30% are semantic (weight=1.0, already set, return_outputs=False)
    
    print(f"   Weight distribution in assistant response:", return_outputs=False)
    print(f"   - Structural (70%): weight = 0.1", return_outputs=False)
    print(f"   - Semantic (30%): weight = 1.0", return_outputs=False)
    
    inputs_weighted = {
        "input_ids": input_ids.clone(),
        "labels": labels.clone(),
        "sample_weights": sample_weights
    }
    
    weighted_loss = trainer.compute_loss(model, inputs_weighted, return_outputs=False, return_outputs=False)
    print(f"   Weighted Loss: {weighted_loss.item():.4f}", return_outputs=False)
    
    # ============= COMPARISON =============
    print(f"\n📈 Comparison:", return_outputs=False)
    print(f"   Standard Loss: {standard_loss.item():.4f}", return_outputs=False)
    print(f"   Weighted Loss: {weighted_loss.item():.4f}", return_outputs=False)
    ratio = weighted_loss.item() / standard_loss.item(, return_outputs=False)
    print(f"   Ratio (weighted/standard): {ratio:.2f}x", return_outputs=False)
    
    if ratio > 1.5:
        print(f"\n❌ ERROR: Weighted loss is {ratio:.2f}x higher than standard!", return_outputs=False)
        print(f"   This shouldn't happen - weighted loss should be LOWER", return_outputs=False)
        print(f"   (because we're reducing weight on 70% of tokens)", return_outputs=False)
    elif ratio < 0.2:
        print(f"\n❌ ERROR: Weighted loss is {ratio:.2f}x lower than standard!", return_outputs=False)
        print(f"   This is too low - might indicate a bug", return_outputs=False)
    else:
        print(f"\n✅ GOOD: Ratio {ratio:.2f}x is reasonable", return_outputs=False)
        print(f"   Weighted loss is lower as expected (down-weighting structural tokens)", return_outputs=False)
    
    # ============= TEST 3: Check if prompt weights matter =============
    print(f"\n🔍 TEST 3: Do prompt weights matter?", return_outputs=False)
    
    # Try with prompt weights set to 1.0 (incorrect but might be what's happening, return_outputs=False)
    sample_weights_wrong = sample_weights.clone(, return_outputs=False)
    sample_weights_wrong[:, :assistant_start] = 1.0  # Set prompt weights to 1.0 (wrong!, return_outputs=False)
    
    inputs_wrong = {
        "input_ids": input_ids.clone(),
        "labels": labels.clone(),
        "sample_weights": sample_weights_wrong
    }
    
    weighted_loss_wrong = trainer.compute_loss(model, inputs_wrong, return_outputs=False, return_outputs=False)
    print(f"   Weighted loss (prompt weights=1.0): {weighted_loss_wrong.item():.4f}", return_outputs=False)
    print(f"   Weighted loss (prompt weights=0.0): {weighted_loss.item():.4f}", return_outputs=False)
    print(f"   Difference: {abs(weighted_loss_wrong.item() - weighted_loss.item()):.6f}", return_outputs=False)
    
    if abs(weighted_loss_wrong.item() - weighted_loss.item()) < 0.001:
        print(f"   ✅ Prompt weights don't matter (as expected, they're masked to -100)", return_outputs=False)
    else:
        print(f"   ❌ Prompt weights DO matter! This shouldn't happen!", return_outputs=False)
    
    print("\n" + "="*80, return_outputs=False)


def test_why_10x_higher():
    """Try to reproduce the '10x higher' issue."""
    print("\n" + "="*80, return_outputs=False)
    print("TEST: Why Might Weighted Loss Be 10x Higher?", return_outputs=False)
    print("="*80, return_outputs=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", return_outputs=False)
    model = TinyModel(vocab_size=100).to(device, return_outputs=False)
    trainer = WeightedLossTrainer(model=model, args=None, return_outputs=False)
    
    batch_size = 2
    seq_len = 1000
    assistant_start = 700
    
    input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device, return_outputs=False)
    labels = torch.randint(0, 100, (batch_size, seq_len), device=device, return_outputs=False)
    labels[:, :assistant_start] = -100
    
    # Create weights with VERY LOW average weight
    sample_weights = torch.full((batch_size, seq_len), 0.01, device=device)  # 1% weight everywhere!
    
    inputs_weighted = {
        "input_ids": input_ids.clone(),
        "labels": labels.clone(),
        "sample_weights": sample_weights
    }
    
    inputs_standard = {
        "input_ids": input_ids.clone(),
        "labels": labels.clone(, return_outputs=False)
    }
    
    weighted_loss = trainer.compute_loss(model, inputs_weighted, return_outputs=False, return_outputs=False)
    standard_loss = trainer.compute_loss(model, inputs_standard, return_outputs=False, return_outputs=False)
    
    print(f"\n🧪 Extreme Case: All weights = 0.01", return_outputs=False)
    print(f"   Standard Loss: {standard_loss.item():.4f}", return_outputs=False)
    print(f"   Weighted Loss: {weighted_loss.item():.4f}", return_outputs=False)
    print(f"   Ratio: {weighted_loss.item() / standard_loss.item():.2f}x", return_outputs=False)
    print(f"   Expected: ~0.01x (1% of standard)", return_outputs=False)
    
    expected = standard_loss.item() * 0.01
    if abs(weighted_loss.item() - expected) / expected < 0.1:
        print(f"   ✅ Matches expected {expected:.4f} (within 10%)", return_outputs=False)
    else:
        print(f"   ❌ Doesn't match expected {expected:.4f}", return_outputs=False)
        print(f"   This might explain the 10x issue!", return_outputs=False)
    
    print("\n" + "="*80, return_outputs=False)


if __name__ == "__main__":
    test_realistic_scenario(, return_outputs=False)
    test_why_10x_higher(, return_outputs=False)
    print("\n🎯 Summary:", return_outputs=False)
    print("   If you see weighted loss 10x HIGHER than standard, this is unexpected.", return_outputs=False)
    print("   The tests above should help diagnose the issue.", return_outputs=False)
    print("   Check the actual weight values being passed to the trainer.\n", return_outputs=False)
