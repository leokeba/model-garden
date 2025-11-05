"""Test weighted loss with prompt masking (train_on_responses_only).

This test simulates the exact scenario that occurs in real training:
1. Prompts are masked to -100 (train_on_responses_only=True)
2. Weights are set for all tokens (including prompts)
3. We verify the loss is computed correctly
"""

import torch
from model_garden.weighted_loss_trainer import WeightedLossTrainer


class TinyModel(torch.nn.Module):
    def __init__(self, vocab_size=100, hidden_size=32):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids, **kwargs):
        hidden = self.embedding(input_ids)
        logits = self.lm_head(hidden)
        return {"logits": logits}


def test_with_prompt_masking():
    """Test weighted loss with realistic prompt masking."""
    print("\n" + "="*80)
    print("TEST: Weighted Loss with Prompt Masking (Real Scenario)")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyModel(vocab_size=100).to(device)
    trainer = WeightedLossTrainer(model=model, args=None, verbose_loss=True)
    
    # Realistic scenario:
    # Total: 1000 tokens
    # - Tokens 0-700: Vision + prompt (masked to -100)
    # - Tokens 700-1000: Assistant response (300 tokens)
    #   - 70% structural (210 tokens, weight=0.1)
    #   - 30% semantic (90 tokens, weight=1.0)
    
    batch_size = 2
    total_len = 1000
    response_start = 700
    response_len = 300
    
    input_ids = torch.randint(0, 100, (batch_size, total_len), device=device)
    labels = torch.randint(0, 100, (batch_size, total_len), device=device)
    
    # Mask prompt tokens (train_on_responses_only behavior)
    labels[:, :response_start] = -100
    
    print(f"\n📊 Sequence Structure:")
    print(f"   Total: {total_len} tokens")
    print(f"   Prompt (masked to -100): {response_start} tokens")
    print(f"   Response (valid): {response_len} tokens")
    
    # ============= TEST 1: Standard Training =============
    print(f"\n🔵 TEST 1: Standard Training (no weighted masking)")
    
    inputs_standard = {
        "input_ids": input_ids.clone(),
        "labels": labels.clone()
    }
    
    standard_loss = trainer.compute_loss(model, inputs_standard, return_outputs=False)
    print(f"   Loss: {standard_loss.item():.4f}")
    
    # ============= TEST 2: Weighted with WRONG prompt weights =============
    print(f"\n🟡 TEST 2: Weighted Masking (WRONG: prompt weights=1.0)")
    print(f"   This simulates the BUG where we set weights=1.0 for ALL tokens")
    
    weights_wrong = torch.ones((batch_size, total_len), device=device)
    # Prompt has weight=1.0 (wrong! but shouldn't matter because labels=-100)
    # Response: 70% structural (weight=0.1), 30% semantic (weight=1.0)
    for i in range(batch_size):
        structural_count = int(response_len * 0.7)
        weights_wrong[i, response_start:response_start + structural_count] = 0.1
    
    inputs_wrong = {
        "input_ids": input_ids.clone(),
        "labels": labels.clone(),
        "sample_weights": weights_wrong
    }
    
    loss_wrong = trainer.compute_loss(model, inputs_wrong, return_outputs=False)
    print(f"   Loss: {loss_wrong.item():.4f}")
    print(f"   Ratio vs standard: {loss_wrong.item() / standard_loss.item():.2f}x")
    
    # ============= TEST 3: Weighted with CORRECT prompt weights =============
    print(f"\n🟢 TEST 3: Weighted Masking (CORRECT: prompt weights=0.0)")
    print(f"   This is the FIXED version where prompt weights are set to 0.0")
    
    weights_correct = torch.zeros((batch_size, total_len), device=device)
    # Prompt has weight=0.0 (correct! cleaner even though doesn't matter)
    # Response: 70% structural (weight=0.1), 30% semantic (weight=1.0)
    weights_correct[:, response_start:] = 1.0  # All response tokens start at 1.0
    for i in range(batch_size):
        structural_count = int(response_len * 0.7)
        weights_correct[i, response_start:response_start + structural_count] = 0.1
    
    inputs_correct = {
        "input_ids": input_ids.clone(),
        "labels": labels.clone(),
        "sample_weights": weights_correct
    }
    
    loss_correct = trainer.compute_loss(model, inputs_correct, return_outputs=False)
    print(f"   Loss: {loss_correct.item():.4f}")
    print(f"   Ratio vs standard: {loss_correct.item() / standard_loss.item():.2f}x")
    
    # ============= COMPARISON =============
    print(f"\n📈 Results:")
    print(f"   Standard loss: {standard_loss.item():.4f}")
    print(f"   Weighted (wrong prompt weights): {loss_wrong.item():.4f} ({loss_wrong.item() / standard_loss.item():.2f}x)")
    print(f"   Weighted (correct prompt weights): {loss_correct.item():.4f} ({loss_correct.item() / standard_loss.item():.2f}x)")
    
    # Verify they're the same (prompt weights shouldn't matter)
    diff = abs(loss_wrong.item() - loss_correct.item())
    print(f"\n   Difference between wrong/correct: {diff:.6f}")
    
    if diff < 0.001:
        print(f"   ✅ Prompt weights don't affect loss (as expected)")
    else:
        print(f"   ❌ Prompt weights DO affect loss (unexpected!)")
    
    # Verify weighted loss is lower than standard
    ratio = loss_correct.item() / standard_loss.item()
    if ratio < 0.2 or ratio > 0.8:
        print(f"\n   ⚠️  Ratio {ratio:.2f}x seems unusual")
        print(f"       Expected: ~0.37x (with 70% at 0.1, 30% at 1.0)")
    else:
        print(f"\n   ✅ Ratio {ratio:.2f}x is reasonable")
    
    # ============= TEST 4: What if we INVERTED weights by mistake? =============
    print(f"\n🔴 TEST 4: What if weights are INVERTED? (Bug Hypothesis)")
    print(f"   Structural=10.0, Semantic=1.0 (instead of 0.1 and 1.0)")
    
    weights_inverted = torch.zeros((batch_size, total_len), device=device)
    weights_inverted[:, response_start:] = 1.0
    for i in range(batch_size):
        structural_count = int(response_len * 0.7)
        weights_inverted[i, response_start:response_start + structural_count] = 10.0  # INVERTED!
    
    inputs_inverted = {
        "input_ids": input_ids.clone(),
        "labels": labels.clone(),
        "sample_weights": weights_inverted
    }
    
    loss_inverted = trainer.compute_loss(model, inputs_inverted, return_outputs=False)
    print(f"   Loss: {loss_inverted.item():.4f}")
    print(f"   Ratio vs standard: {loss_inverted.item() / standard_loss.item():.2f}x")
    
    if loss_inverted.item() / standard_loss.item() > 2.0:
        print(f"\n   🎯 THIS MATCHES YOUR OBSERVED 10-15x HIGHER LOSS!")
        print(f"      The weights might be inverted or misconfigured!")
    
    print("\n" + "="*80)
    print("Summary:")
    print("  - Standard loss should be the baseline")
    print("  - Weighted loss should be LOWER (0.3-0.5x)")
    print("  - If weighted loss is HIGHER, weights are likely inverted/wrong")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_with_prompt_masking()
