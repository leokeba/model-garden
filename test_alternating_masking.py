"""Test the alternating masking strategy for selective loss.

This test verifies that the masking correctly alternates between ON and OFF
states based on the configured step intervals.
"""

import torch
from unittest.mock import MagicMock
from model_garden.selective_loss import SelectiveLossVisionCollator


def test_alternating_masking_pattern():
    """Test that masking alternates correctly based on step count."""
    
    # Mock model and processor
    model = MagicMock()
    processor = MagicMock()
    processor.tokenizer = MagicMock()
    
    # Create collator with alternating strategy
    # Cycle every 10 steps: ON for 5 steps, OFF for 5 steps
    collator = SelectiveLossVisionCollator(
        model=model,
        processor=processor,
        mask_structural_tokens=True,
        masking_strategy="alternating",
        mask_every_n_steps=10,
        mask_for_n_steps=5,
        verbose=True
    )
    
    # Test the masking pattern over 30 steps (3 full cycles)
    expected_pattern = [
        # Cycle 1: steps 0-9
        True, True, True, True, True,      # Steps 0-4: masking ON
        False, False, False, False, False, # Steps 5-9: masking OFF
        # Cycle 2: steps 10-19
        True, True, True, True, True,      # Steps 10-14: masking ON
        False, False, False, False, False, # Steps 15-19: masking OFF
        # Cycle 3: steps 20-29
        True, True, True, True, True,      # Steps 20-24: masking ON
        False, False, False, False, False, # Steps 25-29: masking OFF
    ]
    
    print("\n" + "="*60)
    print("Testing Alternating Masking Pattern")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Cycle length: 10 steps")
    print(f"  - Masking ON: 5 steps per cycle")
    print(f"  - Masking OFF: 5 steps per cycle")
    print(f"\nExpected pattern (30 steps, 3 cycles):")
    print(f"  Steps 0-4:   Masking ON  ✓")
    print(f"  Steps 5-9:   Masking OFF ✗")
    print(f"  Steps 10-14: Masking ON  ✓")
    print(f"  Steps 15-19: Masking OFF ✗")
    print(f"  Steps 20-24: Masking ON  ✓")
    print(f"  Steps 25-29: Masking OFF ✗")
    print("\nVerifying...\n")
    
    results = []
    for step in range(30):
        collator.current_step = step
        should_mask = collator._should_enable_masking()
        expected = expected_pattern[step]
        
        results.append((step, should_mask, expected))
        
        # Print status for each step
        status = "✓ PASS" if should_mask == expected else "✗ FAIL"
        mask_status = "ON " if should_mask else "OFF"
        cycle = step // 10
        cycle_pos = step % 10
        
        print(f"  Step {step:2d} (Cycle {cycle}, Position {cycle_pos}): "
              f"Masking {mask_status} - {status}")
        
        if should_mask != expected:
            print(f"    ERROR: Expected {expected}, got {should_mask}")
    
    # Check all results
    all_passed = all(actual == expected for _, actual, expected in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ All tests PASSED!")
        print("   Alternating masking pattern works correctly.")
    else:
        failed = [(s, a, e) for s, a, e in results if a != e]
        print(f"❌ {len(failed)} tests FAILED!")
        print("   Failed steps:", [s for s, _, _ in failed])
    print("="*60 + "\n")
    
    assert all_passed, "Masking pattern verification failed"


def test_epoch_based_masking():
    """Test epoch-based masking for comparison."""
    
    # Mock model and processor
    model = MagicMock()
    processor = MagicMock()
    processor.tokenizer = MagicMock()
    
    # Create collator with epoch-based strategy
    collator = SelectiveLossVisionCollator(
        model=model,
        processor=processor,
        mask_structural_tokens=True,
        masking_strategy="epoch_based",
        masking_start_epoch=1.0,  # Start after first epoch
        verbose=True
    )
    
    # Mock trainer state
    trainer = MagicMock()
    trainer.state = MagicMock()
    
    print("\n" + "="*60)
    print("Testing Epoch-Based Masking")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Start masking at epoch: 1.0")
    print("\nVerifying...\n")
    
    # Test before threshold
    trainer.state.epoch = 0.5
    collator.set_trainer(trainer)
    should_mask = collator._should_enable_masking()
    print(f"  Epoch 0.5: Masking {'ON' if should_mask else 'OFF'} - "
          f"{'✓ PASS' if not should_mask else '✗ FAIL'}")
    assert not should_mask, "Should not mask before threshold"
    
    # Test at threshold
    trainer.state.epoch = 1.0
    should_mask = collator._should_enable_masking()
    print(f"  Epoch 1.0: Masking {'ON' if should_mask else 'OFF'} - "
          f"{'✓ PASS' if should_mask else '✗ FAIL'}")
    assert should_mask, "Should mask at threshold"
    
    # Test after threshold
    trainer.state.epoch = 1.5
    should_mask = collator._should_enable_masking()
    print(f"  Epoch 1.5: Masking {'ON' if should_mask else 'OFF'} - "
          f"{'✓ PASS' if should_mask else '✗ FAIL'}")
    assert should_mask, "Should mask after threshold"
    
    print("\n✅ All epoch-based tests PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# Selective Loss Masking Strategy Tests")
    print("#"*60)
    
    try:
        test_alternating_masking_pattern()
        test_epoch_based_masking()
        
        print("\n" + "#"*60)
        print("# ✅ ALL TESTS PASSED!")
        print("#"*60 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
        raise
