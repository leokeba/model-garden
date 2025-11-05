"""Test weighted masking strategy for selective loss.

This script demonstrates the new weighted masking feature that applies
soft constraints instead of hard masking (binary on/off).
"""

import torch
from transformers import AutoProcessor
from model_garden.selective_loss import create_selective_loss_collator
from rich.console import Console

console = Console()


def create_mock_model():
    """Create a minimal mock model for testing."""
    class MockEmbeddings:
        def __init__(self):
            self.weight = type('Weight', (), {'dtype': torch.float32})()
    
    class MockModel:
        def __init__(self):
            self.device = "cpu"
            self.config = type('Config', (), {'model_type': 'qwen2_vl'})()
            self._embeddings = MockEmbeddings()
        
        def get_input_embeddings(self):
            return self._embeddings
    
    return MockModel()


def create_mock_batch():
    """Create a mock batch similar to what Unsloth would produce."""
    # Simulate a batch with JSON content
    # Token IDs: [user prompt tokens] + [assistant JSON tokens]
    # For simplicity, using made-up token IDs
    
    batch = {
        "input_ids": torch.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Mock input IDs
        ]),
        "labels": torch.tensor([
            [-100, -100, -100, 50, 51, 52, 53, 54, 55, 56]  # First 3 are prompt (masked), rest is response
        ])
    }
    
    return batch


def test_weighted_vs_binary_masking():
    """Compare weighted masking to binary masking."""
    console.print("\n[bold cyan]Testing Weighted Masking Initialization[/bold cyan]\n")
    
    # Load processor (using Qwen2.5-VL as example)
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        console.print(f"[yellow]Could not load processor: {e}[/yellow]")
        console.print("[yellow]Creating mock processor instead[/yellow]")
        
        # Create a minimal mock processor
        class MockTokenizer:
            def decode(self, token_ids, skip_special_tokens=False):
                # Simulate JSON structure tokens
                token_map = {
                    50: '{', 51: '"', 52: 'name', 53: '"', 54: ':', 
                    55: '"', 56: 'value', 57: '"', 58: '}', 59: ','
                }
                if isinstance(token_ids, list):
                    return ''.join(token_map.get(tid, str(tid)) for tid in token_ids)
                else:
                    return token_map.get(token_ids.item() if hasattr(token_ids, 'item') else token_ids, str(token_ids))
        
        class MockProcessor:
            def __init__(self):
                self.tokenizer = MockTokenizer()
        
        processor = MockProcessor()
    
    model = create_mock_model()
    
    # Test 1: Binary masking (alternating strategy with masking ON)
    console.print("[bold]1. Binary Masking (Alternating Strategy)[/bold]")
    try:
        binary_collator = create_selective_loss_collator(
            model=model,
            processor=processor,
            mask_level="conservative",
            masking_strategy="alternating",
            mask_every_n_steps=100,
            mask_for_n_steps=50,
            verbose=False
        )
        console.print("  [green]✓ Binary collator created successfully[/green]")
        console.print(f"  Strategy: {binary_collator.masking_strategy}")
        console.print(f"  Mask structural: {binary_collator.mask_structural}")
        console.print(f"  Cycle: {binary_collator.mask_every_n_steps} steps")
        console.print(f"  Masking duration: {binary_collator.mask_for_n_steps} steps ON")
    except Exception as e:
        console.print(f"  [red]✗ Failed: {e}[/red]")
    
    # Test 2: Weighted masking
    console.print("\n[bold]2. Weighted Masking (Soft Constraints)[/bold]")
    try:
        weighted_collator = create_selective_loss_collator(
            model=model,
            processor=processor,
            mask_level="conservative",
            masking_strategy="weighted",
            structural_weight=0.1,
            verbose=True  # Show verbose output for this one
        )
        console.print("  [green]✓ Weighted collator created successfully[/green]")
        console.print(f"  Strategy: {weighted_collator.masking_strategy}")
        console.print(f"  Structural weight: {weighted_collator.structural_weight}")
        console.print(f"  Mask structural: {weighted_collator.mask_structural}")
    except Exception as e:
        console.print(f"  [red]✗ Failed: {e}[/red]")
    
    # Test 3: Epoch-based masking
    console.print("\n[bold]3. Epoch-based Masking (Binary, Delayed Start)[/bold]")
    try:
        epoch_collator = create_selective_loss_collator(
            model=model,
            processor=processor,
            mask_level="conservative",
            masking_strategy="epoch_based",
            masking_start_epoch=0.5,
            verbose=False
        )
        console.print("  [green]✓ Epoch-based collator created successfully[/green]")
        console.print(f"  Strategy: {epoch_collator.masking_strategy}")
        console.print(f"  Start epoch: {epoch_collator.masking_start_epoch}")
    except Exception as e:
        console.print(f"  [red]✗ Failed: {e}[/red]")
    
    # Test 4: Different weight values
    console.print("\n[bold]4. Testing Different Weight Values[/bold]")
    for weight in [0.0, 0.05, 0.1, 0.2, 0.5]:
        try:
            collator = create_selective_loss_collator(
                model=model,
                processor=processor,
                mask_level="conservative",
                masking_strategy="weighted",
                structural_weight=weight,
                verbose=False
            )
            console.print(f"  [green]✓[/green] structural_weight={weight}: Structural tokens get {weight*100:.0f}% loss weight")
        except Exception as e:
            console.print(f"  [red]✗[/red] structural_weight={weight}: Failed - {e}")


def test_weighted_masking_with_custom_loss():
    """Demonstrate how to use weighted masking with a custom loss function."""
    console.print("\n[bold cyan]Custom Loss Function for Weighted Masking[/bold cyan]\n")
    
    console.print("[bold]Example Trainer.compute_loss override:[/bold]")
    console.print("""
    from transformers import Trainer
    import torch.nn.functional as F
    
    class WeightedLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            sample_weights = inputs.pop("sample_weights", None)
            
            outputs = model(**inputs)
            logits = outputs.get("logits")
            
            # Compute per-token loss (no reduction)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            if sample_weights is not None:
                # Apply per-token weights
                weights_flat = sample_weights.view(-1)
                loss = loss * weights_flat
            
            # Average only over valid tokens
            valid_mask = (labels != -100).view(-1)
            if sample_weights is not None:
                # Weight the denominator too for proper averaging
                loss = loss[valid_mask].sum() / weights_flat[valid_mask].sum()
            else:
                loss = loss[valid_mask].mean()
            
            return (loss, outputs) if return_outputs else loss
    """)
    
    console.print("\n[bold]Usage:[/bold]")
    console.print("""
    # Create weighted collator
    collator = create_selective_loss_collator(
        model=model,
        processor=processor,
        mask_level="aggressive",
        masking_strategy="weighted",
        structural_weight=0.1,  # Structural tokens contribute 10% to loss
        dataset=train_dataset,
        verbose=True
    )
    
    # Use custom trainer
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        # ... other args
    )
    
    trainer.train()
    """)


def test_strategy_comparison():
    """Compare all three masking strategies."""
    console.print("\n[bold cyan]Comparison of All Three Masking Strategies[/bold cyan]\n")
    
    strategies = [
        {
            "name": "Epoch-based",
            "description": "Hard masking, enabled after epoch threshold",
            "params": {
                "masking_strategy": "epoch_based",
                "masking_start_epoch": 0.5
            },
            "pros": "Simple, clear learning phases",
            "cons": "Binary switch, no gradual transition"
        },
        {
            "name": "Alternating",
            "description": "Hard masking, cycles ON/OFF during training",
            "params": {
                "masking_strategy": "alternating",
                "mask_every_n_steps": 100,
                "mask_for_n_steps": 50
            },
            "pros": "Learns both structure and semantics",
            "cons": "More complex, needs tuning"
        },
        {
            "name": "Weighted",
            "description": "Soft masking, reduced weights for structural tokens",
            "params": {
                "masking_strategy": "weighted",
                "structural_weight": 0.1
            },
            "pros": "Soft constraints, continuous signal",
            "cons": "Requires custom trainer, experimental"
        }
    ]
    
    for strategy in strategies:
        console.print(f"[bold]{strategy['name']} Strategy[/bold]")
        console.print(f"  Description: {strategy['description']}")
        console.print(f"  Parameters: {strategy['params']}")
        console.print(f"  [green]Pros:[/green] {strategy['pros']}")
        console.print(f"  [yellow]Cons:[/yellow] {strategy['cons']}")
        console.print()


if __name__ == "__main__":
    console.print("[bold magenta]Weighted Masking Test Suite[/bold magenta]")
    console.print("=" * 60)
    
    test_weighted_vs_binary_masking()
    test_weighted_masking_with_custom_loss()
    test_strategy_comparison()
    
    console.print("\n[bold green]✓ All tests completed![/bold green]")
    console.print("\n[bold cyan]Next Steps:[/bold cyan]")
    console.print("1. Implement WeightedLossTrainer with custom compute_loss")
    console.print("2. Train with weighted masking: structural_weight=0.1")
    console.print("3. Compare results to alternating strategy")
    console.print("4. Experiment with different weights (0.05, 0.2, 0.5)")
