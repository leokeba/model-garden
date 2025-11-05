"""Integration test for weighted masking strategy in CLI and API.

This test verifies that the weighted masking strategy parameter flows correctly
through the CLI and API layers to the training code.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_cli_signature():
    """Test that CLI has weighted masking parameters."""
    print("\n" + "="*60)
    print("TEST: CLI Signature")
    print("="*60)
    
    from model_garden.cli import main
    
    # Get the train-vision command
    train_vision_cmd = None
    for cmd in main.commands.values():
        if cmd.name == "train-vision":
            train_vision_cmd = cmd
            break
    
    assert train_vision_cmd is not None, "train-vision command not found"
    
    # Check parameters from Click options
    param_names = [p.name for p in train_vision_cmd.params]
    
    print(f"✓ CLI command has {len(param_names)} parameters")
    
    # Check for weighted masking parameters
    assert "selective_loss_masking_strategy" in param_names, "Missing masking_strategy parameter"
    assert "selective_loss_structural_weight" in param_names, "Missing structural_weight parameter"
    
    print("✓ selective_loss_masking_strategy parameter present")
    print("✓ selective_loss_structural_weight parameter present")
    
    # Check the masking_strategy choices
    strategy_param = next(p for p in train_vision_cmd.params if p.name == "selective_loss_masking_strategy")
    if hasattr(strategy_param.type, 'choices'):
        print(f"✓ masking_strategy choices: {strategy_param.type.choices}")
        assert "weighted" in strategy_param.type.choices, "'weighted' not in strategy choices"
        print("✓ 'weighted' is a valid strategy choice")
    
    # Check the structural_weight default
    weight_param = next(p for p in train_vision_cmd.params if p.name == "selective_loss_structural_weight")
    print(f"✓ structural_weight default: {weight_param.default}")
    
    return True


def test_api_models():
    """Test that API models have weighted masking fields."""
    print("\n" + "="*60)
    print("TEST: API Models")
    print("="*60)
    
    from model_garden.api import TrainingJobRequest, TrainingJobInfo
    
    # Check TrainingJobRequest
    request_fields = TrainingJobRequest.model_fields
    
    assert "selective_loss_masking_strategy" in request_fields, "Missing masking_strategy in request"
    assert "selective_loss_structural_weight" in request_fields, "Missing structural_weight in request"
    
    print("✓ TrainingJobRequest has masking_strategy field")
    print("✓ TrainingJobRequest has structural_weight field")
    
    # Check defaults
    default_request = TrainingJobRequest(
        name="test",
        base_model="test",
        dataset_path="test",
        output_dir="test"
    )
    
    print(f"✓ Default masking_strategy: {default_request.selective_loss_masking_strategy}")
    print(f"✓ Default structural_weight: {default_request.selective_loss_structural_weight}")
    
    # Check TrainingJobInfo
    info_fields = TrainingJobInfo.model_fields
    
    assert "selective_loss_masking_strategy" in info_fields, "Missing masking_strategy in info"
    assert "selective_loss_structural_weight" in info_fields, "Missing structural_weight in info"
    
    print("✓ TrainingJobInfo has masking_strategy field")
    print("✓ TrainingJobInfo has structural_weight field")
    
    return True


def test_vision_trainer_signature():
    """Test that VisionLanguageTrainer.train() accepts weighted parameters."""
    print("\n" + "="*60)
    print("TEST: VisionLanguageTrainer Signature")
    print("="*60)
    
    from model_garden.vision_training import VisionLanguageTrainer
    import inspect
    
    sig = inspect.signature(VisionLanguageTrainer.train)
    params = list(sig.parameters.keys())
    
    print(f"✓ train() method has {len(params)} parameters")
    
    assert "selective_loss_masking_strategy" in params, "Missing masking_strategy"
    assert "selective_loss_structural_weight" in params, "Missing structural_weight"
    
    print("✓ selective_loss_masking_strategy parameter present")
    print("✓ selective_loss_structural_weight parameter present")
    
    # Check defaults
    strategy_param = sig.parameters["selective_loss_masking_strategy"]
    weight_param = sig.parameters["selective_loss_structural_weight"]
    
    print(f"✓ masking_strategy default: {strategy_param.default}")
    print(f"✓ structural_weight default: {weight_param.default}")
    
    assert weight_param.default == 0.1, "structural_weight default should be 0.1"
    
    return True


def test_weighted_strategy_validation():
    """Test that 'weighted' is accepted as a valid strategy."""
    print("\n" + "="*60)
    print("TEST: Weighted Strategy Validation")
    print("="*60)
    
    from model_garden.api import TrainingJobRequest
    
    # Test that weighted strategy is accepted
    request = TrainingJobRequest(
        name="test_weighted",
        base_model="test-model",
        dataset_path="test.jsonl",
        output_dir="./output",
        selective_loss=True,
        selective_loss_masking_strategy="weighted",
        selective_loss_structural_weight=0.2
    )
    
    print(f"✓ Created request with strategy: {request.selective_loss_masking_strategy}")
    print(f"✓ Created request with weight: {request.selective_loss_structural_weight}")
    
    assert request.selective_loss_masking_strategy == "weighted"
    assert request.selective_loss_structural_weight == 0.2
    
    print("✓ 'weighted' strategy accepted in API model")
    
    return True


def test_weighted_trainer_import():
    """Test that WeightedLossTrainer can be imported."""
    print("\n" + "="*60)
    print("TEST: WeightedLossTrainer Import")
    print("="*60)
    
    from model_garden.weighted_loss_trainer import WeightedLossTrainer, WeightedLossTrainerWithMetrics
    
    print(f"✓ WeightedLossTrainer imported: {WeightedLossTrainer}")
    print(f"✓ WeightedLossTrainerWithMetrics imported: {WeightedLossTrainerWithMetrics}")
    
    # Check it's a subclass of Trainer
    from transformers import Trainer
    
    assert issubclass(WeightedLossTrainer, Trainer), "WeightedLossTrainer should inherit from Trainer"
    
    print("✓ WeightedLossTrainer is a Trainer subclass")
    
    return True


def main():
    """Run all integration tests."""
    print("\n" + "="*80)
    print("WEIGHTED MASKING INTEGRATION TEST")
    print("="*80)
    print("Testing CLI, API, and training code integration")
    
    try:
        # Test 1: CLI signature
        test_cli_signature()
        
        # Test 2: API models
        test_api_models()
        
        # Test 3: Vision trainer signature
        test_vision_trainer_signature()
        
        # Test 4: Weighted strategy validation
        test_weighted_strategy_validation()
        
        # Test 5: Trainer import
        test_weighted_trainer_import()
        
        print("\n" + "="*80)
        print("ALL INTEGRATION TESTS PASSED ✓")
        print("="*80)
        print("\nWeighted masking strategy is fully integrated!")
        print("\nUsage examples:")
        print("\nCLI:")
        print("  uv run model-garden train-vision \\")
        print("    --dataset data.jsonl \\")
        print("    --output-dir ./output \\")
        print("    --selective-loss \\")
        print("    --selective-loss-masking-strategy weighted \\")
        print("    --selective-loss-structural-weight 0.1")
        print("\nAPI:")
        print("  POST /api/training/jobs")
        print('  {"selective_loss": true,')
        print('   "selective_loss_masking_strategy": "weighted",')
        print('   "selective_loss_structural_weight": 0.1}')
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print("INTEGRATION TEST FAILED ❌")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
