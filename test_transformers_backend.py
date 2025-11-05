#!/usr/bin/env python3
"""Test the Transformers backend with a small model."""

import json
import os
import tempfile
from pathlib import Path

from model_garden.backends import get_backend


def test_transformers_backend():
    """Test that the Transformers backend can load, train, and save a model."""
    print("🧪 Testing Transformers backend...")
    
    # Create a temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a small test dataset
        dataset_path = temp_path / "test_dataset.jsonl"
        with open(dataset_path, "w") as f:
            for i in range(10):
                example = {
                    "instruction": f"Test instruction {i}",
                    "input": "",
                    "output": f"Test output {i}",
                }
                f.write(json.dumps(example) + "\n")
        
        print(f"✓ Created test dataset: {dataset_path}")
        
        # Get the Transformers backend
        backend = get_backend("transformers")
        print(f"✓ Retrieved backend: {backend.name}")
        
        # Create a text trainer with a very small model
        # Using TinyLlama as it's small and fast to test
        trainer = backend.create_text_trainer(
            base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_seq_length=512,  # Small for faster testing
            load_in_4bit=True,  # Use 4-bit to save memory
        )
        print("✓ Created text trainer")
        
        # Load the model
        trainer.load_model()
        print("✓ Model loaded")
        
        # Prepare for training with minimal LoRA config
        trainer.prepare_for_training(
            r=8,  # Small rank for faster testing
            lora_alpha=16,
            lora_dropout=0.0,
            use_gradient_checkpointing=False,  # Disable for faster testing
        )
        print("✓ LoRA adapters configured")
        
        # Load and format the dataset
        dataset = trainer.load_dataset_from_file(str(dataset_path))
        dataset = trainer.format_dataset(dataset)
        print(f"✓ Dataset loaded and formatted: {len(dataset)} examples")
        
        # Train for just a few steps to verify it works
        output_dir = temp_path / "output"
        trainer.train(
            dataset=dataset,
            output_dir=str(output_dir),
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            max_steps=3,  # Just 3 steps to verify training works
            logging_steps=1,
            save_steps=3,
            enable_carbon_tracking=False,  # Disable for testing
        )
        print("✓ Training completed")
        
        # Verify the model was saved
        assert output_dir.exists(), "Output directory not created"
        assert (output_dir / "adapter_config.json").exists(), "LoRA adapter config not saved"
        print("✓ Model saved successfully")
        
        # Test saving as merged model
        merged_dir = temp_path / "merged"
        trainer.save_model(
            output_dir=str(merged_dir),
            save_method="merged_16bit",
        )
        assert merged_dir.exists(), "Merged output directory not created"
        assert (merged_dir / "config.json").exists(), "Merged model config not saved"
        print("✓ Merged model saved successfully")
        
        print("\n✅ All tests passed! Transformers backend is working correctly.")
        return True


if __name__ == "__main__":
    try:
        test_transformers_backend()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
