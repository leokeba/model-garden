#!/usr/bin/env python3
"""Test script for HuggingFace Hub dataset with base64 images."""

from model_garden.vision_training import VisionLanguageTrainer

def test_hub_dataset():
    """Test loading and formatting HuggingFace Hub dataset."""
    print("=== Testing HuggingFace Hub Dataset Integration ===\n")
    
    # Initialize trainer
    trainer = VisionLanguageTrainer(
        base_model="Qwen/Qwen2.5-VL-3B-Instruct",
        max_seq_length=2048,
    )
    
    # Test 1: Load from HuggingFace Hub
    print("Test 1: Loading dataset from HuggingFace Hub...")
    dataset = trainer.load_dataset(
        dataset_path="Barth371/train_pop_valet_no_wrong_doc",
        from_hub=True,
        split="train[:5]"  # Just 5 examples for testing
    )
    print(f"✓ Loaded {len(dataset)} examples")
    print(f"  Columns: {dataset.column_names}\n")
    
    # Test 2: Format dataset (converts base64 to PIL Images)
    print("Test 2: Formatting dataset (base64 → PIL Images)...")
    formatted = trainer.format_dataset(dataset)
    print(f"✓ Formatted {len(formatted)} examples\n")
    
    # Test 3: Verify image conversion
    print("Test 3: Verifying image conversion...")
    from PIL import Image
    
    for i, example in enumerate(formatted[:2]):
        print(f"\nExample {i+1}:")
        messages = example["messages"]
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            print(f"  Role: {role}")
            
            for item in content:
                if item["type"] == "image":
                    img = item["image"]
                    if isinstance(img, Image.Image):
                        print(f"    ✓ Image: {img.format} {img.size} {img.mode}")
                    else:
                        print(f"    ✗ Image type error: {type(img)}")
                elif item["type"] == "text":
                    text = item["text"]
                    preview = text[:80] + "..." if len(text) > 80 else text
                    print(f"    Text: {preview}")
    
    print("\n✅ All tests passed!\n")
    print("Next steps:")
    print("  1. Run full training with: uv run model-garden train-vision --dataset Barth371/train_pop_valet_no_wrong_doc --from-hub ...")
    print("  2. Or use the web UI to start a training job with HuggingFace Hub datasets")

if __name__ == "__main__":
    test_hub_dataset()
