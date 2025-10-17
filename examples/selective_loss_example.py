#!/usr/bin/env python3
"""
Example: Training with Selective Loss for Form Extraction

This script demonstrates how to use selective loss for a structured output task
(form extraction from images).
"""

import json
from pathlib import Path

# Example dataset entry for form extraction
example_data = {
    "messages": [
        {
            "role": "system",
            "content": "You are an expert at extracting structured data from documents."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "data:image/jpeg;base64,..."  # Base64 encoded image
                },
                {
                    "type": "text",
                    "text": "Extract all fields from this vehicle registration form."
                }
            ]
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "Marque": {"contents": "TOYOTA", "confidence_score": 0.98},
                "Modele": {"contents": "COROLLA", "confidence_score": 0.95},
                "Immatriculation": {"contents": "AB-123-CD", "confidence_score": 0.99},
                "Date": {"contents": "2024-01-15", "confidence_score": 0.92}
            })
        }
    ]
}

print("📋 Example Training with Selective Loss for Form Extraction\n")

# Show token distribution
assistant_content = example_data["messages"][-1]["content"]
total_chars = len(assistant_content)

# Count structural tokens
structural_chars = sum(1 for c in assistant_content if c in '{},[]:" \n\t')
semantic_chars = total_chars - structural_chars

print(f"Token Distribution in Response:")
print(f"  Total characters: {total_chars}")
print(f"  Structural (will be masked): {structural_chars} ({100*structural_chars/total_chars:.1f}%)")
print(f"  Semantic (will be trained): {semantic_chars} ({100*semantic_chars/total_chars:.1f}%)")
print()

# Show what gets masked
print("Response breakdown:")
print("  Structure: { } [ ] : , \" and whitespace")
print("  Schema keys: 'Marque', 'Modele', 'contents', 'confidence_score'")
print("  Semantic content: 'TOYOTA', 'COROLLA', 'AB-123-CD', etc.")
print()

# Training command examples
print("=" * 70)
print("TRAINING COMMANDS")
print("=" * 70)
print()

print("1️⃣ CONSERVATIVE MODE (Recommended for first try)")
print("-" * 70)
print("""
uv run model-garden train-vision \\
  --base-model Qwen/Qwen2.5-VL-3B-Instruct \\
  --dataset storage/datasets/openai_finetune_vision_inline_20251017_114023.jsonl \\
  --output-dir ./models/form-extractor-conservative \\
  --epochs 3 \\
  --batch-size 1 \\
  --gradient-accumulation-steps 8 \\
  --learning-rate 2e-5 \\
  --selective-loss \\
  --selective-loss-level conservative \\
  --selective-loss-verbose
""")
print("What this does:")
print("  ✓ Masks: { } [ ] : , and whitespace")
print("  ✗ Trains: Field names, values, null, true, false")
print("  📊 Expected masking: ~17% of tokens")
print()

print("2️⃣ MODERATE MODE (Masks null too)")
print("-" * 70)
print("""
uv run model-garden train-vision \\
  --base-model Qwen/Qwen2.5-VL-3B-Instruct \\
  --dataset storage/datasets/openai_finetune_vision_inline_20251017_114023.jsonl \\
  --output-dir ./models/form-extractor-moderate \\
  --epochs 3 \\
  --selective-loss \\
  --selective-loss-level moderate \\
  --selective-loss-verbose
""")
print("What this does:")
print("  ✓ Masks: Conservative + 'null' keyword")
print("  ✗ Trains: Field names, values, true, false")
print("  📊 Expected masking: ~18-20% of tokens")
print()

print("3️⃣ AGGRESSIVE MODE (Maximum semantic focus)")
print("-" * 70)
print("""
uv run model-garden train-vision \\
  --base-model Qwen/Qwen2.5-VL-3B-Instruct \\
  --dataset storage/datasets/openai_finetune_vision_inline_20251017_114023.jsonl \\
  --output-dir ./models/form-extractor-aggressive \\
  --epochs 3 \\
  --selective-loss \\
  --selective-loss-level aggressive \\
  --selective-loss-schema-keys "Marque,Modele,Immatriculation,Date,contents,confidence_score,bounding_box" \\
  --selective-loss-verbose
""")
print("What this does:")
print("  ✓ Masks: Moderate + specified field names")
print("  ✗ Trains: Only the actual values (TOYOTA, COROLLA, etc.)")
print("  📊 Expected masking: ~35-45% of tokens")
print("  ⚠️  WARNING: Don't mask fields that carry meaning!")
print()

print("4️⃣ BASELINE (No selective loss - for comparison)")
print("-" * 70)
print("""
uv run model-garden train-vision \\
  --base-model Qwen/Qwen2.5-VL-3B-Instruct \\
  --dataset storage/datasets/openai_finetune_vision_inline_20251017_114023.jsonl \\
  --output-dir ./models/form-extractor-baseline \\
  --epochs 3
""")
print("What this does:")
print("  ✓ Standard training on ALL tokens")
print("  📊 Use this as baseline to compare improvements")
print()

# Python API example
print("=" * 70)
print("PYTHON API EXAMPLE")
print("=" * 70)
print("""
from model_garden.vision_training import VisionLanguageTrainer

# Initialize trainer
trainer = VisionLanguageTrainer(
    base_model="Qwen/Qwen2.5-VL-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True
)

# Load model and prepare for training
trainer.load_model()
trainer.prepare_for_training(r=16, lora_alpha=16)

# Load and format dataset
dataset = trainer.load_dataset(
    "storage/datasets/openai_finetune_vision_inline_20251017_114023.jsonl",
    from_hub=False
)
dataset = trainer.format_dataset(dataset)

# Train with selective loss
trainer.train(
    dataset=dataset,
    output_dir="./models/form-extractor",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    # Enable selective loss
    selective_loss=True,
    selective_loss_level="conservative",
    selective_loss_schema_keys=None,
    selective_loss_verbose=True
)

# Save model
trainer.save_model("./models/form-extractor", save_method="merged_16bit")
""")

print()
print("=" * 70)
print("TIPS FOR SUCCESS")
print("=" * 70)
print("""
1. START CONSERVATIVE
   - Use conservative mode first
   - Monitor masking percentage (should be 15-25%)
   - Compare with baseline (no selective loss)

2. ANALYZE YOUR DATA
   - Run: uv run python test_selective_loss.py
   - Check token distribution in your dataset
   - Adjust masking level accordingly

3. TUNE SCHEMA KEYS CAREFULLY
   - Only mask truly predictable field names
   - Don't mask fields that carry semantic meaning
   - Example: "contents" is safe, "type" might not be

4. MONITOR TRAINING
   - Use --selective-loss-verbose
   - Watch for masking statistics
   - Validate on held-out test set

5. COMPARE RESULTS
   - Train both with and without selective loss
   - Measure actual task performance (F1, accuracy, etc.)
   - Selective loss isn't always better - test it!
""")

print()
print("🚀 Ready to try it? Run one of the commands above!")
print("📚 See docs/15-selective-loss-user-guide.md for full documentation")
