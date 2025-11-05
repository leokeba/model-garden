"""Check if model_accepts_loss_kwargs is set correctly."""

import os
os.environ["HF_HOME"] = "/root/model-garden/storage/cache"
os.environ["TRANSFORMERS_CACHE"] = "/root/model-garden/storage/cache/hub"
os.environ["HF_DATASETS_CACHE"] = "/root/model-garden/storage/cache/datasets"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

from unsloth import FastVisionModel
from transformers import TrainingArguments
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
import tempfile

print("Checking model_accepts_loss_kwargs...")

model, processor = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-2B-Instruct",
    load_in_4bit=True,
    max_seq_length=2048,
)

# Add LoRA
model = FastVisionModel.get_peft_model(
    model,
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj"],
)

# Check model properties
print(f"\n1. Model properties:")
print(f"   model_accepts_loss_kwargs: {getattr(model, 'model_accepts_loss_kwargs', 'NOT SET')}")
print(f"   Has config: {hasattr(model, 'config')}")

if hasattr(model, 'config'):
    print(f"   config.model_type: {getattr(model.config, 'model_type', 'NOT SET')}")
    
# Check base model
if hasattr(model, 'base_model'):
    base_model = model.base_model
    print(f"\n2. Base model properties:")
    print(f"   base_model type: {type(base_model)}")
    print(f"   base_model.model_accepts_loss_kwargs: {getattr(base_model, 'model_accepts_loss_kwargs', 'NOT SET')}")

# Check if there's a model.model
if hasattr(model, 'model'):
    inner_model = model.model  
    print(f"\n3. Inner model properties:")
    print(f"   inner_model type: {type(inner_model)}")
    print(f"   inner_model.model_accepts_loss_kwargs: {getattr(inner_model, 'model_accepts_loss_kwargs', 'NOT SET')}")

# Create a minimal trainer to see what it detects
with tempfile.TemporaryDirectory() as tmpdir:
    training_args = SFTConfig(
        output_dir=tmpdir,
        per_device_train_batch_size=1,
        max_steps=1,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    
    # Create dummy dataset
    dummy_data = [{"messages": [{"role": "user", "content": "test"}]}]
    
    try:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dummy_data,
            processing_class=processor,
        )
        
        print(f"\n4. Trainer properties:")
        print(f"   trainer.model_accepts_loss_kwargs: {trainer.model_accepts_loss_kwargs}")
        print(f"   trainer.compute_loss_func: {trainer.compute_loss_func}")
        
        if trainer.model_accepts_loss_kwargs:
            print("\n   ✓ num_items_in_batch WILL be calculated and passed to loss")
        else:
            print("\n   ❌ num_items_in_batch will NOT be calculated!")
            print("   This means loss uses reduction='mean' instead of 'sum'/num_items")
            
    except Exception as e:
        print(f"\n   Error creating trainer: {e}")

print("\n" + "=" * 80)
