"""Simulate actual training to see if there's a difference in how SFTTrainer handles train vs eval."""

import os
os.environ["HF_HOME"] = "/root/model-garden/storage/cache"
os.environ["TRANSFORMERS_CACHE"] = "/root/model-garden/storage/cache/hub"
os.environ["HF_DATASETS_CACHE"] = "/root/model-garden/storage/cache/datasets"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from PIL import Image
import torch
import tempfile

print("=" * 80)
print("SIMULATING ACTUAL SFTTRAINER BEHAVIOR")
print("=" * 80)

# Load model
model, processor = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-2B-Instruct",
    load_in_4bit=True,
    max_seq_length=2048,
)

# Add LoRA adapters (required for training)
model = FastVisionModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)

# Create identical train and eval samples (to isolate the issue)
def create_sample(color):
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": Image.new("RGB", (224, 224), color=color)},
                {"type": "text", "text": "What color?"}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": f"{color.capitalize()}."}
            ]}
        ]
    }

train_data = [create_sample("red"), create_sample("blue")] * 2  # 4 samples
eval_data = [create_sample("green"), create_sample("yellow")] * 2  # 4 different samples

# Create collator WITH prompt masking
print("\n1. Creating data collator with train_on_responses_only=True")
collator = UnslothVisionDataCollator(
    model, 
    processor,
    train_on_responses_only=True,
    instruction_part="<|im_start|>user",
    response_part="<|im_start|>assistant",
    force_match=False
)

# Create a minimal training config
with tempfile.TemporaryDirectory() as tmpdir:
    training_args = SFTConfig(
        output_dir=tmpdir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        max_steps=1,  # Just one step
        eval_strategy="steps",
        eval_steps=1,
        logging_steps=1,
        save_steps=999999,
        report_to="none",
        bf16=True,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    
    print("\n2. Creating SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=processor,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collator,
    )
    
    print("\n3. Manually testing data collator on train and eval samples...")
    
    # Process one train batch
    train_batch = collator(train_data[:2])
    train_batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in train_batch.items()}
    
    # Process one eval batch
    eval_batch = collator(eval_data[:2])
    eval_batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in eval_batch.items()}
    
    # Check masking
    def check_batch(batch, name):
        labels = batch['labels']
        total_tokens = labels.numel()
        non_masked = (labels != -100).sum().item()
        pct = 100 * non_masked / total_tokens
        print(f"\n   {name} batch:")
        print(f"     Total tokens: {total_tokens}")
        print(f"     Non-masked: {non_masked} ({pct:.1f}%)")
        
        # Compute loss manually
        model.eval()
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss.item()
            print(f"     Loss: {loss:.4f}")
        return loss
    
    train_loss = check_batch(train_batch, "TRAIN")
    eval_loss = check_batch(eval_batch, "EVAL")
    
    print(f"\n4. Direct comparison:")
    print(f"   Train loss: {train_loss:.4f}")
    print(f"   Eval loss:  {eval_loss:.4f}")
    print(f"   Ratio: {eval_loss/train_loss:.2f}x")
    
    # Now let's check what the trainer's compute_loss method does
    print("\n5. Testing SFTTrainer.compute_loss() directly...")
    
    # Set model to train mode
    model.train()
    train_loss_from_trainer = trainer.compute_loss(model, train_batch, return_outputs=False)
    print(f"   Train loss (via trainer.compute_loss): {train_loss_from_trainer.item():.4f}")
    
    # Set model to eval mode  
    model.eval()
    eval_loss_from_trainer = trainer.compute_loss(model, eval_batch, return_outputs=False)
    print(f"   Eval loss (via trainer.compute_loss):  {eval_loss_from_trainer.item():.4f}")
    
    print(f"\n   Ratio: {eval_loss_from_trainer.item()/train_loss_from_trainer.item():.2f}x")
    
    if abs(train_loss_from_trainer.item() - eval_loss_from_trainer.item()) < 0.5:
        print("\n   ✓ Losses are similar - no systematic bias")
    else:
        print("\n   ⚠️  Significant difference between train and eval losses!")
        print("       This could explain the discrepancy in your loss curves")

print("\n" + "=" * 80)
