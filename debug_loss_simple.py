#!/usr/bin/env python3
"""
Simplified debug script to check loss computation in eval vs train
"""
import os
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'

import torch
from unsloth import FastVisionModel
from trl import SFTConfig, SFTTrainer

# Load model
print("Loading model...")
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# Configure LoRA
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Create minimal trainer (without dataset)
training_args = SFTConfig(
    output_dir="./test_output",
    per_device_train_batch_size=1,
    max_steps=1,
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
)

# Create a fake batch with some masked labels
print("\nCreating fake batch...")
batch = {
    "input_ids": torch.randint(0, 1000, (2, 100)).to(model.device),
    "attention_mask": torch.ones(2, 100).to(model.device),
    "labels": torch.randint(0, 1000, (2, 100)).to(model.device),
}

# Mask 90% of labels (simulating prompt masking)
mask = torch.rand(2, 100) > 0.9
batch["labels"] = batch["labels"].masked_fill(~mask, -100)

num_valid_tokens = (batch["labels"] != -100).sum()
print(f"Valid tokens: {num_valid_tokens.item()} out of {batch['labels'].numel()}")

print("\n" + "="*80)
print("Testing loss computation paths...")
print("="*80)

print("\n1. Training path: compute_loss(model, batch, num_items_in_batch=...)")
loss_train = trainer.compute_loss(model, batch, num_items_in_batch=num_valid_tokens)
print(f"   Loss shape: {loss_train.shape}")
print(f"   Loss ndim: {loss_train.ndim}")
print(f"   Loss value: {loss_train.item():.6f}")

print("\n2. Eval path: compute_loss(model, batch, return_outputs=True) + .mean()")
loss_eval, outputs = trainer.compute_loss(model, batch, return_outputs=True)
print(f"   Loss before .mean(): shape={loss_eval.shape}, ndim={loss_eval.ndim}")
print(f"   Loss before .mean(): value={loss_eval.item():.6f}")

loss_eval_meaned = loss_eval.detach().mean()
print(f"   Loss after .mean(): shape={loss_eval_meaned.shape}, ndim={loss_eval_meaned.ndim}")
print(f"   Loss after .mean(): value={loss_eval_meaned.item():.6f}")

print(f"\n   Ratio (eval_meaned / train): {(loss_eval_meaned / loss_train).item():.6f}")
print(f"   Difference: {(loss_eval_meaned - loss_train).item():.6f}")

print("\n3. Without num_items_in_batch:")
loss_no_num = trainer.compute_loss(model, batch)
print(f"   Loss value: {loss_no_num.item():.6f}")
print(f"   Ratio vs train: {(loss_no_num / loss_train).item():.6f}")

print("\n" + "="*80)
print("Summary:")
print("="*80)
if abs(loss_train.item() - loss_eval_meaned.item()) < 0.001:
    print("✓ Training and eval losses match - no bug!")
else:
    print("✗ Training and eval losses differ!")
    print(f"  Training loss: {loss_train.item():.6f}")
    print(f"  Eval loss: {loss_eval_meaned.item():.6f}")
    print(f"  Ratio: {(loss_eval_meaned / loss_train).item():.6f}")
