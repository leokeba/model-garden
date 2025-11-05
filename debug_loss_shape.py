#!/usr/bin/env python3
"""
Debug script to check what shape/value compute_loss returns
"""
import os
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'

import torch
from unsloth import FastVisionModel, get_chat_template
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

# Load model
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# Get processor
processor = tokenizer

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

# Get chat template (no need to modify for Qwen2.5-VL)
# tokenizer already has the correct chat template

# Load a tiny dataset
dataset = load_dataset("json", data_files="/root/model-garden/data/vision_test_dataset.jsonl", split="train")
dataset = list(dataset.take(2))  # Just 2 samples

# Create data collator
data_collator = UnslothVisionDataCollator(
    model=model,
    processor=processor,
    train_on_responses_only=True
)

# Create minimal trainer
training_args = SFTConfig(
    output_dir="./test_output",
    per_device_train_batch_size=1,
    max_steps=1,
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# Get a batch
dataloader = trainer.get_train_dataloader()
batch = next(iter(dataloader))

# Move batch to device
batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

print("\n1. Testing compute_loss with num_items_in_batch:")
num_items = (batch["labels"] != -100).sum()
print(f"   num_items_in_batch: {num_items}")

loss = trainer.compute_loss(model, batch, num_items_in_batch=num_items)
print(f"   Loss shape: {loss.shape if hasattr(loss, 'shape') else 'scalar'}")
print(f"   Loss value: {loss}")
print(f"   Loss ndim: {loss.ndim if hasattr(loss, 'ndim') else 'N/A'}")

print("\n2. Testing compute_loss without num_items_in_batch:")
loss2 = trainer.compute_loss(model, batch)
print(f"   Loss shape: {loss2.shape if hasattr(loss2, 'shape') else 'scalar'}")
print(f"   Loss value: {loss2}")

print("\n3. Testing compute_loss with return_outputs=True (eval mode):")
loss3, outputs = trainer.compute_loss(model, batch, return_outputs=True)
print(f"   Loss shape: {loss3.shape if hasattr(loss3, 'shape') else 'scalar'}")
print(f"   Loss value: {loss3}")
print(f"   Loss ndim: {loss3.ndim if hasattr(loss3, 'ndim') else 'N/A'}")
print(f"   After .mean(): {loss3.detach().mean()}")

print("\n4. Checking if num_items_in_batch is passed in eval:")
# Simulate prediction_step call
loss4, outputs = trainer.compute_loss(model, batch, return_outputs=True)
print(f"   Loss before .mean(): {loss4}")
print(f"   Loss after .mean(): {loss4.detach().mean()}")
print(f"   Ratio (should be 1.0 if scalar): {loss4.detach().mean() / loss4}")
