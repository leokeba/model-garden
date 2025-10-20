"""Test memory leak with actual Trainer and UnslothVisionDataCollator.

This test simulates the EXACT production setup:
- Uses SFTTrainer (not manual loop)
- Uses UnslothVisionDataCollator (not manual processor calls)
- Uses callbacks for cleanup (same as production)

Goal: Reproduce the leak in a test environment where we can iterate faster.
"""

import gc
import os
import psutil
import torch
from pathlib import Path
from datasets import load_dataset

# Configure HF cache
HF_HOME = os.getenv('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))
os.environ['HF_HOME'] = HF_HOME
os.environ['TRANSFORMERS_CACHE'] = str(Path(HF_HOME) / 'hub')
os.environ['HF_DATASETS_CACHE'] = str(Path(HF_HOME) / 'datasets')

# CRITICAL: Import unsloth first
from unsloth import FastLanguageModel, UnslothVisionDataCollator

def get_memory_mb():
    """Get current process memory in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

print("=" * 80)
print("PRODUCTION-LIKE MEMORY LEAK TEST")
print("Using: SFTTrainer + UnslothVisionDataCollator + Callbacks")
print("=" * 80)

# Load dataset (use small subset for faster iteration)
print("\n1. Loading dataset...")
mem_start = get_memory_mb()
raw_dataset = load_dataset("Barth371/cmr-all", split="train")
# Use more samples to match production better
raw_dataset = raw_dataset.select(range(300))  # 300 samples instead of 100
print(f"   Memory: {mem_start:.1f} MB -> {get_memory_mb():.1f} MB")

# Format dataset
print("\n2. Formatting dataset...")
from model_garden.vision_training import VisionLanguageTrainer
trainer_instance = VisionLanguageTrainer(
    base_model="Qwen/Qwen2.5-VL-3B-Instruct",
    load_in_4bit=True
)
trainer_instance.load_model()

# CRITICAL: Configure LoRA adapters (required for training quantized models)
print("   Configuring LoRA adapters...")
trainer_instance.prepare_for_training(
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

formatted_dataset_list = trainer_instance.format_dataset(raw_dataset)
print(f"   Memory: {get_memory_mb():.1f} MB (dataset is a list with {len(formatted_dataset_list)} examples)")

# Setup training with MINIMAL steps
print("\n3. Setting up training with SFTTrainer...")
from transformers import TrainerCallback
from trl import SFTTrainer
from trl.trainer.sft_config import SFTConfig

training_args = SFTConfig(
    output_dir="./test_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=100,  # 100 steps to see if leak appears over time
    logging_steps=10,
    save_steps=1000,  # Don't save during test
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    report_to=[],  # Disable logging
    # CRITICAL: Tell Unsloth to skip dataset preparation (which requires formatting_func)
    remove_unused_columns=False,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
)

# Create memory monitoring callback
class MemoryMonitorCallback(TrainerCallback):
    """Monitor memory every 10 steps."""
    
    def __init__(self):
        self.step_memory = []
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            mem_mb = get_memory_mb()
            self.step_memory.append((state.global_step, mem_mb))
            
            # Count tensors
            tensors = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)]
            cpu_tensors = len([t for t in tensors if t.device.type == 'cpu'])
            gpu_tensors = len([t for t in tensors if t.device.type == 'cuda'])
            
            print(f"   Step {state.global_step}: {len(tensors)} tensors (CPU: {cpu_tensors}, GPU: {gpu_tensors}), RAM: {mem_mb:.1f} MB")
        
        return control

memory_callback = MemoryMonitorCallback()

# Create cleanup callback (same as production)
class CleanupCallback(TrainerCallback):
    """Aggressive cleanup after every step."""
    
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get('model')
        if model is not None and hasattr(model, 'zero_grad'):
            model.zero_grad(set_to_none=True)
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return control

cleanup_callback = CleanupCallback()

# Create data collator (EXACT same as production)
data_collator = UnslothVisionDataCollator(trainer_instance.model, trainer_instance.processor)

print(f"   Memory before trainer creation: {get_memory_mb():.1f} MB")

# Create trainer - EXACT same as production (NO formatting_func, list dataset, with data_collator)
trainer = SFTTrainer(
    model=trainer_instance.model,
    tokenizer=trainer_instance.tokenizer,
    args=training_args,
    train_dataset=formatted_dataset_list,  # Pass list directly like production
    data_collator=data_collator,
    callbacks=[memory_callback, cleanup_callback],
)

print(f"   Memory after trainer creation: {get_memory_mb():.1f} MB")

# Run training
print("\n4. Running training (50 steps)...")
print("   Monitoring memory every 10 steps...")
print("-" * 80)

mem_before_train = get_memory_mb()
trainer.train()
mem_after_train = get_memory_mb()

print("-" * 80)
print("\n5. Results:")
print(f"   Memory before training: {mem_before_train:.1f} MB")
print(f"   Memory after training:  {mem_after_train:.1f} MB")
print(f"   Total growth: +{mem_after_train - mem_before_train:.1f} MB")

# Analyze step-by-step growth
if len(memory_callback.step_memory) > 1:
    print("\n6. Step-by-step analysis:")
    for i in range(1, len(memory_callback.step_memory)):
        prev_step, prev_mem = memory_callback.step_memory[i-1]
        curr_step, curr_mem = memory_callback.step_memory[i]
        growth = curr_mem - prev_mem
        steps = curr_step - prev_step
        per_step = growth / steps if steps > 0 else 0
        print(f"   Steps {prev_step}->{curr_step}: +{growth:.1f} MB ({per_step:.1f} MB/step)")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)

# Calculate average leak rate
if len(memory_callback.step_memory) > 1:
    first_step, first_mem = memory_callback.step_memory[0]
    last_step, last_mem = memory_callback.step_memory[-1]
    total_growth = last_mem - first_mem
    total_steps = last_step - first_step
    avg_per_step = total_growth / total_steps if total_steps > 0 else 0
    
    print(f"\nAverage leak rate: {avg_per_step:.1f} MB/step")
    print(f"Extrapolated to 250 steps: {avg_per_step * 250:.1f} MB")
    
    if avg_per_step > 10:
        print("\n⚠️  LEAK CONFIRMED: Memory growing >10 MB/step")
        print("   This matches production behavior!")
    else:
        print("\n✓ No significant leak detected in test")
        print("   Something different between test and production?")
else:
    print("\n⚠️  Not enough data points to analyze")
