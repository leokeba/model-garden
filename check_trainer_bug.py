#!/usr/bin/env python3
"""
Direct test of transformers Trainer.prediction_step to see if it applies .mean()
"""
import os
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'

# Read the trainer.py source code to check prediction_step
trainer_path = "/root/model-garden/.venv/lib/python3.12/site-packages/transformers/trainer.py"

print("Checking Trainer.prediction_step for loss.mean() call...")
print("="*80)

with open(trainer_path, 'r') as f:
    lines = f.readlines()

# Find prediction_step
in_prediction_step = False
in_compute_loss_section = False
line_num = 0
relevant_lines = []

for i, line in enumerate(lines, 1):
    if 'def prediction_step(' in line:
        in_prediction_step = True
        line_num = i
        print(f"\nFound prediction_step at line {i}")
        
    if in_prediction_step:
        if 'compute_loss' in line:
            in_compute_loss_section = True
            
        if in_compute_loss_section:
            relevant_lines.append((i, line.rstrip()))
            
            # Look for the .mean() call after compute_loss
            if '.mean()' in line and 'loss' in line.lower():
                print(f"\n✗ FOUND THE BUG at line {i}:")
                print(f"   {line.strip()}")
                print("\n   Context:")
                for j in range(max(0, len(relevant_lines)-5), len(relevant_lines)):
                    print(f"   {relevant_lines[j][0]}: {relevant_lines[j][1]}")
                break
                
            # Stop when we exit the compute_loss section
            if in_compute_loss_section and line.strip() and not line.strip().startswith('#') and 'if isinstance' in line:
                in_compute_loss_section = False
                
        # Stop when we hit the next method
        if in_prediction_step and line.strip().startswith('def ') and 'prediction_step' not in line:
            break

print("\n" + "="*80)
print("\nNow checking training_step for comparison...")
print("="*80)

in_training_step = False
for i, line in enumerate(lines, 1):
    if 'def training_step(' in line:
        in_training_step = True
        print(f"\nFound training_step at line {i}")
        
    if in_training_step:
        if 'compute_loss' in line:
            print(f"\n{i}: {line.rstrip()}")
            # Show a few lines after
            for j in range(i, min(i+5, len(lines))):
                print(f"{j+1}: {lines[j].rstrip()}")
            break

print("\n" + "="*80)
print("\nConclusion:")
print("="*80)
print("""
If prediction_step has:
    loss = self.compute_loss(...).mean()
    
But training_step has:
    loss = self.compute_loss(...)
    
Then evaluation will incorrectly average the already-normalized loss!
""")
