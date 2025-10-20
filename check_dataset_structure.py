"""Check dataset structure."""

import os
from pathlib import Path
from datasets import load_dataset

# Configure HF cache
HF_HOME = os.getenv('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))
os.environ['HF_HOME'] = HF_HOME
os.environ['TRANSFORMERS_CACHE'] = str(Path(HF_HOME) / 'hub')
os.environ['HF_DATASETS_CACHE'] = str(Path(HF_HOME) / 'datasets')

print("Loading dataset...")
dataset = load_dataset("Barth371/cmr-all", split="train")

print(f"\nDataset size: {len(dataset)}")
print(f"Column names: {dataset.column_names}")
print(f"\nFirst sample:")
sample = dataset[0]
for key, value in sample.items():
    if key == "image":
        print(f"  {key}: <PIL.Image>")
    else:
        print(f"  {key}: {type(value).__name__} = {str(value)[:100]}")
