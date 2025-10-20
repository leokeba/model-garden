#!/usr/bin/env python3
"""Test selective loss with actual dataset sample."""

import json
import sys

# Add parent directory to path
sys.path.insert(0, '/home/leo/Dev/model-garden')

from transformers import AutoProcessor
import torch

def test_with_real_data():
    """Test masking with actual dataset sample."""
    
    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    # Load first example from dataset
    dataset_path = "/home/leo/Dev/model-garden/storage/datasets/openai_finetune_vision_inline_20251017_114023.jsonl"
    
    with open(dataset_path, 'r') as f:
        first_line = f.readline()
    
    data = json.loads(first_line)
    
    # Get assistant response
    assistant_msg = None
    for msg in data['messages']:
        if msg['role'] == 'assistant':
            assistant_msg = msg
            break
    
    if not assistant_msg:
        print("No assistant message found!")
        return
    
    # Extract text content
    if isinstance(assistant_msg['content'], list):
        text_content = None
        for item in assistant_msg['content']:
            if item.get('type') == 'text':
                text_content = item['text']
                break
    else:
        text_content = assistant_msg['content']
    
    if not text_content:
        print("No text content in assistant message!")
        return
    
    print("\n" + "="*80)
    print("ASSISTANT RESPONSE (first 500 chars):")
    print("="*80)
    print(text_content[:500])
    print("...")
    
    # Tokenize
    tokens = processor.tokenizer.encode(text_content, add_special_tokens=False)
    
    print(f"\n" + "="*80)
    print(f"TOKENIZATION ANALYSIS")
    print("="*80)
    print(f"Total tokens: {len(tokens)}")
    
    # Import masking logic
    from model_garden.selective_loss import SelectiveLossVisionCollator
    
    # Check schema keywords
    SCHEMA_KEYWORDS = SelectiveLossVisionCollator.SCHEMA_KEYWORDS
    JSON_TYPE_KEYWORDS = SelectiveLossVisionCollator.JSON_TYPE_KEYWORDS
    STRUCTURAL_CHARS = SelectiveLossVisionCollator.STRUCTURAL_CHARS
    
    print(f"\nChecking tokens for schema contamination...")
    schema_token_count = 0
    type_token_count = 0
    structural_token_count = 0
    kept_token_count = 0
    
    import re
    
    for token_id in tokens[:200]:  # Check first 200 tokens
        token_text = processor.tokenizer.decode([token_id])
        stripped = token_text.strip()
        
        is_schema = False
        is_type = False
        is_structural = False
        
        # Check schema keywords
        for keyword in SCHEMA_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', stripped):
                is_schema = True
                schema_token_count += 1
                break
        
        # Check type keywords
        if not is_schema:
            for type_kw in JSON_TYPE_KEYWORDS:
                if stripped.lower() == type_kw:
                    is_type = True
                    type_token_count += 1
                    break
        
        # Check structural
        if not is_schema and not is_type:
            if not stripped or all(c in STRUCTURAL_CHARS for c in token_text):
                is_structural = True
                structural_token_count += 1
        
        if not is_schema and not is_type and not is_structural:
            kept_token_count += 1
    
    print(f"\nFirst 200 tokens breakdown:")
    print(f"  Schema keywords: {schema_token_count}")
    print(f"  Type keywords: {type_token_count}")
    print(f"  Structural: {structural_token_count}")
    print(f"  Kept (semantic): {kept_token_count}")
    print(f"  Total checked: {schema_token_count + type_token_count + structural_token_count + kept_token_count}")
    
    kept_pct = (kept_token_count / 200) * 100
    print(f"\n✅ Keeping {kept_pct:.1f}% of tokens (target: 20-30%)")
    
    # Show what's being kept
    print("\nSample of KEPT tokens (first 20):")
    kept_sample = []
    for token_id in tokens[:200]:
        token_text = processor.tokenizer.decode([token_id])
        stripped = token_text.strip()
        
        is_schema = any(re.search(r'\b' + re.escape(kw) + r'\b', stripped) for kw in SCHEMA_KEYWORDS)
        is_type = stripped.lower() in JSON_TYPE_KEYWORDS
        is_structural = not stripped or all(c in STRUCTURAL_CHARS for c in token_text)
        
        if not is_schema and not is_type and not is_structural:
            kept_sample.append(repr(token_text))
            if len(kept_sample) >= 20:
                break
    
    for i, tok in enumerate(kept_sample):
        print(f"  {i+1:2d}. {tok}")
    
    if schema_token_count > 0:
        print(f"\n⚠️  WARNING: Found {schema_token_count} schema keyword tokens!")
        print("     This suggests the assistant response contains schema definitions")
        print("     which should only be in the user prompt.")
    else:
        print("\n✅ No schema keywords found in assistant response - looks good!")

if __name__ == "__main__":
    test_with_real_data()
