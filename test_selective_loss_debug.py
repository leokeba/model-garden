#!/usr/bin/env python3
"""Debug script to test selective loss masking logic."""

import torch
from transformers import AutoTokenizer

# Simulate the schema keywords
SCHEMA_KEYWORDS = {
    '$schema', '$id', '$ref', '$defs', 'definitions',
    'type', 'properties', 'items', 'required', 'additionalProperties',
    'enum', 'const', 'anyOf', 'oneOf', 'allOf', 'not',
    'minimum', 'maximum', 'minLength', 'maxLength', 'pattern',
    'minItems', 'maxItems', 'uniqueItems', 'format',
    'description', 'default', 'examples',
}

JSON_TYPE_KEYWORDS = {'object', 'array', 'string', 'number', 'integer', 'boolean', 'null'}

STRUCTURAL_CHARS = {'{', '}', '[', ']', ':', ',', '"', ' ', '\n', '\t', '\r'}

def is_structural_token(token_text: str) -> bool:
    """Check if token should be masked."""
    stripped = token_text.strip()
    
    # 1. Mask pure whitespace
    if not stripped:
        return True
    
    # 2. Mask pure structural characters
    if all(c in STRUCTURAL_CHARS for c in token_text):
        return True
    
    # 3. Mask schema keywords
    import re
    for keyword in SCHEMA_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', stripped):
            return True
    
    # 4. Mask JSON type keywords
    for type_keyword in JSON_TYPE_KEYWORDS:
        if stripped.lower() == type_keyword:
            return True
    
    return False

def test_masking():
    """Test the masking logic on sample JSON."""
    # Load a tokenizer (use Qwen2.5-VL tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    # Sample JSON response (simplified)
    sample_json = '''{
  "type": "object",
  "properties": {
    "Marque": {
      "type": "string",
      "description": "Car brand"
    },
    "Modele": {
      "contents": "Peugeot 208"
    }
  }
}'''
    
    print("=" * 80)
    print("Sample JSON:")
    print(sample_json)
    print("=" * 80)
    
    # Tokenize
    tokens = tokenizer.encode(sample_json, add_special_tokens=False)
    
    print(f"\nTotal tokens: {len(tokens)}")
    print("\nToken-by-token analysis:")
    print("-" * 80)
    
    masked_count = 0
    kept_count = 0
    kept_tokens = []
    
    for i, token_id in enumerate(tokens):
        token_text = tokenizer.decode([token_id])
        should_mask = is_structural_token(token_text)
        
        if should_mask:
            status = "MASK"
            masked_count += 1
        else:
            status = "KEEP"
            kept_count += 1
            kept_tokens.append(token_id)
        
        print(f"{i:3d} | {status:4s} | {repr(token_text):30s} | token_id={token_id}")
    
    print("-" * 80)
    print(f"\nStatistics:")
    print(f"  Masked: {masked_count}/{len(tokens)} ({masked_count/len(tokens)*100:.1f}%)")
    print(f"  Kept:   {kept_count}/{len(tokens)} ({kept_count/len(tokens)*100:.1f}%)")
    
    print(f"\nUnmasked (semantic) content:")
    unmasked_text = tokenizer.decode(kept_tokens)
    print(f"  {repr(unmasked_text)}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_masking()
