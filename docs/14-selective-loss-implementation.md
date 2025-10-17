# Selective Loss Implementation for Unsloth Training

## Overview

This document describes the **Unsloth-compatible** implementation of selective loss for structured output training.

## Key Constraint: Unsloth Compatibility

Your training pipeline uses:
- `UnslothVisionDataCollator` - handles PIL images and vision-language data
- `FastLanguageModel` - Unsloth's optimized model wrapper
- `SFTTrainer` with `SFTConfig` - from TRL library

**Critical**: We must preserve Unsloth's optimizations while adding selective loss.

## Recommended Approach: Extend UnslothVisionDataCollator

### Why This Works

1. **Preserves Optimizations**: Inherits all Unsloth's vision handling
2. **Clean Integration**: Drops into existing training pipeline
3. **Easy to Toggle**: Can enable/disable via parameter

### Implementation

Create `model_garden/selective_loss.py`:

```python
"""Selective loss computation for structured outputs with Unsloth compatibility."""

from unsloth.trainer import UnslothVisionDataCollator
import torch
import re
from typing import List, Set, Dict, Optional
from rich.console import Console

console = Console()


class SelectiveLossVisionCollator(UnslothVisionDataCollator):
    """
    Extends UnslothVisionDataCollator to mask structural JSON tokens in loss.
    
    For structured output tasks (e.g., JSON form extraction), we want to:
    - Compute loss ONLY on semantic content (field values)
    - Ignore structural tokens (braces, brackets, colons, etc.)
    - Optionally ignore schema keys (field names)
    
    This focuses training on the hard task of content extraction while
    letting the model rely on schema constraints for structure.
    """
    
    # JSON structural characters to mask
    STRUCTURAL_CHARS = {'{', '}', '[', ']', ':', ',', '"'}
    
    # Common JSON keywords to mask
    JSON_KEYWORDS = {'null', 'true', 'false'}
    
    def __init__(
        self,
        model,
        processor,
        mask_structural_tokens: bool = True,