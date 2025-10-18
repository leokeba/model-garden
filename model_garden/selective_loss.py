"""Selective loss computation for structured output training.

This module provides data collators that mask structural JSON tokens during training,
allowing the model to focus on learning semantic content rather than trivial structure.

Key Features:
- Extends Unsloth's UnslothVisionDataCollator for compatibility
- Masks JSON structural characters ({, }, [, ], :, ,) and whitespace
- Optional masking of schema keys and null keyword
- Maintains all Unsloth optimizations for vision-language models

Usage:
    from model_garden.selective_loss import create_selective_loss_collator
    
    collator = create_selective_loss_collator(
        model=model,
        processor=processor,
        mask_level="conservative",
        verbose=True
    )
"""

import torch
import re
from typing import List, Dict, Set, Optional, Any
from unsloth.trainer import UnslothVisionDataCollator
from rich.console import Console

console = Console()

from unsloth.trainer import UnslothVisionDataCollator
import torch
import re
from typing import List, Set, Dict, Optional, Any
from rich.console import Console

console = Console()


class SelectiveLossVisionCollator(UnslothVisionDataCollator):
    """Extends UnslothVisionDataCollator to mask structural JSON tokens in loss computation.
    
    For structured output tasks (e.g., JSON form extraction), standard training computes
    loss on all tokens including structural JSON characters. This wastes training signal
    on trivial predictions.
    
    This collator masks structural tokens so the model focuses on semantic content.
    
    Schema Key Auto-Detection:
    - In aggressive mode, schema keys are automatically detected from the training data
    - Keys are extracted from JSON responses during the first few batches
    - This handles varying schemas across examples gracefully
    
    Args:
        model: The model being trained
        processor: Vision processor for handling images
        mask_structural_tokens: Whether to mask JSON structure (default: True)
        mask_schema_keys: Whether to mask field names (default: False)
        schema_keys: List of field names to mask (required if mask_schema_keys=True)
        verbose: Whether to print masking statistics (default: False)
    
    Note:
        Schema keys should be pre-detected using detect_schema_keys_from_dataset()
        or create_selective_loss_collator() which handles detection automatically.
    
    Example:
        >>> # Pre-detect schema keys
        >>> detected_keys = detect_schema_keys_from_dataset(dataset, processor)
        >>> collator = SelectiveLossVisionCollator(
        ...     model=model,
        ...     processor=processor,
        ...     mask_structural_tokens=True,
        ...     mask_schema_keys=True,
        ...     schema_keys=list(detected_keys)
        ... )
    """
    
    # JSON structural characters to mask (including whitespace and quotes)
    # Quotes are structural in JSON - they delimit strings but carry no semantic meaning
    STRUCTURAL_CHARS = {'{', '}', '[', ']', ':', ',', '"', ' ', '\n', '\t', '\r'}
    
    # Only null is truly structural - true/false can be semantic content
    JSON_KEYWORDS = {'null'}
    
    def __init__(
        self,
        model,
        processor,
        mask_structural_tokens: bool = True,
        mask_schema_keys: bool = False,
        schema_keys: Optional[List[str]] = None,
        mask_json_keywords: bool = False,
        masking_start_step: int = 0,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(model, processor, **kwargs)
        self.mask_structural = mask_structural_tokens
        self.mask_keys = mask_schema_keys
        self.schema_keys = set(schema_keys) if schema_keys else set()
        self.mask_keywords = mask_json_keywords
        self.masking_start_step = masking_start_step
        self.verbose = verbose
        
        # Statistics for debugging
        self.total_tokens = 0
        self.masked_tokens = 0
        self.batch_count = 0
        self.current_step = 0
        
        if self.verbose:
            console.print("[cyan]Initialized SelectiveLossVisionCollator[/cyan]")
            console.print(f"  Mask structural tokens: {self.mask_structural}")
            console.print(f"  Mask schema keys: {self.mask_keys}")
            if self.schema_keys:
                console.print(f"  Schema keys to mask ({len(self.schema_keys)}): {list(self.schema_keys)[:10]}")
            if self.masking_start_step > 0:
                console.print(f"  [yellow]Masking delayed until step {self.masking_start_step}[/yellow]")
                console.print(f"  [yellow]Model will learn JSON structure first, then apply selective loss[/yellow]")
    
    def __call__(self, features):
        """Process batch and apply selective loss masking."""
        # First, use Unsloth's collator to handle vision data properly
        batch = super().__call__(features)
        
        # Increment step counter
        self.current_step += 1
        
        # DEBUG: Log configuration on first call
        if self.current_step == 1 and self.verbose:
            console.print(f"[cyan]DEBUG: masking_start_step={self.masking_start_step}, mask_structural={self.mask_structural}[/cyan]")
        
        # Check if we should apply masking yet
        if not self.mask_structural:
            return batch  # No masking, use standard Unsloth behavior
        
        if self.current_step <= self.masking_start_step:
            # Before masking_start_step: let model learn structure normally
            if self.verbose and self.current_step % 10 == 0:
                console.print(
                    f"[dim]Step {self.current_step}/{self.masking_start_step}: "
                    f"Learning structure (masking disabled)[/dim]"
                )
            return batch  # No masking yet
        
        # Log when masking starts (now at step masking_start_step + 1)
        if self.current_step == self.masking_start_step + 1 and self.verbose:
            console.print(
                f"[green]✓ Step {self.current_step}: Masking activated! (after {self.masking_start_step} steps of structure learning)[/green]"
            )
        
        # Apply selective loss masking
        if "labels" in batch:
            original_labels = batch["labels"].clone()
            batch["labels"] = self._apply_selective_masking(
                batch["labels"], 
                batch.get("input_ids", None)
            )
            
            # Increment batch count
            self.batch_count += 1
            
            # Update statistics
            if self.verbose:
                newly_masked = (batch["labels"] == -100).sum() - (original_labels == -100).sum()
                total = batch["labels"].numel()
                self.total_tokens += total
                self.masked_tokens += newly_masked.item()
                
                if self.batch_count % 10 == 0:  # Print every 10 batches
                    mask_pct = (self.masked_tokens / self.total_tokens) * 100
                    console.print(
                        f"[dim]Batch {self.batch_count}: Masked {mask_pct:.1f}% of tokens "
                        f"({self.masked_tokens}/{self.total_tokens}) [Step {self.current_step}][/dim]"
                    )
        
        return batch
    
    def _apply_selective_masking(self, labels, input_ids=None):
        """Mask structural tokens in labels tensor.
        
        Args:
            labels: Tensor of label token IDs [batch_size, seq_len]
            input_ids: Optional input token IDs for context
            
        Returns:
            Modified labels with -100 for structural tokens
        """
        masked_labels = labels.clone()
        
        for i in range(labels.size(0)):
            # Get the label sequence for this example
            label_tokens = labels[i]
            
            # Find positions to mask
            mask_indices = self._find_structural_indices(label_tokens)
            
            # Apply masking (-100 is ignored by PyTorch loss functions)
            if mask_indices:
                masked_labels[i, mask_indices] = -100
        
        return masked_labels
    
    def _find_structural_indices(self, token_ids):
        """Identify which token positions are structural (not semantic).
        
        Strategy:
        1. Decode tokens to text
        2. Use character-level analysis to find structural chars
        3. Map character positions back to token indices
        
        Args:
            token_ids: Tensor of token IDs for one sequence
            
        Returns:
            List of token indices to mask
        """
        # Skip if already fully masked or empty
        valid_mask = token_ids != -100
        if not valid_mask.any():
            return []
        
        # Decode to text (excluding already-masked tokens)
        valid_tokens = token_ids[valid_mask]
        try:
            decoded_text = self.processor.tokenizer.decode(
                valid_tokens, 
                skip_special_tokens=False
            )
        except Exception as e:
            # If decoding fails, don't mask anything (safety)
            if self.verbose:
                console.print(f"[yellow]Warning: Failed to decode tokens: {e}[/yellow]")
            return []
        
        # Build mapping from character position to token index
        char_to_token = self._build_char_to_token_map(valid_tokens, decoded_text)
        
        # Find structural character positions
        structural_positions = self._find_structural_char_positions(decoded_text)
        
        # Map character positions to token indices
        token_indices = []
        for char_pos in structural_positions:
            token_idx = char_to_token.get(char_pos)
            if token_idx is not None:
                # Map back to original sequence position (accounting for masked tokens)
                original_idx = torch.where(valid_mask)[0][token_idx].item()
                token_indices.append(original_idx)
        
        return list(set(token_indices))
    
    def _find_structural_char_positions(self, text: str) -> Set[int]:
        """Find character positions of structural JSON tokens.
        
        Args:
            text: Decoded text to analyze
            
        Returns:
            Set of character positions to mask
        """
        positions = set()
        
        # Mask single structural characters
        for i, char in enumerate(text):
            if char in self.STRUCTURAL_CHARS:
                positions.add(i)
        
        # Mask JSON keywords if enabled (only 'null' - true/false can be semantic)
        if self.mask_keywords:
            for keyword in self.JSON_KEYWORDS:
                # Use word boundaries to avoid masking parts of real content
                pattern = r'\b' + re.escape(keyword) + r'\b'
                for match in re.finditer(pattern, text):
                    positions.update(range(match.start(), match.end()))
        
        # Mask schema keys if enabled
        if self.mask_keys and self.schema_keys:
            for key in self.schema_keys:
                # Match quoted keys like "Marque": or "contents":
                pattern = r'"' + re.escape(key) + r'"\s*:'
                for match in re.finditer(pattern, text):
                    # Mask the key and quotes, but not the colon (already masked)
                    positions.update(range(match.start(), match.end() - 1))
        
        return positions
    
    def _build_char_to_token_map(self, token_ids, text: str) -> Dict[int, int]:
        """Build mapping from character position to token index.
        
        This is needed because we identify structural positions at the character level
        (using regex/string analysis) but need to mask at the token level.
        
        Args:
            token_ids: Tensor of token IDs
            text: Full decoded text
            
        Returns:
            Dictionary mapping character position to token index
        """
        char_to_token = {}
        current_pos = 0
        
        for token_idx, token_id in enumerate(token_ids):
            # Decode individual token
            token_text = self.processor.tokenizer.decode([token_id.item()])
            token_len = len(token_text)
            
            # Map all characters in this token to this token index
            for offset in range(token_len):
                char_pos = current_pos + offset
                if char_pos < len(text):
                    char_to_token[char_pos] = token_idx
            
            current_pos += token_len
        
        return char_to_token
    
    def get_masking_stats(self) -> Dict[str, Any]:
        """Get statistics about token masking.
        
        Returns:
            Dictionary with masking statistics
        """
        if self.total_tokens == 0:
            return {
                'total_tokens': 0,
                'masked_tokens': 0,
                'mask_percentage': 0.0,
                'batch_count': 0
            }
        
        return {
            'total_tokens': self.total_tokens,
            'masked_tokens': self.masked_tokens,
            'mask_percentage': (self.masked_tokens / self.total_tokens) * 100,
            'batch_count': self.batch_count
        }
    
    def print_stats(self):
        """Print masking statistics."""
        stats = self.get_masking_stats()
        console.print("\n[bold cyan]Selective Loss Masking Statistics:[/bold cyan]")
        console.print(f"  Total tokens processed: {stats['total_tokens']:,}")
        console.print(f"  Tokens masked: {stats['masked_tokens']:,}")
        console.print(f"  Mask percentage: {stats['mask_percentage']:.2f}%")
        console.print(f"  Batches processed: {stats['batch_count']}")


# Helper function for easy usage
def detect_schema_keys_from_dataset(
    dataset,
    processor,
    num_samples: int = 50,
    threshold: float = 0.3,
    verbose: bool = False
) -> Set[str]:
    """Pre-analyze dataset to detect schema keys before training starts.
    
    Args:
        dataset: Training dataset with formatted messages
        processor: Vision processor with tokenizer
        num_samples: Number of samples to analyze (default: 50)
        threshold: Minimum frequency (0-1) for a key to be included (default: 0.3)
        verbose: Whether to print detection progress
        
    Returns:
        Set of detected schema keys
    """
    import json
    from datasets import Dataset
    
    if verbose:
        console.print(f"[cyan]🔍 Pre-analyzing {num_samples} samples to detect schema keys...[/cyan]")
    
    detected_keys_counter = {}
    num_samples = min(num_samples, len(dataset))
    
    for idx in range(num_samples):
        sample = dataset[idx]
        
        # Extract assistant response from messages
        if "messages" in sample:
            messages = sample["messages"]
            assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)
            
            if assistant_msg:
                # Get text content from assistant message
                content = assistant_msg.get("content", [])
                if isinstance(content, list):
                    text_content = next((c.get("text") for c in content if c.get("type") == "text"), None)
                else:
                    text_content = content
                
                if text_content:
                    # Try to parse as JSON
                    try:
                        json_data = json.loads(text_content)
                        
                        # Extract all field names recursively
                        def extract_keys(obj, keys_found):
                            if isinstance(obj, dict):
                                for key in obj.keys():
                                    keys_found.add(key)
                                    extract_keys(obj[key], keys_found)
                            elif isinstance(obj, list):
                                for item in obj:
                                    extract_keys(item, keys_found)
                        
                        keys_found = set()
                        extract_keys(json_data, keys_found)
                        
                        # Update counter
                        for key in keys_found:
                            detected_keys_counter[key] = detected_keys_counter.get(key, 0) + 1
                            
                    except Exception:
                        # Not valid JSON, skip
                        pass
    
    # Select keys that appear in at least threshold% of samples
    min_count = int(num_samples * threshold)
    detected_keys = {
        key for key, count in detected_keys_counter.items()
        if count >= min_count
    }
    
    if verbose:
        if detected_keys:
            console.print(f"[green]✅ Detected {len(detected_keys)} schema keys from {num_samples} samples:[/green]")
            # Show top 20 most common keys
            sorted_keys = sorted(
                detected_keys_counter.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
            for key, count in sorted_keys:
                if key in detected_keys:
                    pct = (count / num_samples) * 100
                    console.print(f"  ✓ {key} ({pct:.1f}% of samples)")
        else:
            console.print("[yellow]⚠️  No schema keys detected! Check dataset format.[/yellow]")
            if detected_keys_counter:
                console.print(f"[dim]   Keys seen (below {threshold*100}% threshold):[/dim]")
                for key, count in list(detected_keys_counter.items())[:10]:
                    pct = (count / num_samples) * 100
                    console.print(f"[dim]   - {key}: {pct:.1f}%[/dim]")
    
    return detected_keys


def create_selective_loss_collator(
    model,
    processor,
    mask_level: str = "conservative",
    schema_keys: Optional[List[str]] = None,
    dataset = None,
    masking_start_step: int = 0,
    verbose: bool = False
):
    """Create a SelectiveLossVisionCollator with preset masking levels.
    
    Args:
        model: The model being trained
        processor: Vision processor
        mask_level: Masking aggressiveness:
            - "none": No masking (standard training)
            - "conservative": Mask only structural characters ({, }, [, ], :, ,, ")
            - "moderate": Conservative + 'null' keyword
            - "aggressive": Moderate + schema keys (auto-detected if not specified)
        schema_keys: Optional list of field names to mask (auto-detected if None)
        dataset: Training dataset (required for auto-detection in aggressive mode)
        masking_start_step: Delay masking until this step (0 = immediate, >0 = learn structure first)
        verbose: Whether to print statistics
        
    Returns:
        Configured SelectiveLossVisionCollator
        
    Example:
        >>> # Auto-detect schema keys (recommended)
        >>> collator = create_selective_loss_collator(
        ...     model, processor, 
        ...     mask_level="aggressive",
        ...     dataset=train_dataset,  # Required for auto-detection
        ...     verbose=True
        ... )
        >>> 
        >>> # Manual schema keys (optional)
        >>> collator = create_selective_loss_collator(
        ...     model, processor, 
        ...     mask_level="aggressive",
        ...     schema_keys=['Marque', 'Modele', 'contents'],
        ...     verbose=True
        ... )
    """
    if mask_level == "none":
        # Return standard Unsloth collator
        return UnslothVisionDataCollator(model, processor)
    
    elif mask_level == "conservative":
        return SelectiveLossVisionCollator(
            model=model,
            processor=processor,
            mask_structural_tokens=True,
            mask_schema_keys=False,
            mask_json_keywords=False,
            masking_start_step=masking_start_step,
            verbose=verbose
        )
    
    elif mask_level == "moderate":
        return SelectiveLossVisionCollator(
            model=model,
            processor=processor,
            mask_structural_tokens=True,
            mask_schema_keys=False,
            mask_json_keywords=True,
            masking_start_step=masking_start_step,
            verbose=verbose
        )
    
    elif mask_level == "aggressive":
        # Auto-detect schema keys from dataset if not provided
        if schema_keys is None:
            if dataset is None:
                raise ValueError(
                    "For aggressive mode with auto-detection, you must provide the 'dataset' parameter. "
                    "Either pass dataset=train_dataset or specify schema_keys manually."
                )
            
            # Pre-analyze dataset to detect schema keys
            detected_keys = detect_schema_keys_from_dataset(
                dataset=dataset,
                processor=processor,
                num_samples=min(50, len(dataset)),
                threshold=0.3,
                verbose=verbose
            )
            schema_keys = list(detected_keys)
        
        return SelectiveLossVisionCollator(
            model=model,
            processor=processor,
            mask_structural_tokens=True,
            mask_schema_keys=True,
            schema_keys=schema_keys,
            mask_json_keywords=True,
            masking_start_step=masking_start_step,
            verbose=verbose
        )
    
    else:
        raise ValueError(
            f"Unknown mask_level: {mask_level}. "
            f"Choose from: 'none', 'conservative', 'moderate', 'aggressive'"
        )

