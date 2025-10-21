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
    
Implementation Note:
    Import order matters! Unsloth MUST be imported before torch/transformers/trl
    to apply optimizations. This module is lazy-loaded (only imported when training),
    so having unsloth at the top doesn't cause worker spawn at API startup.
"""

# Standard library imports
import re
from typing import List, Dict, Set, Optional, Any, TYPE_CHECKING

# CRITICAL: Import unsloth FIRST before torch (Unsloth requires this for optimizations)
from unsloth.trainer import UnslothVisionDataCollator

# Then import torch and other ML libraries
import torch
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
    
    # JSON structural characters to mask (NOT including < and > which are for XML/HTML tags)
    # Quotes are structural in JSON - they delimit strings but carry no semantic meaning
    STRUCTURAL_CHARS = {'{', '}', '[', ']', ':', ',', '"', ' ', '\n', '\t', '\r'}
    
    # Only null is truly structural - true/false can be semantic content
    JSON_KEYWORDS = {'null'}
    
    # JSON type keywords (these appear as values but are not semantic for form extraction)
    JSON_TYPE_KEYWORDS = {'object', 'array', 'string', 'number', 'integer', 'boolean', 'null'}
    
    # JSON Schema keywords that should be masked (not semantic content for form extraction)
    SCHEMA_KEYWORDS = {
        # Schema structure
        '$schema', '$id', '$ref', '$defs', 'definitions',
        # Type keywords
        'type', 'properties', 'items', 'required', 'additionalProperties',
        'enum', 'const', 'anyOf', 'oneOf', 'allOf', 'not',
        # Validation keywords
        'minimum', 'maximum', 'minLength', 'maxLength', 'pattern',
        'minItems', 'maxItems', 'uniqueItems', 'format',
        # Metadata (keep 'title' as it might be semantic)
        'description', 'default', 'examples',
    }
    
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
        # Show initial batch info only once for debugging
        if self.current_step == 0 and self.verbose:
            console.print(f"[cyan]SelectiveLossVisionCollator processing batch[/cyan]")
            if len(features) > 0 and 'messages' in features[0]:
                msgs = features[0]['messages']
                console.print(f"[cyan]  Messages format: {len(msgs)} messages[/cyan]")  # type: ignore[arg-type]
        
        # First, use Unsloth's collator to handle vision data properly
        batch = super().__call__(features)
        
        # Show batch shape info only on first call
        if self.current_step == 0 and self.verbose:
            console.print(f"[cyan]After parent.__call__:[/cyan]")
            if 'labels' in batch:
                labels_sample = batch['labels'][0]
                total = len(labels_sample)
                masked = (labels_sample == -100).sum().item()
                console.print(f"[cyan]  Labels: {total} tokens, {masked} masked, {total-masked} unmasked[/cyan]")
            if 'input_ids' in batch:
                console.print(f"[cyan]  Input IDs shape: {batch['input_ids'].shape}[/cyan]")
        
        # CRITICAL: Always check labels to diagnose NaN eval_loss
        # This runs on EVERY call (training and eval) to catch the root cause
        if "labels" in batch and len(batch["labels"]) > 0:
            first_labels = batch["labels"][0]
            total_tokens = len(first_labels)
            masked_tokens = (first_labels == -100).sum().item()
            unmasked_tokens = total_tokens - masked_tokens
            
            # Always warn if ALL tokens are masked (regardless of train/eval mode)
            if unmasked_tokens == 0:
                console.print(f"[red]❌ CRITICAL: ALL tokens masked in batch![/red]")
                console.print(f"[red]   Total tokens: {total_tokens}, Step: {self.current_step}[/red]")
                console.print(f"[red]   This will cause NaN loss![/red]")
                
                # Check if this is due to sequence truncation (common with vision models)
                if "input_ids" in batch and len(batch["input_ids"]) > 0:
                    try:
                        input_ids = batch["input_ids"][0]
                        # Quick check for truncation without excessive logging
                        full_text = self.processor.tokenizer.decode(input_ids, skip_special_tokens=False)  # type: ignore[attr-defined]
                        user_count = full_text.count("<|im_start|>user")
                        assistant_count = full_text.count("<|im_start|>assistant")
                        
                        # CRITICAL WARNING: Check if assistant response was truncated
                        if assistant_count == 0 and user_count > 0:
                            console.print(f"[red]⚠️  SEQUENCE TRUNCATION DETECTED![/red]")
                            console.print(f"[red]   Assistant response was CUT OFF by max_seq_length![/red]")
                            console.print(f"[red]   Current sequence length: {len(input_ids)} tokens[/red]")
                            console.print(f"[red]   → SOLUTION: Increase max_seq_length (currently at {len(input_ids)})[/red]")
                            console.print(f"[red]   → For vision models with images, use 8192+ tokens[/red]")
                        else:
                            console.print(f"[red]   Markers found: user={user_count}, assistant={assistant_count}[/red]")
                            console.print(f"[red]   Check train_on_responses_only configuration[/red]")
                    except Exception as e:
                        console.print(f"[red]   Could not analyze sequence: {e}[/red]")
        
        # Only increment step counter during training (not evaluation)
        # During eval, PyTorch uses torch.no_grad() context, so gradients are disabled
        is_training = torch.is_grad_enabled()
        if is_training:
            self.current_step += 1
        
        # DEBUG: Log configuration on first call
        if self.current_step == 1 and self.verbose:
            console.print(f"[cyan]DEBUG: masking_start_step={self.masking_start_step}, mask_structural={self.mask_structural}[/cyan]")
        
        # Check if we should apply masking yet
        if not self.mask_structural:
            return batch  # No masking, use standard Unsloth behavior
        
        if self.current_step <= self.masking_start_step:
            # Before masking_start_step: let model learn structure normally
            if self.verbose and self.current_step % 10 == 0 and is_training:
                console.print(
                    f"[dim]Step {self.current_step}/{self.masking_start_step}: "
                    f"Learning structure (masking disabled)[/dim]"
                )
            return batch  # No masking yet
        
        # Log when masking starts (now at step masking_start_step + 1)
        if self.current_step == self.masking_start_step + 1 and self.verbose and is_training:
            console.print(
                f"[green]✓ Step {self.current_step}: Masking activated! (after {self.masking_start_step} steps of structure learning)[/green]"
            )
        
        # Apply selective loss masking
        if "labels" in batch:
            # MEMORY FIX: Only clone for statistics if verbose, and detach immediately
            original_masked_count = 0
            if self.verbose and is_training:
                # Count original masked tokens (use item() to convert to Python int immediately)
                original_masked_count = (batch["labels"] == -100).sum().item()
            
            # DEBUG: On first masked batch, check what Unsloth gave us
            if is_training and self.batch_count == 0 and self.verbose:
                first_seq = batch["labels"][0]
                total_tokens = len(first_seq)
                prompt_masked = (first_seq == -100).sum().item()
                console.print(f"[cyan]🔍 Batch structure check (first sequence):[/cyan]")
                console.print(f"   Total tokens: {total_tokens}")
                console.print(f"   Prompt tokens (masked to -100): {prompt_masked} ({prompt_masked/total_tokens*100:.1f}%)")
                console.print(f"   Assistant tokens (not -100): {total_tokens - prompt_masked} ({(total_tokens - prompt_masked)/total_tokens*100:.1f}%)")
            
            # Increment batch count (only during training) - do this BEFORE checking for logging
            if is_training:
                self.batch_count += 1
            
            # Store original for logging only if needed
            original_labels = None
            should_print_sample = self.verbose and is_training and self.batch_count % 10 == 0
            if should_print_sample:
                # Clone only when we need to print, and detach to avoid gradient tracking
                original_labels = batch["labels"][0].clone().detach()
            
            batch["labels"] = self._apply_selective_masking(
                batch["labels"], 
                batch.get("input_ids", None)
            )
            
            # Update statistics
            if self.verbose and is_training:
                # MEMORY FIX: Calculate on-the-fly without storing tensors
                newly_masked_count = (batch["labels"] == -100).sum().item() - original_masked_count
                # FIX: Only count assistant tokens (originally not -100) as the denominator
                # We're measuring "what % of assistant tokens did we mask", not "what % of all tokens"
                assistant_tokens_count = batch["labels"].numel() - original_masked_count
                self.total_tokens += assistant_tokens_count
                self.masked_tokens += newly_masked_count
                
                if should_print_sample:  # Print every 10 batches
                    mask_pct = (self.masked_tokens / self.total_tokens) * 100
                    console.print(
                        f"[dim]Batch {self.batch_count}: Masked {mask_pct:.1f}% of tokens "
                        f"({self.masked_tokens}/{self.total_tokens}) [Step {self.current_step}][/dim]"
                    )
                    
                    # Show sample of unmasked content from first example in batch
                    if original_labels is not None:
                        self._print_unmasked_sample(batch["labels"][0], original_labels)
                        # MEMORY FIX: Explicitly delete the cloned tensor
                        del original_labels
        
        return batch
    
    def _apply_selective_masking(self, labels, input_ids=None):
        """Mask structural tokens in labels tensor.
        
        Args:
            labels: Tensor of label token IDs [batch_size, seq_len]
            input_ids: Optional input token IDs for context
            
        Returns:
            Modified labels with -100 for structural tokens
        """
        # MEMORY FIX: Modify labels in-place instead of cloning
        # This reduces peak memory during training/evaluation
        for i in range(labels.size(0)):
            # Get the label sequence for this example
            label_tokens = labels[i]
            
            # Find positions to mask
            mask_indices = self._find_structural_indices(label_tokens)
            
            # Apply masking (-100 is ignored by PyTorch loss functions)
            if mask_indices:
                labels[i, mask_indices] = -100
        
        return labels
    
    def _find_structural_indices(self, token_ids):
        """Identify which token positions are structural (not semantic).
        
        Strategy:
        1. Decode each token individually
        2. Check if token is structural based on its text content
        3. For schema keys, use sliding window to match multi-token keys
        4. Mark structural tokens for masking
        
        Args:
            token_ids: Tensor of token IDs for one sequence
            
        Returns:
            List of token indices to mask
        """
        # Skip if already fully masked or empty
        valid_mask = token_ids != -100
        if not valid_mask.any():
            return []
        
        # Decode to full text for context
        valid_tokens = token_ids[valid_mask]
        try:
            full_text = self.processor.tokenizer.decode(
                valid_tokens, 
                skip_special_tokens=False
            )
        except Exception as e:
            # If decoding fails, don't mask anything (safety)
            if self.verbose:
                console.print(f"[yellow]Warning: Failed to decode tokens: {e}[/yellow]")
            return []
        
        # Identify structural tokens
        token_indices_to_mask = set()  # Use set to avoid duplicates
        
        # First pass: mask individual structural tokens
        for i, token_id in enumerate(valid_tokens):
            # Decode individual token
            try:
                token_text = self.processor.tokenizer.decode([token_id.item()], skip_special_tokens=False)
            except:
                continue
            
            # Check if this token should be masked (non-schema-key checks)
            if self._is_structural_token(token_text, full_text, check_schema_keys=False):
                # Map back to original sequence position (accounting for masked tokens)
                original_idx = torch.where(valid_mask)[0][i].item()
                token_indices_to_mask.add(original_idx)
        
        # Second pass: mask schema keys using sliding window
        if self.mask_keys and self.schema_keys:
            token_indices_to_mask.update(self._find_schema_key_spans(valid_tokens, valid_mask))
        
        return list(token_indices_to_mask)
    
    def _find_schema_key_spans(self, valid_tokens, valid_mask):
        """Find token spans that correspond to schema keys using sliding window.
        
        Args:
            valid_tokens: Tensor of non-masked token IDs
            valid_mask: Boolean mask of valid positions in original sequence
            
        Returns:
            Set of token indices to mask
        """
        indices_to_mask = set()
        
        for key in self.schema_keys:
            # Try sliding windows of different sizes (1 to 10 tokens)
            for window_size in range(1, min(11, len(valid_tokens) + 1)):
                for start_idx in range(len(valid_tokens) - window_size + 1):
                    # Decode this window
                    window_tokens = valid_tokens[start_idx:start_idx + window_size]
                    try:
                        window_text = self.processor.tokenizer.decode(
                            window_tokens, 
                            skip_special_tokens=False
                        ).strip()
                    except:
                        continue
                    
                    # Check if this window matches the schema key (with or without quotes)
                    if window_text == key or window_text == f'"{key}"' or window_text == f"'{key}'":
                        # Mask all tokens in this window
                        for i in range(start_idx, start_idx + window_size):
                            original_idx = torch.where(valid_mask)[0][i].item()
                            indices_to_mask.add(original_idx)
        
        return indices_to_mask
    
    def _is_structural_token(self, token_text: str, full_context: str, check_schema_keys: bool = True) -> bool:
        """Determine if a token is structural (should be masked).
        
        Args:
            token_text: The decoded text of the individual token
            full_context: The full decoded text for context
            check_schema_keys: Whether to check for schema keys (set False to avoid duplication with sliding window)
            
        Returns:
            True if token should be masked, False if it's semantic content
        """
        # Strip whitespace for checking
        stripped = token_text.strip()
        
        # 1. Mask pure whitespace tokens
        if not stripped:
            return True
        
        # 2. Mask pure structural character tokens
        if all(c in self.STRUCTURAL_CHARS for c in token_text):
            return True
        
        # 3. Mask JSON Schema keywords
        for keyword in self.SCHEMA_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', stripped):
                return True
        
        # 4. Mask JSON type keywords (object, string, number, etc.)
        for type_keyword in self.JSON_TYPE_KEYWORDS:
            if stripped.lower() == type_keyword:
                return True
        
        # 5. Mask JSON null keyword
        if self.mask_keywords and stripped == 'null':
            return True
        
        # 6. Mask schema field names if enabled (handled by sliding window now)
        # This is kept for backwards compatibility but won't match multi-token keys
        if check_schema_keys and self.mask_keys and self.schema_keys:
            for key in self.schema_keys:
                # Check if token is exactly the schema key (single-token keys only)
                if stripped == key or stripped == f'"{key}"':
                    return True
        
        # 7. Keep semantic content
        return False
    
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
    
    def _print_unmasked_sample(self, masked_labels, original_labels):
        """Print a sample of the unmasked (semantic) content being learned.
        
        Args:
            masked_labels: Labels tensor after masking [seq_len]
            original_labels: Labels tensor before masking [seq_len]
        """
        try:
            # Get indices of tokens that are NOT masked AFTER our masking
            unmasked_after = (masked_labels != -100)
            
            # Get indices of tokens that were NOT masked BEFORE our masking (i.e., assistant response only)
            unmasked_before = (original_labels != -100)
            
            if not unmasked_after.any():
                console.print("[dim]  └─ No unmasked tokens in this example[/dim]")
                return
            
            # Only consider tokens that were valid in the original (not prompt tokens)
            valid_original_count = unmasked_before.sum().item()
            if valid_original_count == 0:
                console.print("[dim]  └─ No valid tokens in original (all prompt)[/dim]")
                return
            
            # DEBUG: Show what tokens are in the original labels (before our masking)
            if self.batch_count == 10:  # Only on first verbose batch to avoid spam
                console.print("[yellow]  🔍 DEBUG: Checking original labels (what Unsloth gave us):[/yellow]")
                
                # Count masked vs unmasked in original
                total_tokens = len(original_labels)
                original_masked = (original_labels == -100).sum().item()
                original_unmasked = total_tokens - original_masked
                console.print(f"[yellow]     Total tokens in sequence: {total_tokens}[/yellow]")
                console.print(f"[yellow]     Masked by Unsloth (prompt): {original_masked} ({original_masked/total_tokens*100:.1f}%)[/yellow]")
                console.print(f"[yellow]     Unmasked (assistant): {original_unmasked} ({original_unmasked/total_tokens*100:.1f}%)[/yellow]")
                
                # Decode ONLY the unmasked tokens (should be assistant only)
                if unmasked_before.any():
                    original_valid_tokens = original_labels[unmasked_before]
                    original_decoded = self.processor.tokenizer.decode(
                        original_valid_tokens[:100],  # First 100 tokens
                        skip_special_tokens=False
                    )
                    console.print(f"[yellow]     First 100 unmasked tokens (should be assistant only): {repr(original_decoded[:300])}[/yellow]")
                else:
                    console.print(f"[yellow]     WARNING: No unmasked tokens found![/yellow]")
            
            # Extract unmasked token IDs (what we're keeping for training)
            unmasked_tokens = masked_labels[unmasked_after]
            
            # Decode to text
            unmasked_text = self.processor.tokenizer.decode(
                unmasked_tokens,
                skip_special_tokens=True
            )
            
            # Also get the full original text for comparison
            original_tokens = original_labels[unmasked_before]
            original_text = self.processor.tokenizer.decode(
                original_tokens,
                skip_special_tokens=True
            )
            
            # Calculate how much was kept vs masked (only count originally valid tokens)
            kept_tokens = unmasked_after.sum().item()
            kept_pct = (kept_tokens / valid_original_count * 100) if valid_original_count > 0 else 0
            
            console.print(f"[dim]  └─ Unmasked content ({kept_pct:.1f}% kept, {kept_tokens}/{valid_original_count} tokens):[/dim]")
            
            # Truncate if too long
            max_display = 500
            if len(unmasked_text) > max_display:
                display_text = unmasked_text[:max_display] + "..."
            else:
                display_text = unmasked_text
            
            console.print(f"[green]     {repr(display_text)}[/green]")
            
            # Show original if not too long (for comparison)
            if len(original_text) <= 400:
                console.print(f"[dim]     Original ({len(original_text)} chars): {repr(original_text[:200])}{'...' if len(original_text) > 200 else ''}[/dim]")
                
        except Exception as e:
            console.print(f"[yellow]  └─ Could not decode unmasked tokens: {e}[/yellow]")
            import traceback
            if self.verbose:
                traceback.print_exc()
    
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
    verbose: bool = False,
    train_on_responses_only: bool = False,
    instruction_part: Optional[str] = None,
    response_part: Optional[str] = None
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
        train_on_responses_only: Whether to mask prompts (train only on assistant responses)
        instruction_part: Chat template marker for user messages (e.g., "<|im_start|>user")
        response_part: Chat template marker for assistant messages (e.g., "<|im_start|>assistant")
        
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
    # Validate parameters
    if train_on_responses_only:
        if instruction_part is None or response_part is None:
            raise ValueError(
                "train_on_responses_only=True requires instruction_part and response_part. "
                "For Qwen models, use: instruction_part='<|im_start|>user', response_part='<|im_start|>assistant'"
            )
    
    if mask_level == "none":
        # Return standard Unsloth collator
        kwargs = {}
        if train_on_responses_only:
            kwargs["train_on_responses_only"] = True
            kwargs["instruction_part"] = instruction_part
            kwargs["response_part"] = response_part
            kwargs["force_match"] = False  # CRITICAL: Don't mask everything if markers not found
        
        return UnslothVisionDataCollator(model, processor, **kwargs)
    
    elif mask_level == "conservative":
        kwargs = {}
        if train_on_responses_only:
            kwargs["train_on_responses_only"] = True
            kwargs["instruction_part"] = instruction_part
            kwargs["response_part"] = response_part
            kwargs["force_match"] = False  # CRITICAL: Don't mask everything if markers not found
        
        return SelectiveLossVisionCollator(
            model=model,
            processor=processor,
            mask_structural_tokens=True,
            mask_schema_keys=False,
            mask_json_keywords=False,
            masking_start_step=masking_start_step,
            verbose=verbose,
            **kwargs
        )
    
    elif mask_level == "moderate":
        kwargs = {}
        if train_on_responses_only:
            kwargs["train_on_responses_only"] = True
            kwargs["instruction_part"] = instruction_part
            kwargs["response_part"] = response_part
            kwargs["force_match"] = False  # CRITICAL: Don't mask everything if markers not found
        
        return SelectiveLossVisionCollator(
            model=model,
            processor=processor,
            mask_structural_tokens=True,
            mask_schema_keys=False,
            mask_json_keywords=True,
            masking_start_step=masking_start_step,
            verbose=verbose,
            **kwargs
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
        
        kwargs = {}
        if train_on_responses_only:
            kwargs["train_on_responses_only"] = True
            kwargs["instruction_part"] = instruction_part
            kwargs["response_part"] = response_part
            kwargs["force_match"] = False  # CRITICAL: Don't mask everything if markers not found
        
        return SelectiveLossVisionCollator(
            model=model,
            processor=processor,
            mask_structural_tokens=True,
            mask_schema_keys=True,
            schema_keys=schema_keys,
            mask_json_keywords=True,
            masking_start_step=masking_start_step,
            verbose=verbose,
            **kwargs
        )
    
    else:
        raise ValueError(
            f"Unknown mask_level: {mask_level}. "
            f"Choose from: 'none', 'conservative', 'moderate', 'aggressive'"
        )

