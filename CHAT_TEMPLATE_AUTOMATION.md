# Automatic Chat Template Detection

## Problem

Currently, Model Garden only works with Qwen-VL models and a few other model families because chat templates are applied manually using hardcoded markers:

```python
# Hardcoded in vision_training.py
instruction_part="<|im_start|>user"
response_part="<|im_start|>assistant"

# Hardcoded in inference.py
prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    f"{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
```

This creates several issues:
1. **Limited model support** - Only Qwen-VL and models with identical chat formats work
2. **Maintenance burden** - Each new model family requires manual template implementation
3. **Error-prone** - Easy to get markers wrong, breaking training/inference
4. **Not future-proof** - New models with different templates won't work

## Solution

HuggingFace Transformers provides `tokenizer.apply_chat_template()` which automatically applies the correct chat template for any model. Every modern chat model includes its template in the tokenizer config.

### How It Works

```python
# Instead of manual formatting:
text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

# Use automatic template detection:
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

The tokenizer knows its own chat format and applies it automatically!

### Benefits

1. **Universal model support** - Works with ANY chat model (Llama, Mistral, Phi, Gemma, etc.)
2. **Zero maintenance** - No need to update code for new models
3. **Always correct** - Uses the official template from model authors
4. **Multimodal support** - Vision models also support this (Qwen2-VL, LLaVA, etc.)

### Implementation Plan

#### Phase 1: Inference Service (`inference.py`)

**Current code (lines 976-986):**
```python
if self.is_vision_model and images:
    # Hardcoded Qwen format
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
```

**New code:**
```python
def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
    """Format chat messages using the model's native chat template."""
    try:
        # Use the tokenizer's built-in chat template
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted
    except Exception as e:
        console.print(f"[yellow]Warning: Could not apply chat template: {e}[/yellow]")
        console.print("[yellow]Falling back to simple format[/yellow]")
        # Fallback for models without chat templates
        return self._format_simple(messages)
```

#### Phase 2: Training (`vision_training.py`)

**Current code (lines 870-871, 886-887):**
```python
instruction_part="<|im_start|>user",  # Hardcoded Qwen markers
response_part="<|im_start|>assistant"
```

**New approach:**

1. **Auto-detect chat markers** from tokenizer's template
2. **Extract markers** by applying template to sample messages
3. **Use extracted markers** for selective loss masking

```python
def _detect_chat_markers(self, processor) -> tuple[str, str]:
    """Detect instruction and response markers from tokenizer's chat template.
    
    Returns:
        (instruction_marker, response_marker) - e.g. ("<|im_start|>user", "<|im_start|>assistant")
    """
    # Apply template to sample messages
    sample = [
        {"role": "user", "content": "__USER__"},
        {"role": "assistant", "content": "__ASSISTANT__"}
    ]
    
    try:
        formatted = processor.apply_chat_template(sample, tokenize=False)
        
        # Extract markers by finding what comes before our placeholders
        user_idx = formatted.find("__USER__")
        assistant_idx = formatted.find("__ASSISTANT__")
        
        if user_idx > 0 and assistant_idx > 0:
            # Extract marker before user content (e.g., "<|im_start|>user\n")
            user_marker_start = formatted.rfind("<", 0, user_idx)
            instruction_part = formatted[user_marker_start:user_idx].strip()
            
            # Extract marker before assistant content
            assistant_marker_start = formatted.rfind("<", 0, assistant_idx)
            response_part = formatted[assistant_marker_start:assistant_idx].strip()
            
            console.print(f"[green]✓ Detected chat markers:[/green]")
            console.print(f"  Instruction: {instruction_part}")
            console.print(f"  Response: {response_part}")
            
            return instruction_part, response_part
    except Exception as e:
        console.print(f"[yellow]Warning: Could not detect chat markers: {e}[/yellow]")
    
    # Fallback: Try common patterns
    return self._fallback_markers(processor)

def _fallback_markers(self, processor) -> tuple[str, str]:
    """Fallback chat markers for models without templates."""
    model_type = processor.tokenizer.config.model_type.lower()
    
    # Common patterns
    if "qwen" in model_type:
        return "<|im_start|>user", "<|im_start|>assistant"
    elif "llama" in model_type:
        return "[INST]", "[/INST]"
    elif "phi" in model_type:
        return "<|user|>", "<|assistant|>"
    elif "mistral" in model_type:
        return "[INST]", "[/INST]"
    else:
        # Generic fallback
        console.print("[yellow]⚠️  Using generic markers - training may not work optimally[/yellow]")
        return "User:", "Assistant:"
```

#### Phase 3: Update Data Collators (`selective_loss.py`)

Make chat markers **optional** with automatic detection:

```python
def create_selective_loss_collator(
    model,
    processor,
    mask_level: str = "none",
    instruction_part: Optional[str] = None,  # Now optional!
    response_part: Optional[str] = None,     # Now optional!
    **kwargs
):
    """Create selective loss collator with automatic chat marker detection."""
    
    # Auto-detect markers if not provided
    if instruction_part is None or response_part is None:
        from model_garden.vision_training import VisionLanguageTrainer
        trainer = VisionLanguageTrainer.__new__(VisionLanguageTrainer)
        instruction_part, response_part = trainer._detect_chat_markers(processor)
    
    # Rest of function unchanged
    ...
```

### Testing Strategy

1. **Test with Qwen models** (current working baseline)
2. **Test with Llama-Vision** (different template: `[INST]`/`[/INST]`)
3. **Test with Phi-3-Vision** (different template: `<|user|>`/`<|assistant|>`)
4. **Test with LLaVA** (different template variations)
5. **Test fallback** for models without templates

### Migration Path

1. **Phase 1** (Backward compatible):
   - Add new automatic detection alongside existing hardcoded markers
   - Use environment variable to toggle: `USE_AUTO_CHAT_TEMPLATE=true`
   - Default to old behavior for stability

2. **Phase 2** (Deprecation):
   - Make automatic detection the default
   - Print deprecation warnings for hardcoded markers
   - Update documentation

3. **Phase 3** (Cleanup):
   - Remove hardcoded template code
   - Remove fallback logic after sufficient testing

### Example: Multi-Model Support

After implementation, users can train/serve ANY chat model:

```bash
# Qwen (currently works)
uv run model-garden train-vision --base-model Qwen/Qwen2.5-VL-7B

# Llama-Vision (will work automatically!)
uv run model-garden train-vision --base-model meta-llama/Llama-3.2-11B-Vision

# Phi-3-Vision (will work automatically!)
uv run model-garden train-vision --base-model microsoft/Phi-3-vision-128k-instruct

# LLaVA (will work automatically!)
uv run model-garden train-vision --base-model llava-hf/llava-1.5-7b-hf
```

### References

- [HuggingFace Chat Templating Guide](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [apply_chat_template() API](https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template)
- [Multimodal Chat Templates](https://huggingface.co/docs/transformers/main/en/chat_templating_multimodal)

### Related Files

Files that need updates:
- `model_garden/inference.py` - Lines 976-986, 1205-1227 (chat formatting)
- `model_garden/vision_training.py` - Lines 870-871, 886-887 (hardcoded markers)
- `model_garden/selective_loss.py` - Lines 785-806 (marker parameters)
- `inspect_vision_preprocessing.py` - Documentation/examples

### Priority

**Medium-High** - This is a significant improvement that would:
- Unlock support for many popular VLM models (Llama-Vision, Phi-3-Vision, etc.)
- Reduce maintenance burden
- Future-proof the codebase

However, it's not blocking current functionality since Qwen models work well.

### Estimated Effort

- **Implementation**: 2-3 days
- **Testing**: 1-2 days  
- **Documentation**: 1 day
- **Total**: ~5-6 days

### Success Criteria

1. ✅ Can train Llama-3.2-Vision without code changes
2. ✅ Can train Phi-3-Vision without code changes
3. ✅ Qwen models still work (backward compatibility)
4. ✅ Clear error messages for unsupported models
5. ✅ Documentation updated with multi-model examples
