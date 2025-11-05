"""Test if chat marker detection is working correctly."""

import os
os.environ["HF_HOME"] = "/root/model-garden/storage/cache"
os.environ["TRANSFORMERS_CACHE"] = "/root/model-garden/storage/cache/hub"
os.environ["HF_DATASETS_CACHE"] = "/root/model-garden/storage/cache/datasets"

from unsloth import FastVisionModel

print("Testing chat marker detection...")

model, processor = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-2B-Instruct",
    load_in_4bit=True,
    max_seq_length=2048,
)

# Simulate what _detect_chat_markers does
test_messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are helpful."}]},
    {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Hi there"}]}
]

print("\n1. Applying chat template:")
result = processor.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=False)
print(f"Result: {repr(result[:300])}")

# Check if markers are present
print("\n2. Checking for common markers:")
markers_to_check = [
    ("<|im_start|>user", "<|im_start|>assistant"),
    ("[INST]", "[/INST]"),
    ("<|user|>", "<|assistant|>"),
    ("### Instruction:", "### Response:"),
]

for inst_marker, resp_marker in markers_to_check:
    has_inst = inst_marker in result
    has_resp = resp_marker in result
    print(f"   {inst_marker:30s} -> {has_inst}")
    print(f"   {resp_marker:30s} -> {has_resp}")
    if has_inst and has_resp:
        print(f"   ✓ FOUND: These markers should work!")
    print()

print("\n3. What if the markers are WRONG?")
print("   If _detect_chat_markers returns incorrect markers, and force_match=False,")
print("   then NO masking happens - ALL tokens contribute to loss!")
print("   This would explain:")
print("   - Training loss being very low (model learns full sequence including prompt)")
print("   - Validation loss being higher (different prompts, model hasn't seen them)")
