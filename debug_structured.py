"""Minimal test to debug structured output issue."""

import asyncio
import json


async def test_basic_text_only():
    """Test 1: Basic structured output with text only (no images)."""
    print("\n" + "="*60)
    print("TEST 1: Text-only structured output")
    print("="*60)
    
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    
    # Create engine with minimal config
    print("Loading model...")
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2.5-3B-Instruct",  # Use text-only model first
        max_model_len=2048,
        gpu_memory_utilization=0.3,  # Low to avoid conflicts
        trust_remote_code=True,
    )
    
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("✅ Model loaded")
    
    # Test without structured output first
    print("\n1a. Testing normal generation...")
    sampling_params = SamplingParams(
        max_tokens=50,
        temperature=0.7,
    )
    
    prompt = "What is the capital of France?"
    request_id = "test-1a"
    
    results_gen = engine.generate(prompt, sampling_params, request_id)
    async for result in results_gen:
        final_result = result
    
    print(f"✅ Normal output: {final_result.outputs[0].text[:100]}")
    
    # Now test WITH structured output
    print("\n1b. Testing structured output (JSON)...")
    
    schema = {
        "type": "object",
        "properties": {
            "capital": {"type": "string"},
            "country": {"type": "string"}
        },
        "required": ["capital", "country"]
    }
    
    try:
        structured_params = StructuredOutputsParams(json=schema)
        sampling_params_structured = SamplingParams(
            max_tokens=100,
            temperature=0.7,
            structured_outputs=structured_params,
        )
        
        prompt2 = "What is the capital of France? Answer in JSON format."
        request_id2 = "test-1b"
        
        results_gen2 = engine.generate(prompt2, sampling_params_structured, request_id2)
        async for result in results_gen2:
            final_result2 = result
        
        output = final_result2.outputs[0].text
        print(f"✅ Structured output: {output}")
        
        # Validate JSON
        parsed = json.loads(output)
        print(f"✅ Valid JSON: {parsed}")
        print(f"   Capital: {parsed.get('capital')}")
        print(f"   Country: {parsed.get('country')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


async def test_vision_with_structured():
    """Test 2: Structured output with vision model."""
    print("\n" + "="*60)
    print("TEST 2: Vision model + structured output")
    print("="*60)
    
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.inputs import TextPrompt
    
    # Create engine with vision model
    print("Loading vision model...")
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        max_model_len=2048,
        gpu_memory_utilization=0.4,
        trust_remote_code=True,
    )
    
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("✅ Vision model loaded")
    
    # Test with text only first (no image)
    print("\n2a. Testing vision model with text only + structured output...")
    
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"}
        },
        "required": ["answer"]
    }
    
    try:
        structured_params = StructuredOutputsParams(json=schema)
        sampling_params = SamplingParams(
            max_tokens=100,
            temperature=0.7,
            structured_outputs=structured_params,
        )
        
        prompt = "What is 2+2? Answer in JSON format with an 'answer' field."
        request_id = "test-2a"
        
        results_gen = engine.generate(prompt, sampling_params, request_id)
        async for result in results_gen:
            final_result = result
        
        output = final_result.outputs[0].text
        print(f"✅ Output: {output}")
        
        parsed = json.loads(output)
        print(f"✅ Valid JSON: {parsed}")
        
    except Exception as e:
        print(f"❌ Error with text-only: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Now test with an image
    print("\n2b. Testing vision model with image + structured output...")
    
    # Create a simple test image
    from PIL import Image
    import tempfile
    
    # Create a red square image
    img = Image.new('RGB', (100, 100), color='red')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        img.save(tmp, format='PNG')
        image_path = tmp.name
    
    print(f"   Created test image: {image_path}")
    
    schema2 = {
        "type": "object",
        "properties": {
            "color": {"type": "string"},
            "description": {"type": "string"}
        },
        "required": ["color", "description"]
    }
    
    try:
        structured_params2 = StructuredOutputsParams(json=schema2)
        sampling_params2 = SamplingParams(
            max_tokens=150,
            temperature=0.7,
            structured_outputs=structured_params2,
        )
        
        # Create multimodal input
        prompt_text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What color is this image? Answer in JSON format with 'color' and 'description' fields.<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = TextPrompt(
            prompt=prompt_text,
            multi_modal_data={"image": [image_path]}
        )
        
        request_id2 = "test-2b"
        
        results_gen2 = engine.generate(inputs, sampling_params2, request_id2)
        async for result in results_gen2:
            final_result2 = result
        
        output2 = final_result2.outputs[0].text
        print(f"✅ Output with image: {output2}")
        
        parsed2 = json.loads(output2)
        print(f"✅ Valid JSON: {parsed2}")
        print(f"   Color: {parsed2.get('color')}")
        print(f"   Description: {parsed2.get('description')}")
        
    except Exception as e:
        print(f"❌ Error with image: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


async def main():
    print("🧪 Debugging Structured Output Issue")
    print("="*60)
    
    # Test 1: Text-only model
    try:
        success1 = await test_basic_text_only()
        if not success1:
            print("\n❌ Test 1 failed. Stopping here.")
            return
    except Exception as e:
        print(f"\n❌ Test 1 crashed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("✅ Test 1 passed! Moving to vision model...")
    print("="*60)
    
    # Test 2: Vision model
    try:
        success2 = await test_vision_with_structured()
        if success2:
            print("\n" + "="*60)
            print("✅✅✅ All tests passed!")
            print("="*60)
    except Exception as e:
        print(f"\n❌ Test 2 crashed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
