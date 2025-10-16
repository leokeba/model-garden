"""Simple test for structured outputs without API."""

import asyncio
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

async def test_structured_output():
    print("Testing structured output with vLLM directly...")
    
    # Create engine
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        max_model_len=4096,
        trust_remote_code=True,
        enable_chunked_prefill=False,  # Disable V1 engine
        gpu_memory_utilization=0.3,  # Use only 30% to avoid conflicts with running server
    )
    
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # Test 1: Simple text generation without structured output
    print("\n1. Testing without structured output...")
    sampling_params = SamplingParams(
        max_tokens=100,
        temperature=0.7,
    )
    
    prompt = "Tell me about Paris in one sentence."
    request_id = "test-1"
    
    results_gen = engine.generate(prompt, sampling_params, request_id)
    async for result in results_gen:
        final_result = result
    
    print(f"✅ Result: {final_result.outputs[0].text}")
    
    # Test 2: With structured output
    print("\n2. Testing WITH structured output (JSON)...")
    
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"},
            "population": {"type": "integer"}
        },
        "required": ["city", "country"]
    }
    
    structured_params = StructuredOutputsParams(json=schema)
    sampling_params2 = SamplingParams(
        max_tokens=200,
        temperature=0.7,
        structured_outputs=structured_params,
    )
    
    prompt2 = "Provide information about Paris in JSON format with city, country, and population."
    request_id2 = "test-2"
    
    try:
        results_gen2 = engine.generate(prompt2, sampling_params2, request_id2)
        async for result in results_gen2:
            final_result2 = result
        
        print(f"✅ Structured Result: {final_result2.outputs[0].text}")
        
        # Try to parse as JSON
        import json
        parsed = json.loads(final_result2.outputs[0].text)
        print(f"✅ Valid JSON: {parsed}")
        
    except Exception as e:
        print(f"❌ Error with structured output: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_structured_output())
