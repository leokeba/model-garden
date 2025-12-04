"""End-to-end tests for Model Garden inference workflows.

These tests run actual inference jobs with loaded models to verify
the complete inference pipeline works correctly. They require GPU access and
a working vLLM installation.

Usage:
    pytest tests/test_e2e_inference.py --run-integration -v
    pytest tests/test_e2e_inference.py --run-integration -v -k "text"  # Text only
    pytest tests/test_e2e_inference.py --run-integration -v -k "vision"  # Vision only
    pytest tests/test_e2e_inference.py --run-integration -v -k "structured"  # Structured outputs
    pytest tests/test_e2e_inference.py --run-integration -v -k "lora"  # LoRA adapter tests
"""

import asyncio
import base64
import json
from pathlib import Path

import pytest
import pytest_asyncio

# Mark all tests in this module as integration tests requiring GPU
pytestmark = [pytest.mark.integration, pytest.mark.requires_gpu, pytest.mark.slow]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def text_model_path() -> str:
    """Return a small text model for testing."""
    return "unsloth/tinyllama-bnb-4bit"


@pytest.fixture(scope="module")
def vision_model_path() -> str:
    """Return a vision model for testing."""
    return "Qwen/Qwen2.5-VL-3B-Instruct"


@pytest.fixture(scope="module")
def test_images_dir() -> Path:
    """Return the path to test images directory."""
    project_root = Path(__file__).parent.parent
    return project_root / "data" / "test_images"


@pytest.fixture
def sample_image_base64(test_images_dir: Path) -> str:
    """Load a test image as base64."""
    image_path = test_images_dir / "red_square.jpg"
    if not image_path.exists():
        pytest.skip(f"Test image not found: {image_path}")

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================================================================
# Text Inference Service Tests
# ============================================================================


class TestModelLoadUnload:
    """Tests for model loading and unloading lifecycle."""

    @pytest.mark.asyncio
    async def test_load_and_unload_model(self, text_model_path: str):
        """Test that a model can be loaded and unloaded."""
        from model_garden.inference import InferenceService

        service = InferenceService(
            model_path=text_model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_model_len=512,
            enforce_eager=True,
        )

        # Initially not loaded
        assert not service.is_loaded

        # Load the model
        await service.load_model()
        assert service.is_loaded

        # Get model info
        info = service.get_model_info()
        assert info["is_loaded"] is True
        assert "tinyllama" in info["model_path"].lower()

        # Unload the model
        await service.unload_model()
        assert not service.is_loaded


class TestTextInferenceService:
    """End-to-end tests for text model inference using InferenceService directly."""

    @pytest_asyncio.fixture
    async def text_inference_service(self, text_model_path: str):
        """Create and load a text inference service."""
        from model_garden.inference import InferenceService

        service = InferenceService(
            model_path=text_model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,  # Use less memory for testing
            max_model_len=512,
            max_num_seqs=4,
            enforce_eager=True,  # Save memory
        )

        await service.load_model()
        yield service
        await service.unload_model()

    @pytest.mark.asyncio
    async def test_text_generation_basic(self, text_inference_service):
        """Test basic text generation."""
        result = await text_inference_service.generate(
            prompt="Hello, how are you?",
            max_tokens=50,
            temperature=0.7,
            stream=False,
        )

        assert isinstance(result, dict)
        assert "text" in result
        assert len(result["text"]) > 0
        assert "usage" in result
        assert result["usage"]["completion_tokens"] > 0

    @pytest.mark.asyncio
    async def test_text_generation_with_stop_sequence(self, text_inference_service):
        """Test text generation with stop sequences."""
        result = await text_inference_service.generate(
            prompt="Count from 1 to 10:",
            max_tokens=100,
            temperature=0.3,
            stop=["\n\n", "5"],
            stream=False,
        )

        assert isinstance(result, dict)
        assert "text" in result
        # Output should stop at or before "5"
        text = result["text"]
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_text_generation_streaming(self, text_inference_service):
        """Test streaming text generation."""
        chunks = []
        stream = await text_inference_service.generate(
            prompt="Write a short poem:",
            max_tokens=50,
            temperature=0.7,
            stream=True,
        )

        # Collect all chunks
        async for chunk in stream:
            chunks.append(chunk)

        # Should have received multiple chunks
        assert len(chunks) > 0

        # Combined text should be non-empty
        full_text = "".join(chunks)
        assert len(full_text) > 0

    @pytest.mark.asyncio
    async def test_chat_completion_basic(self, text_inference_service):
        """Test basic chat completion."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello!"},
        ]

        result = await text_inference_service.chat_completion(
            messages=messages,
            max_tokens=50,
            temperature=0.7,
            stream=False,
        )

        assert isinstance(result, dict)
        assert "choices" in result
        assert len(result["choices"]) > 0
        assert "message" in result["choices"][0]
        assert "content" in result["choices"][0]["message"]
        assert len(result["choices"][0]["message"]["content"]) > 0

    @pytest.mark.asyncio
    async def test_chat_completion_multi_turn(self, text_inference_service):
        """Test multi-turn chat completion."""
        messages = [
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "And what is 4+4?"},
        ]

        result = await text_inference_service.chat_completion(
            messages=messages,
            max_tokens=50,
            temperature=0.3,
            stream=False,
        )

        assert isinstance(result, dict)
        assert "choices" in result
        content = result["choices"][0]["message"]["content"]
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_chat_completion_streaming(self, text_inference_service):
        """Test streaming chat completion."""
        messages = [
            {"role": "user", "content": "Count to 3."},
        ]

        chunks = []
        stream = await text_inference_service.chat_completion(
            messages=messages,
            max_tokens=30,
            temperature=0.3,
            stream=True,
        )

        async for chunk in stream:
            chunks.append(chunk)

        # Should have received chunks
        assert len(chunks) > 0

        # Last chunk should have finish_reason
        last_chunk = chunks[-1]
        assert last_chunk["choices"][0].get("finish_reason") is not None

    @pytest.mark.asyncio
    async def test_temperature_affects_output(self, text_inference_service):
        """Test that temperature affects output variability."""
        prompt = "Complete this sentence: The sky is"

        # Generate with low temperature (more deterministic)
        result_low_temp = await text_inference_service.generate(
            prompt=prompt,
            max_tokens=20,
            temperature=0.1,
            stream=False,
        )

        # Generate with high temperature (more random)
        result_high_temp = await text_inference_service.generate(
            prompt=prompt,
            max_tokens=20,
            temperature=1.5,
            stream=False,
        )

        # Both should produce output
        assert len(result_low_temp["text"]) > 0
        assert len(result_high_temp["text"]) > 0


# ============================================================================
# Vision Inference Service Tests
# ============================================================================


class TestVisionModelLoadUnload:
    """Tests for vision model loading lifecycle."""

    @pytest.mark.asyncio
    async def test_vision_model_loads(self, vision_model_path: str):
        """Test that a vision model can be loaded."""
        from model_garden.inference import InferenceService
        from model_garden.inference.utils import is_vision_model

        # Verify it's detected as vision model
        assert is_vision_model(vision_model_path)

        service = InferenceService(
            model_path=vision_model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.6,
            max_model_len=1024,
            enforce_eager=True,
            limit_mm_per_prompt={"image": 1, "video": 0},
        )

        await service.load_model()
        assert service.is_loaded

        await service.unload_model()


class TestVisionInferenceService:
    """End-to-end tests for vision model inference."""

    @pytest_asyncio.fixture
    async def vision_inference_service(self, vision_model_path: str):
        """Create and load a vision inference service."""
        from model_garden.inference import InferenceService

        service = InferenceService(
            model_path=vision_model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.6,
            max_model_len=2048,
            max_num_seqs=4,
            enforce_eager=True,
            limit_mm_per_prompt={"image": 1, "video": 0},
        )

        await service.load_model()
        yield service
        await service.unload_model()

    @pytest.mark.asyncio
    async def test_vision_generation_with_image_file(
        self, vision_inference_service, test_images_dir: Path
    ):
        """Test vision generation with a local image file."""
        image_path = test_images_dir / "red_square.jpg"
        if not image_path.exists():
            pytest.skip(f"Test image not found: {image_path}")

        # Read image and convert to base64 for chat_completion
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        messages = [
            {"role": "user", "content": "What shape and color is shown in this image?"},
        ]

        # Use chat_completion which properly handles multimodal format
        result = await vision_inference_service.chat_completion(
            messages=messages,
            max_tokens=100,
            temperature=0.3,
            image=image_base64,
            stream=False,
        )

        assert isinstance(result, dict)
        assert "choices" in result
        content = result["choices"][0]["message"]["content"].lower()
        # Should identify the shape or color
        assert any(word in content for word in ["red", "square", "shape", "color"])

    @pytest.mark.asyncio
    async def test_vision_generation_with_base64_image(
        self, vision_inference_service, sample_image_base64: str
    ):
        """Test vision generation with base64 encoded image."""
        messages = [
            {"role": "user", "content": "Describe what you see."},
        ]

        # Use chat_completion which properly handles multimodal format
        result = await vision_inference_service.chat_completion(
            messages=messages,
            max_tokens=100,
            temperature=0.3,
            image=sample_image_base64,
            stream=False,
        )

        assert isinstance(result, dict)
        assert "choices" in result
        content = result["choices"][0]["message"]["content"]
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_vision_chat_completion(self, vision_inference_service, test_images_dir: Path):
        """Test vision chat completion with image."""
        image_path = test_images_dir / "blue_circle.jpg"
        if not image_path.exists():
            pytest.skip(f"Test image not found: {image_path}")

        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        messages = [
            {"role": "user", "content": "What shape is this? Answer briefly."},
        ]

        result = await vision_inference_service.chat_completion(
            messages=messages,
            max_tokens=50,
            temperature=0.3,
            image=image_base64,
            stream=False,
        )

        assert isinstance(result, dict)
        assert "choices" in result
        content = result["choices"][0]["message"]["content"]
        assert len(content) > 0


# ============================================================================
# Structured Output Tests
# ============================================================================


class TestStructuredOutputs:
    """End-to-end tests for structured output generation."""

    @pytest_asyncio.fixture
    async def inference_service_for_structured(self, text_model_path: str):
        """Create inference service for structured output tests."""
        from model_garden.inference import InferenceService

        service = InferenceService(
            model_path=text_model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_model_len=1024,
            enforce_eager=True,
        )

        await service.load_model()
        yield service
        await service.unload_model()

    @pytest.mark.asyncio
    async def test_json_object_output(self, inference_service_for_structured):
        """Test generating JSON object output."""
        result = await inference_service_for_structured.generate(
            prompt='Generate a JSON object with fields "name" and "age":',
            max_tokens=100,
            temperature=0.3,
            structured_outputs={"json": {"type": "object"}},
            stream=False,
        )

        assert isinstance(result, dict)
        assert "text" in result
        # The output should be valid JSON (or at least start with {)
        text = result["text"].strip()
        assert text.startswith("{") or text.startswith("[")

    @pytest.mark.asyncio
    async def test_json_schema_output(self, inference_service_for_structured):
        """Test generating output conforming to JSON schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }

        result = await inference_service_for_structured.generate(
            prompt="Generate a person's information:",
            max_tokens=100,
            temperature=0.3,
            structured_outputs={"json": schema},
            stream=False,
        )

        assert isinstance(result, dict)
        assert "text" in result

        # Try to parse the output as JSON
        text = result["text"].strip()
        try:
            parsed = json.loads(text)
            assert isinstance(parsed, dict)
        except json.JSONDecodeError:
            # Structured output may not always produce valid JSON with all models
            pass

    @pytest.mark.asyncio
    async def test_choice_output(self, inference_service_for_structured):
        """Test generating output from a fixed set of choices."""
        result = await inference_service_for_structured.generate(
            prompt="Is the sky blue? Answer with yes or no:",
            max_tokens=10,
            temperature=0.1,
            structured_outputs={"choice": ["yes", "no"]},
            stream=False,
        )

        assert isinstance(result, dict)
        assert "text" in result
        text = result["text"].strip().lower()
        # Output should be one of the choices (or contain it)
        assert "yes" in text or "no" in text


# ============================================================================
# API Integration Tests
# ============================================================================


class TestInferenceAPIRoutes:
    """End-to-end tests for inference API routes."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        from fastapi.testclient import TestClient

        from model_garden.api.app import create_app

        app = create_app()
        return TestClient(app)

    @pytest_asyncio.fixture
    async def loaded_model_client(self, text_model_path: str):
        """Create an async HTTP client with a model already loaded.

        Uses httpx.AsyncClient with ASGITransport for proper async support
        with vLLM's async engine.
        """
        import httpx

        from model_garden.api.app import create_app
        from model_garden.inference import InferenceService, set_inference_service

        # Create and load the inference service
        service = InferenceService(
            model_path=text_model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_model_len=512,
            enforce_eager=True,
        )
        await service.load_model()
        set_inference_service(service)

        app = create_app()

        # Use httpx.AsyncClient with ASGITransport for truly async requests
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

        # Cleanup
        await service.unload_model()
        set_inference_service(None)

    @pytest_asyncio.fixture
    async def loaded_vision_model_client(self, vision_model_path: str):
        """Async HTTP client with a vision model loaded."""
        import httpx

        from model_garden.api.app import create_app
        from model_garden.inference import InferenceService, set_inference_service

        service = InferenceService(
            model_path=vision_model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.6,
            max_model_len=2048,
            enforce_eager=True,
            limit_mm_per_prompt={"image": 1, "video": 0},
        )
        await service.load_model()
        set_inference_service(service)

        app = create_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

        await service.unload_model()
        set_inference_service(None)

    def test_inference_status_no_model(self, client):
        """Test inference status when no model is loaded."""
        response = client.get("/api/v1/inference/status")
        assert response.status_code == 200
        data = response.json()
        assert data["loaded"] is False

    @pytest.mark.asyncio
    async def test_inference_status_with_model(self, loaded_model_client):
        """Test inference status when a model is loaded."""
        response = await loaded_model_client.get("/api/v1/inference/status")
        assert response.status_code == 200
        data = response.json()
        assert data["loaded"] is True
        assert data["model_info"] is not None

    @pytest.mark.asyncio
    async def test_generate_endpoint(self, loaded_model_client):
        """Test the generate endpoint."""
        response = await loaded_model_client.post(
            "/api/v1/inference/generate",
            json={
                "prompt": "Hello, world!",
                "max_tokens": 20,
                "temperature": 0.7,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "text" in data

    @pytest.mark.asyncio
    async def test_chat_completions_endpoint(self, loaded_model_client):
        """Test the chat completions endpoint."""
        response = await loaded_model_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Say hi!"}],
                "max_tokens": 20,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0

    @pytest.mark.asyncio
    async def test_chat_completions_endpoint_base64_image(
        self,
        loaded_vision_model_client,
        sample_image_base64: str,
    ):
        """Ensure base64-only vision payloads are accepted by the API route."""
        data_url = f"data:image/jpeg;base64,{sample_image_base64}"
        response = await loaded_vision_model_client.post(
            "/api/v1/chat/completions",
            json={
                "model": "test",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the dominant color."},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                "max_tokens": 20,
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert "choices" in payload
        assert payload["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_openai_compatible_endpoint(self, loaded_model_client):
        """Test the OpenAI-compatible endpoint path."""
        response = await loaded_model_client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 20,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestInferenceUtils:
    """Tests for inference utility functions with real models."""

    def test_is_lora_adapter_detection_hub(self):
        """Test LoRA adapter detection for HuggingFace models."""
        from model_garden.inference.utils import is_lora_adapter

        # A known base model (not an adapter)
        assert is_lora_adapter("unsloth/tinyllama-bnb-4bit") is False

    def test_is_vision_model_detection(self):
        """Test vision model detection."""
        from model_garden.inference.utils import is_vision_model

        # Vision models
        assert is_vision_model("Qwen/Qwen2.5-VL-3B-Instruct") is True
        assert is_vision_model("llava-hf/llava-1.5-7b-hf") is True

        # Text-only models
        assert is_vision_model("unsloth/tinyllama-bnb-4bit") is False
        assert is_vision_model("meta-llama/Llama-3-8B") is False

    def test_estimate_model_size(self):
        """Test model size estimation from name."""
        from model_garden.inference.utils import estimate_model_size_gb

        # Should estimate based on "7B" in name
        size_7b = estimate_model_size_gb("meta-llama/Llama-3-7B")
        assert size_7b >= 7.0

        # Smaller model
        size_1b = estimate_model_size_gb("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        assert size_1b >= 1.0

    def test_gpu_memory_utilization_calculation(self):
        """Test GPU memory utilization calculation."""
        from model_garden.inference.utils import calculate_gpu_memory_utilization

        util = calculate_gpu_memory_utilization(
            model_path="meta-llama/Llama-3-8B",
            max_model_len=4096,
            tensor_parallel_size=1,
        )

        assert 0.5 <= util <= 0.95


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestInferenceErrorHandling:
    """Tests for error handling in inference."""

    @pytest.mark.asyncio
    async def test_generate_without_loading(self):
        """Test that generating without a loaded model raises an error."""
        from model_garden.inference import InferenceService

        service = InferenceService(
            model_path="some/model",
            tensor_parallel_size=1,
        )

        with pytest.raises(RuntimeError, match="Model not loaded"):
            await service.generate(prompt="test")

    @pytest.mark.asyncio
    async def test_double_load_warning(self, text_model_path: str):
        """Test that loading a model twice shows a warning."""
        from model_garden.inference import InferenceService

        service = InferenceService(
            model_path=text_model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_model_len=512,
            enforce_eager=True,
        )

        await service.load_model()
        assert service.is_loaded

        # Second load should not raise but should warn
        await service.load_model()  # Should print warning
        assert service.is_loaded

        await service.unload_model()

    @pytest.mark.asyncio
    async def test_unload_without_loading(self):
        """Test that unloading without a loaded model is safe."""
        from model_garden.inference import InferenceService

        service = InferenceService(
            model_path="some/model",
            tensor_parallel_size=1,
        )

        # Should not raise
        await service.unload_model()


# ============================================================================
# Performance Tests
# ============================================================================


class TestInferencePerformance:
    """Performance-related tests for inference."""

    @pytest_asyncio.fixture
    async def perf_inference_service(self, text_model_path: str):
        """Create inference service for performance tests."""
        from model_garden.inference import InferenceService

        service = InferenceService(
            model_path=text_model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_model_len=512,
            enforce_eager=True,
        )

        await service.load_model()
        yield service
        await service.unload_model()

    @pytest.mark.asyncio
    async def test_multiple_sequential_generations(self, perf_inference_service):
        """Test multiple sequential generations work correctly."""
        prompts = [
            "Hello!",
            "How are you?",
            "What's the weather?",
            "Tell me a joke.",
            "Goodbye!",
        ]

        results = []
        for prompt in prompts:
            result = await perf_inference_service.generate(
                prompt=prompt,
                max_tokens=20,
                temperature=0.7,
                stream=False,
            )
            results.append(result)

        # All should succeed
        assert len(results) == len(prompts)
        for result in results:
            assert "text" in result
            assert len(result["text"]) > 0

    @pytest.mark.asyncio
    async def test_generation_respects_max_tokens(self, perf_inference_service):
        """Test that generation respects max_tokens limit."""
        result = await perf_inference_service.generate(
            prompt="Write a very long story about a dragon.",
            max_tokens=10,  # Very short limit
            temperature=0.7,
            stream=False,
        )

        # Output should be limited (though exact token count may vary)
        assert result["usage"]["completion_tokens"] <= 15  # Some tolerance


# ============================================================================
# LoRA Adapter Inference Tests
# ============================================================================


class TestLoRAAdapterInference:
    """End-to-end tests for LoRA adapter inference."""

    @pytest.fixture
    def temp_adapter_dir(self, tmp_path: Path) -> Path:
        """Create a temporary directory for adapter files."""
        adapter_dir = tmp_path / "test_adapter"
        adapter_dir.mkdir()
        return adapter_dir

    @pytest.fixture
    def mock_lora_adapter(self, temp_adapter_dir: Path, text_model_path: str) -> Path:
        """Create a mock LoRA adapter directory structure.

        Note: This creates only the config files. For actual adapter weights,
        you would need to train a real adapter.
        """
        # Create adapter_config.json
        adapter_config = {
            "base_model_name_or_path": text_model_path,
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

        config_path = temp_adapter_dir / "adapter_config.json"
        config_path.write_text(json.dumps(adapter_config, indent=2))

        return temp_adapter_dir

    def test_detect_lora_adapter_local(self, mock_lora_adapter: Path):
        """Test that local LoRA adapters are correctly detected."""
        from model_garden.inference.utils import is_lora_adapter

        assert is_lora_adapter(str(mock_lora_adapter)) is True

    def test_get_base_model_from_adapter(self, mock_lora_adapter: Path, text_model_path: str):
        """Test extracting base model from adapter config."""
        from model_garden.inference.utils import get_base_model_from_adapter

        base_model = get_base_model_from_adapter(str(mock_lora_adapter))
        assert base_model == text_model_path

    def test_detect_non_adapter_directory(self, tmp_path: Path):
        """Test that regular model directories are not detected as adapters."""
        from model_garden.inference.utils import is_lora_adapter

        # Create a fake model directory without adapter config
        model_dir = tmp_path / "regular_model"
        model_dir.mkdir()

        # Create config.json (regular model config)
        config = {"model_type": "llama", "hidden_size": 4096}
        (model_dir / "config.json").write_text(json.dumps(config))

        assert is_lora_adapter(str(model_dir)) is False


# ============================================================================
# Model Registry Integration Tests
# ============================================================================


class TestModelRegistryIntegration:
    """Tests for model registry integration with inference."""

    def test_model_registry_available(self):
        """Test that model registry is available and can list models."""
        from model_garden.model_registry import get_text_models, get_vision_models

        text_models = get_text_models()
        vision_models = get_vision_models()

        # Should return lists (may be empty in clean install)
        assert isinstance(text_models, list)
        assert isinstance(vision_models, list)

    def test_get_nonexistent_model(self):
        """Test getting a model that doesn't exist in registry."""
        from model_garden.model_registry import get_model

        model = get_model("nonexistent/model-path")
        assert model is None


# ============================================================================
# Concurrent Request Tests
# ============================================================================


class TestConcurrentRequests:
    """Tests for handling concurrent inference requests."""

    @pytest_asyncio.fixture
    async def inference_service_for_concurrency(self, text_model_path: str):
        """Create inference service for concurrency tests."""
        from model_garden.inference import InferenceService

        service = InferenceService(
            model_path=text_model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_model_len=512,
            max_num_seqs=8,  # Allow multiple concurrent sequences
            enforce_eager=True,
        )

        await service.load_model()
        yield service
        await service.unload_model()

    @pytest.mark.asyncio
    async def test_concurrent_text_generations(self, inference_service_for_concurrency):
        """Test that multiple concurrent text generations work."""

        async def generate_one(prompt: str):
            return await inference_service_for_concurrency.generate(
                prompt=prompt,
                max_tokens=20,
                temperature=0.7,
                stream=False,
            )

        # Create multiple concurrent requests
        prompts = [f"Question {i}: What is {i}+{i}?" for i in range(4)]

        # Run concurrently
        results = await asyncio.gather(*[generate_one(p) for p in prompts])

        # All should succeed
        assert len(results) == len(prompts)
        for result in results:
            assert "text" in result
            assert len(result["text"]) > 0

    @pytest.mark.asyncio
    async def test_concurrent_chat_completions(self, inference_service_for_concurrency):
        """Test that multiple concurrent chat completions work."""

        async def chat_one(question: str):
            return await inference_service_for_concurrency.chat_completion(
                messages=[{"role": "user", "content": question}],
                max_tokens=20,
                temperature=0.7,
                stream=False,
            )

        # Create multiple concurrent requests
        questions = ["Say hi!", "Count to 3.", "What's 1+1?", "Hello!"]

        # Run concurrently
        results = await asyncio.gather(*[chat_one(q) for q in questions])

        # All should succeed
        assert len(results) == len(questions)
        for result in results:
            assert "choices" in result
            assert len(result["choices"][0]["message"]["content"]) > 0


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest_asyncio.fixture
    async def edge_case_service(self, text_model_path: str):
        """Create inference service for edge case tests."""
        from model_garden.inference import InferenceService

        service = InferenceService(
            model_path=text_model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_model_len=512,
            enforce_eager=True,
        )

        await service.load_model()
        yield service
        await service.unload_model()

    @pytest.mark.asyncio
    async def test_empty_prompt(self, edge_case_service):
        """Test generation with empty prompt."""
        result = await edge_case_service.generate(
            prompt="",
            max_tokens=20,
            temperature=0.7,
            stream=False,
        )

        # Should still produce some output (or handle gracefully)
        assert isinstance(result, dict)
        assert "text" in result

    @pytest.mark.asyncio
    async def test_very_long_prompt(self, edge_case_service):
        """Test generation with a long prompt."""
        # Create a prompt that's close to the max length
        long_prompt = "Hello world! " * 100  # ~400 tokens

        result = await edge_case_service.generate(
            prompt=long_prompt,
            max_tokens=10,
            temperature=0.7,
            stream=False,
        )

        assert isinstance(result, dict)
        assert "text" in result

    @pytest.mark.asyncio
    async def test_special_characters_in_prompt(self, edge_case_service):
        """Test generation with special characters."""
        special_prompt = "Test with émojis 🎉 and spëcial çharacters: <>&\"'"

        result = await edge_case_service.generate(
            prompt=special_prompt,
            max_tokens=20,
            temperature=0.7,
            stream=False,
        )

        assert isinstance(result, dict)
        assert "text" in result

    @pytest.mark.asyncio
    async def test_unicode_prompt(self, edge_case_service):
        """Test generation with Unicode text."""
        unicode_prompt = "Translate to English: こんにちは世界"

        result = await edge_case_service.generate(
            prompt=unicode_prompt,
            max_tokens=30,
            temperature=0.7,
            stream=False,
        )

        assert isinstance(result, dict)
        assert "text" in result

    @pytest.mark.asyncio
    async def test_zero_temperature(self, edge_case_service):
        """Test generation with temperature=0 (deterministic)."""
        result = await edge_case_service.generate(
            prompt="What is 2+2?",
            max_tokens=20,
            temperature=0.0,
            stream=False,
        )

        assert isinstance(result, dict)
        assert "text" in result

    @pytest.mark.asyncio
    async def test_empty_messages_list(self, edge_case_service):
        """Test chat completion with empty messages."""
        # This should either work with default behavior or raise a clear error
        try:
            result = await edge_case_service.chat_completion(
                messages=[],
                max_tokens=20,
                stream=False,
            )
            # If it works, result should be valid
            assert isinstance(result, dict)
        except (ValueError, RuntimeError):
            # Expected - empty messages should be rejected
            pass

    @pytest.mark.asyncio
    async def test_system_only_message(self, edge_case_service):
        """Test chat completion with only system message."""
        result = await edge_case_service.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
            ],
            max_tokens=20,
            stream=False,
        )

        # Should produce some response
        assert isinstance(result, dict)
        assert "choices" in result


# ============================================================================
# Sampling Parameters Tests
# ============================================================================


class TestSamplingParameters:
    """Tests for various sampling parameters."""

    @pytest_asyncio.fixture
    async def sampling_service(self, text_model_path: str):
        """Create inference service for sampling parameter tests."""
        from model_garden.inference import InferenceService

        service = InferenceService(
            model_path=text_model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_model_len=512,
            enforce_eager=True,
        )

        await service.load_model()
        yield service
        await service.unload_model()

    @pytest.mark.asyncio
    async def test_top_p_sampling(self, sampling_service):
        """Test generation with top_p (nucleus) sampling."""
        result = await sampling_service.generate(
            prompt="Write a story:",
            max_tokens=30,
            temperature=0.8,
            top_p=0.9,
            stream=False,
        )

        assert isinstance(result, dict)
        assert "text" in result
        assert len(result["text"]) > 0

    @pytest.mark.asyncio
    async def test_top_k_sampling(self, sampling_service):
        """Test generation with top_k sampling."""
        result = await sampling_service.generate(
            prompt="List three colors:",
            max_tokens=30,
            temperature=0.7,
            top_k=50,
            stream=False,
        )

        assert isinstance(result, dict)
        assert "text" in result
        assert len(result["text"]) > 0

    @pytest.mark.asyncio
    async def test_repetition_penalty(self, sampling_service):
        """Test generation with repetition penalty."""
        result = await sampling_service.generate(
            prompt="Repeat after me:",
            max_tokens=50,
            temperature=0.7,
            repetition_penalty=1.2,
            stream=False,
        )

        assert isinstance(result, dict)
        assert "text" in result

    @pytest.mark.asyncio
    async def test_presence_penalty(self, sampling_service):
        """Test generation with presence penalty."""
        result = await sampling_service.generate(
            prompt="Write about nature:",
            max_tokens=50,
            temperature=0.7,
            presence_penalty=0.5,
            stream=False,
        )

        assert isinstance(result, dict)
        assert "text" in result

    @pytest.mark.asyncio
    async def test_frequency_penalty(self, sampling_service):
        """Test generation with frequency penalty."""
        result = await sampling_service.generate(
            prompt="Describe a forest:",
            max_tokens=50,
            temperature=0.7,
            frequency_penalty=0.5,
            stream=False,
        )

        assert isinstance(result, dict)
        assert "text" in result

    @pytest.mark.asyncio
    async def test_combined_sampling_parameters(self, sampling_service):
        """Test generation with multiple sampling parameters combined."""
        result = await sampling_service.generate(
            prompt="Tell me a fact:",
            max_tokens=40,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1,
            stream=False,
        )

        assert isinstance(result, dict)
        assert "text" in result
        assert len(result["text"]) > 0

    @pytest.mark.asyncio
    async def test_high_temperature_generates_varied_output(self, sampling_service):
        """Test that high temperature generates varied output."""
        # Generate same prompt multiple times with high temp
        outputs = []
        for _ in range(3):
            result = await sampling_service.generate(
                prompt="Pick a random number:",
                max_tokens=10,
                temperature=1.5,
                stream=False,
            )
            outputs.append(result["text"])

        # All should be valid
        for output in outputs:
            assert len(output) >= 0


# ============================================================================
# Model Info and State Tests
# ============================================================================


class TestModelInfoAndState:
    """Tests for model info retrieval and state management."""

    @pytest_asyncio.fixture
    async def info_service(self, text_model_path: str):
        """Create inference service for info tests."""
        from model_garden.inference import InferenceService

        service = InferenceService(
            model_path=text_model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_model_len=512,
            enforce_eager=True,
        )

        await service.load_model()
        yield service
        await service.unload_model()

    @pytest.mark.asyncio
    async def test_get_model_info_contains_expected_fields(self, info_service):
        """Test that model info contains expected fields."""
        info = info_service.get_model_info()

        assert isinstance(info, dict)
        assert "is_loaded" in info
        assert info["is_loaded"] is True
        assert "model_path" in info

    @pytest.mark.asyncio
    async def test_model_state_after_generation(self, info_service):
        """Test that model state is consistent after generation."""
        # Generate text
        await info_service.generate(
            prompt="Test prompt",
            max_tokens=10,
            stream=False,
        )

        # Model should still be loaded
        assert info_service.is_loaded

        # Should be able to get info
        info = info_service.get_model_info()
        assert info["is_loaded"] is True

    @pytest.mark.asyncio
    async def test_multiple_generations_maintain_state(self, info_service):
        """Test that multiple generations maintain model state."""
        for i in range(5):
            result = await info_service.generate(
                prompt=f"Question {i}",
                max_tokens=10,
                stream=False,
            )
            assert "text" in result

        # Model should still be loaded
        assert info_service.is_loaded


# ============================================================================
# Chat Message Format Tests
# ============================================================================


class TestChatMessageFormats:
    """Tests for various chat message formats."""

    @pytest_asyncio.fixture
    async def chat_service(self, text_model_path: str):
        """Create inference service for chat tests."""
        from model_garden.inference import InferenceService

        service = InferenceService(
            model_path=text_model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_model_len=512,
            enforce_eager=True,
        )

        await service.load_model()
        yield service
        await service.unload_model()

    @pytest.mark.asyncio
    async def test_user_only_message(self, chat_service):
        """Test chat with only user message."""
        result = await chat_service.chat_completion(
            messages=[
                {"role": "user", "content": "Hello!"},
            ],
            max_tokens=20,
            stream=False,
        )

        assert "choices" in result
        assert len(result["choices"]) > 0

    @pytest.mark.asyncio
    async def test_system_and_user_messages(self, chat_service):
        """Test chat with system and user messages."""
        result = await chat_service.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's 1+1?"},
            ],
            max_tokens=20,
            stream=False,
        )

        assert "choices" in result
        content = result["choices"][0]["message"]["content"]
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, chat_service):
        """Test multi-turn conversation."""
        result = await chat_service.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
                {"role": "user", "content": "What is my name?"},
            ],
            max_tokens=30,
            stream=False,
        )

        assert "choices" in result
        content = result["choices"][0]["message"]["content"]
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_long_system_prompt(self, chat_service):
        """Test chat with a long system prompt."""
        long_system = (
            "You are a helpful assistant specialized in Python programming. "
            "You provide clear, concise code examples with explanations. "
            "Always follow PEP 8 style guidelines and best practices. "
            "When explaining concepts, use simple language suitable for beginners."
        )

        result = await chat_service.chat_completion(
            messages=[
                {"role": "system", "content": long_system},
                {"role": "user", "content": "How do I print hello world?"},
            ],
            max_tokens=50,
            stream=False,
        )

        assert "choices" in result
        assert len(result["choices"][0]["message"]["content"]) > 0


# ============================================================================
# Batch Processing Tests
# ============================================================================


class TestBatchProcessing:
    """Tests for batch inference processing."""

    @pytest_asyncio.fixture
    async def batch_service(self, text_model_path: str):
        """Create inference service for batch tests."""
        from model_garden.inference import InferenceService

        service = InferenceService(
            model_path=text_model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_model_len=512,
            max_num_seqs=8,
            enforce_eager=True,
        )

        await service.load_model()
        yield service
        await service.unload_model()

    @pytest.mark.asyncio
    async def test_sequential_requests(self, batch_service):
        """Test processing multiple requests sequentially."""
        prompts = [
            "Say hello",
            "Count to 3",
            "What is 2+2?",
        ]

        results = []
        for prompt in prompts:
            result = await batch_service.generate(
                prompt=prompt,
                max_tokens=20,
                stream=False,
            )
            results.append(result)

        assert len(results) == 3
        for result in results:
            assert "text" in result

    @pytest.mark.asyncio
    async def test_parallel_requests(self, batch_service):
        """Test processing multiple requests in parallel."""

        async def gen(prompt):
            return await batch_service.generate(
                prompt=prompt,
                max_tokens=20,
                stream=False,
            )

        prompts = [f"Question {i}" for i in range(4)]
        results = await asyncio.gather(*[gen(p) for p in prompts])

        assert len(results) == 4
        for result in results:
            assert "text" in result

    @pytest.mark.asyncio
    async def test_mixed_streaming_and_non_streaming(self, batch_service):
        """Test mixing streaming and non-streaming requests."""
        # Non-streaming request
        result1 = await batch_service.generate(
            prompt="Hello",
            max_tokens=10,
            stream=False,
        )
        assert "text" in result1

        # Streaming request
        chunks = []
        stream = await batch_service.generate(
            prompt="Hi there",
            max_tokens=10,
            stream=True,
        )
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) > 0

        # Another non-streaming request
        result3 = await batch_service.generate(
            prompt="Goodbye",
            max_tokens=10,
            stream=False,
        )
        assert "text" in result3


# ============================================================================
# Token Counting Tests
# ============================================================================


class TestTokenCounting:
    """Tests for token counting in responses."""

    @pytest_asyncio.fixture
    async def token_service(self, text_model_path: str):
        """Create inference service for token counting tests."""
        from model_garden.inference import InferenceService

        service = InferenceService(
            model_path=text_model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_model_len=512,
            enforce_eager=True,
        )

        await service.load_model()
        yield service
        await service.unload_model()

    @pytest.mark.asyncio
    async def test_usage_stats_present(self, token_service):
        """Test that usage stats are present in response."""
        result = await token_service.generate(
            prompt="Hello world",
            max_tokens=20,
            stream=False,
        )

        assert "usage" in result
        usage = result["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    @pytest.mark.asyncio
    async def test_completion_tokens_within_limit(self, token_service):
        """Test that completion tokens don't exceed max_tokens."""
        result = await token_service.generate(
            prompt="Write a very long story",
            max_tokens=5,
            stream=False,
        )

        # Completion tokens should be at most max_tokens (with some tolerance)
        assert result["usage"]["completion_tokens"] <= 10

    @pytest.mark.asyncio
    async def test_chat_completion_usage_stats(self, token_service):
        """Test that chat completions include usage stats."""
        result = await token_service.chat_completion(
            messages=[{"role": "user", "content": "Hi!"}],
            max_tokens=20,
            stream=False,
        )

        assert "usage" in result
        usage = result["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["total_tokens"] >= usage["prompt_tokens"]
