"""Tests for inference API routes.

These tests verify the inference and model serving endpoints work correctly.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Mark all tests to bypass mock_heavy_imports
pytestmark = pytest.mark.requires_gpu


@pytest.fixture
def app():
    """Create a test FastAPI application."""
    from model_garden.api.app import create_app

    return create_app()


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_inference_service():
    """Mock inference service."""
    with patch("model_garden.inference.get_inference_service") as mock:
        service = MagicMock()
        service.is_loaded = False
        service.model_path = None
        mock.return_value = service
        yield service, mock


@pytest.fixture
def mock_loaded_inference_service():
    """Mock inference service with a loaded model."""
    with patch("model_garden.inference.get_inference_service") as mock:
        service = MagicMock()
        service.is_loaded = True
        service.model_path = "/models/test-model"
        service.get_model_info.return_value = {
            "model_path": "/models/test-model",
            "dtype": "bfloat16",
        }
        service.generate = AsyncMock(return_value="Generated text")
        service.chat_completion = AsyncMock(
            return_value={
                "text": "Hello! How can I help you?",
                "finish_reason": "stop",
            }
        )
        mock.return_value = service
        yield service, mock


@pytest.fixture
def mock_job_queue():
    """Mock job queue."""
    with patch("model_garden.queue.get_job_queue") as mock:
        queue = AsyncMock()
        queue.add_job = AsyncMock()
        queue.get_running_job = AsyncMock(return_value=None)
        queue.list_jobs = AsyncMock(return_value=[])
        mock.return_value = queue
        yield queue


class TestInferenceStatus:
    """Tests for GET /api/v1/inference/status."""

    def test_status_no_model_loaded(
        self, client: TestClient, mock_inference_service, mock_job_queue
    ):
        """Test status when no model is loaded."""
        service, _ = mock_inference_service
        service.is_loaded = False

        response = client.get("/api/v1/inference/status")
        assert response.status_code == 200

        data = response.json()
        assert data["loaded"] is False

    def test_status_model_loaded(
        self, client: TestClient, mock_loaded_inference_service, mock_job_queue
    ):
        """Test status when a model is loaded."""
        service, _ = mock_loaded_inference_service

        response = client.get("/api/v1/inference/status")
        assert response.status_code == 200

        data = response.json()
        assert data["loaded"] is True
        assert data["model_info"] is not None


class TestLoadModel:
    """Tests for POST /api/v1/inference/load."""

    def test_load_model_already_loaded(
        self, client: TestClient, mock_loaded_inference_service, mock_job_queue
    ):
        """Test loading when a model is already loaded."""
        response = client.post(
            "/api/v1/inference/load",
            json={"model_path": "/models/new-model"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is False
        assert "already loaded" in data["message"]

    def test_load_model_loading_in_progress(
        self, client: TestClient, mock_inference_service, mock_job_queue
    ):
        """Test loading when another model is loading."""
        mock_job_queue.get_running_job.return_value = {
            "job_id": "loading-1",
            "job_config": {"model_path": "/models/other"},
        }

        response = client.post(
            "/api/v1/inference/load",
            json={"model_path": "/models/new-model"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is False
        assert "already loading" in data["message"]

    def test_load_model_starts_loading(
        self, client: TestClient, mock_inference_service, mock_job_queue
    ):
        """Test successfully starting model loading."""
        with patch("model_garden.api.run_model_loading"):
            response = client.post(
                "/api/v1/inference/load",
                json={
                    "model_path": "/models/new-model",
                    "tensor_parallel_size": 1,
                    "gpu_memory_utilization": 0.9,
                },
            )
            assert response.status_code == 200

            data = response.json()
            assert data["success"] is True
            assert "loading started" in data["message"]


class TestUnloadModel:
    """Tests for POST /api/v1/inference/unload."""

    def test_unload_no_model(self, client: TestClient, mock_inference_service):
        """Test unloading when no model is loaded."""
        response = client.post("/api/v1/inference/unload")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is False
        assert "No model" in data["message"]

    def test_unload_model_success(self, client: TestClient, mock_loaded_inference_service):
        """Test successfully unloading a model."""
        service, _ = mock_loaded_inference_service
        service.unload_model = AsyncMock()

        with patch("model_garden.inference.set_inference_service"):
            with patch("model_garden.carbon.stop_inference_tracker", return_value=None):
                response = client.post("/api/v1/inference/unload")
                assert response.status_code == 200

                data = response.json()
                assert data["success"] is True
                assert "unloaded" in data["message"]


class TestInferenceGenerate:
    """Tests for POST /api/v1/inference/generate."""

    def test_generate_no_model(self, client: TestClient, mock_inference_service):
        """Test generation when no model is loaded."""
        response = client.post(
            "/api/v1/inference/generate",
            json={"prompt": "Hello, world!"},
        )
        assert response.status_code == 400
        assert "No model loaded" in response.json()["detail"]

    def test_generate_success(self, client: TestClient, mock_loaded_inference_service):
        """Test successful text generation."""
        response = client.post(
            "/api/v1/inference/generate",
            json={
                "prompt": "Hello, world!",
                "max_tokens": 100,
                "temperature": 0.7,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert "text" in data

    def test_generate_with_all_params(self, client: TestClient, mock_loaded_inference_service):
        """Test generation with all parameters."""
        response = client.post(
            "/api/v1/inference/generate",
            json={
                "prompt": "Hello!",
                "max_tokens": 50,
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 50,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.5,
                "stop": [".", "!"],
                "stream": False,
            },
        )
        assert response.status_code == 200


class TestChatCompletions:
    """Tests for POST /api/v1/chat/completions."""

    def test_chat_no_model(self, client: TestClient, mock_inference_service):
        """Test chat when no model is loaded."""
        response = client.post(
            "/api/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 400
        assert "No model loaded" in response.json()["detail"]

    def test_chat_success(self, client: TestClient, mock_loaded_inference_service):
        """Test successful chat completion."""
        response = client.post(
            "/api/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello!"},
                ],
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert "choices" in data

    def test_chat_with_multimodal_content(self, client: TestClient, mock_loaded_inference_service):
        """Test chat with multimodal content (text + image)."""
        response = client.post(
            "/api/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                                },
                            },
                        ],
                    }
                ],
            },
        )
        assert response.status_code == 200

    def test_chat_with_structured_output(self, client: TestClient, mock_loaded_inference_service):
        """Test chat with structured output response format."""
        response = client.post(
            "/api/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Generate JSON"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "person",
                        "schema": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                        },
                    },
                },
            },
        )
        assert response.status_code == 200


class TestOpenAIChatCompletions:
    """Tests for POST /v1/chat/completions (OpenAI standard path)."""

    def test_openai_path_works(self, client: TestClient, mock_loaded_inference_service):
        """Test the OpenAI-standard path works."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 200


class TestModelLoadingQueue:
    """Tests for GET /api/v1/inference/queue."""

    def test_get_queue_empty(self, client: TestClient, mock_job_queue):
        """Test getting empty queue."""
        response = client.get("/api/v1/inference/queue")
        assert response.status_code == 200

        data = response.json()
        assert "summary" in data
        assert data["summary"]["running_count"] == 0

    def test_get_queue_with_jobs(self, client: TestClient, mock_job_queue):
        """Test getting queue with jobs."""
        from model_garden.queue import JobStatus

        mock_job_queue.list_jobs.return_value = [
            {
                "job_id": "job-1",
                "status": JobStatus.RUNNING,
                "job_config": {"model_path": "/models/model-1"},
                "started_at": "2024-01-01T00:00:00Z",
            }
        ]

        response = client.get("/api/v1/inference/queue")
        assert response.status_code == 200


class TestResponseFormatConversion:
    """Tests for response_format to structured_outputs conversion."""

    def test_convert_text_format(self):
        """Test converting text format (no structured output)."""
        from model_garden.api.routes.inference import (
            ResponseFormat,
            convert_response_format_to_structured_outputs,
        )

        response_format = ResponseFormat(type="text")
        result = convert_response_format_to_structured_outputs(response_format)
        assert result is None

    def test_convert_json_object_format(self):
        """Test converting json_object format."""
        from model_garden.api.routes.inference import (
            ResponseFormat,
            convert_response_format_to_structured_outputs,
        )

        response_format = ResponseFormat(type="json_object")
        result = convert_response_format_to_structured_outputs(response_format)
        assert result is not None
        assert result["type"] == "json_object"

    def test_convert_json_schema_format(self):
        """Test converting json_schema format."""
        from model_garden.api.routes.inference import (
            ResponseFormat,
            convert_response_format_to_structured_outputs,
        )

        response_format = ResponseFormat(
            type="json_schema",
            json_schema={
                "name": "test",
                "schema": {"type": "object", "properties": {}},
            },
        )
        result = convert_response_format_to_structured_outputs(response_format)
        assert result is not None
        assert result["type"] == "json_schema"
        assert "json_schema" in result

    def test_convert_none_format(self):
        """Test converting None format."""
        from model_garden.api.routes.inference import (
            convert_response_format_to_structured_outputs,
        )

        result = convert_response_format_to_structured_outputs(None)
        assert result is None


class TestEnsureOpenAIChatResponseFormat:
    """Tests for ensuring OpenAI-compatible chat response format."""

    def test_already_openai_format(self):
        """Test response already in OpenAI format."""
        from model_garden.api.routes.inference import (
            ensure_openai_chat_response_format,
        )

        response = {
            "id": "chat-123",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
        }

        result = ensure_openai_chat_response_format(response, "test-model")
        assert result == response

    def test_convert_text_to_openai_format(self):
        """Test converting text response to OpenAI format."""
        from model_garden.api.routes.inference import (
            ensure_openai_chat_response_format,
        )

        response = {"text": "Hello, world!"}

        result = ensure_openai_chat_response_format(response, "test-model")

        assert "choices" in result
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["content"] == "Hello, world!"

    def test_preserve_carbon_trace(self):
        """Test that carbon trace is preserved."""
        from model_garden.api.routes.inference import (
            ensure_openai_chat_response_format,
        )

        response = {
            "text": "Hello!",
            "x_carbon_trace": {"emissions_g_co2": 0.001},
        }

        result = ensure_openai_chat_response_format(response, "test-model")

        assert "x_carbon_trace" in result
        assert result["x_carbon_trace"]["emissions_g_co2"] == 0.001


class TestResolveModelPath:
    """Tests for model path resolution."""

    def test_resolve_huggingface_id(self):
        """Test resolving HuggingFace model ID."""
        from model_garden.api.routes.inference import resolve_model_path

        result = resolve_model_path("meta-llama/Llama-3.2-1B")
        assert result == "meta-llama/Llama-3.2-1B"

    def test_resolve_local_path(self):
        """Test resolving local path."""
        from model_garden.api.routes.inference import resolve_model_path

        result = resolve_model_path("/absolute/path/model")
        assert result == "/absolute/path/model"

    def test_resolve_relative_path(self):
        """Test resolving relative path."""
        from model_garden.api.routes.inference import resolve_model_path

        result = resolve_model_path("./models/my-model")
        assert "/models/my-model" in result
