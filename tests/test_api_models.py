"""Tests for model_garden.api.models module."""

from typing import Any

import pytest
from pydantic import ValidationError

from model_garden.api.models.common import APIResponse, PaginatedResponse
from model_garden.api.models.datasets import (
    DatasetValidationRequest,
    DatasetValidationResponse,
)
from model_garden.api.models.inference import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    CompletionRequest,
)
from model_garden.api.models.models import ModelInfo, ModelRenameRequest
from model_garden.api.models.training import TrainingJobInfo, TrainingJobRequest


class TestAPIResponse:
    """Tests for APIResponse model."""

    def test_success_response(self):
        """Test creating a success response."""
        response = APIResponse(
            success=True,
            message="Operation completed",
            data={"result": "value"},
        )
        assert response.success is True
        assert response.message == "Operation completed"
        assert response.data == {"result": "value"}

    def test_error_response(self):
        """Test creating an error response."""
        response = APIResponse(
            success=False,
            message="Something went wrong",
        )
        assert response.success is False
        assert response.data is None

    def test_data_optional(self):
        """Test that data field is optional."""
        response = APIResponse(success=True, message="OK")
        assert response.data is None

    def test_required_fields(self):
        """Test that success and message are required."""
        with pytest.raises(ValidationError):
            APIResponse()  # type: ignore[call-arg]


class TestPaginatedResponse:
    """Tests for PaginatedResponse model."""

    def test_basic_pagination(self):
        """Test creating a paginated response."""
        response = PaginatedResponse(
            items=[{"id": 1}, {"id": 2}],
            total=10,
            page=1,
            page_size=2,
            pages=5,
        )
        assert len(response.items) == 2
        assert response.total == 10
        assert response.page == 1
        assert response.page_size == 2
        assert response.pages == 5

    def test_empty_items(self):
        """Test pagination with empty items."""
        response = PaginatedResponse(
            items=[],
            total=0,
            page=1,
            page_size=10,
            pages=0,
        )
        assert response.items == []
        assert response.total == 0


class TestModelInfo:
    """Tests for ModelInfo model."""

    def test_full_model_info(self):
        """Test creating model info with all fields."""
        info = ModelInfo(
            id="model-123",
            name="my-model",
            base_model="unsloth/llama-3.2-1b",
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-02T00:00:00Z",
            size_bytes=1000000,
            path="/models/my-model",
            training_job_id="job-456",
            config={"type": "lora"},
            metrics={"loss": 0.5},
        )
        assert info.id == "model-123"
        assert info.name == "my-model"
        assert info.size_bytes == 1000000

    def test_minimal_model_info(self):
        """Test creating model info with required fields only."""
        info = ModelInfo(
            id="model-123",
            name="my-model",
            base_model="llama",
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            path="/models/my-model",
        )
        assert info.size_bytes is None
        assert info.training_job_id is None
        assert info.config is None


class TestModelRenameRequest:
    """Tests for ModelRenameRequest model."""

    def test_rename_request(self):
        """Test creating a rename request."""
        request = ModelRenameRequest(name="new-model-name")
        assert request.name == "new-model-name"

    def test_name_required(self):
        """Test that name is required."""
        with pytest.raises(ValidationError):
            ModelRenameRequest()  # type: ignore[call-arg]


class TestTrainingJobRequest:
    """Tests for TrainingJobRequest model."""

    def test_minimal_request(self):
        """Test creating a request with required fields only."""
        request = TrainingJobRequest(
            name="my-training-job",
            base_model="unsloth/tinyllama-bnb-4bit",
            dataset_path="/data/train.jsonl",
            output_dir="/models/output",
        )
        assert request.name == "my-training-job"
        assert request.save_method == "merged_16bit"
        assert request.backend == "unsloth"
        assert request.is_vision is False

    def test_full_request(self):
        """Test creating a request with all fields."""
        request = TrainingJobRequest(
            name="full-job",
            base_model="Qwen/Qwen2.5-VL-3B-Instruct",
            dataset_path="/data/vision.jsonl",
            validation_dataset_path="/data/val.jsonl",
            output_dir="/models/output",
            hyperparameters={"learning_rate": 2e-5},
            lora_config={"r": 16, "alpha": 32},
            from_hub=True,
            validation_from_hub=True,
            is_vision=True,
            model_type="vision",
            save_method="lora",
            backend="transformers",
            selective_loss=True,
            selective_loss_level="aggressive",
            quality_mode=True,
            early_stopping_enabled=True,
            early_stopping_patience=5,
        )
        assert request.is_vision is True
        assert request.selective_loss is True
        assert request.quality_mode is True
        assert request.early_stopping_patience == 5

    def test_default_values(self):
        """Test default values are set correctly."""
        request = TrainingJobRequest(
            name="test",
            base_model="model",
            dataset_path="/data/test.jsonl",
            output_dir="/output",
        )
        assert request.validation_dataset_path is None
        assert request.hyperparameters is None
        assert request.from_hub is False
        assert request.selective_loss is False
        assert request.selective_loss_level == "conservative"
        assert request.selective_loss_masking_strategy == "epoch_based"
        assert request.early_stopping_enabled is False
        assert request.load_in_16bit is False
        assert request.load_in_8bit is False


class TestTrainingJobInfo:
    """Tests for TrainingJobInfo model."""

    def test_minimal_info(self):
        """Test creating job info with required fields."""
        info = TrainingJobInfo(
            id="job-123",
            name="my-job",
            status="running",
            base_model="llama",
            dataset_path="/data/train.jsonl",
            output_dir="/output",
            created_at="2024-01-01T00:00:00Z",
        )
        assert info.id == "job-123"
        assert info.status == "running"

    def test_full_info(self):
        """Test creating job info with all fields."""
        info = TrainingJobInfo(
            id="job-123",
            name="my-job",
            status="completed",
            base_model="llama",
            dataset_path="/data/train.jsonl",
            output_dir="/output",
            created_at="2024-01-01T00:00:00Z",
            started_at="2024-01-01T00:05:00Z",
            completed_at="2024-01-01T01:00:00Z",
            progress={"step": 100, "total": 100},
            current_step=100,
            total_steps=100,
            current_epoch=3,
            metrics={"train_loss": 0.5, "eval_loss": 0.6},
            queue_position=None,
            rerun_from="job-original",
        )
        assert info.completed_at is not None
        assert info.current_step == 100
        assert info.rerun_from == "job-original"


class TestDatasetValidationRequest:
    """Tests for DatasetValidationRequest model."""

    def test_minimal_request(self):
        """Test request with just path."""
        request = DatasetValidationRequest(path="/data/train.jsonl")
        assert request.path == "/data/train.jsonl"
        assert request.schema_type is None

    def test_with_schema_type(self):
        """Test request with schema type."""
        request = DatasetValidationRequest(
            path="/data/vision.jsonl",
            schema_type="vision",
        )
        assert request.schema_type == "vision"


class TestDatasetValidationResponse:
    """Tests for DatasetValidationResponse model."""

    def test_valid_response(self):
        """Test creating a valid response."""
        response = DatasetValidationResponse(
            valid=True,
            total_rows=100,
            format="jsonl",
            schema_type="alpaca",
            fields=["instruction", "input", "output"],
            field_types={"instruction": "str", "input": "str", "output": "str"},
            missing_fields={},
            sample_rows=[{"instruction": "test", "input": "", "output": "result"}],
            file_size_bytes=5000,
            errors=[],
            warnings=[],
            avg_input_length=50.5,
            avg_output_length=100.2,
            total_tokens_estimate=15000,
        )
        assert response.valid is True
        assert response.total_rows == 100
        assert response.has_images is False

    def test_vision_response(self):
        """Test vision dataset response."""
        response = DatasetValidationResponse(
            valid=True,
            total_rows=50,
            format="jsonl",
            schema_type="vision",
            fields=["text", "image", "response"],
            field_types={"text": "str", "image": "str", "response": "str"},
            missing_fields={},
            sample_rows=[],
            file_size_bytes=10000,
            errors=[],
            warnings=[],
            has_images=True,
            image_count=50,
        )
        assert response.has_images is True
        assert response.image_count == 50

    def test_invalid_response(self):
        """Test invalid dataset response."""
        response = DatasetValidationResponse(
            valid=False,
            total_rows=0,
            format="unknown",
            schema_type="unknown",
            fields=[],
            field_types={},
            missing_fields={"instruction": 100},
            sample_rows=[],
            file_size_bytes=0,
            errors=["Missing required field: instruction"],
            warnings=["File appears empty"],
        )
        assert response.valid is False
        assert len(response.errors) == 1


class TestChatCompletionMessage:
    """Tests for ChatCompletionMessage model."""

    def test_text_message(self):
        """Test simple text message."""
        message = ChatCompletionMessage(
            role="user",
            content="Hello, how are you?",
        )
        assert message.role == "user"
        assert message.content == "Hello, how are you?"

    def test_multimodal_message(self):
        """Test multimodal message with image."""
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        ]
        message = ChatCompletionMessage(
            role="user",
            content=content,
        )
        assert isinstance(message.content, list)
        assert len(message.content) == 2


class TestChatCompletionRequest:
    """Tests for ChatCompletionRequest model."""

    def test_minimal_request(self):
        """Test minimal chat completion request."""
        request = ChatCompletionRequest(
            messages=[
                ChatCompletionMessage(role="user", content="Hello"),
            ],
        )
        assert request.model == "default"
        assert len(request.messages) == 1
        assert request.temperature == 0.7
        assert request.stream is False

    def test_full_request(self):
        """Test chat completion with all options."""
        request = ChatCompletionRequest(
            model="my-model",
            messages=[
                ChatCompletionMessage(role="system", content="You are helpful."),
                ChatCompletionMessage(role="user", content="What is 2+2?"),
            ],
            max_tokens=100,
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            repetition_penalty=1.1,
            stop=["END", "\n\n"],
            stream=True,
        )
        assert request.max_tokens == 100
        assert request.temperature == 0.5
        assert request.stream is True
        assert len(request.stop) == 2

    def test_structured_output_request(self):
        """Test request with structured output."""
        request = ChatCompletionRequest(
            messages=[
                ChatCompletionMessage(role="user", content="Generate JSON"),
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                },
            },
        )
        assert request.response_format is not None
        assert request.response_format["type"] == "json_schema"

    def test_guided_generation(self):
        """Test request with guided generation."""
        request = ChatCompletionRequest(
            messages=[
                ChatCompletionMessage(role="user", content="Pick a color"),
            ],
            guided_choice=["red", "green", "blue"],
        )
        assert request.guided_choice == ["red", "green", "blue"]


class TestCompletionRequest:
    """Tests for CompletionRequest model."""

    def test_minimal_request(self):
        """Test minimal completion request."""
        request = CompletionRequest(prompt="Once upon a time")
        assert request.prompt == "Once upon a time"
        assert request.model == "default"

    def test_with_images(self):
        """Test completion with images for vision models."""
        request = CompletionRequest(
            prompt="Describe this image",
            images=["data:image/png;base64,..."],
        )
        assert request.images is not None
        assert len(request.images) == 1

    def test_with_structured_output(self):
        """Test completion with structured output."""
        request = CompletionRequest(
            prompt="Generate JSON",
            response_format={"type": "json_object"},
            guided_json={"type": "object"},
        )
        assert request.response_format is not None
        assert request.guided_json is not None

    def test_with_regex_guide(self):
        """Test completion with regex guidance."""
        request = CompletionRequest(
            prompt="Generate a phone number",
            guided_regex=r"\d{3}-\d{3}-\d{4}",
        )
        assert request.guided_regex == r"\d{3}-\d{3}-\d{4}"
