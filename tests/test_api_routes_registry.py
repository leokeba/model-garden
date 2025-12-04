"""Tests for registry API routes.

These tests verify the model registry endpoints work correctly.
"""

from unittest.mock import MagicMock, patch

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
def mock_registry():
    """Mock model registry."""
    with patch("model_garden.api.routes.registry.get_registry") as mock:
        registry = MagicMock()
        registry.get_model_list_for_ui.return_value = []
        registry.get_model.return_value = None
        registry.get_categories.return_value = {}
        registry.validate_model_for_training.return_value = (True, None)
        registry.validate_model_for_inference.return_value = (True, None)
        mock.return_value = registry
        yield registry


@pytest.fixture
def sample_model_info():
    """Create a sample model info object."""
    model_info = MagicMock()
    model_info.id = "unsloth/tinyllama-bnb-4bit"
    model_info.name = "TinyLlama 1.1B BNB 4-bit"
    model_info.category = "tinyllama"
    model_info.provider = "Unsloth"
    model_info.base_architecture = "llama"
    model_info.parameters = "1.1B"
    model_info.description = "TinyLlama optimized with Unsloth"
    model_info.tags = ["llama", "tiny", "4bit"]
    model_info.status = "active"
    model_info.is_vision_model = False
    model_info.is_quantized = True
    model_info.urls = {"huggingface": "https://huggingface.co/unsloth/tinyllama-bnb-4bit"}

    # Requirements
    requirements = MagicMock()
    requirements.min_vram_gb = 4
    requirements.recommended_vram_gb = 8
    requirements.min_ram_gb = 8
    model_info.requirements = requirements

    # Capabilities
    capabilities = MagicMock()
    capabilities.vision = False
    capabilities.function_calling = False
    capabilities.structured_outputs = True
    capabilities.streaming = True
    model_info.capabilities = capabilities

    # Inference defaults
    inference_defaults = MagicMock()
    inference_defaults.max_model_len = 4096
    inference_defaults.dtype = "bfloat16"
    inference_defaults.gpu_memory_utilization = 0.9
    inference_defaults.max_num_seqs = 16
    inference_defaults.enforce_eager = False
    inference_defaults.limit_mm_per_prompt = None
    inference_defaults.quantization = None
    inference_defaults.tensor_parallel_size = 1
    model_info.inference_defaults = inference_defaults

    # Training defaults
    model_info.training_defaults = {"save_method": "merged_16bit"}
    model_info.get_training_hyperparameters.return_value = {"learning_rate": 2e-4}
    model_info.get_lora_config.return_value = {"r": 16}
    model_info.get_inference_config.return_value = {"dtype": "bfloat16"}

    return model_info


class TestListRegistryModels:
    """Tests for GET /api/v1/registry/models."""

    def test_list_empty_registry(self, client: TestClient, mock_registry):
        """Test listing when registry is empty."""
        response = client.get("/api/v1/registry/models")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["total"] == 0

    def test_list_models_with_data(
        self, client: TestClient, mock_registry, sample_model_info
    ):
        """Test listing models with existing data."""
        mock_registry.get_model_list_for_ui.return_value = [
            {
                "id": "unsloth/tinyllama-bnb-4bit",
                "name": "TinyLlama 1.1B BNB 4-bit",
                "category": "tinyllama",
            }
        ]
        mock_registry.get_model.return_value = sample_model_info

        response = client.get("/api/v1/registry/models")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["total"] == 1
        assert len(data["data"]) == 1

    def test_list_models_filter_by_category(
        self, client: TestClient, mock_registry, sample_model_info
    ):
        """Test filtering models by category."""
        mock_registry.get_model_list_for_ui.return_value = [
            {
                "id": "unsloth/tinyllama-bnb-4bit",
                "name": "TinyLlama",
                "category": "tinyllama",
            }
        ]
        mock_registry.get_model.return_value = sample_model_info

        response = client.get("/api/v1/registry/models?category=tinyllama")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        mock_registry.get_model_list_for_ui.assert_called_with(category="tinyllama")

    def test_list_models_includes_training_defaults(
        self, client: TestClient, mock_registry, sample_model_info
    ):
        """Test that response includes training defaults."""
        mock_registry.get_model_list_for_ui.return_value = [
            {"id": "unsloth/tinyllama-bnb-4bit", "name": "TinyLlama"}
        ]
        mock_registry.get_model.return_value = sample_model_info

        response = client.get("/api/v1/registry/models")
        assert response.status_code == 200

        data = response.json()
        model = data["data"][0]
        assert "training_defaults" in model
        assert "inference_defaults" in model
        assert "requirements" in model
        assert "capabilities" in model

    def test_list_models_registry_not_found(self, client: TestClient, mock_registry):
        """Test handling when registry file is not found."""
        mock_registry.get_model_list_for_ui.side_effect = FileNotFoundError("Not found")

        response = client.get("/api/v1/registry/models")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["total"] == 0


class TestGetRegistryModel:
    """Tests for GET /api/v1/registry/models/{model_id}."""

    def test_get_existing_model(
        self, client: TestClient, mock_registry, sample_model_info
    ):
        """Test getting an existing model."""
        mock_registry.get_model.return_value = sample_model_info

        response = client.get("/api/v1/registry/models/unsloth/tinyllama-bnb-4bit")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["data"]["id"] == "unsloth/tinyllama-bnb-4bit"

    def test_get_nonexistent_model(self, client: TestClient, mock_registry):
        """Test getting a non-existent model."""
        mock_registry.get_model.return_value = None

        response = client.get("/api/v1/registry/models/nonexistent/model")
        assert response.status_code == 404

    def test_get_model_full_details(
        self, client: TestClient, mock_registry, sample_model_info
    ):
        """Test that model details include all expected fields."""
        mock_registry.get_model.return_value = sample_model_info

        response = client.get("/api/v1/registry/models/unsloth/tinyllama-bnb-4bit")
        assert response.status_code == 200

        data = response.json()["data"]
        assert "id" in data
        assert "name" in data
        assert "category" in data
        assert "provider" in data
        assert "requirements" in data
        assert "capabilities" in data
        assert "training_defaults" in data
        assert "inference_defaults" in data
        assert "urls" in data


class TestListCategories:
    """Tests for GET /api/v1/registry/categories."""

    def test_list_categories_empty(self, client: TestClient, mock_registry):
        """Test listing categories when none exist."""
        mock_registry.get_categories.return_value = {}

        response = client.get("/api/v1/registry/categories")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["data"] == {}

    def test_list_categories_with_data(self, client: TestClient, mock_registry):
        """Test listing categories with existing data."""
        mock_registry.get_categories.return_value = {
            "tinyllama": {"name": "TinyLlama", "count": 3},
            "llama": {"name": "Llama", "count": 10},
        }

        response = client.get("/api/v1/registry/categories")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "tinyllama" in data["data"]
        assert "llama" in data["data"]

    def test_list_categories_registry_not_found(
        self, client: TestClient, mock_registry
    ):
        """Test handling when registry file is not found."""
        mock_registry.get_categories.side_effect = FileNotFoundError("Not found")

        response = client.get("/api/v1/registry/categories")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["data"] == {}


class TestValidateForTraining:
    """Tests for POST /api/v1/registry/validate/training."""

    def test_validate_valid_model(self, client: TestClient, mock_registry):
        """Test validating a valid model for training."""
        mock_registry.validate_model_for_training.return_value = (True, None)

        response = client.post(
            "/api/v1/registry/validate/training",
            json={"model_id": "unsloth/tinyllama-bnb-4bit"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["valid"] is True
        assert data["errors"] == []

    def test_validate_invalid_model(self, client: TestClient, mock_registry):
        """Test validating an invalid model for training."""
        mock_registry.validate_model_for_training.return_value = (
            False,
            "Model not supported for training",
        )

        response = client.post(
            "/api/v1/registry/validate/training",
            json={"model_id": "invalid/model"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["valid"] is False
        assert len(data["errors"]) == 1

    def test_validate_with_config_warnings(
        self, client: TestClient, mock_registry, sample_model_info
    ):
        """Test validation with config that triggers warnings."""
        mock_registry.validate_model_for_training.return_value = (True, None)
        mock_registry.get_model.return_value = sample_model_info

        response = client.post(
            "/api/v1/registry/validate/training",
            json={
                "model_id": "unsloth/tinyllama-bnb-4bit",
                "config": {"batch_size": 8},  # Large batch size
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["valid"] is True
        assert len(data["warnings"]) > 0


class TestValidateForInference:
    """Tests for POST /api/v1/registry/validate/inference."""

    def test_validate_valid_model(self, client: TestClient, mock_registry):
        """Test validating a valid model for inference."""
        mock_registry.validate_model_for_inference.return_value = (True, None)

        response = client.post(
            "/api/v1/registry/validate/inference",
            json={"model_id": "unsloth/tinyllama-bnb-4bit"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["valid"] is True
        assert data["errors"] == []

    def test_validate_invalid_model(self, client: TestClient, mock_registry):
        """Test validating an invalid model for inference."""
        mock_registry.validate_model_for_inference.return_value = (
            False,
            "Model not supported for inference",
        )

        response = client.post(
            "/api/v1/registry/validate/inference",
            json={"model_id": "invalid/model"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["valid"] is False
        assert len(data["errors"]) == 1

    def test_validate_missing_model_id(self, client: TestClient, mock_registry):
        """Test validation with missing model_id."""
        response = client.post(
            "/api/v1/registry/validate/inference",
            json={},
        )
        assert response.status_code == 422  # Validation error
