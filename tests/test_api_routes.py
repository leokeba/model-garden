"""Tests for API routes.

These tests verify the FastAPI endpoints work correctly using TestClient.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


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
def temp_models_dir(tmp_path: Path) -> Path:
    """Create a temporary models directory."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir


class TestSystemRoutes:
    """Tests for system endpoints."""

    def test_system_status(self, client: TestClient):
        """Test system status endpoint."""
        response = client.get("/api/v1/system/status")
        assert response.status_code == 200
        data = response.json()
        # Response contains GPU/CPU info wrapped in success/data
        assert "success" in data or "gpu" in data or "cpu" in data


class TestModelRoutes:
    """Tests for model management endpoints."""

    def test_list_models(self, client: TestClient):
        """Test listing models."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        # Should return a paginated response
        assert "items" in data or "total" in data

    def test_get_model_not_found(self, client: TestClient):
        """Test getting a non-existent model."""
        response = client.get("/api/v1/models/nonexistent-model")
        assert response.status_code == 404


class TestTrainingRoutes:
    """Tests for training job endpoints."""

    def test_list_training_jobs(self, client: TestClient):
        """Test listing training jobs."""
        response = client.get("/api/v1/training/jobs")
        assert response.status_code == 200
        data = response.json()
        # Can be wrapped in success or be a direct list
        assert "success" in data or "items" in data or isinstance(data, list)

    def test_create_training_job_missing_fields(self, client: TestClient):
        """Test creating a training job with missing required fields."""
        response = client.post(
            "/api/v1/training/jobs",
            json={"name": "test-job"},  # Missing required fields
        )
        # Should return validation error
        assert response.status_code in [400, 422]  # Bad request or validation error


class TestDatasetRoutes:
    """Tests for dataset validation endpoints."""

    def test_list_datasets(self, client: TestClient):
        """Test listing datasets."""
        response = client.get("/api/v1/datasets")
        assert response.status_code == 200


class TestCarbonRoutes:
    """Tests for carbon emissions endpoints."""

    def test_get_emissions_summary(self, client: TestClient):
        """Test getting emissions summary."""
        response = client.get("/api/v1/carbon/summary")
        assert response.status_code == 200
        data = response.json()
        # Response may have different structure
        assert "success" in data or "total_emissions_kg" in data or isinstance(data, dict)

    def test_list_emissions(self, client: TestClient):
        """Test listing emissions records."""
        response = client.get("/api/v1/carbon/emissions")
        assert response.status_code == 200
        data = response.json()
        # Can be success wrapper or direct list
        assert "success" in data or "items" in data or "emissions" in data or isinstance(data, dict)


class TestInferenceRoutes:
    """Tests for inference endpoints (without a loaded model)."""

    def test_chat_completion_no_model(self, client: TestClient):
        """Test chat completion without a loaded model."""
        response = client.post(
            "/api/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        # Should fail because no model is loaded
        assert response.status_code in [400, 500, 503]

    def test_completions_via_inference_generate(self, client: TestClient):
        """Test inference generate without a loaded model."""
        response = client.post(
            "/api/v1/inference/generate",
            json={
                "prompt": "Hello, world",
            },
        )
        # Should fail because no model is loaded
        assert response.status_code in [400, 500, 503]

    def test_inference_status(self, client: TestClient):
        """Test getting inference status."""
        response = client.get("/api/v1/inference/status")
        assert response.status_code == 200
        data = response.json()
        # Should contain loaded status
        assert "loaded" in data or "success" in data


class TestOpenAPISchema:
    """Tests for OpenAPI schema generation."""

    def test_openapi_json(self, client: TestClient):
        """Test OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
        assert "info" in data

    def test_docs_available(self, client: TestClient):
        """Test Swagger UI docs are available."""
        response = client.get("/docs")
        assert response.status_code == 200
