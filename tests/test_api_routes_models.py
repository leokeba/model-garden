"""Tests for models API routes.

These tests verify the model management endpoints work correctly.
"""

import json
from pathlib import Path
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
def mock_storage():
    """Mock storage manager."""
    with patch("model_garden.api.routes.models.get_storage_manager") as mock:
        storage = MagicMock()
        storage.load_models.return_value = {}
        storage.save_models.return_value = None
        storage.load_training_jobs.return_value = {}
        mock.return_value = storage
        yield storage


@pytest.fixture
def sample_models():
    """Sample models data."""
    return {
        "model-1": {
            "id": "model-1",
            "name": "Test Model 1",
            "path": "/models/model-1",
            "created_at": "2024-01-01T00:00:00Z",
            "model_type": "text",
            "base_model": "unsloth/tinyllama-bnb-4bit",
        },
        "model-2": {
            "id": "model-2",
            "name": "Vision Model",
            "path": "/models/model-2",
            "created_at": "2024-01-02T00:00:00Z",
            "model_type": "vision",
            "base_model": "Qwen/Qwen2.5-VL-3B-Instruct",
        },
    }


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary model directory with config files."""
    model_dir = tmp_path / "test-model"
    model_dir.mkdir()

    # Create a basic config.json
    config = {
        "architectures": ["LlamaForCausalLM"],
        "_name_or_path": "unsloth/tinyllama-bnb-4bit",
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    return model_dir


class TestListModels:
    """Tests for GET /api/v1/models."""

    def test_list_empty_models(self, client: TestClient):
        """Test listing when no models exist."""
        with patch("model_garden.api.routes.models.get_storage_manager") as mock_storage:
            storage = MagicMock()
            storage.load_models.return_value = {}
            storage.save_models.return_value = None
            storage.load_training_jobs.return_value = {}
            mock_storage.return_value = storage

            response = client.get("/api/v1/models")
            assert response.status_code == 200

            data = response.json()
            assert "items" in data

    def test_list_models_with_data(
        self, client: TestClient, sample_models
    ):
        """Test listing models with existing data."""
        with patch("model_garden.api.routes.models.get_storage_manager") as mock_storage:
            storage = MagicMock()
            storage.load_models.return_value = sample_models
            storage.save_models.return_value = None
            storage.load_training_jobs.return_value = {}
            mock_storage.return_value = storage

            response = client.get("/api/v1/models")
            assert response.status_code == 200

            data = response.json()
            # The mock models are mixed with real models from disk
            assert data["total"] >= 2

    def test_list_models_filter_by_type(
        self, client: TestClient, sample_models
    ):
        """Test filtering models by type."""
        with patch("model_garden.api.routes.models.get_storage_manager") as mock_storage:
            storage = MagicMock()
            storage.load_models.return_value = sample_models
            storage.save_models.return_value = None
            storage.load_training_jobs.return_value = {}
            mock_storage.return_value = storage

            response = client.get("/api/v1/models?model_type=vision")
            assert response.status_code == 200

            data = response.json()
            assert data["total"] >= 1
            # All returned items should be vision type
            for item in data["items"]:
                assert item["model_type"] == "vision"

    def test_list_models_pagination(
        self, client: TestClient, mock_storage, sample_models
    ):
        """Test pagination of models."""
        mock_storage.load_models.return_value = sample_models

        response = client.get("/api/v1/models?page=1&page_size=1")
        assert response.status_code == 200

        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 1
        assert len(data["items"]) == 1


class TestGetModel:
    """Tests for GET /api/v1/models/{model_id}."""

    def test_get_existing_model(
        self, client: TestClient, mock_storage, sample_models
    ):
        """Test getting an existing model."""
        mock_storage.load_models.return_value = sample_models

        response = client.get("/api/v1/models/model-1")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == "model-1"
        assert data["name"] == "Test Model 1"

    def test_get_nonexistent_model(self, client: TestClient, mock_storage):
        """Test getting a non-existent model."""
        mock_storage.load_models.return_value = {}

        response = client.get("/api/v1/models/nonexistent")
        assert response.status_code == 404


class TestRenameModel:
    """Tests for PUT /api/v1/models/{model_id}."""

    def test_rename_model(
        self, client: TestClient, mock_storage, sample_models
    ):
        """Test renaming a model."""
        mock_storage.load_models.return_value = sample_models.copy()

        response = client.put(
            "/api/v1/models/model-1",
            json={"name": "Renamed Model"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True

        # Verify storage was updated
        mock_storage.save_models.assert_called_once()

    def test_rename_nonexistent_model(self, client: TestClient, mock_storage):
        """Test renaming a non-existent model."""
        mock_storage.load_models.return_value = {}

        response = client.put(
            "/api/v1/models/nonexistent",
            json={"name": "New Name"},
        )
        assert response.status_code == 404


class TestDeleteModel:
    """Tests for DELETE /api/v1/models/{model_id}."""

    def test_delete_model_from_storage_only(
        self, client: TestClient, mock_storage, sample_models
    ):
        """Test deleting a model from storage only."""
        mock_storage.load_models.return_value = sample_models.copy()

        response = client.delete("/api/v1/models/model-1")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True

    def test_delete_model_with_files(
        self, client: TestClient, mock_storage, sample_models, temp_model_dir
    ):
        """Test deleting a model including files."""
        # Update the model path to point to temp dir
        models = sample_models.copy()
        models["model-1"]["path"] = str(temp_model_dir)
        mock_storage.load_models.return_value = models

        response = client.delete("/api/v1/models/model-1?delete_files=true")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "files removed" in data["message"]

    def test_delete_nonexistent_model(self, client: TestClient, mock_storage):
        """Test deleting a non-existent model."""
        mock_storage.load_models.return_value = {}

        response = client.delete("/api/v1/models/nonexistent")
        assert response.status_code == 404


class TestExtractModelMetadata:
    """Tests for extract_model_metadata utility."""

    def test_extract_from_adapter_config(self, tmp_path):
        """Test extracting metadata from adapter_config.json."""
        from model_garden.api.routes.models import extract_model_metadata

        model_dir = tmp_path / "lora-model"
        model_dir.mkdir()

        adapter_config = {
            "base_model_name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct",
            "auto_mapping": {"base_model_class": "Qwen2VLForConditionalGeneration"},
        }
        (model_dir / "adapter_config.json").write_text(json.dumps(adapter_config))

        metadata = extract_model_metadata(model_dir)

        assert metadata["is_adapter"] is True
        assert "Qwen" in metadata["base_model"]
        assert metadata["model_type"] == "vision"

    def test_extract_from_config_json(self, tmp_path):
        """Test extracting metadata from config.json."""
        from model_garden.api.routes.models import extract_model_metadata

        model_dir = tmp_path / "merged-model"
        model_dir.mkdir()

        config = {
            "_name_or_path": "unsloth/llama-3.2-1b",
            "architectures": ["LlamaForCausalLM"],
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        metadata = extract_model_metadata(model_dir)

        assert metadata["is_adapter"] is False
        assert metadata["base_model"] == "unsloth/llama-3.2-1b"
        assert metadata["model_type"] == "text"

    def test_extract_from_nonexistent_path(self, tmp_path):
        """Test extracting metadata from non-existent path."""
        from model_garden.api.routes.models import extract_model_metadata

        metadata = extract_model_metadata(tmp_path / "nonexistent")

        assert metadata["base_model"] is None
        assert metadata["model_type"] is None


class TestEnrichModelFromTrainingJobs:
    """Tests for enrich_model_from_training_jobs utility."""

    def test_enrich_by_training_job_id(self):
        """Test enriching model data by training job ID."""
        from model_garden.api.routes.models import enrich_model_from_training_jobs

        model_data = {
            "id": "my-model",
            "training_job_id": "job-1",
            "base_model": "unknown",
            "model_type": "unknown",
        }

        training_jobs = {
            "job-1": {
                "base_model": "unsloth/tinyllama-bnb-4bit",
                "is_vision": False,
            }
        }

        enriched = enrich_model_from_training_jobs(model_data, training_jobs)

        assert enriched["base_model"] == "unsloth/tinyllama-bnb-4bit"
        assert enriched["model_type"] == "text"

    def test_enrich_by_output_dir(self):
        """Test enriching model data by matching output directory."""
        from model_garden.api.routes.models import enrich_model_from_training_jobs

        model_data = {
            "id": "my-model",
            "path": "/models/output",
            "base_model": "unknown",
        }

        training_jobs = {
            "job-1": {
                "output_dir": "/models/output",
                "base_model": "unsloth/llama-3.2-1b",
                "is_vision": False,
            }
        }

        enriched = enrich_model_from_training_jobs(model_data, training_jobs)

        assert enriched["training_job_id"] == "job-1"
        assert enriched["base_model"] == "unsloth/llama-3.2-1b"

    def test_enrich_no_match(self):
        """Test enriching when no matching job found."""
        from model_garden.api.routes.models import enrich_model_from_training_jobs

        model_data = {
            "id": "my-model",
            "path": "/models/some-path",
            "base_model": "unknown",
        }

        training_jobs = {
            "job-1": {
                "output_dir": "/models/different-path",
                "base_model": "unsloth/llama-3.2-1b",
            }
        }

        enriched = enrich_model_from_training_jobs(model_data, training_jobs)

        assert "training_job_id" not in enriched
        assert enriched["base_model"] == "unknown"


class TestGetModelFilesInfo:
    """Tests for get_model_files_info utility."""

    def test_get_files_info(self, tmp_path):
        """Test getting model files information."""
        from model_garden.api.routes.models import get_model_files_info

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Create some model files
        (model_dir / "config.json").write_text("{}")
        (model_dir / "model.safetensors").write_bytes(b"x" * 1000)
        (model_dir / "adapter_config.json").write_text("{}")

        info = get_model_files_info(model_dir)

        assert info["file_count"] == 3
        assert info["total_size"] > 0
        assert info["has_adapter"] is True
        assert info["has_safetensors"] is True

    def test_get_files_info_nonexistent(self, tmp_path):
        """Test getting files info for non-existent path."""
        from model_garden.api.routes.models import get_model_files_info

        info = get_model_files_info(tmp_path / "nonexistent")

        assert info["file_count"] == 0
        assert info["total_size"] == 0


class TestUploadToHub:
    """Tests for POST /api/v1/models/{model_id}/upload-to-hub."""

    def test_upload_no_token(self, client: TestClient, tmp_path, sample_models):
        """Test uploading without HuggingFace token."""
        # Create a valid model path
        model_path = tmp_path / "model-1"
        model_path.mkdir()
        (model_path / "config.json").write_text('{}')
        
        sample_models["model-1"]["path"] = str(model_path)
        
        with patch("model_garden.api.routes.models.get_storage_manager") as mock_storage:
            storage = MagicMock()
            storage.load_models.return_value = sample_models
            mock_storage.return_value = storage

            with patch("model_garden.api.routes.models.get_hf_token", return_value=None):
                response = client.post("/api/v1/models/model-1/upload-to-hub")
                assert response.status_code == 400
                assert "token" in response.json()["detail"].lower()

    def test_upload_nonexistent_model(self, client: TestClient, mock_storage):
        """Test uploading a non-existent model."""
        mock_storage.load_models.return_value = {}

        response = client.post("/api/v1/models/nonexistent/upload-to-hub")
        assert response.status_code == 404

    def test_upload_model_path_not_exists(
        self, client: TestClient, sample_models
    ):
        """Test uploading when model path doesn't exist."""
        models = sample_models.copy()
        models["model-1"]["path"] = "/nonexistent/path"

        with patch("model_garden.api.routes.models.get_storage_manager") as mock_storage:
            storage = MagicMock()
            storage.load_models.return_value = models
            mock_storage.return_value = storage

            with patch(
                "model_garden.api.routes.models.get_hf_token", return_value="hf_test_token"
            ):
                response = client.post("/api/v1/models/model-1/upload-to-hub")
                assert response.status_code == 400
                assert "path does not exist" in response.json()["detail"]
