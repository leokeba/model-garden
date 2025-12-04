"""Tests for training API routes.

These tests verify the training job management endpoints work correctly.
"""

import json
from pathlib import Path
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
def mock_storage():
    """Mock storage manager."""
    with patch("model_garden.api.routes.training.get_storage_manager") as mock:
        storage = MagicMock()
        storage.load_training_jobs.return_value = {}
        storage.save_training_jobs.return_value = None
        mock.return_value = storage
        yield storage


@pytest.fixture
def mock_job_queue():
    """Mock job queue."""
    with patch("model_garden.queue.get_job_queue") as mock:
        queue = AsyncMock()
        queue.add_job = AsyncMock()
        queue.get_queue_position = AsyncMock(return_value=1)
        queue.cancel_job = AsyncMock(return_value=True)
        queue.list_jobs = AsyncMock(return_value=[])
        mock.return_value = queue
        yield queue


@pytest.fixture
def mock_websocket_manager():
    """Mock WebSocket connection manager."""
    with patch("model_garden.api.routes.training.get_connection_manager") as mock:
        manager = MagicMock()
        manager.send_update = AsyncMock()
        mock.return_value = manager
        yield manager


@pytest.fixture
def sample_training_jobs():
    """Sample training jobs data."""
    return {
        "job-1": {
            "id": "job-1",
            "name": "Test Job 1",
            "status": "completed",
            "base_model": "unsloth/tinyllama-bnb-4bit",
            "dataset_path": "/data/train.jsonl",
            "output_dir": "/models/output1",
            "created_at": "2024-01-01T00:00:00Z",
            "hyperparameters": {},
            "lora_config": {},
        },
        "job-2": {
            "id": "job-2",
            "name": "Test Job 2",
            "status": "running",
            "base_model": "unsloth/llama-3.2-1b",
            "dataset_path": "/data/train2.jsonl",
            "output_dir": "/models/output2",
            "created_at": "2024-01-02T00:00:00Z",
            "hyperparameters": {},
            "lora_config": {},
        },
        "job-3": {
            "id": "job-3",
            "name": "Test Job 3",
            "status": "queued",
            "base_model": "unsloth/qwen2.5-1.5b",
            "dataset_path": "/data/train3.jsonl",
            "output_dir": "/models/output3",
            "created_at": "2024-01-03T00:00:00Z",
            "hyperparameters": {},
            "lora_config": {},
        },
    }


class TestListTrainingJobs:
    """Tests for GET /api/v1/training/jobs."""

    def test_list_empty_jobs(self, client: TestClient, mock_storage):
        """Test listing when no jobs exist."""
        mock_storage.load_training_jobs.return_value = {}

        response = client.get("/api/v1/training/jobs")
        assert response.status_code == 200

        data = response.json()
        assert "items" in data
        assert data["total"] == 0

    def test_list_jobs_with_data(
        self, client: TestClient, mock_storage, sample_training_jobs
    ):
        """Test listing jobs with existing data."""
        mock_storage.load_training_jobs.return_value = sample_training_jobs

        response = client.get("/api/v1/training/jobs")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 3
        assert len(data["items"]) == 3

    def test_list_jobs_with_status_filter(
        self, client: TestClient, mock_storage, sample_training_jobs
    ):
        """Test filtering jobs by status."""
        mock_storage.load_training_jobs.return_value = sample_training_jobs

        response = client.get("/api/v1/training/jobs?status_filter=running")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["status"] == "running"

    def test_list_jobs_pagination(
        self, client: TestClient, mock_storage, sample_training_jobs
    ):
        """Test pagination of jobs."""
        mock_storage.load_training_jobs.return_value = sample_training_jobs

        response = client.get("/api/v1/training/jobs?page=1&page_size=2")
        assert response.status_code == 200

        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 2
        assert len(data["items"]) == 2

    def test_list_jobs_sorted_by_created_at(
        self, client: TestClient, mock_storage, sample_training_jobs
    ):
        """Test jobs are sorted by creation date (newest first)."""
        mock_storage.load_training_jobs.return_value = sample_training_jobs

        response = client.get("/api/v1/training/jobs")
        assert response.status_code == 200

        data = response.json()
        items = data["items"]
        # Should be sorted newest first
        assert items[0]["created_at"] >= items[1]["created_at"]


class TestCreateTrainingJob:
    """Tests for POST /api/v1/training/jobs."""

    def test_create_job_minimal(
        self, client: TestClient, mock_storage, mock_job_queue
    ):
        """Test creating a job with minimal required fields."""
        mock_storage.load_training_jobs.return_value = {}

        response = client.post(
            "/api/v1/training/jobs",
            json={
                "name": "New Training Job",
                "base_model": "unsloth/tinyllama-bnb-4bit",
                "dataset_path": "/data/train.jsonl",
                "output_dir": "/models/output",
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "job_id" in data["data"]
        assert "queue_position" in data["data"]

    def test_create_job_full_config(
        self, client: TestClient, mock_storage, mock_job_queue
    ):
        """Test creating a job with full configuration."""
        mock_storage.load_training_jobs.return_value = {}

        response = client.post(
            "/api/v1/training/jobs",
            json={
                "name": "Full Training Job",
                "base_model": "Qwen/Qwen2.5-VL-3B-Instruct",
                "dataset_path": "/data/vision.jsonl",
                "validation_dataset_path": "/data/val.jsonl",
                "output_dir": "/models/output",
                "hyperparameters": {
                    "learning_rate": 2e-5,
                    "num_epochs": 3,
                },
                "lora_config": {
                    "r": 16,
                    "lora_alpha": 32,
                },
                "is_vision": True,
                "save_method": "lora",
                "selective_loss": True,
                "quality_mode": True,
                "early_stopping_enabled": True,
                "early_stopping_patience": 5,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True

    def test_create_job_hub_dataset(
        self, client: TestClient, mock_storage, mock_job_queue
    ):
        """Test creating a job with HuggingFace Hub dataset."""
        mock_storage.load_training_jobs.return_value = {}

        response = client.post(
            "/api/v1/training/jobs",
            json={
                "name": "Hub Dataset Job",
                "base_model": "unsloth/tinyllama-bnb-4bit",
                "dataset_path": "tatsu-lab/alpaca",
                "output_dir": "/models/output",
                "from_hub": True,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True

    def test_create_job_missing_required_fields(self, client: TestClient):
        """Test creating a job with missing required fields."""
        response = client.post(
            "/api/v1/training/jobs",
            json={
                "name": "Incomplete Job",
            },
        )
        assert response.status_code == 422  # Validation error

    def test_create_job_saves_to_storage(
        self, client: TestClient, mock_storage, mock_job_queue
    ):
        """Test that job is saved to storage."""
        mock_storage.load_training_jobs.return_value = {}

        response = client.post(
            "/api/v1/training/jobs",
            json={
                "name": "Test Job",
                "base_model": "unsloth/tinyllama-bnb-4bit",
                "dataset_path": "/data/train.jsonl",
                "output_dir": "/models/output",
            },
        )
        assert response.status_code == 200

        # Verify storage was called
        mock_storage.save_training_jobs.assert_called_once()


class TestGetTrainingJob:
    """Tests for GET /api/v1/training/jobs/{job_id}."""

    def test_get_existing_job(
        self, client: TestClient, mock_storage, sample_training_jobs
    ):
        """Test getting an existing job."""
        mock_storage.load_training_jobs.return_value = sample_training_jobs

        response = client.get("/api/v1/training/jobs/job-1")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == "job-1"
        assert data["name"] == "Test Job 1"

    def test_get_nonexistent_job(self, client: TestClient, mock_storage):
        """Test getting a non-existent job."""
        mock_storage.load_training_jobs.return_value = {}

        response = client.get("/api/v1/training/jobs/nonexistent")
        assert response.status_code == 404

    def test_get_queued_job_includes_position(
        self, client: TestClient, mock_storage, mock_job_queue, sample_training_jobs
    ):
        """Test that queued jobs include queue position."""
        mock_storage.load_training_jobs.return_value = sample_training_jobs
        mock_job_queue.get_queue_position.return_value = 2

        response = client.get("/api/v1/training/jobs/job-3")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "queued"
        assert data.get("queue_position") == 2


class TestDeleteTrainingJob:
    """Tests for DELETE /api/v1/training/jobs/{job_id}."""

    def test_delete_completed_job(
        self,
        client: TestClient,
        mock_storage,
        mock_job_queue,
        mock_websocket_manager,
        sample_training_jobs,
    ):
        """Test deleting a completed job."""
        mock_storage.load_training_jobs.return_value = sample_training_jobs.copy()

        response = client.delete("/api/v1/training/jobs/job-1")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "deleted" in data["message"]

    def test_cancel_queued_job(
        self,
        client: TestClient,
        mock_storage,
        mock_job_queue,
        mock_websocket_manager,
        sample_training_jobs,
    ):
        """Test cancelling a queued job."""
        mock_storage.load_training_jobs.return_value = sample_training_jobs.copy()

        response = client.delete("/api/v1/training/jobs/job-3")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True

    def test_delete_nonexistent_job(
        self, client: TestClient, mock_storage, mock_job_queue
    ):
        """Test deleting a non-existent job."""
        mock_storage.load_training_jobs.return_value = {}

        response = client.delete("/api/v1/training/jobs/nonexistent")
        assert response.status_code == 404


class TestEarlyStopJob:
    """Tests for POST /api/v1/training/jobs/{job_id}/stop."""

    def test_early_stop_running_job(
        self,
        client: TestClient,
        mock_storage,
        mock_websocket_manager,
        sample_training_jobs,
    ):
        """Test requesting early stop for a running job."""
        mock_storage.load_training_jobs.return_value = sample_training_jobs

        response = client.post("/api/v1/training/jobs/job-2/stop")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True

    def test_early_stop_non_running_job(
        self, client: TestClient, mock_storage, sample_training_jobs
    ):
        """Test early stop for a non-running job fails."""
        mock_storage.load_training_jobs.return_value = sample_training_jobs

        response = client.post("/api/v1/training/jobs/job-1/stop")  # completed
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is False

    def test_early_stop_nonexistent_job(self, client: TestClient, mock_storage):
        """Test early stop for non-existent job."""
        mock_storage.load_training_jobs.return_value = {}

        response = client.post("/api/v1/training/jobs/nonexistent/stop")
        assert response.status_code == 404


class TestRerunTrainingJob:
    """Tests for POST /api/v1/training/jobs/{job_id}/rerun."""

    def test_rerun_completed_job(
        self, client: TestClient, mock_storage, mock_job_queue, sample_training_jobs
    ):
        """Test rerunning a completed job."""
        mock_storage.load_training_jobs.return_value = sample_training_jobs.copy()

        response = client.post("/api/v1/training/jobs/job-1/rerun")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "job_id" in data["data"]
        assert data["data"]["original_job_id"] == "job-1"

    def test_rerun_running_job_fails(
        self, client: TestClient, mock_storage, sample_training_jobs
    ):
        """Test that rerunning a running job fails."""
        mock_storage.load_training_jobs.return_value = sample_training_jobs

        response = client.post("/api/v1/training/jobs/job-2/rerun")  # running
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is False

    def test_rerun_nonexistent_job(self, client: TestClient, mock_storage):
        """Test rerunning a non-existent job."""
        mock_storage.load_training_jobs.return_value = {}

        response = client.post("/api/v1/training/jobs/nonexistent/rerun")
        assert response.status_code == 404


class TestTrainingQueue:
    """Tests for GET /api/v1/training/queue."""

    def test_get_queue_status(self, client: TestClient, mock_job_queue):
        """Test getting queue status."""
        mock_job_queue.list_jobs.return_value = []

        response = client.get("/api/v1/training/queue")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "queued" in data["data"]
        assert "running" in data["data"]

    def test_get_queue_with_jobs(self, client: TestClient, mock_job_queue):
        """Test getting queue status with jobs."""
        mock_job_queue.list_jobs.side_effect = [
            [  # queued jobs
                {
                    "job_id": "job-1",
                    "job_config": {"name": "Queued Job"},
                    "queued_at": "2024-01-01T00:00:00Z",
                    "priority": 0,
                }
            ],
            [  # running jobs
                {
                    "job_id": "job-2",
                    "job_config": {"name": "Running Job"},
                    "started_at": "2024-01-01T00:00:00Z",
                    "status_message": "Training...",
                }
            ],
        ]

        response = client.get("/api/v1/training/queue")
        assert response.status_code == 200

        data = response.json()
        assert data["data"]["queued"] == 1
        assert data["data"]["running"] == 1


class TestPathResolution:
    """Tests for path resolution utilities."""

    def test_resolve_relative_path(self):
        """Test resolving relative paths."""
        from model_garden.api.routes.training import resolve_path

        path = resolve_path("data/train.jsonl")
        assert Path(path).is_absolute()

    def test_resolve_absolute_path(self):
        """Test resolving absolute paths."""
        from model_garden.api.routes.training import resolve_path

        path = resolve_path("/absolute/path/train.jsonl")
        assert path == "/absolute/path/train.jsonl"

    def test_resolve_model_path_simple_name(self):
        """Test resolving simple model names."""
        from model_garden.api.routes.training import resolve_model_path

        path = resolve_model_path("my-model")
        assert "models/my-model" in path

    def test_resolve_model_path_with_slash(self):
        """Test resolving paths with slashes."""
        from model_garden.api.routes.training import resolve_model_path

        path = resolve_model_path("./models/my-model")
        assert Path(path).is_absolute()


class TestCreateTrainingJobRecord:
    """Tests for creating training job records."""

    def test_create_job_record(self):
        """Test creating a complete job record."""
        from model_garden.api.models import TrainingJobRequest
        from model_garden.api.routes.training import create_training_job_record

        request = TrainingJobRequest(
            name="Test Job",
            base_model="unsloth/tinyllama-bnb-4bit",
            dataset_path="/data/train.jsonl",
            output_dir="/models/output",
            hyperparameters={"learning_rate": 2e-5},
            lora_config={"r": 16},
            is_vision=False,
            save_method="merged_16bit",
        )

        record = create_training_job_record(
            job_id="test-id",
            job_request=request,
            dataset_path="/data/train.jsonl",
            validation_dataset_path=None,
            output_dir="/models/output",
        )

        assert record.id == "test-id"
        assert record.name == "Test Job"
        assert record.status == "queued"
        assert record.base_model == "unsloth/tinyllama-bnb-4bit"
        assert record.hyperparameters == {"learning_rate": 2e-5}

    def test_create_vision_job_record(self):
        """Test creating a vision job record."""
        from model_garden.api.models import TrainingJobRequest
        from model_garden.api.routes.training import create_training_job_record

        request = TrainingJobRequest(
            name="Vision Job",
            base_model="Qwen/Qwen2.5-VL-3B-Instruct",
            dataset_path="/data/vision.jsonl",
            output_dir="/models/output",
            is_vision=True,
            save_method="lora",
        )

        record = create_training_job_record(
            job_id="vision-id",
            job_request=request,
            dataset_path="/data/vision.jsonl",
            validation_dataset_path=None,
            output_dir="/models/output",
        )

        assert record.is_vision is True
        assert record.model_type == "vision"
        assert record.save_method == "lora"
