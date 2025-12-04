"""Tests for datasets API routes.

These tests verify the dataset management endpoints work correctly.
"""

from io import BytesIO
from unittest.mock import patch

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


class TestListDatasets:
    """Tests for GET /api/v1/datasets."""

    def test_list_datasets(self, client: TestClient):
        """Test listing datasets."""
        response = client.get("/api/v1/datasets")
        assert response.status_code == 200

        data = response.json()
        assert "datasets" in data


class TestUploadDataset:
    """Tests for POST /api/v1/datasets/upload."""

    def test_upload_invalid_extension(self, client: TestClient):
        """Test uploading file with invalid extension."""
        content = b"some content"

        response = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("test.exe", BytesIO(content), "application/octet-stream")},
        )

        assert response.status_code == 400
        assert "Invalid file format" in response.json()["detail"]

    def test_upload_no_filename(self, client: TestClient):
        """Test uploading without a filename."""
        content = b'{"test": "data"}'

        # Create file without name
        response = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("", BytesIO(content), "application/json")},
        )

        # Either 400 (no filename) or 422 (validation error) is acceptable
        assert response.status_code in [400, 422]


class TestGetDatasetStats:
    """Tests for GET /api/v1/datasets/{dataset_name}/stats."""

    def test_get_stats_nonexistent(self, client: TestClient):
        """Test getting stats for non-existent dataset."""
        response = client.get("/api/v1/datasets/nonexistent.jsonl/stats")
        assert response.status_code == 404


class TestPreviewDataset:
    """Tests for GET /api/v1/datasets/{dataset_name}/preview."""

    def test_preview_nonexistent(self, client: TestClient):
        """Test previewing non-existent dataset."""
        response = client.get("/api/v1/datasets/nonexistent.jsonl/preview")
        assert response.status_code == 404


class TestDeleteDataset:
    """Tests for DELETE /api/v1/datasets/{dataset_name}."""

    def test_delete_nonexistent(self, client: TestClient):
        """Test deleting non-existent dataset."""
        response = client.delete("/api/v1/datasets/nonexistent.jsonl")
        assert response.status_code == 404


class TestLoadFromHub:
    """Tests for POST /api/v1/datasets/from-hub."""

    def test_load_from_hub_missing_id(self, client: TestClient):
        """Test loading from Hub without dataset_id."""
        response = client.post(
            "/api/v1/datasets/from-hub",
            json={},
        )
        # Server returns error for missing dataset_id
        assert response.status_code in [400, 500]
        # The error message should indicate the issue
        detail = response.json().get("detail", "")
        assert "dataset_id" in detail.lower() or "required" in detail.lower()

    def test_load_from_hub_error(self, client: TestClient):
        """Test handling Hub loading errors."""
        with patch("datasets.load_dataset", side_effect=Exception("Hub error")):
            response = client.post(
                "/api/v1/datasets/from-hub",
                json={"dataset_id": "invalid/dataset"},
            )
            assert response.status_code == 500
