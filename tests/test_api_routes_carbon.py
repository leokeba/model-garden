"""Tests for carbon emissions API routes.

These tests verify the carbon tracking endpoints work correctly.
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
def mock_storage():
    """Mock storage manager."""
    with patch("model_garden.api.routes.carbon.get_storage_manager") as mock:
        storage = MagicMock()
        storage.load_training_jobs.return_value = {}
        mock.return_value = storage
        yield storage


@pytest.fixture
def mock_emissions_db():
    """Mock emissions database."""
    with patch("model_garden.carbon.get_emissions_db") as mock:
        db = MagicMock()
        db.get_all_emissions.return_value = []
        db.get_total_emissions.return_value = {
            "total_emissions_kg_co2": 0.0,
            "total_energy_kwh": 0.0,
            "total_duration_seconds": 0.0,
            "total_count": 0,
            "by_type": {},
            "equivalents": {},
        }
        db.get_emission.return_value = None
        mock.return_value = db
        yield db


@pytest.fixture
def sample_emissions():
    """Sample emissions data."""
    return [
        {
            "job_id": "job-1",
            "job_type": "training",
            "model_name": "Test Model 1",
            "base_model": "unsloth/tinyllama-bnb-4bit",
            "timestamp": "2024-01-01T00:00:00Z",
            "duration_seconds": 3600,
            "energy_consumed_kwh": 0.5,
            "emissions_kg_co2": 0.2,
            "emissions_rate_kg_per_sec": 0.0001,
            "cpu_energy_kwh": 0.1,
            "gpu_energy_kwh": 0.35,
            "ram_energy_kwh": 0.05,
            "carbon_intensity_g_per_kwh": 400.0,
            "country_name": "France",
            "region": "Europe",
            "equivalents": {},
        },
        {
            "job_id": "job-2",
            "job_type": "inference",
            "model_name": "Test Model 2",
            "timestamp": "2024-01-02T00:00:00Z",
            "duration_seconds": 1800,
            "energy_consumed_kwh": 0.25,
            "emissions_kg_co2": 0.1,
            "emissions_rate_kg_per_sec": 0.00005,
            "carbon_intensity_g_per_kwh": 400.0,
            "country_name": "France",
        },
    ]


class TestListEmissions:
    """Tests for GET /api/v1/carbon/emissions."""

    def test_list_empty_emissions(self, client: TestClient, mock_storage, mock_emissions_db):
        """Test listing when no emissions exist."""
        response = client.get("/api/v1/carbon/emissions")
        assert response.status_code == 200

        data = response.json()
        assert "emissions" in data
        assert data["count"] == 0

    def test_list_emissions_with_data(
        self, client: TestClient, mock_storage, mock_emissions_db, sample_emissions
    ):
        """Test listing emissions with existing data."""
        mock_emissions_db.get_all_emissions.return_value = sample_emissions

        response = client.get("/api/v1/carbon/emissions")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 2
        assert len(data["emissions"]) == 2

    def test_list_emissions_filter_by_type(
        self, client: TestClient, mock_storage, mock_emissions_db, sample_emissions
    ):
        """Test filtering emissions by job type."""
        mock_emissions_db.get_all_emissions.return_value = [sample_emissions[0]]

        response = client.get("/api/v1/carbon/emissions?job_type=training")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 1
        assert data["emissions"][0]["stage"] == "training"

    def test_list_emissions_with_limit(
        self, client: TestClient, mock_storage, mock_emissions_db, sample_emissions
    ):
        """Test limiting emissions results."""
        mock_emissions_db.get_all_emissions.return_value = sample_emissions[:1]

        response = client.get("/api/v1/carbon/emissions?limit=1")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 1


class TestEmissionsSummary:
    """Tests for GET /api/v1/carbon/summary."""

    def test_get_empty_summary(self, client: TestClient, mock_storage, mock_emissions_db):
        """Test getting summary when no emissions exist."""
        response = client.get("/api/v1/carbon/summary")
        assert response.status_code == 200

        data = response.json()
        assert data["total_emissions_kg_co2"] == 0.0
        assert data["total_energy_kwh"] == 0.0

    def test_get_summary_with_data(self, client: TestClient, mock_storage, mock_emissions_db):
        """Test getting summary with emissions data."""
        mock_emissions_db.get_total_emissions.return_value = {
            "total_emissions_kg_co2": 0.5,
            "total_energy_kwh": 1.0,
            "total_duration_seconds": 7200,
            "total_count": 5,
            "by_type": {"training": 3, "inference": 2},
            "equivalents": {"km_driven": 2.5, "tree_months": 0.1},
        }

        response = client.get("/api/v1/carbon/summary")
        assert response.status_code == 200

        data = response.json()
        assert data["total_emissions_kg_co2"] == 0.5
        assert data["total_count"] == 5


class TestInferenceStats:
    """Tests for GET /api/v1/carbon/inference/stats."""

    def test_get_stats_no_tracker(self, client: TestClient):
        """Test getting stats when no tracker is active."""
        with patch("model_garden.carbon.get_inference_tracker", return_value=None):
            response = client.get("/api/v1/carbon/inference/stats")
            assert response.status_code == 200

            data = response.json()
            assert data["tracking"] is False

    def test_get_stats_with_tracker(self, client: TestClient):
        """Test getting stats when tracker is active."""
        mock_tracker = MagicMock()
        mock_tracker.get_current_stats.return_value = {
            "emissions_kg_co2": 0.01,
            "energy_consumed_kwh": 0.05,
            "duration_seconds": 600,
        }

        with patch("model_garden.carbon.get_inference_tracker", return_value=mock_tracker):
            response = client.get("/api/v1/carbon/inference/stats")
            assert response.status_code == 200

            data = response.json()
            assert data["tracking"] is True
            assert "emissions_kg_co2" in data


class TestBoampsReport:
    """Tests for GET /api/v1/carbon/boamps/{job_id}."""

    def test_get_report_not_found(self, client: TestClient, mock_storage, mock_emissions_db):
        """Test getting report for non-existent job."""
        mock_emissions_db.get_emission.return_value = None

        response = client.get("/api/v1/carbon/boamps/nonexistent")
        assert response.status_code == 404

    def test_get_report_success(
        self, client: TestClient, mock_storage, mock_emissions_db, sample_emissions
    ):
        """Test getting a valid BoAmps report."""
        mock_emissions_db.get_emission.return_value = sample_emissions[0]

        mock_generator = MagicMock()
        mock_generator.generate_report.return_value = {
            "version": "1.1.0",
            "task": {},
            "energy": {},
            "infrastructure": {},
        }

        with patch("model_garden.carbon.get_boamps_generator", return_value=mock_generator):
            response = client.get("/api/v1/carbon/boamps/job-1")
            assert response.status_code == 200

            data = response.json()
            assert "version" in data

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_get_report_includes_dataset_info(
        self,
        mock_hw_detector,
        client: TestClient,
        mock_storage,
        mock_emissions_db,
        sample_emissions,
    ):
        """BoAmps report should include dataset metadata from training job storage."""

        # Minimal hardware info to satisfy generator
        hw = MagicMock()
        hw.get_gpu_info.return_value = {}
        hw.get_cpu_info.return_value = {"family": "CPU"}
        hw.get_ram_info.return_value = {"total_gb": 16}
        hw.get_system_info.return_value = {
            "os_name": "Linux",
            "os_version": "6.5",
            "python_version": "3.11",
        }
        mock_hw_detector.return_value = hw

        # Emission and training job metadata
        mock_emissions_db.get_emission.return_value = sample_emissions[0]
        mock_storage.load_training_jobs.return_value = {
            "job-1": {
                "base_model": "meta-llama/Meta-Llama-3-8B",
                "dataset_path": "org/sample-dataset",
                "from_hub": True,
                "dataset_size": 1024**3,  # 1 GiB
                "dataset_num_samples": 42,
            }
        }

        response = client.get("/api/v1/carbon/boamps/job-1")
        assert response.status_code == 200

        report = response.json()
        dataset = report["task"]["dataset"]

        assert dataset
        first = dataset[0]
        assert first["dataSize"] == pytest.approx(1.0)
        assert first["dataQuantity"] == 42
        assert first["source"] == "public"
        assert first.get("sourceUri", "").endswith("org/sample-dataset")


class TestAnalyticsTrends:
    """Tests for GET /api/v1/carbon/analytics/trends."""

    def test_get_trends_empty(self, client: TestClient, mock_storage, mock_emissions_db):
        """Test getting trends when no data exists."""
        response = client.get("/api/v1/carbon/analytics/trends")
        assert response.status_code == 200

        data = response.json()
        assert data["data_points"] == []
        assert data["totals"]["emissions_kg"] == 0

    def test_get_trends_with_data(
        self, client: TestClient, mock_storage, mock_emissions_db, sample_emissions
    ):
        """Test getting trends with data."""
        mock_emissions_db.get_all_emissions.return_value = sample_emissions

        response = client.get("/api/v1/carbon/analytics/trends")
        assert response.status_code == 200

        data = response.json()
        assert "period" in data
        assert "granularity" in data

    def test_get_trends_different_periods(
        self, client: TestClient, mock_storage, mock_emissions_db, sample_emissions
    ):
        """Test getting trends with different periods."""
        mock_emissions_db.get_all_emissions.return_value = sample_emissions

        for period in ["7d", "30d", "90d", "all"]:
            response = client.get(f"/api/v1/carbon/analytics/trends?period={period}")
            assert response.status_code == 200
            assert response.json()["period"] == period

    def test_get_trends_different_granularity(
        self, client: TestClient, mock_storage, mock_emissions_db, sample_emissions
    ):
        """Test getting trends with different granularity."""
        mock_emissions_db.get_all_emissions.return_value = sample_emissions

        for granularity in ["hour", "day", "week", "month"]:
            response = client.get(f"/api/v1/carbon/analytics/trends?granularity={granularity}")
            assert response.status_code == 200
            assert response.json()["granularity"] == granularity


class TestAnalyticsComparisons:
    """Tests for GET /api/v1/carbon/analytics/comparisons."""

    def test_get_comparisons_empty(self, client: TestClient, mock_storage, mock_emissions_db):
        """Test getting comparisons when no data exists."""
        response = client.get("/api/v1/carbon/analytics/comparisons")
        assert response.status_code == 200

        data = response.json()
        assert data["by_model"] == []
        assert data["efficiency_ranking"] == []

    def test_get_comparisons_with_data(
        self, client: TestClient, mock_storage, mock_emissions_db, sample_emissions
    ):
        """Test getting comparisons with data."""
        mock_emissions_db.get_all_emissions.return_value = sample_emissions

        response = client.get("/api/v1/carbon/analytics/comparisons")
        assert response.status_code == 200

        data = response.json()
        assert "by_model" in data
        assert "by_type" in data
        assert "efficiency_ranking" in data
        assert "top_emitters" in data


class TestAnalyticsRecommendations:
    """Tests for GET /api/v1/carbon/analytics/recommendations."""

    def test_get_recommendations_empty(self, client: TestClient, mock_storage, mock_emissions_db):
        """Test getting recommendations when no data exists."""
        response = client.get("/api/v1/carbon/analytics/recommendations")
        assert response.status_code == 200

        data = response.json()
        assert "recommendations" in data
        assert "insights" in data
        assert "summary" in data
        # Should have at least a start-tracking recommendation
        assert len(data["recommendations"]) >= 1

    def test_get_recommendations_with_data(
        self, client: TestClient, mock_storage, mock_emissions_db, sample_emissions
    ):
        """Test getting recommendations with data."""
        mock_emissions_db.get_all_emissions.return_value = sample_emissions

        response = client.get("/api/v1/carbon/analytics/recommendations")
        assert response.status_code == 200

        data = response.json()
        assert "recommendations" in data
        assert "summary" in data
        assert "efficiency_score" in data["summary"]

    def test_recommendations_structure(
        self, client: TestClient, mock_storage, mock_emissions_db, sample_emissions
    ):
        """Test that recommendations have proper structure."""
        mock_emissions_db.get_all_emissions.return_value = sample_emissions

        response = client.get("/api/v1/carbon/analytics/recommendations")
        assert response.status_code == 200

        data = response.json()
        for rec in data["recommendations"]:
            assert "id" in rec
            assert "priority" in rec
            assert "title" in rec
            assert "description" in rec
