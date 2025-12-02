"""Tests for BoAmps report generation.

These tests verify the BoAmps emissions report generator creates
properly formatted reports compliant with the BoAmps v1.1.0 spec.
"""

import json
from unittest.mock import MagicMock, patch


def create_mock_hardware_detector():
    """Create a properly configured mock hardware detector."""
    mock_detector = MagicMock()

    # GPU info
    mock_detector.get_gpu_info.return_value = {
        "primary": {
            "model": "RTX 4090",
            "manufacturer": "NVIDIA",
            "memory": "24564 MiB",
            "family": "Ada Lovelace",
        }
    }

    # CPU info
    mock_detector.get_cpu_info.return_value = {
        "manufacturer": "Intel",
        "model": "Core i9-13900K",
        "family": "Core i9",
    }

    # RAM info
    mock_detector.get_ram_info.return_value = {"total_gb": 64}

    # System info
    mock_detector.get_system_info.return_value = {
        "os_name": "Linux",
        "os_version": "6.5.0",
        "os_distribution": "Ubuntu 22.04",
        "python_version": "3.11.0",
    }

    return mock_detector


class TestBoAmpsReportGenerator:
    """Tests for BoAmpsReportGenerator class."""

    def test_init_default_values(self):
        """Test default initialization values."""
        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator()
        assert generator.publisher_name == "Model Garden"
        assert generator.publisher_division is None
        assert generator.confidentiality_level == "public"
        assert generator.BOAMPS_VERSION == "1.1.0"

    def test_init_custom_values(self):
        """Test custom initialization values."""
        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator(
            publisher_name="My Company",
            publisher_division="ML Team",
            confidentiality_level="internal",
        )
        assert generator.publisher_name == "My Company"
        assert generator.publisher_division == "ML Team"
        assert generator.confidentiality_level == "internal"

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_generate_report_structure(self, mock_hw_detector):
        """Test that generated report has correct structure."""
        from model_garden.carbon.boamps import BoAmpsReportGenerator

        mock_hw_detector.return_value = create_mock_hardware_detector()

        generator = BoAmpsReportGenerator()

        emissions_data = {
            "job_id": "test-job-123",
            "job_type": "training",
            "emissions_kg_co2": 0.5,
            "energy_consumed_kwh": 1.2,
            "duration_seconds": 3600,
            "timestamp": "2024-01-01T00:00:00Z",
        }

        report = generator.generate_report(emissions_data)

        # Check top-level sections exist
        assert "header" in report
        assert "task" in report
        assert "measures" in report
        assert "infrastructure" in report
        assert "system" in report
        assert "software" in report
        assert "environment" in report

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_generate_header(self, mock_hw_detector):
        """Test header section generation."""
        mock_hw_detector.return_value = create_mock_hardware_detector()

        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator(
            publisher_name="Test Publisher", publisher_division="Test Division"
        )

        emissions_data = {"job_id": "job-abc", "timestamp": "2024-06-15T12:00:00Z"}

        report = generator.generate_report(emissions_data, report_status="draft")
        header = report["header"]

        assert header["licensing"] == "Creative Commons 4.0"
        assert header["formatVersion"] == "1.1.0"
        assert header["reportId"] == "job-abc"
        assert header["reportDatetime"] == "2024-06-15T12:00:00Z"
        assert header["reportStatus"] == "draft"
        assert header["publisher"]["name"] == "Test Publisher"
        assert header["publisher"]["division"] == "Test Division"

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_generate_task_training(self, mock_hw_detector):
        """Test task section for training job."""
        mock_hw_detector.return_value = create_mock_hardware_detector()

        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator()

        emissions_data = {
            "job_id": "train-123",
            "job_type": "training",
            "model_name": "llama-3-8b",
        }

        job_config = {
            "base_model": "llama-3-8b",
            "dataset_path": "/data/train.jsonl",
            "hyperparameters": {"num_epochs": 3, "learning_rate": 2e-4},
        }

        report = generator.generate_report(emissions_data, job_config=job_config)
        task = report["task"]

        # Task should have BoAmps v1.1.0 compliant structure
        assert "taskFamily" in task
        assert "taskStage" in task
        assert task["taskStage"] == "training"
        assert "algorithms" in task

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_generate_task_inference(self, mock_hw_detector):
        """Test task section for inference job."""
        mock_hw_detector.return_value = create_mock_hardware_detector()

        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator()

        emissions_data = {
            "job_id": "infer-456",
            "job_type": "inference",
        }

        report = generator.generate_report(emissions_data)
        task = report["task"]

        assert isinstance(task, dict)
        assert task["taskStage"] == "inference"

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_generate_measures(self, mock_hw_detector):
        """Test measures section with emissions data."""
        mock_hw_detector.return_value = create_mock_hardware_detector()

        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator()

        emissions_data = {
            "job_id": "test-job",
            "emissions_kg_co2": 0.123,
            "energy_consumed_kwh": 0.456,
            "duration_seconds": 1800,
            "timestamp": "2024-01-01T12:00:00Z",
        }

        report = generator.generate_report(emissions_data)
        measures = report["measures"]

        # Measures is a list in BoAmps v1.1.0
        assert isinstance(measures, list)
        assert len(measures) > 0

        # First measure should have required fields
        measure = measures[0]
        assert "measurementMethod" in measure
        assert "powerConsumption" in measure
        assert "measurementDuration" in measure


class TestBoAmpsReportValidation:
    """Tests for BoAmps report validation and compliance."""

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_report_serializable_to_json(self, mock_hw_detector):
        """Test that generated report is JSON serializable."""
        mock_hw_detector.return_value = create_mock_hardware_detector()

        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator()

        emissions_data = {
            "job_id": "test",
            "emissions_kg_co2": 0.1,
            "energy_consumed_kwh": 0.2,
            "timestamp": "2024-01-01T00:00:00Z",
        }

        report = generator.generate_report(emissions_data)

        # Should not raise
        json_str = json.dumps(report)
        assert isinstance(json_str, str)

        # Should be valid JSON that can be parsed back
        parsed = json.loads(json_str)
        assert parsed == report

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_report_without_optional_fields(self, mock_hw_detector):
        """Test report generation with minimal data."""
        mock_hw_detector.return_value = create_mock_hardware_detector()

        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator()

        # Minimal emissions data
        emissions_data = {}

        # Should not raise
        report = generator.generate_report(emissions_data)
        assert isinstance(report, dict)


class TestBoAmpsVersionCompliance:
    """Tests for BoAmps version compliance."""

    def test_version_constant(self):
        """Test that version constant is correct."""
        from model_garden.carbon.boamps import BoAmpsReportGenerator

        assert BoAmpsReportGenerator.BOAMPS_VERSION == "1.1.0"

    def test_licensing_constant(self):
        """Test that licensing constant is correct."""
        from model_garden.carbon.boamps import BoAmpsReportGenerator

        assert BoAmpsReportGenerator.LICENSING == "Creative Commons 4.0"
