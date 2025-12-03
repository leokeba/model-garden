"""Tests for BoAmps report generation.

These tests verify the BoAmps emissions report generator creates
properly formatted reports compliant with the BoAmps v1.1.0 spec.

BoAmps Specification: https://github.com/Boavizta/BoAmps
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
        """Test that generated report has correct structure per BoAmps spec."""
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

        # Check top-level sections exist per BoAmps spec
        assert "header" in report
        assert "task" in report
        assert "measures" in report
        assert "infrastructure" in report
        assert "system" in report
        assert "software" in report
        assert "environment" in report
        assert "quality" in report  # BoAmps quality field

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_generate_header(self, mock_hw_detector):
        """Test header section generation with BoAmps compliant datetime format."""
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
        # BoAmps requires YYYY-MM-DD HH:MM:SS format
        assert header["reportDatetime"] == "2024-06-15 12:00:00"
        assert header["reportStatus"] == "draft"
        assert header["publisher"]["name"] == "Test Publisher"
        assert header["publisher"]["division"] == "Test Division"
        # BoAmps requires formatVersionSpecificationUri
        assert "formatVersionSpecificationUri" in header

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_generate_task_training(self, mock_hw_detector):
        """Test task section for training job (fine-tuning with base model)."""
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
        # With a base_model, this is fine-tuning, not training from scratch
        assert task["taskStage"] == "finetuning"
        assert "algorithms" in task
        # BoAmps uses singular "dataset" (not "datasets")
        assert "dataset" in task
        assert isinstance(task["dataset"], list)

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
        # BoAmps requires string format for measurementDateTime
        assert "measurementDateTime" in measure
        assert isinstance(measure["measurementDateTime"], str)

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_infrastructure_components_have_component_type(self, mock_hw_detector):
        """Test that infrastructure components have required componentType field."""
        mock_hw_detector.return_value = create_mock_hardware_detector()

        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator()

        emissions_data = {
            "job_id": "test-job",
            "gpu_energy_kwh": 0.5,
            "cpu_energy_kwh": 0.3,
            "energy_consumed_kwh": 0.8,
        }

        report = generator.generate_report(emissions_data)
        components = report["infrastructure"]["components"]

        # Each component must have componentType (required by BoAmps schema)
        for component in components:
            assert "componentType" in component
            assert component["componentType"] in ["gpu", "cpu", "ram"]
            assert "nbComponent" in component

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_dataset_valid_enum_values(self, mock_hw_detector):
        """Test that dataset uses valid BoAmps enum values."""
        mock_hw_detector.return_value = create_mock_hardware_detector()

        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator()

        emissions_data = {"job_id": "test"}
        job_config = {
            "dataset_path": "/data/train.jsonl",
            "from_hub": True,
        }

        report = generator.generate_report(emissions_data, job_config=job_config)
        dataset = report["task"]["dataset"]

        assert len(dataset) > 0
        # dataUsage must be "input" or "output"
        assert dataset[0]["dataUsage"] in ["input", "output"]
        # dataType must be valid BoAmps enum
        valid_data_types = [
            "tabular",
            "audio",
            "boolean",
            "image",
            "video",
            "object",
            "text",
            "token",
            "word",
            "other",
        ]
        assert dataset[0]["dataType"] in valid_data_types
        # source must be valid BoAmps enum
        assert dataset[0]["source"] in ["public", "private", "other"]

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_quantization_is_string(self, mock_hw_detector):
        """Test that quantization is a string per BoAmps schema."""
        mock_hw_detector.return_value = create_mock_hardware_detector()

        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator()

        emissions_data = {"job_id": "test"}
        job_config = {
            "load_in_4bit": True,
        }

        report = generator.generate_report(emissions_data, job_config=job_config)
        algorithms = report["task"]["algorithms"]

        assert len(algorithms) > 0
        if "quantization" in algorithms[0]:
            # BoAmps requires string like "fp32", "fp16", "int8", "q4", etc.
            assert isinstance(algorithms[0]["quantization"], str)


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

        # Required BoAmps fields should still be present
        assert "task" in report
        assert "measures" in report
        assert "infrastructure" in report

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_quality_field_values(self, mock_hw_detector):
        """Test that quality field has valid BoAmps enum value."""
        mock_hw_detector.return_value = create_mock_hardware_detector()

        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator()
        report = generator.generate_report({})

        # quality must be "high", "medium", or "low"
        assert report["quality"] in ["high", "medium", "low"]


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

    def test_spec_uri_constant(self):
        """Test that specification URI is set."""
        from model_garden.carbon.boamps import BoAmpsReportGenerator

        assert hasattr(BoAmpsReportGenerator, "BOAMPS_SPEC_URI")
        assert "Boavizta/BoAmps" in BoAmpsReportGenerator.BOAMPS_SPEC_URI


class TestBoAmpsVisionModels:
    """Tests for vision-language model support."""

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_vision_model_detection_from_job_id(self, mock_hw_detector):
        """Test that VL models are detected from job_id."""
        mock_hw_detector.return_value = create_mock_hardware_detector()

        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator()

        emissions_data = {
            "job_id": "inference-Qwen-Qwen3-VL-8B-Instruct-123456",
            "job_type": "inference",
        }

        report = generator.generate_report(emissions_data)
        task = report["task"]

        # Should detect as vision-language model
        assert task["taskFamily"] == "multiModalTextGeneration"
        assert task["algorithms"][0]["algorithmType"] == "vlm"

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_vision_model_detection_from_config(self, mock_hw_detector):
        """Test that is_vision config is respected."""
        mock_hw_detector.return_value = create_mock_hardware_detector()

        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator()

        emissions_data = {"job_id": "training-123", "job_type": "training"}
        job_config = {"is_vision": True, "base_model": "Qwen/Qwen2.5-VL-3B"}

        report = generator.generate_report(emissions_data, job_config=job_config)
        task = report["task"]

        assert task["taskFamily"] == "multiModalTextGeneration"
        assert task["algorithms"][0]["algorithmType"] == "vlm"
        assert task["dataset"][0]["dataType"] == "image"

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_vision_model_task_description(self, mock_hw_detector):
        """Test that vision models get proper task description."""
        mock_hw_detector.return_value = create_mock_hardware_detector()

        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator()

        emissions_data = {
            "job_id": "vision-training-123456",
            "job_type": "training",
        }

        report = generator.generate_report(emissions_data)
        task = report["task"]

        assert "taskDescription" in task
        assert "Vision-language" in task["taskDescription"]


class TestBoAmpsModelExtraction:
    """Tests for model name and parameter extraction."""

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_extract_model_from_job_id(self, mock_hw_detector):
        """Test model name extraction from job_id."""
        mock_hw_detector.return_value = create_mock_hardware_detector()

        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator()

        emissions_data = {
            "job_id": "inference-Qwen-Qwen3-VL-8B-Instruct-123",
            "job_type": "inference",
        }

        report = generator.generate_report(emissions_data)
        algorithms = report["task"]["algorithms"]

        assert len(algorithms) > 0
        assert "foundationModelName" in algorithms[0]
        # Should extract Qwen-related model name
        assert (
            "Qwen" in algorithms[0]["foundationModelName"]
            or "qwen" in algorithms[0]["foundationModelName"].lower()
        )

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_extract_parameters_from_model_name(self, mock_hw_detector):
        """Test parameter extraction from model name."""
        mock_hw_detector.return_value = create_mock_hardware_detector()

        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator()

        emissions_data = {"job_id": "test"}
        job_config = {"base_model": "Meta-Llama/Llama-3.1-8B-Instruct"}

        report = generator.generate_report(emissions_data, job_config=job_config)
        algorithms = report["task"]["algorithms"]

        assert algorithms[0]["parametersNumber"] == 8.0

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_huggingface_uri_generation(self, mock_hw_detector):
        """Test HuggingFace URI generation."""
        mock_hw_detector.return_value = create_mock_hardware_detector()

        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator()

        emissions_data = {"job_id": "test"}
        job_config = {"base_model": "Qwen/Qwen2.5-VL-7B-Instruct"}

        report = generator.generate_report(emissions_data, job_config=job_config)
        algorithms = report["task"]["algorithms"]

        assert "foundationModelUri" in algorithms[0]
        assert "huggingface.co" in algorithms[0]["foundationModelUri"]


class TestBoAmpsDatasetMetadata:
    """Tests for dataset metadata in reports."""

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_dataset_quantity_from_config(self, mock_hw_detector):
        """Test that dataset quantity is included when available."""
        mock_hw_detector.return_value = create_mock_hardware_detector()

        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator()

        emissions_data = {"job_id": "test"}
        job_config = {
            "dataset_path": "org/dataset",
            "from_hub": True,
            "dataset_num_samples": 10000,
        }

        report = generator.generate_report(emissions_data, job_config=job_config)
        dataset = report["task"]["dataset"][0]

        assert dataset["dataQuantity"] == 10000

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_dataset_owner_extraction(self, mock_hw_detector):
        """Test that dataset owner is extracted from HuggingFace path."""
        mock_hw_detector.return_value = create_mock_hardware_detector()

        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator()

        emissions_data = {"job_id": "test"}
        job_config = {
            "dataset_path": "Barth371/cmr-all",
            "from_hub": True,
        }

        report = generator.generate_report(emissions_data, job_config=job_config)
        dataset = report["task"]["dataset"][0]

        assert dataset["owner"] == "Barth371"
        assert "huggingface.co/datasets" in dataset["sourceUri"]


class TestBoAmpsQualityEstimation:
    """Tests for quality field estimation."""

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_high_quality_with_full_data(self, mock_hw_detector):
        """Test high quality when all data is available."""
        mock_hw_detector.return_value = create_mock_hardware_detector()

        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator()

        emissions_data = {
            "job_id": "test",
            "tracking_mode": "process",
            "gpu_energy_kwh": 0.5,
            "cpu_energy_kwh": 0.1,
            "ram_energy_kwh": 0.05,
            "duration_seconds": 3600,
            "gpu_power_watts": 300,
            "cpu_power_watts": 50,
        }

        report = generator.generate_report(emissions_data)

        assert report["quality"] == "high"

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_medium_quality_with_partial_data(self, mock_hw_detector):
        """Test medium quality with partial data."""
        mock_hw_detector.return_value = create_mock_hardware_detector()

        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator()

        emissions_data = {
            "job_id": "test",
            "tracking_mode": "process",
            "gpu_energy_kwh": 0.5,
            "cpu_energy_kwh": 0.1,
            "duration_seconds": 3600,
            # No power data or RAM data
        }

        report = generator.generate_report(emissions_data)

        assert report["quality"] == "medium"

    @patch("model_garden.carbon.boamps.get_hardware_detector")
    def test_low_quality_with_constant_tracking(self, mock_hw_detector):
        """Test low quality with constant tracking mode."""
        mock_hw_detector.return_value = create_mock_hardware_detector()

        from model_garden.carbon.boamps import BoAmpsReportGenerator

        generator = BoAmpsReportGenerator()

        emissions_data = {
            "job_id": "test",
            "tracking_mode": "constant",
            "duration_seconds": 3600,
        }

        report = generator.generate_report(emissions_data)

        assert report["quality"] == "low"
