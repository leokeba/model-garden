"""Tests for model_garden.utils module."""

import json
from pathlib import Path

from model_garden.utils.dataset_validator import DatasetStats, DatasetValidator


class TestDatasetValidator:
    """Tests for DatasetValidator class."""

    def test_detect_format_jsonl(self, temp_dir: Path):
        """Test JSONL format detection."""
        file_path = temp_dir / "test.jsonl"
        file_path.touch()
        assert DatasetValidator.detect_format(file_path) == "jsonl"

    def test_detect_format_json(self, temp_dir: Path):
        """Test JSON format detection."""
        file_path = temp_dir / "test.json"
        file_path.touch()
        assert DatasetValidator.detect_format(file_path) == "json"

    def test_detect_format_csv(self, temp_dir: Path):
        """Test CSV format detection."""
        file_path = temp_dir / "test.csv"
        file_path.touch()
        assert DatasetValidator.detect_format(file_path) == "csv"

    def test_detect_format_unknown(self, temp_dir: Path):
        """Test unknown format detection."""
        file_path = temp_dir / "test.txt"
        file_path.touch()
        assert DatasetValidator.detect_format(file_path) == "unknown"

    def test_detect_schema_type_vision(self):
        """Test vision schema detection."""
        sample_data = [{"text": "query", "image": "path.jpg", "response": "answer"}]
        assert DatasetValidator.detect_schema_type(sample_data) == "vision"

    def test_detect_schema_type_alpaca(self):
        """Test Alpaca schema detection."""
        sample_data = [{"instruction": "Do something", "input": "", "output": "Done"}]
        assert DatasetValidator.detect_schema_type(sample_data) == "alpaca"

    def test_detect_schema_type_text(self):
        """Test text schema detection."""
        sample_data = [{"input": "query", "output": "response"}]
        assert DatasetValidator.detect_schema_type(sample_data) == "text"

    def test_detect_schema_type_empty(self):
        """Test schema detection with empty data."""
        assert DatasetValidator.detect_schema_type([]) == "unknown"

    def test_estimate_tokens(self):
        """Test token estimation."""
        # Empty string
        assert DatasetValidator.estimate_tokens("") == 0

        # Short text (4 chars = 1 token)
        assert DatasetValidator.estimate_tokens("test") == 1

        # Longer text (12 chars = 3 tokens)
        assert DatasetValidator.estimate_tokens("hello world!") == 3

    def test_load_jsonl_dataset(self, temp_dir: Path):
        """Test loading JSONL dataset."""
        dataset_path = temp_dir / "test.jsonl"
        samples = [
            {"instruction": "Hello", "output": "Hi"},
            {"instruction": "Bye", "output": "Goodbye"},
        ]

        with open(dataset_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        data, fmt = DatasetValidator.load_dataset(dataset_path)
        assert fmt == "jsonl"
        assert len(data) == 2
        assert data[0]["instruction"] == "Hello"

    def test_load_json_dataset(self, temp_dir: Path):
        """Test loading JSON dataset."""
        dataset_path = temp_dir / "test.json"
        samples = [
            {"instruction": "Hello", "output": "Hi"},
            {"instruction": "Bye", "output": "Goodbye"},
        ]

        with open(dataset_path, "w") as f:
            json.dump(samples, f)

        data, fmt = DatasetValidator.load_dataset(dataset_path)
        assert fmt == "json"
        assert len(data) == 2

    def test_load_csv_dataset(self, temp_dir: Path):
        """Test loading CSV dataset."""
        dataset_path = temp_dir / "test.csv"

        with open(dataset_path, "w") as f:
            f.write("instruction,output\n")
            f.write("Hello,Hi\n")
            f.write("Bye,Goodbye\n")

        data, fmt = DatasetValidator.load_dataset(dataset_path)
        assert fmt == "csv"
        assert len(data) == 2
        assert data[0]["instruction"] == "Hello"

    def test_load_dataset_max_rows(self, temp_dir: Path):
        """Test loading dataset with max_rows limit."""
        dataset_path = temp_dir / "test.jsonl"

        with open(dataset_path, "w") as f:
            for i in range(100):
                f.write(json.dumps({"instruction": f"q{i}", "output": f"a{i}"}) + "\n")

        data, _ = DatasetValidator.load_dataset(dataset_path, max_rows=10)
        assert len(data) == 10

    def test_validate_dataset_file_not_found(self, temp_dir: Path):
        """Test validation with non-existent file."""
        stats = DatasetValidator.validate_dataset(temp_dir / "nonexistent.jsonl")
        assert len(stats.validation_errors) > 0
        assert "not found" in stats.validation_errors[0].lower()

    def test_validate_dataset_empty(self, temp_dir: Path):
        """Test validation with empty file."""
        dataset_path = temp_dir / "empty.jsonl"
        dataset_path.touch()

        stats = DatasetValidator.validate_dataset(dataset_path)
        assert len(stats.validation_errors) > 0
        assert "empty" in stats.validation_errors[0].lower()

    def test_validate_dataset_alpaca_format(self, temp_dir: Path):
        """Test validation of Alpaca format dataset."""
        dataset_path = temp_dir / "alpaca.jsonl"
        samples = [
            {"instruction": "Say hello", "input": "", "output": "Hello!"},
            {"instruction": "Count", "input": "to 3", "output": "1, 2, 3"},
        ]

        with open(dataset_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        stats = DatasetValidator.validate_dataset(dataset_path)
        assert stats.total_rows == 2
        assert stats.format == "jsonl"
        assert "instruction" in stats.fields
        assert "output" in stats.fields
        assert len(stats.validation_errors) == 0

    def test_validate_dataset_missing_required_field(self, temp_dir: Path):
        """Test validation detects missing required fields."""
        dataset_path = temp_dir / "missing.jsonl"
        samples = [
            {"instruction": "Hello"},  # Missing 'output'
            {"instruction": "Bye"},  # Missing 'output'
        ]

        with open(dataset_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        stats = DatasetValidator.validate_dataset(dataset_path, schema_type="alpaca")
        # Should have errors or warnings about missing 'output' field
        assert stats.missing_fields.get("output", 0) == 2 or len(stats.validation_errors) > 0

    def test_validate_dataset_vision_format(self, temp_dir: Path):
        """Test validation of vision format dataset."""
        dataset_path = temp_dir / "vision.jsonl"
        samples = [
            {"text": "What is this?", "image": "path/to/image.jpg", "response": "An image"},
        ]

        with open(dataset_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        stats = DatasetValidator.validate_dataset(dataset_path)
        assert stats.has_images is True
        assert stats.image_count == 1

    def test_validate_dataset_warnings_small_dataset(self, temp_dir: Path):
        """Test that warnings are generated for small datasets."""
        dataset_path = temp_dir / "small.jsonl"
        samples = [{"instruction": "Hi", "output": "Hello"}] * 5

        with open(dataset_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        stats = DatasetValidator.validate_dataset(dataset_path)
        assert any("fewer than" in w for w in stats.warnings)

    def test_convert_csv_to_jsonl(self, temp_dir: Path):
        """Test CSV to JSONL conversion."""
        csv_path = temp_dir / "data.csv"
        jsonl_path = temp_dir / "data.jsonl"

        with open(csv_path, "w") as f:
            f.write("instruction,output\n")
            f.write("Hello,Hi\n")
            f.write("Bye,Goodbye\n")

        count = DatasetValidator.convert_csv_to_jsonl(csv_path, jsonl_path)
        assert count == 2
        assert jsonl_path.exists()

        # Verify content
        with open(jsonl_path) as f:
            lines = f.readlines()
            assert len(lines) == 2
            assert json.loads(lines[0])["instruction"] == "Hello"

    def test_convert_jsonl_to_csv(self, temp_dir: Path):
        """Test JSONL to CSV conversion."""
        jsonl_path = temp_dir / "data.jsonl"
        csv_path = temp_dir / "data.csv"

        samples = [
            {"instruction": "Hello", "output": "Hi"},
            {"instruction": "Bye", "output": "Goodbye"},
        ]

        with open(jsonl_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        count = DatasetValidator.convert_jsonl_to_csv(jsonl_path, csv_path)
        assert count == 2
        assert csv_path.exists()


class TestDatasetStats:
    """Tests for DatasetStats dataclass."""

    def test_dataclass_creation(self):
        """Test DatasetStats can be created with required fields."""
        stats = DatasetStats(
            total_rows=100,
            format="jsonl",
            fields=["instruction", "output"],
            field_types={"instruction": "str", "output": "str"},
            missing_fields={},
            sample_rows=[],
            file_size_bytes=1024,
            validation_errors=[],
            warnings=[],
        )

        assert stats.total_rows == 100
        assert stats.format == "jsonl"
        assert len(stats.fields) == 2

    def test_dataclass_optional_fields(self):
        """Test DatasetStats optional fields have correct defaults."""
        stats = DatasetStats(
            total_rows=0,
            format="unknown",
            fields=[],
            field_types={},
            missing_fields={},
            sample_rows=[],
            file_size_bytes=0,
            validation_errors=[],
            warnings=[],
        )

        assert stats.avg_input_length is None
        assert stats.avg_output_length is None
        assert stats.total_tokens_estimate is None
        assert stats.has_images is False
        assert stats.image_count == 0
