"""Pytest configuration and shared fixtures for Model Garden tests."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# Prevent actual GPU/ML imports during tests unless explicitly needed
@pytest.fixture(autouse=True)
def mock_heavy_imports(request):
    """Mock heavy ML imports by default to speed up tests.

    Use @pytest.mark.requires_gpu to skip this fixture for integration tests.
    """
    if "requires_gpu" in [marker.name for marker in request.node.iter_markers()]:
        yield
        return

    # Mock torch if not already imported
    with patch.dict(
        "sys.modules",
        {
            "torch": MagicMock(),
            "torch.cuda": MagicMock(),
        },
    ):
        yield


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_jsonl_dataset(temp_dir: Path) -> Path:
    """Create a sample JSONL dataset for testing."""
    dataset_path = temp_dir / "test_dataset.jsonl"

    samples = [
        '{"instruction": "Say hello", "input": "", "output": "Hello!"}',
        '{"instruction": "Count to 3", "input": "", "output": "1, 2, 3"}',
        '{"instruction": "What is 2+2?", "input": "", "output": "4"}',
    ]

    dataset_path.write_text("\n".join(samples))
    return dataset_path


@pytest.fixture
def sample_vision_dataset(temp_dir: Path) -> Path:
    """Create a sample vision-language dataset for testing."""
    dataset_path = temp_dir / "test_vision_dataset.jsonl"

    # Create a small test image (1x1 pixel PNG as base64)
    # This is a minimal valid PNG
    import base64

    minimal_png = base64.b64encode(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f"
        b"\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    ).decode()

    samples = [
        f'{{"text": "What is in this image?", "image": "data:image/png;base64,{minimal_png}", "response": "A test image"}}',
    ]

    dataset_path.write_text("\n".join(samples))
    return dataset_path


@pytest.fixture
def mock_training_job() -> dict[str, Any]:
    """Create a mock training job configuration."""
    return {
        "id": "test-job-123",
        "name": "Test Training Job",
        "status": "pending",
        "base_model": "unsloth/tinyllama-bnb-4bit",
        "dataset_path": "/path/to/dataset.jsonl",
        "output_dir": "/path/to/output",
        "created_at": "2024-01-01T00:00:00Z",
        "hyperparameters": {
            "learning_rate": 2e-4,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
        },
        "lora_config": {
            "r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
        },
    }


@pytest.fixture
def mock_model_info() -> dict[str, Any]:
    """Create a mock model info configuration."""
    return {
        "id": "test-model-123",
        "name": "Test Model",
        "base_model": "unsloth/tinyllama-bnb-4bit",
        "status": "ready",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "path": "/models/test-model",
        "size_bytes": 1024000,
    }


# Markers for test categorization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU (skipped without GPU)"
    )
    config.addinivalue_line("markers", "slow: mark test as slow (skipped with --fast)")
    config.addinivalue_line("markers", "integration: mark test as integration test")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Skip slow tests",
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers and options."""
    skip_slow = pytest.mark.skip(reason="Skipped with --fast option")
    skip_integration = pytest.mark.skip(reason="Need --run-integration to run")

    for item in items:
        if config.getoption("--fast") and "slow" in [m.name for m in item.iter_markers()]:
            item.add_marker(skip_slow)

        if not config.getoption("--run-integration") and "integration" in [
            m.name for m in item.iter_markers()
        ]:
            item.add_marker(skip_integration)
