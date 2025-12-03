"""Tests for API storage management."""

from pathlib import Path

from model_garden.api.storage import StorageManager


class TestStorageManager:
    """Tests for the StorageManager class."""

    def test_init_creates_directory(self, temp_dir: Path):
        """Test that StorageManager creates storage directory on init."""
        storage_dir = temp_dir / "new_storage"
        assert not storage_dir.exists()

        manager = StorageManager(storage_dir)

        assert storage_dir.exists()
        assert manager.storage_dir == storage_dir
        assert manager.jobs_file == storage_dir / "training_jobs.json"
        assert manager.models_file == storage_dir / "models.json"

    def test_load_training_jobs_empty(self, temp_dir: Path):
        """Test loading jobs when no file exists."""
        manager = StorageManager(temp_dir)
        jobs = manager.load_training_jobs()
        assert jobs == {}

    def test_save_and_load_training_jobs(self, temp_dir: Path):
        """Test saving and loading training jobs."""
        manager = StorageManager(temp_dir)

        test_jobs = {
            "job-1": {
                "id": "job-1",
                "status": "running",
                "base_model": "test-model",
            },
            "job-2": {
                "id": "job-2",
                "status": "completed",
                "base_model": "another-model",
            },
        }

        manager.save_training_jobs(test_jobs)

        # Verify file was created
        assert manager.jobs_file.exists()

        # Load and verify content
        loaded_jobs = manager.load_training_jobs()
        assert loaded_jobs == test_jobs

    def test_load_training_jobs_corrupted_file(self, temp_dir: Path):
        """Test loading jobs from corrupted JSON file."""
        manager = StorageManager(temp_dir)

        # Write invalid JSON
        manager.jobs_file.write_text("{ not valid json }")

        # Should return empty dict without crashing
        jobs = manager.load_training_jobs()
        assert jobs == {}

    def test_load_models_empty(self, temp_dir: Path):
        """Test loading models when no file exists."""
        manager = StorageManager(temp_dir)
        models = manager.load_models()
        assert models == {}

    def test_save_and_load_models(self, temp_dir: Path):
        """Test saving and loading models."""
        manager = StorageManager(temp_dir)

        test_models = {
            "model-1": {
                "id": "model-1",
                "name": "Test Model",
                "path": "/models/test",
                "status": "ready",
            },
            "model-2": {
                "id": "model-2",
                "name": "Another Model",
                "path": "/models/another",
                "status": "loading",
            },
        }

        manager.save_models(test_models)

        # Verify file was created
        assert manager.models_file.exists()

        # Load and verify content
        loaded_models = manager.load_models()
        assert loaded_models == test_models

    def test_load_models_corrupted_file(self, temp_dir: Path):
        """Test loading models from corrupted JSON file."""
        manager = StorageManager(temp_dir)

        # Write invalid JSON
        manager.models_file.write_text("invalid json content")

        # Should return empty dict without crashing
        models = manager.load_models()
        assert models == {}

    def test_save_jobs_handles_nested_data(self, temp_dir: Path):
        """Test saving jobs with nested/complex data structures."""
        manager = StorageManager(temp_dir)

        test_jobs = {
            "job-complex": {
                "id": "job-complex",
                "hyperparameters": {
                    "learning_rate": 2e-4,
                    "num_epochs": 3,
                    "nested": {"deep": {"value": [1, 2, 3]}},
                },
                "lora_config": {
                    "r": 16,
                    "alpha": 16,
                    "target_modules": ["q_proj", "v_proj"],
                },
                "metrics": [
                    {"epoch": 1, "loss": 0.5},
                    {"epoch": 2, "loss": 0.3},
                ],
            }
        }

        manager.save_training_jobs(test_jobs)
        loaded_jobs = manager.load_training_jobs()

        assert loaded_jobs == test_jobs
        assert loaded_jobs["job-complex"]["hyperparameters"]["nested"]["deep"]["value"] == [1, 2, 3]

    def test_multiple_save_load_cycles(self, temp_dir: Path):
        """Test multiple save/load cycles maintain data integrity."""
        manager = StorageManager(temp_dir)

        for i in range(5):
            jobs = {f"job-{i}": {"id": f"job-{i}", "iteration": i}}
            manager.save_training_jobs(jobs)

            loaded = manager.load_training_jobs()
            assert loaded == jobs

    def test_independent_jobs_and_models(self, temp_dir: Path):
        """Test that jobs and models are stored independently."""
        manager = StorageManager(temp_dir)

        jobs = {"job-1": {"id": "job-1"}}
        models = {"model-1": {"id": "model-1"}}

        manager.save_training_jobs(jobs)
        manager.save_models(models)

        # Each should be independent
        assert manager.load_training_jobs() == jobs
        assert manager.load_models() == models

        # Modifying one should not affect the other
        manager.save_training_jobs({"job-2": {"id": "job-2"}})
        assert manager.load_models() == models

    def test_save_handles_unicode(self, temp_dir: Path):
        """Test saving and loading Unicode content."""
        manager = StorageManager(temp_dir)

        test_jobs = {
            "job-unicode": {
                "id": "job-unicode",
                "name": "テスト 🚀 модель",
                "description": "Ümlauts and émojis: 日本語",
            }
        }

        manager.save_training_jobs(test_jobs)
        loaded = manager.load_training_jobs()

        assert loaded == test_jobs
        assert loaded["job-unicode"]["name"] == "テスト 🚀 модель"
