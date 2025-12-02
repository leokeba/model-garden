"""Storage management for persistent data."""

import json
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


class StorageManager:
    """Manages persistent storage of training jobs and models."""

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_file = storage_dir / "training_jobs.json"
        self.models_file = storage_dir / "models.json"

    def load_training_jobs(self) -> dict[str, dict]:
        """Load training jobs from disk."""
        if self.jobs_file.exists():
            try:
                with open(self.jobs_file) as f:
                    data = json.load(f)
                    return data
            except Exception as e:
                print(f"⚠️  Error loading training jobs: {e}")
                import traceback

                traceback.print_exc()
                return {}
        return {}

    def save_training_jobs(self, jobs: dict[str, dict]) -> None:
        """Save training jobs to disk."""
        try:
            with open(self.jobs_file, "w") as f:
                json.dump(jobs, f, indent=2)
        except Exception as e:
            print(f"⚠️  Error saving training jobs: {e}")

    def load_models(self) -> dict[str, dict]:
        """Load models from disk."""
        if self.models_file.exists():
            try:
                with open(self.models_file) as f:
                    data = json.load(f)
                    return data
            except Exception as e:
                print(f"⚠️  Error loading models: {e}")
                import traceback

                traceback.print_exc()
                return {}
        return {}

    def save_models(self, models: dict[str, dict]) -> None:
        """Save models to disk."""
        try:
            with open(self.models_file, "w") as f:
                json.dump(models, f, indent=2)
        except Exception as e:
            print(f"⚠️  Error saving models: {e}")


# Singleton instance
_storage_manager: StorageManager | None = None


def get_storage_manager() -> StorageManager:
    """Get the global storage manager instance."""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = StorageManager(PROJECT_ROOT / "storage")
    return _storage_manager
