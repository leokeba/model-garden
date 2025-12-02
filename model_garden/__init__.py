"""Model Garden - Fine-tune and serve LLMs with carbon footprint tracking.

Package Structure:
    model_garden/
    ├── api/            - FastAPI application and routes
    ├── backends/       - Training backend implementations (Unsloth, HuggingFace)
    ├── carbon/         - Carbon footprint tracking
    ├── experiments/    - Hyperparameter exploration and visualization
    ├── inference/      - vLLM inference service
    ├── queue/          - Job queue and worker management
    ├── training/       - Model trainers and training utilities
    └── utils/          - General utilities (memory, dataset validation)

For convenience, commonly used components can be imported directly:
    from model_garden import InferenceService
    from model_garden import ModelTrainer, VisionLanguageTrainer
"""

__version__ = "0.1.0"

# Expose main classes for convenience imports
# These are lazy-loaded to avoid importing heavy dependencies at module load time


def __getattr__(name):
    """Lazy-load main components to avoid importing heavy dependencies."""
    if name == "InferenceService":
        from model_garden.inference import InferenceService

        return InferenceService
    elif name == "ModelTrainer":
        from model_garden.training import ModelTrainer

        return ModelTrainer
    elif name == "VisionLanguageTrainer":
        from model_garden.training import VisionLanguageTrainer

        return VisionLanguageTrainer
    elif name == "create_text_trainer":
        from model_garden.training import create_text_trainer

        return create_text_trainer
    elif name == "create_vision_trainer":
        from model_garden.training import create_vision_trainer

        return create_vision_trainer
    elif name == "DatasetValidator":
        from model_garden.utils import DatasetValidator

        return DatasetValidator
    elif name == "HyperparameterExplorer":
        from model_garden.experiments import HyperparameterExplorer

        return HyperparameterExplorer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Version
    "__version__",
    # Main classes (lazy-loaded)
    "InferenceService",
    "ModelTrainer",
    "VisionLanguageTrainer",
    "create_text_trainer",
    "create_vision_trainer",
    "DatasetValidator",
    "HyperparameterExplorer",
]
