"""Command-line interface for Model Garden.

This module re-exports the CLI from the cli package for backward compatibility.
The actual implementation is in model_garden/cli/.

Package Structure:
    cli/
    ├── __init__.py      - Main entry point, registers all commands
    ├── train.py         - Training commands (train, train-vision)
    ├── inference.py     - Inference commands (serve-model, inference-generate, inference-chat)
    ├── dataset.py       - Dataset commands (create-dataset, create-vision-dataset)
    ├── server.py        - Server commands (serve)
    └── utils.py         - Utility commands (generate, list-backends)
"""

# Re-export main from the cli package
from model_garden.cli import console, main

__all__ = ["main", "console"]


if __name__ == "__main__":
    main()
