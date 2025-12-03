"""Dataset service - backend-agnostic dataset operations.

This module provides the DatasetService class that consolidates all dataset-related
operations used by both CLI and API.

Example:
    >>> from model_garden.services import DatasetService
    >>>
    >>> service = DatasetService()
    >>> result = service.validate_dataset("./data/train.jsonl")
    >>> print(f"Examples: {result.total_rows}")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class DatasetInfo:
    """Information about a dataset.

    Attributes:
        name: Dataset file name
        path: Full path to dataset
        size_bytes: File size in bytes
        format: File format (jsonl, json, csv, parquet)
        examples: Number of examples
        created_at: Creation timestamp (ISO format)
        modified_at: Modification timestamp (ISO format)
    """

    name: str
    path: str
    size_bytes: int
    format: str
    examples: int
    created_at: str | None = None
    modified_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "size": self.size_bytes,
            "format": self.format,
            "examples": self.examples,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
        }


@dataclass
class DatasetValidationResult:
    """Result of dataset validation.

    Attributes:
        valid: Whether the dataset is valid
        total_rows: Number of rows/examples
        format: Detected format
        fields: List of field names
        field_types: Dict mapping field names to types
        schema_type: Detected schema type (alpaca, openai, vision, etc.)
        errors: List of validation errors
        warnings: List of warnings
        has_images: Whether dataset contains image references
        image_count: Number of image references found
        avg_input_length: Average input text length
        avg_output_length: Average output text length
        total_tokens_estimate: Estimated total tokens
        sample_rows: Sample rows for preview
    """

    valid: bool = True
    total_rows: int = 0
    format: str = "unknown"
    fields: list[str] = field(default_factory=list)
    field_types: dict[str, str] = field(default_factory=dict)
    schema_type: str = "unknown"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    has_images: bool = False
    image_count: int = 0
    image_paths: list[str] = field(default_factory=list)
    avg_input_length: float | None = None
    avg_output_length: float | None = None
    total_tokens_estimate: int | None = None
    sample_rows: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "valid": self.valid,
            "total_rows": self.total_rows,
            "format": self.format,
            "fields": self.fields,
            "field_types": self.field_types,
            "schema_type": self.schema_type,
            "errors": self.errors,
            "warnings": self.warnings,
        }

        if self.has_images:
            result["vision_stats"] = {
                "has_images": self.has_images,
                "image_count": self.image_count,
                "sample_image_paths": self.image_paths[:5],
            }

        if self.avg_input_length is not None:
            result["text_stats"] = {
                "avg_input_length": self.avg_input_length,
                "avg_output_length": self.avg_output_length,
                "total_tokens_estimate": self.total_tokens_estimate,
            }

        return result


class DatasetService:
    """Backend-agnostic dataset service.

    This service consolidates all dataset operations used by CLI and API:
    - Listing datasets
    - Validating datasets
    - Loading from files or HuggingFace Hub
    - Creating sample datasets
    - Getting statistics and previews

    Example:
        >>> service = DatasetService()
        >>> datasets = service.list_datasets()
        >>> result = service.validate_dataset("./data/train.jsonl")
    """

    def __init__(self, storage_dir: str = "./storage/datasets"):
        """Initialize the dataset service.

        Args:
            storage_dir: Directory for dataset storage
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> list[DatasetInfo]:
        """List all datasets in storage.

        Returns:
            List of DatasetInfo objects
        """
        from datetime import datetime

        datasets = []

        for dataset_file in self.storage_dir.iterdir():
            if not dataset_file.is_file():
                continue

            stat = dataset_file.stat()

            # Count examples
            example_count = self._count_examples(dataset_file)

            datasets.append(
                DatasetInfo(
                    name=dataset_file.name,
                    path=str(dataset_file),
                    size_bytes=stat.st_size,
                    format=dataset_file.suffix.lstrip("."),
                    examples=example_count,
                    created_at=datetime.fromtimestamp(stat.st_ctime).isoformat() + "Z",
                    modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat() + "Z",
                )
            )

        # Sort by modified date (newest first)
        datasets.sort(key=lambda x: x.modified_at or "", reverse=True)

        return datasets

    def _count_examples(self, file_path: Path) -> int:
        """Count examples in a dataset file.

        Args:
            file_path: Path to dataset file

        Returns:
            Number of examples
        """
        try:
            suffix = file_path.suffix.lower()

            if suffix == ".jsonl":
                with open(file_path) as f:
                    return sum(1 for _ in f)
            elif suffix == ".json":
                with open(file_path) as f:
                    data = json.load(f)
                    return len(data) if isinstance(data, list) else 1
            elif suffix == ".csv":
                import pandas as pd

                df = pd.read_csv(file_path)
                return len(df)
            elif suffix == ".parquet":
                import pandas as pd

                df = pd.read_parquet(file_path)
                return len(df)
        except Exception:
            pass

        return 0

    def validate_dataset(
        self,
        file_path: str | Path,
        check_images: bool = True,
    ) -> DatasetValidationResult:
        """Validate a dataset file.

        Args:
            file_path: Path to dataset file
            check_images: Whether to check image references

        Returns:
            DatasetValidationResult with validation details
        """
        from model_garden.utils import DatasetValidator

        path = Path(file_path)

        if not path.exists():
            return DatasetValidationResult(
                valid=False,
                errors=[f"File not found: {file_path}"],
            )

        try:
            stats = DatasetValidator.validate_dataset(path)

            # Detect schema type
            schema_type = "unknown"
            if stats.sample_rows:
                schema_type = DatasetValidator.detect_schema_type(stats.sample_rows)

            return DatasetValidationResult(
                valid=len(stats.validation_errors) == 0,
                total_rows=stats.total_rows,
                format=stats.format,
                fields=stats.fields,
                field_types=stats.field_types,
                schema_type=schema_type,
                errors=stats.validation_errors,
                warnings=stats.warnings,
                has_images=stats.has_images,
                image_count=stats.image_count,
                image_paths=stats.image_paths or [],
                avg_input_length=stats.avg_input_length,
                avg_output_length=stats.avg_output_length,
                total_tokens_estimate=stats.total_tokens_estimate,
                sample_rows=stats.sample_rows or [],
            )

        except Exception as e:
            return DatasetValidationResult(
                valid=False,
                errors=[f"Validation failed: {str(e)}"],
            )

    def preview_dataset(
        self,
        file_path: str | Path,
        limit: int = 10,
    ) -> list[dict]:
        """Preview samples from a dataset.

        Args:
            file_path: Path to dataset file
            limit: Maximum samples to return

        Returns:
            List of sample dictionaries
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        samples = []
        suffix = path.suffix.lower()

        try:
            if suffix == ".jsonl":
                with open(path) as f:
                    for i, line in enumerate(f):
                        if i >= limit:
                            break
                        try:
                            samples.append(json.loads(line))
                        except json.JSONDecodeError:
                            samples.append({"_error": "Invalid JSON", "_raw": line})

            elif suffix == ".json":
                with open(path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples = data[:limit]
                    else:
                        samples = [data]

            elif suffix == ".csv":
                import pandas as pd

                df = pd.read_csv(path, nrows=limit)
                samples = df.to_dict("records")

            elif suffix == ".parquet":
                import pandas as pd

                df = pd.read_parquet(path)
                samples = df.head(limit).to_dict("records")

            elif suffix == ".txt":
                with open(path) as f:
                    for i, line in enumerate(f):
                        if i >= limit:
                            break
                        samples.append({"text": line.strip()})

            else:
                raise ValueError(f"Cannot preview {suffix} files")

        except Exception as e:
            raise RuntimeError(f"Failed to preview dataset: {e}") from e

        return samples

    def load_from_hub(
        self,
        dataset_id: str,
        split: str = "train",
        output_name: str | None = None,
    ) -> DatasetInfo:
        """Load a dataset from HuggingFace Hub.

        Args:
            dataset_id: HuggingFace dataset ID
            split: Dataset split to load
            output_name: Custom output filename (default: derived from dataset_id)

        Returns:
            DatasetInfo for the saved dataset
        """
        from datetime import datetime

        from datasets import load_dataset

        from model_garden.utils.console import console

        console.print(f"[cyan]📥 Loading dataset {dataset_id} from HuggingFace Hub...[/cyan]")

        # Load from Hub
        dataset = load_dataset(dataset_id, split=split)

        # Determine output filename
        if output_name is None:
            output_name = dataset_id.replace("/", "_") + ".jsonl"

        output_path = self.storage_dir / output_name

        # Convert to JSONL
        example_count = 0
        with open(output_path, "w") as f:
            for item in dataset:
                # Convert dataset item to dict (HF datasets use mapping-like interface)
                item_dict = {k: item[k] for k in item.keys()}  # type: ignore[union-attr]
                f.write(json.dumps(item_dict) + "\n")
                example_count += 1

        console.print(f"[green]✓[/green] Dataset saved to {output_path}")
        console.print(f"[green]✓[/green] Total examples: {example_count}")

        stat = output_path.stat()

        return DatasetInfo(
            name=output_name,
            path=str(output_path),
            size_bytes=stat.st_size,
            format="jsonl",
            examples=example_count,
            created_at=datetime.fromtimestamp(stat.st_ctime).isoformat() + "Z",
            modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat() + "Z",
        )

    def create_sample_dataset(
        self,
        output_path: str,
        num_examples: int = 100,
        dataset_type: Literal["text", "vision"] = "text",
    ) -> DatasetInfo:
        """Create a sample dataset for testing.

        Args:
            output_path: Path to save the dataset
            num_examples: Number of examples to generate
            dataset_type: Type of dataset ("text" or "vision")

        Returns:
            DatasetInfo for the created dataset
        """
        from datetime import datetime

        from model_garden.utils.console import console

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        console.print(f"[cyan]Creating {dataset_type} sample dataset...[/cyan]")

        if dataset_type == "vision":
            from model_garden.training import create_vision_sample_dataset

            create_vision_sample_dataset(str(path), num_examples)
            console.print(
                "[yellow]⚠️  Remember to replace placeholder image paths with real images[/yellow]"
            )
        else:
            from model_garden.training import create_sample_dataset

            create_sample_dataset(str(path), num_examples)

        stat = path.stat()
        example_count = self._count_examples(path)

        console.print(f"[green]✓[/green] Created {example_count} examples at {path}")

        return DatasetInfo(
            name=path.name,
            path=str(path),
            size_bytes=stat.st_size,
            format=path.suffix.lstrip("."),
            examples=example_count,
            created_at=datetime.fromtimestamp(stat.st_ctime).isoformat() + "Z",
            modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat() + "Z",
        )

    def delete_dataset(self, name: str) -> bool:
        """Delete a dataset.

        Args:
            name: Dataset filename

        Returns:
            True if deleted successfully

        Raises:
            FileNotFoundError: If dataset doesn't exist
        """
        file_path = self.storage_dir / name

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {name}")

        file_path.unlink()
        return True

    def get_dataset_path(self, name_or_path: str, from_hub: bool = False) -> str:
        """Resolve a dataset path.

        Args:
            name_or_path: Dataset name (in storage) or path
            from_hub: If True, treat as HuggingFace dataset ID

        Returns:
            Resolved path to the dataset
        """
        if from_hub:
            # Return as-is for Hub datasets
            return name_or_path

        path = Path(name_or_path)

        # Check if it's already an absolute path
        if path.is_absolute():
            return str(path)

        # Check if it exists relative to storage
        storage_path = self.storage_dir / name_or_path
        if storage_path.exists():
            return str(storage_path)

        # Check if it exists relative to CWD
        if path.exists():
            return str(path.resolve())

        # Return as-is (may be created later)
        return name_or_path
