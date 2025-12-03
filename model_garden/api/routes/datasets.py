# Dataset routes
"""
Routes for dataset management:
- GET /api/v1/datasets - List datasets
- POST /api/v1/datasets/upload - Upload a dataset
- GET /api/v1/datasets/{name}/stats - Get dataset statistics
- GET /api/v1/datasets/{name}/preview - Preview dataset samples
- DELETE /api/v1/datasets/{name} - Delete a dataset
- POST /api/v1/datasets/from-hub - Load from HuggingFace Hub
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile, status

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


@router.get("")
async def list_datasets():
    """List all available datasets."""
    datasets_dir = Path("./storage/datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    for dataset_file in datasets_dir.iterdir():
        if dataset_file.is_file():
            stat = dataset_file.stat()

            # Count examples
            example_count = 0
            try:
                if dataset_file.suffix == ".jsonl":
                    with open(dataset_file) as f:
                        example_count = sum(1 for _ in f)
                elif dataset_file.suffix == ".json":
                    with open(dataset_file) as f:
                        data = json.load(f)
                        example_count = len(data) if isinstance(data, list) else 1
                elif dataset_file.suffix == ".csv":
                    df = pd.read_csv(dataset_file)
                    example_count = len(df)
                elif dataset_file.suffix == ".parquet":
                    df = pd.read_parquet(dataset_file)
                    example_count = len(df)
            except Exception as e:
                print(f"Warning: Could not count examples in {dataset_file.name}: {e}")

            datasets.append(
                {
                    "name": dataset_file.name,
                    "path": str(dataset_file),
                    "size": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat() + "Z",
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat() + "Z",
                    "format": dataset_file.suffix.lstrip("."),
                    "examples": example_count,
                }
            )

    # Sort by modified date (newest first)
    datasets.sort(key=lambda x: x["modified_at"], reverse=True)

    return {"datasets": datasets}


# Module-level default for File upload parameter to avoid B008 error
_file_upload_default = File(...)


@router.post("/upload")
async def upload_dataset(file: UploadFile = _file_upload_default):
    """Upload a dataset file."""
    from model_garden.utils import DatasetValidator

    datasets_dir = Path("./storage/datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # Validate file extension
    allowed_extensions = [".json", ".jsonl", ".csv", ".txt", ".parquet"]

    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must have a name")

    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format. Allowed formats: {', '.join(allowed_extensions)}",
        )

    file_path = datasets_dir / file.filename

    # Check if file exists
    if file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=f"Dataset {file.filename} already exists"
        )

    try:
        # Write file in chunks
        with open(file_path, "wb") as f:
            while chunk := await file.read(8192):
                f.write(chunk)

        # Validate dataset
        validation_stats = None
        if file_ext in [".json", ".jsonl", ".csv"]:
            try:
                validation_stats = DatasetValidator.validate_dataset(file_path)

                if validation_stats.validation_errors:
                    file_path.unlink()
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "message": "Dataset validation failed",
                            "errors": validation_stats.validation_errors,
                            "warnings": validation_stats.warnings,
                        },
                    )
            except HTTPException:
                raise
            except Exception as e:
                print(f"Warning: Dataset validation failed: {e}")

        stat = file_path.stat()

        # Count examples
        example_count = 0
        if validation_stats:
            example_count = validation_stats.total_rows
        else:
            try:
                if file_ext == ".jsonl":
                    with open(file_path) as f:
                        example_count = sum(1 for _ in f)
                elif file_ext == ".json":
                    with open(file_path) as f:
                        data = json.load(f)
                        example_count = len(data) if isinstance(data, list) else 1
                elif file_ext == ".csv":
                    df = pd.read_csv(file_path)
                    example_count = len(df)
                elif file_ext == ".parquet":
                    df = pd.read_parquet(file_path)
                    example_count = len(df)
            except Exception as e:
                print(f"Warning: Could not count examples: {e}")

        response_data = {
            "success": True,
            "message": f"Dataset {file.filename} uploaded successfully",
            "dataset": {
                "name": file.filename,
                "path": str(file_path),
                "size": stat.st_size,
                "format": file_ext.lstrip("."),
                "examples": example_count,
            },
        }

        if validation_stats:
            response_data["validation"] = {
                "schema_type": DatasetValidator.detect_schema_type(validation_stats.sample_rows)
                if validation_stats.sample_rows
                else "unknown",
                "fields": validation_stats.fields,
                "warnings": validation_stats.warnings,
                "has_images": validation_stats.has_images,
                "image_count": validation_stats.image_count,
                "avg_input_length": validation_stats.avg_input_length,
                "avg_output_length": validation_stats.avg_output_length,
                "total_tokens_estimate": validation_stats.total_tokens_estimate,
            }

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload dataset: {str(e)}",
        ) from None


@router.get("/{dataset_name}/stats")
async def get_dataset_stats(dataset_name: str):
    """Get detailed statistics for a dataset."""
    from model_garden.utils import DatasetValidator

    datasets_dir = Path("./storage/datasets")
    file_path = datasets_dir / dataset_name

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset {dataset_name} not found"
        )

    try:
        stats = DatasetValidator.validate_dataset(file_path)

        return {
            "name": dataset_name,
            "total_rows": stats.total_rows,
            "format": stats.format,
            "fields": stats.fields,
            "field_types": stats.field_types,
            "missing_fields": stats.missing_fields,
            "file_size_bytes": stats.file_size_bytes,
            "validation_errors": stats.validation_errors,
            "warnings": stats.warnings,
            "sample_rows": stats.sample_rows,
            "schema_type": DatasetValidator.detect_schema_type(stats.sample_rows)
            if stats.sample_rows
            else "unknown",
            "text_stats": {
                "avg_input_length": stats.avg_input_length,
                "avg_output_length": stats.avg_output_length,
                "total_tokens_estimate": stats.total_tokens_estimate,
            }
            if stats.avg_input_length is not None
            else None,
            "vision_stats": {
                "has_images": stats.has_images,
                "image_count": stats.image_count,
                "sample_image_paths": stats.image_paths,
            }
            if stats.has_images
            else None,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze dataset: {str(e)}",
        ) from None


@router.get("/{dataset_name}/preview")
async def preview_dataset(dataset_name: str, limit: int = 10):
    """Preview samples from a dataset."""
    datasets_dir = Path("./storage/datasets")
    file_path = datasets_dir / dataset_name

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset {dataset_name} not found"
        )

    try:
        samples = []
        file_ext = file_path.suffix.lower()

        if file_ext == ".jsonl":
            with open(file_path) as f:
                for i, line in enumerate(f):
                    if i >= limit:
                        break
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        samples.append({"_error": "Invalid JSON", "_raw": line})

        elif file_ext == ".json":
            with open(file_path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data[:limit]
                else:
                    samples = [data]

        elif file_ext == ".csv":
            df = pd.read_csv(file_path, nrows=limit)
            samples = df.to_dict("records")

        elif file_ext == ".parquet":
            df = pd.read_parquet(file_path)
            samples = df.head(limit).to_dict("records")

        elif file_ext == ".txt":
            with open(file_path) as f:
                for i, line in enumerate(f):
                    if i >= limit:
                        break
                    samples.append({"text": line.strip()})

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Cannot preview {file_ext} files"
            )

        return {"samples": samples, "count": len(samples)}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to preview dataset: {str(e)}",
        ) from None


@router.delete("/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """Delete a dataset."""
    datasets_dir = Path("./storage/datasets")
    file_path = datasets_dir / dataset_name

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset {dataset_name} not found"
        )

    try:
        file_path.unlink()
        return {"success": True, "message": f"Dataset {dataset_name} deleted successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete dataset: {str(e)}",
        ) from None


@router.post("/from-hub")
async def load_dataset_from_hub(request: dict):
    """Load a dataset from HuggingFace Hub."""
    try:
        from datasets import load_dataset

        dataset_id = request.get("dataset_id")
        split = request.get("split", "train")

        if not dataset_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="dataset_id is required"
            )

        print(f"📥 Loading dataset {dataset_id} from HuggingFace Hub...")

        # Load from Hub
        dataset = load_dataset(dataset_id, split=split)

        # Save to storage
        datasets_dir = Path("./storage/datasets")
        datasets_dir.mkdir(parents=True, exist_ok=True)

        safe_name = dataset_id.replace("/", "_") + ".jsonl"
        output_path = datasets_dir / safe_name

        # Convert to JSONL
        example_count = 0
        with open(output_path, "w") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")
                example_count += 1

        print(f"✓ Dataset saved to {output_path}")
        print(f"✓ Total examples: {example_count}")

        return {
            "success": True,
            "message": f"Dataset {dataset_id} loaded successfully",
            "dataset_name": safe_name,
            "examples": example_count,
            "path": str(output_path),
        }

    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load dataset from Hub: {str(e)}",
        ) from None
