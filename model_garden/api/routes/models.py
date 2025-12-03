# Model management routes
"""
Routes for model management:
- GET /api/v1/models - List available models
- GET /api/v1/models/{model_id} - Get model details
- PUT /api/v1/models/{model_id} - Rename a model
- DELETE /api/v1/models/{model_id} - Delete a model
- POST /api/v1/models/{model_id}/upload-to-hub - Upload to HuggingFace Hub
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from model_garden.utils.hf_cache import get_hf_token

from ..models import APIResponse, ModelRenameRequest, PaginatedResponse
from ..storage import get_storage_manager

router = APIRouter(prefix="/api/v1/models", tags=["models"])


def extract_model_metadata(model_path: Path) -> dict:
    """Extract metadata from a model directory.

    Reads config.json, adapter_config.json, or README.md to extract:
    - base_model: The base model used for fine-tuning
    - model_type: Whether it's a vision or text model
    - is_adapter: Whether this is a LoRA adapter

    Args:
        model_path: Path to the model directory

    Returns:
        Dictionary with extracted metadata
    """
    metadata = {
        "base_model": None,
        "model_type": None,
        "is_adapter": False,
    }

    if not model_path.exists():
        return metadata

    # Try adapter_config.json first (for LoRA models)
    adapter_config_path = model_path / "adapter_config.json"
    if adapter_config_path.exists():
        try:
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)

            metadata["is_adapter"] = True

            # Get base model from adapter config
            base_model = adapter_config.get("base_model_name_or_path", "")
            if base_model:
                # Clean up unsloth model names to show the original model
                if "unsloth/" in base_model:
                    # e.g., "unsloth/qwen2.5-vl-7b-instruct-unsloth-bnb-4bit" -> "Qwen/Qwen2.5-VL-7B-Instruct"
                    clean_name = base_model.replace("unsloth/", "")
                    # Try to infer original model name
                    if "qwen2.5-vl-7b" in clean_name.lower():
                        metadata["base_model"] = "Qwen/Qwen2.5-VL-7B-Instruct"
                    elif "qwen2.5-vl-3b" in clean_name.lower():
                        metadata["base_model"] = "Qwen/Qwen2.5-VL-3B-Instruct"
                    elif "qwen2.5-vl-72b" in clean_name.lower():
                        metadata["base_model"] = "Qwen/Qwen2.5-VL-72B-Instruct"
                    elif "qwen3-vl-8b" in clean_name.lower():
                        metadata["base_model"] = "Qwen/Qwen3-VL-8B-Instruct"
                    else:
                        metadata["base_model"] = base_model
                else:
                    metadata["base_model"] = base_model

            # Detect model type from auto_mapping or target modules
            auto_mapping = adapter_config.get("auto_mapping", {})
            base_class = auto_mapping.get("base_model_class", "")
            if "VL" in base_class or "vision" in base_class.lower():
                metadata["model_type"] = "vision"
            else:
                metadata["model_type"] = "text"

        except Exception as e:
            print(f"Warning: Could not parse adapter_config.json: {e}")

    # Try config.json for merged models
    config_path = model_path / "config.json"
    if config_path.exists() and not metadata["base_model"]:
        try:
            with open(config_path) as f:
                config = json.load(f)

            # Check for _name_or_path which sometimes contains the base model
            name_or_path = config.get("_name_or_path", "")
            if name_or_path and "/" in name_or_path:
                metadata["base_model"] = name_or_path

            # Detect model type from architectures
            architectures = config.get("architectures", [])
            if architectures:
                arch = architectures[0].lower()
                if "vl" in arch or "vision" in arch or "image" in arch:
                    metadata["model_type"] = "vision"
                else:
                    metadata["model_type"] = "text"

        except Exception as e:
            print(f"Warning: Could not parse config.json: {e}")

    return metadata


def enrich_model_from_training_jobs(model_data: dict, training_jobs: dict) -> dict:
    """Enrich model data by linking to its training job.

    Tries to find the training job that produced this model by matching:
    1. training_job_id field
    2. output_dir matching the model path
    3. model name matching job name

    Args:
        model_data: The model data dict to enrich
        training_jobs: Dict of all training jobs

    Returns:
        Enriched model data dict
    """
    # Already has a training job link
    if model_data.get("training_job_id") and model_data["training_job_id"] in training_jobs:
        job = training_jobs[model_data["training_job_id"]]
        if model_data.get("base_model") in (None, "unknown", ""):
            model_data["base_model"] = job.get("base_model", "unknown")
        if model_data.get("model_type") in (None, "unknown", ""):
            model_data["model_type"] = "vision" if job.get("is_vision") else "text"
        return model_data

    model_path = model_data.get("path", "")
    model_name = model_data.get("name", model_data.get("id", ""))

    # Try to find matching training job
    for job_id, job in training_jobs.items():
        job_output = job.get("output_dir", "")
        job_name = job.get("name", "")

        # Match by output directory
        if (
            job_output
            and model_path
            and (
                model_path == job_output
                or model_path.rstrip("/") == job_output.rstrip("/")
                or Path(model_path).resolve() == Path(job_output).resolve()
            )
        ):
            model_data["training_job_id"] = job_id
            if model_data.get("base_model") in (None, "unknown", ""):
                model_data["base_model"] = job.get("base_model", "unknown")
            if model_data.get("model_type") in (None, "unknown", ""):
                model_data["model_type"] = "vision" if job.get("is_vision") else "text"
            if model_data.get("created_at") in (None, "unknown", ""):
                model_data["created_at"] = job.get("created_at")
            return model_data

        # Match by name (less reliable, only if names match exactly)
        if model_name and job_name and model_name == job_name:
            # Additional check: job should be completed
            if job.get("status") == "completed":
                model_data["training_job_id"] = job_id
                if model_data.get("base_model") in (None, "unknown", ""):
                    model_data["base_model"] = job.get("base_model", "unknown")
                if model_data.get("model_type") in (None, "unknown", ""):
                    model_data["model_type"] = "vision" if job.get("is_vision") else "text"
                return model_data

    return model_data


def get_model_files_info(model_path: Path) -> dict:
    """Get information about model files."""
    info = {
        "total_size": 0,
        "file_count": 0,
        "has_adapter": False,
        "has_safetensors": False,
        "has_pytorch": False,
    }

    if not model_path.exists():
        return info

    for file in model_path.rglob("*"):
        if file.is_file():
            info["file_count"] += 1
            info["total_size"] += file.stat().st_size

            if file.name == "adapter_config.json":
                info["has_adapter"] = True
            elif file.suffix == ".safetensors":
                info["has_safetensors"] = True
            elif file.suffix in [".bin", ".pt", ".pth"]:
                info["has_pytorch"] = True

    return info


@router.get("", response_model=PaginatedResponse)
async def list_models(
    page: int = 1,
    page_size: int = 20,
    model_type: str | None = None,
):
    """List all available models."""
    storage = get_storage_manager()
    models_storage = storage.load_models()

    # Load training jobs for enrichment
    training_jobs = storage.load_training_jobs()

    # Include models saved to default models directory
    models_dir = Path("./models")
    if models_dir.exists():
        for model_folder in models_dir.iterdir():
            if model_folder.is_dir():
                model_id = model_folder.name
                if model_id not in models_storage:
                    # Check if it's a valid model folder
                    config_file = model_folder / "config.json"
                    adapter_config = model_folder / "adapter_config.json"

                    if config_file.exists() or adapter_config.exists():
                        # Extract metadata from model files
                        metadata = extract_model_metadata(model_folder)

                        # Auto-register discovered model with extracted metadata
                        model_data = {
                            "id": model_id,
                            "name": model_id,
                            "path": str(model_folder.resolve()),
                            "created_at": datetime.fromtimestamp(
                                model_folder.stat().st_ctime
                            ).isoformat()
                            + "Z",
                            "model_type": metadata.get("model_type", "unknown"),
                            "base_model": metadata.get("base_model", "unknown"),
                        }

                        # Try to enrich from training jobs
                        model_data = enrich_model_from_training_jobs(model_data, training_jobs)
                        models_storage[model_id] = model_data

    # Enrich existing models that have "unknown" values
    updated = False
    for model_id, model_data in models_storage.items():
        if model_data.get("base_model") == "unknown" or model_data.get("model_type") == "unknown":
            model_path = Path(model_data.get("path", ""))
            if model_path.exists():
                metadata = extract_model_metadata(model_path)
                if metadata.get("base_model") and model_data.get("base_model") == "unknown":
                    model_data["base_model"] = metadata["base_model"]
                    updated = True
                if metadata.get("model_type") and model_data.get("model_type") == "unknown":
                    model_data["model_type"] = metadata["model_type"]
                    updated = True

            # Try training jobs enrichment
            enriched = enrich_model_from_training_jobs(model_data, training_jobs)
            if enriched.get("base_model") != model_data.get("base_model"):
                model_data.update(enriched)
                updated = True

    # Save enriched data back to storage
    if updated:
        storage.save_models(models_storage)

    # Filter models
    filtered_models = list(models_storage.values())

    if model_type:
        filtered_models = [m for m in filtered_models if m.get("model_type") == model_type]

    # Sort by created_at
    filtered_models.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    # Pagination
    total = len(filtered_models)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    items = filtered_models[start_idx:end_idx]

    pages = (total + page_size - 1) // page_size

    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )


@router.get("/{model_id}")
async def get_model(model_id: str):
    """Get information about a specific model."""
    storage = get_storage_manager()
    models_storage = storage.load_models()

    if model_id not in models_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {model_id} not found"
        )

    model_data = models_storage[model_id].copy()

    # Add file information if path exists
    model_path = Path(model_data.get("path", ""))
    if model_path.exists():
        file_info = get_model_files_info(model_path)
        model_data["files"] = file_info

    return model_data


@router.put("/{model_id}", response_model=APIResponse)
async def rename_model(model_id: str, request: ModelRenameRequest):
    """Rename a model."""
    storage = get_storage_manager()
    models_storage = storage.load_models()

    if model_id not in models_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {model_id} not found"
        )

    # Update the name
    models_storage[model_id]["name"] = request.name
    storage.save_models(models_storage)

    return APIResponse(success=True, message=f"Model renamed to {request.name}")


@router.delete("/{model_id}", response_model=APIResponse)
async def delete_model(model_id: str, delete_files: bool = False):
    """Delete a model from storage and optionally from disk."""
    storage = get_storage_manager()
    models_storage = storage.load_models()

    if model_id not in models_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {model_id} not found"
        )

    model_data = models_storage[model_id]
    model_path = Path(model_data.get("path", ""))

    # Delete files if requested
    if delete_files and model_path.exists():
        try:
            shutil.rmtree(model_path)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete model files: {str(e)}",
            ) from None

    # Remove from storage
    del models_storage[model_id]
    storage.save_models(models_storage)

    message = f"Model {model_id} deleted"
    if delete_files:
        message += " (files removed)"

    return APIResponse(success=True, message=message)


@router.post("/{model_id}/upload-to-hub", response_model=APIResponse)
async def upload_model_to_hub(
    model_id: str,
    repo_id: str | None = None,
    private: bool = True,
    hf_token: str | None = None,
):
    """Upload a model to HuggingFace Hub."""

    from huggingface_hub import HfApi, create_repo

    storage = get_storage_manager()
    models_storage = storage.load_models()

    if model_id not in models_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {model_id} not found"
        )

    model_data = models_storage[model_id]
    model_path = Path(model_data.get("path", ""))

    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model path does not exist: {model_path}",
        )

    # Get token from request or environment
    token = hf_token or get_hf_token()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="HuggingFace token required. Set HF_TOKEN environment variable or provide hf_token parameter.",
        )

    try:
        api = HfApi(token=token)

        # Get user info for repo name
        user_info = api.whoami()
        username = user_info.get("name", user_info.get("user", "unknown"))

        # Generate repo ID if not provided
        if not repo_id:
            model_name = model_data.get("name", model_id).replace(" ", "-")
            repo_id = f"{username}/{model_name}"

        # Generate description
        repo_description = f"Fine-tuned model: {model_data.get('name', model_id)}"
        if model_data.get("base_model"):
            repo_description += f" (based on {model_data.get('base_model')})"

        # Create repo if it doesn't exist
        try:
            create_repo(
                repo_id=repo_id,
                token=token,
                private=private,
                exist_ok=True,
                repo_type="model",
            )
            print(f"✓ Repository created/verified: {repo_id}")
        except Exception as e:
            print(f"Warning: Could not create repository: {e}")

        # Upload model files
        print(f"📤 Uploading model to {repo_id}...")
        url = api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            token=token,
            commit_message=f"Upload {model_data.get('name', model_id)} via Model Garden",
        )

        if not url:
            raise RuntimeError("Failed to upload model to HuggingFace Hub: no URL returned")

        # Create README if it doesn't exist
        readme_path = model_path / "README.md"
        if not readme_path.exists():
            try:
                readme_content = f"""---
license: apache-2.0
base_model: {model_data.get("base_model", "unknown")}
tags:
  - model-garden
  - fine-tuned
  - {model_data.get("model_type", "language-model")}
---

# {model_data.get("name", model_id)}

{repo_description}

## Model Details

- **Base Model**: {model_data.get("base_model", "unknown")}
- **Fine-tuned with**: [Model Garden](https://github.com/leokeba/model-garden)
- **Training Date**: {model_data.get("created_at", "unknown")}
- **Model Type**: {model_data.get("model_type", "unknown")}

## Usage

### With Model Garden

```bash
# Serve the model
uv run model-garden serve-model --model-path {repo_id}

# Generate text
uv run model-garden inference-generate \\
    --model-path {repo_id} \\
    --prompt "Your prompt here"
```

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Generate text
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

---

*Generated with [Model Garden](https://github.com/leokeba/model-garden)*
"""
                api.upload_file(
                    path_or_fileobj=readme_content.encode(),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    token=token,
                    commit_message="Add model card",
                )
                print("✓ README.md created and uploaded")
            except Exception as e:
                print(f"Warning: Could not create README: {e}")

        # Update model storage with Hub URL
        models_storage[model_id]["hub_url"] = f"https://huggingface.co/{repo_id}"
        models_storage[model_id]["hub_repo_id"] = repo_id
        storage.save_models(models_storage)

        return APIResponse(
            success=True,
            data={
                "repo_id": repo_id,
                "url": f"https://huggingface.co/{repo_id}",
                "commit_url": url,
            },
            message="Model uploaded successfully to HuggingFace Hub",
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload model: {str(e)}",
        ) from None
