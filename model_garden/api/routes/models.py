# Model management routes
"""
Routes for model management:
- GET /api/v1/models - List available models
- GET /api/v1/models/{model_id} - Get model details
- PUT /api/v1/models/{model_id} - Rename a model
- DELETE /api/v1/models/{model_id} - Delete a model
- POST /api/v1/models/{model_id}/upload-to-hub - Upload to HuggingFace Hub
"""

import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from ..models import APIResponse, ModelRenameRequest, PaginatedResponse
from ..storage import get_storage_manager

router = APIRouter(prefix="/api/v1/models", tags=["models"])


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
                        # Auto-register discovered model
                        models_storage[model_id] = {
                            "id": model_id,
                            "name": model_id,
                            "path": str(model_folder.resolve()),
                            "created_at": datetime.fromtimestamp(
                                model_folder.stat().st_ctime
                            ).isoformat()
                            + "Z",
                            "model_type": "unknown",
                            "base_model": "unknown",
                        }

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
    import os

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
    token = hf_token or os.environ.get("HF_TOKEN")
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
