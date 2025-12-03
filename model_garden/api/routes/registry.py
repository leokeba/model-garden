# Model Registry routes
"""
Routes for model registry management:
- GET /api/v1/registry/models - List models from registry
- GET /api/v1/registry/models/{model_id} - Get model details
- GET /api/v1/registry/categories - List model categories
- POST /api/v1/registry/validate/training - Validate model for training
- POST /api/v1/registry/validate/inference - Validate model for inference
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Any

from model_garden.model_registry import get_registry

router = APIRouter(prefix="/api/v1/registry", tags=["registry"])


class ValidateRequest(BaseModel):
    """Request to validate a model."""

    model_id: str
    config: dict | None = None


@router.get("/models")
async def list_registry_models(category: str | None = None):
    """List all models from the registry, optionally filtered by category."""
    try:
        registry = get_registry()

        if category:
            models = registry.get_model_list_for_ui(category=category)
        else:
            models = registry.get_model_list_for_ui()

        # For the frontend, also include training_defaults and inference_defaults
        full_models = []
        for model_summary in models:
            model_info = registry.get_model(model_summary["id"])
            if model_info:
                model_data = model_summary.copy()
                model_data["training_defaults"] = {
                    "hyperparameters": model_info.get_training_hyperparameters(),
                    "lora_config": model_info.get_lora_config(),
                    "save_method": model_info.training_defaults.get("save_method", "merged_16bit"),
                }
                model_data["inference_defaults"] = model_info.get_inference_config()
                model_data["requirements"] = {
                    "min_vram_gb": model_info.requirements.min_vram_gb,
                    "recommended_vram_gb": model_info.requirements.recommended_vram_gb,
                    "gpu_required": model_info.requirements.min_vram_gb > 0,
                }
                model_data["capabilities"] = {
                    "supports_vision": model_info.capabilities.vision,
                    "supports_chat": True,  # All our models support chat
                    "supports_function_calling": model_info.capabilities.function_calling,
                    "supports_structured_output": model_info.capabilities.structured_outputs,
                    "max_sequence_length": model_info.inference_defaults.max_model_len,
                }
                full_models.append(model_data)

        return {
            "success": True,
            "data": full_models,
            "total": len(full_models),
        }
    except FileNotFoundError as e:
        # Registry file not found - return empty list
        return {
            "success": True,
            "data": [],
            "total": 0,
            "message": f"Registry not found: {e}",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load registry: {str(e)}",
        )


@router.get("/models/{model_id:path}")
async def get_registry_model(model_id: str):
    """Get details for a specific model from the registry."""
    try:
        registry = get_registry()
        model_info = registry.get_model(model_id)

        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_id}' not found in registry",
            )

        return {
            "success": True,
            "data": {
                "id": model_info.id,
                "name": model_info.name,
                "category": model_info.category,
                "provider": model_info.provider,
                "base_architecture": model_info.base_architecture,
                "parameters": model_info.parameters,
                "description": model_info.description,
                "tags": model_info.tags,
                "status": model_info.status,
                "is_vision": model_info.is_vision_model,
                "is_quantized": model_info.is_quantized,
                "requirements": {
                    "min_vram_gb": model_info.requirements.min_vram_gb,
                    "recommended_vram_gb": model_info.requirements.recommended_vram_gb,
                    "min_ram_gb": model_info.requirements.min_ram_gb,
                    "gpu_required": model_info.requirements.min_vram_gb > 0,
                },
                "capabilities": {
                    "supports_vision": model_info.capabilities.vision,
                    "supports_chat": True,
                    "supports_function_calling": model_info.capabilities.function_calling,
                    "supports_structured_output": model_info.capabilities.structured_outputs,
                    "supports_streaming": model_info.capabilities.streaming,
                    "max_sequence_length": model_info.inference_defaults.max_model_len,
                },
                "training_defaults": {
                    "hyperparameters": model_info.get_training_hyperparameters(),
                    "lora_config": model_info.get_lora_config(),
                    "save_method": model_info.training_defaults.get("save_method", "merged_16bit"),
                },
                "inference_defaults": model_info.get_inference_config(),
                "urls": model_info.urls,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model: {str(e)}",
        )


@router.get("/categories")
async def list_categories():
    """List all model categories."""
    try:
        registry = get_registry()
        categories = registry.get_categories()

        return {
            "success": True,
            "data": categories,
        }
    except FileNotFoundError:
        return {
            "success": True,
            "data": {},
            "message": "Registry not found",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load categories: {str(e)}",
        )


@router.post("/validate/training")
async def validate_for_training(request: ValidateRequest):
    """Validate if a model can be used for training."""
    try:
        registry = get_registry()
        is_valid, error = registry.validate_model_for_training(request.model_id)

        errors = [error] if error else []
        warnings: list[str] = []

        # Additional validation based on config if provided
        if is_valid and request.config:
            model_info = registry.get_model(request.model_id)
            if model_info:
                # Check VRAM requirements
                if request.config.get("batch_size", 2) > 4:
                    warnings.append(
                        f"Batch size > 4 may require more than {model_info.requirements.min_vram_gb}GB VRAM"
                    )

        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}",
        )


@router.post("/validate/inference")
async def validate_for_inference(request: ValidateRequest):
    """Validate if a model can be used for inference."""
    try:
        registry = get_registry()
        is_valid, error = registry.validate_model_for_inference(request.model_id)

        errors = [error] if error else []
        warnings: list[str] = []

        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}",
        )
