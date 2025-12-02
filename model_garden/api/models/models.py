"""Pydantic models for model management API endpoints."""

from pydantic import BaseModel


class ModelInfo(BaseModel):
    """Model information response."""

    id: str
    name: str
    base_model: str
    status: str
    created_at: str
    updated_at: str
    size_bytes: int | None = None
    path: str
    training_job_id: str | None = None
    config: dict | None = None
    metrics: dict | None = None


class ModelRenameRequest(BaseModel):
    """Request body for renaming a model."""

    name: str  # The new name for the model
