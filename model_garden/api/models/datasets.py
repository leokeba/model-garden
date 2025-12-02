"""Pydantic models for dataset-related API endpoints."""

from typing import Any

from pydantic import BaseModel


class DatasetValidationRequest(BaseModel):
    """Request body for dataset validation."""

    path: str
    schema_type: str | None = None  # 'text', 'vision', 'alpaca', or auto-detect


class DatasetValidationResponse(BaseModel):
    """Response body for dataset validation."""

    valid: bool
    total_rows: int
    format: str
    schema_type: str
    fields: list[str]
    field_types: dict[str, str]
    missing_fields: dict[str, int]
    sample_rows: list[dict[str, Any]]
    file_size_bytes: int
    errors: list[str]
    warnings: list[str]
    # Text-specific stats
    avg_input_length: float | None = None
    avg_output_length: float | None = None
    total_tokens_estimate: int | None = None
    # Vision-specific stats
    has_images: bool = False
    image_count: int = 0
