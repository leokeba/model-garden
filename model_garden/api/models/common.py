"""Common Pydantic models shared across the API."""

from pydantic import BaseModel


class APIResponse(BaseModel):
    """Standard API response format."""

    success: bool
    data: dict | None = None
    message: str


class PaginatedResponse(BaseModel):
    """Paginated response format."""

    items: list[dict]
    total: int
    page: int
    page_size: int
    pages: int
