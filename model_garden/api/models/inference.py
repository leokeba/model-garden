"""Pydantic models for inference API endpoints."""

from typing import Any

from pydantic import BaseModel, Field


class ChatCompletionMessage(BaseModel):
    """A single message in a chat completion request."""

    role: str
    content: str | list[dict[str, Any]]  # Can be string or multimodal content


class ChatCompletionRequest(BaseModel):
    """Request body for chat completions (OpenAI-compatible)."""

    model: str = Field(default="default", description="Model to use (ignored, uses loaded model)")
    messages: list[ChatCompletionMessage]
    max_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    stop: list[str] | None = None
    stream: bool = False
    # Structured outputs (JSON mode, schema validation, etc.)
    response_format: dict[str, Any] | None = None
    # Legacy structured output fields (deprecated, use response_format)
    json_schema: dict[str, Any] | None = None
    guided_json: dict[str, Any] | None = None
    guided_regex: str | None = None
    guided_choice: list[str] | None = None
    guided_grammar: str | None = None


class CompletionRequest(BaseModel):
    """Request body for text completions (OpenAI-compatible)."""

    model: str = Field(default="default", description="Model to use (ignored, uses loaded model)")
    prompt: str
    max_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    stop: list[str] | None = None
    stream: bool = False
    # Image support for vision models
    images: list[str] | None = None
    # Structured outputs
    response_format: dict[str, Any] | None = None
    guided_json: dict[str, Any] | None = None
    guided_regex: str | None = None
    guided_choice: list[str] | None = None
    guided_grammar: str | None = None
