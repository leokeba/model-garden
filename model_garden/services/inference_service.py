"""Inference service - backend-agnostic model serving.

This module provides request classes and a service wrapper that consolidates
inference-related logic from CLI and API into a single place.

The actual inference implementation is delegated to model_garden.inference.InferenceService,
but this module provides:
- Unified request classes for model loading and inference
- Registry integration for model defaults
- Consistent parameter handling across CLI and API

Example:
    >>> from model_garden.services import InferenceService, ModelLoadRequest
    >>>
    >>> service = InferenceService()
    >>> request = ModelLoadRequest(model_path="./models/my-model")
    >>> await service.load_model(request)
    >>> response = await service.generate("Hello, world!")
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelLoadRequest:
    """Request to load a model for inference.

    This dataclass consolidates all model loading parameters. Both CLI and API
    convert their inputs to ModelLoadRequest before calling InferenceService.

    Attributes:
        model_path: Path to model or HuggingFace model ID
        base_model: Optional explicit base model (for adapters without config)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory fraction (0.0 = auto)
        max_model_len: Maximum sequence length
        max_num_seqs: Maximum concurrent sequences
        enforce_eager: Disable CUDA graphs (saves memory)
        limit_mm_per_prompt: Limit multimodal inputs per prompt
        dtype: Data type (auto, float16, bfloat16, float32)
        quantization: Quantization method (auto, awq, gptq, etc.)
        enable_lora: Enable LoRA adapter support
        max_loras: Maximum concurrent LoRA adapters
        max_lora_rank: Maximum LoRA rank to support
    """

    model_path: str
    base_model: str | None = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.0  # 0 = auto
    max_model_len: int | None = None
    max_num_seqs: int = 16
    enforce_eager: bool = False
    limit_mm_per_prompt: dict[str, int] | None = None
    dtype: str = "auto"
    quantization: str | None = "auto"
    enable_lora: bool = True
    max_loras: int = 1
    max_lora_rank: int = 64
    trust_remote_code: bool = True

    def apply_registry_defaults(self) -> ModelLoadRequest:
        """Apply defaults from the model registry if available.

        This looks up the model in the registry and fills in any unspecified
        parameters with registry defaults.

        Returns:
            Self with registry defaults applied
        """
        import copy

        try:
            from model_garden.model_registry import get_model

            model_info = get_model(self.model_path)
            if model_info is None:
                return self

            result = copy.deepcopy(self)

            # Apply defaults from registry only if not explicitly set
            defaults = model_info.inference_defaults

            if result.tensor_parallel_size == 1 and defaults.tensor_parallel_size > 1:
                result.tensor_parallel_size = defaults.tensor_parallel_size

            if result.gpu_memory_utilization == 0.0 and defaults.gpu_memory_utilization:
                result.gpu_memory_utilization = defaults.gpu_memory_utilization

            if result.max_model_len is None and defaults.max_model_len:
                result.max_model_len = defaults.max_model_len

            if result.max_num_seqs == 16 and defaults.max_num_seqs:
                result.max_num_seqs = defaults.max_num_seqs

            if result.enforce_eager is False and defaults.enforce_eager:
                result.enforce_eager = defaults.enforce_eager

            if result.dtype == "auto" and defaults.dtype:
                result.dtype = defaults.dtype

            if result.quantization in ("auto", None) and defaults.quantization:
                result.quantization = defaults.quantization

            if result.limit_mm_per_prompt is None and defaults.limit_mm_per_prompt:
                result.limit_mm_per_prompt = defaults.limit_mm_per_prompt

            return result

        except Exception:
            # Registry lookup failed, return unchanged
            return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_path": self.model_path,
            "base_model": self.base_model,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "enforce_eager": self.enforce_eager,
            "limit_mm_per_prompt": self.limit_mm_per_prompt,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "enable_lora": self.enable_lora,
            "max_loras": self.max_loras,
            "max_lora_rank": self.max_lora_rank,
            "trust_remote_code": self.trust_remote_code,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelLoadRequest:
        """Create from dictionary."""
        return cls(
            model_path=data.get("model_path", ""),
            base_model=data.get("base_model"),
            tensor_parallel_size=data.get("tensor_parallel_size", 1),
            gpu_memory_utilization=data.get("gpu_memory_utilization", 0.0),
            max_model_len=data.get("max_model_len"),
            max_num_seqs=data.get("max_num_seqs", 16),
            enforce_eager=data.get("enforce_eager", False),
            limit_mm_per_prompt=data.get("limit_mm_per_prompt"),
            dtype=data.get("dtype", "auto"),
            quantization=data.get("quantization", "auto"),
            enable_lora=data.get("enable_lora", True),
            max_loras=data.get("max_loras", 1),
            max_lora_rank=data.get("max_lora_rank", 64),
            trust_remote_code=data.get("trust_remote_code", True),
        )


@dataclass
class InferenceRequest:
    """Request for text generation.

    Attributes:
        prompt: Input text prompt
        messages: Chat messages (alternative to prompt)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling probability
        top_k: Top-k sampling (-1 to disable)
        frequency_penalty: Frequency penalty
        presence_penalty: Presence penalty
        repetition_penalty: Repetition penalty
        stop: Stop sequences
        stream: Whether to stream response
        image: Optional image for vision models
        images: Optional list of images for vision models
        structured_outputs: Structured output parameters
    """

    # Input (one of these required)
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None

    # Generation parameters
    max_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    stop: list[str] | None = None
    stream: bool = False

    # Vision inputs
    image: str | None = None
    images: list[str] | None = None

    # Structured outputs
    structured_outputs: dict | None = None


class InferenceService:
    """Backend-agnostic inference service wrapper.

    This service wraps the underlying vLLM-based InferenceService and provides:
    - Registry integration for model defaults
    - Consistent parameter handling
    - Unified interface for CLI and API

    Example:
        >>> service = InferenceService()
        >>> request = ModelLoadRequest(model_path="./models/my-model")
        >>> await service.load_model(request)
        >>> response = await service.chat([{"role": "user", "content": "Hello"}])
    """

    def __init__(self):
        """Initialize the inference service wrapper."""
        self._service = None

    @property
    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self._service is not None and self._service.is_loaded

    @property
    def model_path(self) -> str | None:
        """Get the loaded model path."""
        return self._service.model_path if self._service else None

    async def load_model(self, request: ModelLoadRequest) -> dict[str, Any]:
        """Load a model for inference.

        Args:
            request: Model load request with all parameters

        Returns:
            Model info dictionary

        Raises:
            RuntimeError: If a model is already loaded
        """
        from model_garden.inference import InferenceService as VLLMInferenceService
        from model_garden.utils.console import console

        if self.is_loaded:
            assert self._service is not None  # Type narrowing
            raise RuntimeError(
                f"Model already loaded: {self._service.model_path}. Unload it first."
            )

        # Apply registry defaults
        request = request.apply_registry_defaults()

        console.print(f"[cyan]Loading model: {request.model_path}[/cyan]")

        # Create the underlying vLLM service
        self._service = VLLMInferenceService(
            model_path=request.model_path,
            tensor_parallel_size=request.tensor_parallel_size,
            gpu_memory_utilization=request.gpu_memory_utilization,
            max_model_len=request.max_model_len,
            max_num_seqs=request.max_num_seqs,
            enforce_eager=request.enforce_eager,
            limit_mm_per_prompt=request.limit_mm_per_prompt,
            dtype=request.dtype,
            quantization=request.quantization,
            trust_remote_code=request.trust_remote_code,
            enable_lora=request.enable_lora,
            max_loras=request.max_loras,
            max_lora_rank=request.max_lora_rank,
        )

        # Override base model if explicitly provided
        if request.base_model:
            self._service.base_model_path = request.base_model
            self._service.is_adapter = True
            self._service.adapter_path = request.model_path

        # Load the model
        await self._service.load_model()

        # Initialize carbon tracking
        try:
            from model_garden.carbon import init_inference_tracker

            init_inference_tracker(request.model_path)
            console.print("[green]✓[/green] Carbon tracking initialized")
        except Exception as e:
            console.print(f"[yellow]⚠️  Carbon tracking not available: {e}[/yellow]")

        return self._service.get_model_info()

    async def unload_model(self) -> None:
        """Unload the current model."""
        if not self.is_loaded:
            return

        # Stop carbon tracking
        try:
            from model_garden.carbon import stop_inference_tracker

            emissions_data = stop_inference_tracker()
            if emissions_data:
                from model_garden.utils.console import console

                console.print(
                    f"[green]✓[/green] Inference emissions: {emissions_data['emissions_kg_co2']:.6f} kg CO2"
                )
        except Exception:
            pass

        assert self._service is not None  # Type narrowing
        await self._service.unload_model()
        self._service = None

    async def generate(self, request: InferenceRequest) -> dict | AsyncIterator[str]:
        """Generate text from a prompt.

        Args:
            request: Inference request with prompt and parameters

        Returns:
            Generation result or async iterator if streaming
        """
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")

        assert self._service is not None  # Type narrowing

        # Combine image and images into single list
        images = request.images or []
        if request.image:
            images = [request.image] + images

        return await self._service.generate(
            prompt=request.prompt or "",
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            repetition_penalty=request.repetition_penalty,
            stop=request.stop,
            stream=request.stream,
            images=images if images else None,
            structured_outputs=request.structured_outputs,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = False,
        image: str | None = None,
        structured_outputs: dict | None = None,
        **kwargs,
    ) -> dict | AsyncIterator[dict]:
        """Chat completion (OpenAI-compatible).

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            stream: Whether to stream response
            image: Optional image for vision models
            structured_outputs: Structured output parameters
            **kwargs: Additional parameters

        Returns:
            Chat completion response or async iterator if streaming
        """
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")

        assert self._service is not None  # Type narrowing
        return await self._service.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            image=image,
            structured_outputs=structured_outputs,
            **kwargs,
        )

    def get_model_info(self) -> dict[str, Any] | None:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return None
        assert self._service is not None  # Type narrowing
        return self._service.get_model_info()


# Singleton instance management (for global state in API)
_global_service: InferenceService | None = None


def get_inference_service() -> InferenceService | None:
    """Get the global inference service instance."""
    global _global_service
    return _global_service


def set_inference_service(service: InferenceService | None) -> None:
    """Set the global inference service instance."""
    global _global_service
    _global_service = service
