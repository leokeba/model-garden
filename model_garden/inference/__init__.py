"""
Inference package for Model Garden.

This package provides the vLLM-powered inference service for serving models.
"""

from .service import (
    InferenceService,
    get_inference_service,
    set_inference_service,
)
from .utils import (
    calculate_gpu_memory_utilization,
    detect_quantization_method,
    estimate_model_size_gb,
    get_base_model_from_adapter,
    get_gpu_memory_gb,
    is_lora_adapter,
    is_vision_model,
)

__all__ = [
    # Service
    "InferenceService",
    "get_inference_service",
    "set_inference_service",
    # Utils
    "get_gpu_memory_gb",
    "estimate_model_size_gb",
    "calculate_gpu_memory_utilization",
    "is_lora_adapter",
    "get_base_model_from_adapter",
    "is_vision_model",
    "detect_quantization_method",
]
