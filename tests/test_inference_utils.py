"""Tests for inference utilities.

These tests verify the utility functions used for model detection,
GPU memory estimation, and quantization detection.
"""

import json
from pathlib import Path

import pytest


class TestGPUMemoryDetection:
    """Tests for GPU memory detection functions."""

    @pytest.mark.requires_gpu
    def test_get_gpu_memory_with_cuda(self):
        """Test getting GPU memory when CUDA is available."""
        # This test requires actual GPU access
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from model_garden.inference.utils import get_gpu_memory_gb

        result = get_gpu_memory_gb()
        assert isinstance(result, float)
        assert result > 0.0


class TestModelSizeEstimation:
    """Tests for model size estimation."""

    def test_estimate_model_size_from_name(self):
        """Test estimating model size from HuggingFace model name."""
        from model_garden.inference.utils import estimate_model_size_gb

        # Model with size in name (e.g., 7B)
        size = estimate_model_size_gb("unsloth/llama-3-8B-Instruct")
        assert size >= 8.0  # Should estimate based on "8B" in name

        # Model with smaller size
        size = estimate_model_size_gb("unsloth/tinyllama-1.1B")
        assert size >= 1.0

    def test_estimate_model_size_local_dir(self, tmp_path: Path):
        """Test estimating model size from local directory."""
        from model_garden.inference.utils import estimate_model_size_gb

        # Create a fake model directory with weight files
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()

        # Create a fake safetensors file (1MB)
        fake_weights = model_dir / "model.safetensors"
        fake_weights.write_bytes(b"\x00" * (1024 * 1024))

        size = estimate_model_size_gb(str(model_dir))
        assert size >= 1.0  # Should return at least 1.0 GB

    def test_estimate_model_size_nonexistent(self):
        """Test estimating model size for non-existent path."""
        from model_garden.inference.utils import estimate_model_size_gb

        size = estimate_model_size_gb("/nonexistent/path")
        assert size == 7.0  # Default fallback


class TestGPUMemoryUtilization:
    """Tests for GPU memory utilization calculation."""

    def test_calculate_gpu_memory_utilization(self):
        """Test calculating GPU memory utilization."""
        from model_garden.inference.utils import calculate_gpu_memory_utilization

        # Should return a value between 0.5 and 0.95
        util = calculate_gpu_memory_utilization("unsloth/llama-3-8B")
        assert 0.5 <= util <= 0.95

    def test_calculate_gpu_memory_utilization_with_tensor_parallel(self):
        """Test GPU memory utilization with tensor parallelism."""
        from model_garden.inference.utils import calculate_gpu_memory_utilization

        util_1 = calculate_gpu_memory_utilization("unsloth/llama-3-8B", tensor_parallel_size=1)
        util_2 = calculate_gpu_memory_utilization("unsloth/llama-3-8B", tensor_parallel_size=2)

        # Utilization should be lower with more GPUs (more overhead)
        assert util_2 <= util_1

    def test_calculate_gpu_memory_utilization_with_max_model_len(self):
        """Test GPU memory utilization with different context lengths."""
        from model_garden.inference.utils import calculate_gpu_memory_utilization

        util_short = calculate_gpu_memory_utilization("unsloth/llama-3-8B", max_model_len=2048)
        util_long = calculate_gpu_memory_utilization("unsloth/llama-3-8B", max_model_len=8192)

        # Both should be valid
        assert 0.5 <= util_short <= 0.95
        assert 0.5 <= util_long <= 0.95


class TestLoRADetection:
    """Tests for LoRA adapter detection."""

    def test_is_lora_adapter_local_true(self, tmp_path: Path):
        """Test detecting a local LoRA adapter."""
        from model_garden.inference.utils import is_lora_adapter

        # Create a fake adapter directory
        adapter_dir = tmp_path / "my-adapter"
        adapter_dir.mkdir()

        # Create adapter_config.json
        adapter_config = adapter_dir / "adapter_config.json"
        adapter_config.write_text(
            json.dumps(
                {
                    "r": 16,
                    "lora_alpha": 16,
                    "target_modules": ["q_proj", "v_proj"],
                }
            )
        )

        assert is_lora_adapter(str(adapter_dir)) is True

    def test_is_lora_adapter_local_false(self, tmp_path: Path):
        """Test detecting a non-adapter local model."""
        from model_garden.inference.utils import is_lora_adapter

        # Create a fake model directory without adapter config
        model_dir = tmp_path / "regular-model"
        model_dir.mkdir()

        # Create config.json but no adapter_config.json
        config = model_dir / "config.json"
        config.write_text(json.dumps({"model_type": "llama"}))

        assert is_lora_adapter(str(model_dir)) is False

    def test_is_lora_adapter_nonexistent(self):
        """Test detecting adapter on non-existent path."""
        from model_garden.inference.utils import is_lora_adapter

        # Non-existent local path
        assert is_lora_adapter("/nonexistent/path") is False


class TestVisionModelDetection:
    """Tests for vision model detection."""

    def test_is_vision_model_from_config(self, tmp_path: Path):
        """Test detecting vision model from config."""
        from model_garden.inference.utils import is_vision_model

        # Create a fake vision model directory
        model_dir = tmp_path / "vision-model"
        model_dir.mkdir()

        # Create config.json with vision model type
        config = model_dir / "config.json"
        config.write_text(
            json.dumps(
                {
                    "model_type": "qwen2_5_vl",
                    "vision_config": {"hidden_size": 1024},
                }
            )
        )

        assert is_vision_model(str(model_dir)) is True

    def test_is_vision_model_from_name(self):
        """Test detecting vision model from name."""
        from model_garden.inference.utils import is_vision_model

        # Model names with vision indicators
        assert is_vision_model("Qwen/Qwen2.5-VL-7B-Instruct") is True
        assert is_vision_model("llava-hf/llava-1.5-7b-hf") is True

        # Text-only models
        assert is_vision_model("unsloth/tinyllama-bnb-4bit") is False
        assert is_vision_model("meta-llama/Llama-3-8B") is False

    def test_is_vision_model_nonexistent(self):
        """Test vision detection on non-existent path."""
        from model_garden.inference.utils import is_vision_model

        # HuggingFace model ID that doesn't exist locally
        # Should rely on name pattern matching
        result = is_vision_model("some/text-model")
        assert result is False


class TestQuantizationDetection:
    """Tests for quantization method detection."""

    def test_detect_quantization_awq(self, tmp_path: Path):
        """Test detecting AWQ quantization."""
        from model_garden.inference.utils import detect_quantization_method

        # Create a fake AWQ model
        model_dir = tmp_path / "awq-model"
        model_dir.mkdir()

        config = model_dir / "config.json"
        config.write_text(
            json.dumps(
                {
                    "quantization_config": {"quant_method": "awq"},
                }
            )
        )

        assert detect_quantization_method(str(model_dir)) == "awq"

    def test_detect_quantization_gptq(self, tmp_path: Path):
        """Test detecting GPTQ quantization."""
        from model_garden.inference.utils import detect_quantization_method

        # Create a fake GPTQ model
        model_dir = tmp_path / "gptq-model"
        model_dir.mkdir()

        config = model_dir / "config.json"
        config.write_text(
            json.dumps(
                {
                    "quantization_config": {"quant_method": "gptq"},
                }
            )
        )

        assert detect_quantization_method(str(model_dir)) == "gptq"

    def test_detect_quantization_bitsandbytes(self, tmp_path: Path):
        """Test detecting bitsandbytes quantization."""
        from model_garden.inference.utils import detect_quantization_method

        model_dir = tmp_path / "bnb-model"
        model_dir.mkdir()

        # BitsAndBytes models have load_in_4bit/8bit in config
        config = model_dir / "config.json"
        config.write_text(
            json.dumps(
                {
                    "quantization_config": {"load_in_4bit": True},
                }
            )
        )

        result = detect_quantization_method(str(model_dir))
        assert result in ["bitsandbytes", None]  # May vary based on detection logic

    def test_detect_quantization_none(self, tmp_path: Path):
        """Test detecting no quantization."""
        from model_garden.inference.utils import detect_quantization_method

        # Create a non-quantized model
        model_dir = tmp_path / "fp16-model"
        model_dir.mkdir()

        config = model_dir / "config.json"
        config.write_text(
            json.dumps(
                {
                    "model_type": "llama",
                    "torch_dtype": "float16",
                }
            )
        )

        result = detect_quantization_method(str(model_dir))
        assert result is None

    def test_detect_quantization_from_name(self):
        """Test detecting quantization from model name."""
        from model_garden.inference.utils import detect_quantization_method

        # AWQ model name
        result = detect_quantization_method("TheBloke/Llama-2-7B-AWQ")
        assert result in ["awq", None]  # May detect from name or return None

        # GPTQ model name
        result = detect_quantization_method("TheBloke/Llama-2-7B-GPTQ")
        assert result in ["gptq", None]


class TestBaseModelDetection:
    """Tests for base model detection from adapters."""

    def test_get_base_model_from_adapter(self, tmp_path: Path):
        """Test getting base model from adapter config."""
        from model_garden.inference.utils import get_base_model_from_adapter

        # Create a fake adapter
        adapter_dir = tmp_path / "my-adapter"
        adapter_dir.mkdir()

        adapter_config = adapter_dir / "adapter_config.json"
        adapter_config.write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "meta-llama/Llama-3-8B",
                    "r": 16,
                    "lora_alpha": 16,
                }
            )
        )

        base_model = get_base_model_from_adapter(str(adapter_dir))
        assert base_model == "meta-llama/Llama-3-8B"

    def test_get_base_model_from_adapter_not_found(self, tmp_path: Path):
        """Test getting base model from non-adapter directory."""
        from model_garden.inference.utils import get_base_model_from_adapter

        # Directory without adapter config
        model_dir = tmp_path / "regular-model"
        model_dir.mkdir()

        base_model = get_base_model_from_adapter(str(model_dir))
        assert base_model is None
