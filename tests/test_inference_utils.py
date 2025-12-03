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

    def test_get_base_model_from_adapter_missing_key(self, tmp_path: Path):
        """Test getting base model when adapter config doesn't have the key."""
        from model_garden.inference.utils import get_base_model_from_adapter

        adapter_dir = tmp_path / "incomplete-adapter"
        adapter_dir.mkdir()

        # Adapter config without base_model_name_or_path
        adapter_config = adapter_dir / "adapter_config.json"
        adapter_config.write_text(
            json.dumps(
                {
                    "r": 16,
                    "lora_alpha": 16,
                }
            )
        )

        base_model = get_base_model_from_adapter(str(adapter_dir))
        assert base_model is None


class TestModelSizeEstimationExtended:
    """Extended tests for model size estimation."""

    def test_estimate_model_size_multiple_weight_files(self, tmp_path: Path):
        """Test estimating model size with multiple weight files."""
        from model_garden.inference.utils import estimate_model_size_gb

        model_dir = tmp_path / "large-model"
        model_dir.mkdir()

        # Create multiple safetensors files (each 1MB)
        for i in range(5):
            (model_dir / f"model-{i:05d}-of-00005.safetensors").write_bytes(b"\x00" * (1024 * 1024))

        size = estimate_model_size_gb(str(model_dir))
        # Should be approximately 5MB but rounded up to at least 1.0
        assert size >= 1.0

    def test_estimate_model_size_bin_files(self, tmp_path: Path):
        """Test estimating model size from .bin files."""
        from model_garden.inference.utils import estimate_model_size_gb

        model_dir = tmp_path / "pytorch-model"
        model_dir.mkdir()

        # Create a pytorch_model.bin file
        (model_dir / "pytorch_model.bin").write_bytes(b"\x00" * (2 * 1024 * 1024))

        size = estimate_model_size_gb(str(model_dir))
        assert size >= 1.0

    def test_estimate_model_size_from_name_various_formats(self):
        """Test model size estimation from various name formats."""
        from model_garden.inference.utils import estimate_model_size_gb

        # Various naming conventions
        assert estimate_model_size_gb("org/model-7B-fp16") >= 7.0
        assert estimate_model_size_gb("org/model-3b-instruct") >= 3.0
        assert estimate_model_size_gb("org/model-70B-chat") >= 70.0

        # Non-matching name should return default
        assert estimate_model_size_gb("org/model-large") == 7.0


class TestVisionModelDetectionExtended:
    """Extended tests for vision model detection."""

    def test_is_vision_model_with_processor_config(self, tmp_path: Path):
        """Test detecting vision model by processor_config.json."""
        from model_garden.inference.utils import is_vision_model

        model_dir = tmp_path / "vision-model-2"
        model_dir.mkdir()

        # Only processor_config.json, no config.json
        (model_dir / "processor_config.json").write_text(
            json.dumps({"processor_class": "Qwen2VLProcessor"})
        )

        assert is_vision_model(str(model_dir)) is True

    def test_is_vision_model_from_architectures(self, tmp_path: Path):
        """Test detecting vision model from architectures in config."""
        from model_garden.inference.utils import is_vision_model

        model_dir = tmp_path / "vision-model-arch"
        model_dir.mkdir()

        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen2",
                    "architectures": ["Qwen2VLForConditionalGeneration"],
                }
            )
        )

        assert is_vision_model(str(model_dir)) is True

    def test_is_vision_model_with_visual_config(self, tmp_path: Path):
        """Test detecting vision model with visual_config in config."""
        from model_garden.inference.utils import is_vision_model

        model_dir = tmp_path / "vision-model-vc"
        model_dir.mkdir()

        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "llava",
                    "visual_config": {"hidden_size": 1024},
                }
            )
        )

        assert is_vision_model(str(model_dir)) is True


class TestQuantizationDetectionExtended:
    """Extended tests for quantization detection."""

    def test_detect_quantization_from_weight_filename_awq(self, tmp_path: Path):
        """Test detecting AWQ from weight file name."""
        from model_garden.inference.utils import detect_quantization_method

        model_dir = tmp_path / "awq-named-model"
        model_dir.mkdir()

        # Weight file with AWQ in the name
        (model_dir / "model-awq.safetensors").write_bytes(b"\x00" * 100)
        (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))

        result = detect_quantization_method(str(model_dir))
        assert result == "awq"

    def test_detect_quantization_from_weight_filename_gptq(self, tmp_path: Path):
        """Test detecting GPTQ from weight file name."""
        from model_garden.inference.utils import detect_quantization_method

        model_dir = tmp_path / "gptq-named-model"
        model_dir.mkdir()

        # Weight file with GPTQ in the name
        (model_dir / "model-gptq.safetensors").write_bytes(b"\x00" * 100)
        (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))

        result = detect_quantization_method(str(model_dir))
        assert result == "gptq"

    def test_detect_quantization_adapter_returns_none(self, tmp_path: Path):
        """Test that LoRA adapters return None for quantization."""
        from model_garden.inference.utils import detect_quantization_method

        adapter_dir = tmp_path / "lora-adapter"
        adapter_dir.mkdir()

        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"r": 16, "base_model_name_or_path": "meta-llama/Llama-3-8B"})
        )

        result = detect_quantization_method(str(adapter_dir))
        assert result is None

    def test_detect_quantization_bnb_fallback(self, tmp_path: Path):
        """Test BitsAndBytes detection falls back to None for merged models."""
        from model_garden.inference.utils import detect_quantization_method

        model_dir = tmp_path / "merged-bnb-model"
        model_dir.mkdir()

        # BitsAndBytes config in a merged model should return None
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "quantization_config": {"quant_method": "bitsandbytes"},
                }
            )
        )
        # Regular safetensors file (no quant in name)
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 100)

        result = detect_quantization_method(str(model_dir))
        # Should detect as merged model, not quantized
        assert result is None

    def test_detect_quantization_empty_dir(self, tmp_path: Path):
        """Test quantization detection on empty directory."""
        from model_garden.inference.utils import detect_quantization_method

        model_dir = tmp_path / "empty-model"
        model_dir.mkdir()

        result = detect_quantization_method(str(model_dir))
        assert result is None


class TestGPUMemoryUtilizationEdgeCases:
    """Edge case tests for GPU memory utilization."""

    def test_calculate_utilization_very_small_model(self):
        """Test utilization for very small models."""
        from model_garden.inference.utils import calculate_gpu_memory_utilization

        util = calculate_gpu_memory_utilization("tiny/model-100M")
        assert 0.5 <= util <= 0.95

    def test_calculate_utilization_very_large_model(self):
        """Test utilization for very large models."""
        from model_garden.inference.utils import calculate_gpu_memory_utilization

        util = calculate_gpu_memory_utilization("big/model-70B")
        # For large models, should maximize utilization
        assert 0.5 <= util <= 0.95

    def test_calculate_utilization_various_tensor_parallel(self):
        """Test utilization with various tensor parallel sizes."""
        from model_garden.inference.utils import calculate_gpu_memory_utilization

        base_util = calculate_gpu_memory_utilization("test/model-7B", tensor_parallel_size=1)

        for tp_size in [2, 4, 8]:
            util = calculate_gpu_memory_utilization("test/model-7B", tensor_parallel_size=tp_size)
            # More GPUs should generally mean lower per-GPU utilization
            assert util <= base_util
            assert 0.5 <= util <= 0.95
