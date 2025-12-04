"""Tests for training/merge.py - LoRA adapter merging utilities.

These tests verify the LoRA adapter merging functionality without requiring
actual GPU operations by using mocks.

Note: These tests require real torch imports, so they're marked as requires_gpu
to bypass the mock_heavy_imports fixture in conftest.py.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# Mark all tests in this module to use real torch (not mocked)
pytestmark = pytest.mark.requires_gpu


class TestCleanupMemory:
    """Tests for _cleanup_memory function."""

    def test_cleanup_memory_with_cuda_available(self):
        """Test memory cleanup when CUDA is available."""
        from model_garden.training.merge import _cleanup_memory

        with patch("model_garden.training.merge.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True

            _cleanup_memory()

            mock_torch.cuda.empty_cache.assert_called_once()
            mock_torch.cuda.synchronize.assert_called_once()

    def test_cleanup_memory_without_cuda(self):
        """Test memory cleanup when CUDA is not available."""
        from model_garden.training.merge import _cleanup_memory

        with patch("model_garden.training.merge.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            _cleanup_memory()

            mock_torch.cuda.empty_cache.assert_not_called()
            mock_torch.cuda.synchronize.assert_not_called()


class TestMergeVisionLoraAdapter:
    """Tests for merge_vision_lora_adapter function."""

    @pytest.fixture
    def temp_adapter_dir(self, temp_dir: Path) -> Path:
        """Create a temporary adapter directory with config."""
        adapter_dir = temp_dir / "adapter"
        adapter_dir.mkdir()

        # Create adapter_config.json
        config = {"base_model_name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct"}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        return adapter_dir

    def test_raises_when_adapter_config_missing(self, temp_dir: Path):
        """Test that FileNotFoundError is raised when adapter_config.json is missing."""
        from model_garden.training.merge import merge_vision_lora_adapter

        adapter_dir = temp_dir / "empty_adapter"
        adapter_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="adapter_config.json not found"):
            merge_vision_lora_adapter(
                adapter_path=str(adapter_dir),
                output_dir=str(temp_dir / "output"),
                base_model=None,
            )

    def test_raises_when_base_model_not_in_config(self, temp_dir: Path):
        """Test that ValueError is raised when base model not in config."""
        from model_garden.training.merge import merge_vision_lora_adapter

        adapter_dir = temp_dir / "adapter"
        adapter_dir.mkdir()

        # Create config without base_model_name_or_path
        config = {"r": 16, "lora_alpha": 16}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        with pytest.raises(ValueError, match="Could not find base_model_name_or_path"):
            merge_vision_lora_adapter(
                adapter_path=str(adapter_dir),
                output_dir=str(temp_dir / "output"),
                base_model=None,
            )

    def test_creates_output_directory(self, temp_adapter_dir: Path, temp_dir: Path):
        """Test that output directory is created even if merge fails."""
        from model_garden.training.merge import merge_vision_lora_adapter

        output_dir = temp_dir / "nested" / "output" / "dir"
        assert not output_dir.exists()

        # Mock the model loading to fail after output dir creation
        with patch("model_garden.training.merge.AutoModelForVision2Seq") as mock_model:
            mock_model.from_pretrained.side_effect = Exception("Model load failed")

            with pytest.raises(Exception, match="Model load failed"):
                merge_vision_lora_adapter(
                    adapter_path=str(temp_adapter_dir),
                    output_dir=str(output_dir),
                    base_model="test/model",
                )

            # Output directory should have been created before the error
            assert output_dir.exists()

    def test_base_model_detection_from_local_config(self, temp_adapter_dir: Path, temp_dir: Path):
        """Test that base model is correctly read from adapter_config.json."""
        from model_garden.training.merge import merge_vision_lora_adapter

        # Mock to fail early but after reading the config
        with patch("model_garden.training.merge.AutoModelForVision2Seq") as mock_model:
            mock_model.from_pretrained.side_effect = Exception("Expected failure")

            with pytest.raises(Exception):
                merge_vision_lora_adapter(
                    adapter_path=str(temp_adapter_dir),
                    output_dir=str(temp_dir / "output"),
                    base_model=None,  # Should auto-detect
                )

            # Verify the auto-detected base model was used
            call_args = mock_model.from_pretrained.call_args
            assert call_args[0][0] == "Qwen/Qwen2.5-VL-3B-Instruct"

    def test_explicit_base_model_overrides_config(self, temp_adapter_dir: Path, temp_dir: Path):
        """Test that explicitly provided base_model overrides config."""
        from model_garden.training.merge import merge_vision_lora_adapter

        with patch("model_garden.training.merge.AutoModelForVision2Seq") as mock_model:
            mock_model.from_pretrained.side_effect = Exception("Expected failure")

            with pytest.raises(Exception):
                merge_vision_lora_adapter(
                    adapter_path=str(temp_adapter_dir),
                    output_dir=str(temp_dir / "output"),
                    base_model="custom/base-model",  # Override config
                )

            # Verify custom base model was used
            call_args = mock_model.from_pretrained.call_args
            assert call_args[0][0] == "custom/base-model"


class TestMergeTextLoraAdapter:
    """Tests for merge_text_lora_adapter function."""

    @pytest.fixture
    def temp_text_adapter_dir(self, temp_dir: Path) -> Path:
        """Create a temporary adapter directory for text models."""
        adapter_dir = temp_dir / "text_adapter"
        adapter_dir.mkdir()

        config = {"base_model_name_or_path": "unsloth/llama-3.2-3b-bnb-4bit"}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

        return adapter_dir

    def test_raises_when_adapter_config_missing(self, temp_dir: Path):
        """Test FileNotFoundError when adapter_config.json is missing."""
        from model_garden.training.merge import merge_text_lora_adapter

        adapter_dir = temp_dir / "empty_adapter"
        adapter_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="adapter_config.json not found"):
            merge_text_lora_adapter(
                adapter_path=str(adapter_dir),
                output_dir=str(temp_dir / "output"),
                base_model=None,
            )

    def test_base_model_detection_from_local_config(
        self, temp_text_adapter_dir: Path, temp_dir: Path
    ):
        """Test that base model is correctly read from adapter_config.json."""
        from model_garden.training.merge import merge_text_lora_adapter

        with patch("transformers.AutoModelForCausalLM") as mock_model:
            mock_model.from_pretrained.side_effect = Exception("Expected failure")

            with pytest.raises(Exception):
                merge_text_lora_adapter(
                    adapter_path=str(temp_text_adapter_dir),
                    output_dir=str(temp_dir / "output"),
                    base_model=None,
                )

            # Verify base model was detected
            call_args = mock_model.from_pretrained.call_args
            assert call_args[0][0] == "unsloth/llama-3.2-3b-bnb-4bit"

    def test_load_in_4bit_option(self, temp_text_adapter_dir: Path, temp_dir: Path):
        """Test that load_in_4bit is passed correctly."""
        from model_garden.training.merge import merge_text_lora_adapter

        with patch("transformers.AutoModelForCausalLM") as mock_model:
            mock_model.from_pretrained.side_effect = Exception("Expected failure")

            with pytest.raises(Exception):
                merge_text_lora_adapter(
                    adapter_path=str(temp_text_adapter_dir),
                    output_dir=str(temp_dir / "output"),
                    base_model="test/model",
                    load_in_4bit=True,
                )

            call_kwargs = mock_model.from_pretrained.call_args[1]
            assert call_kwargs["load_in_4bit"] is True


class TestHuggingFaceHubAdapters:
    """Tests for loading adapters from HuggingFace Hub."""

    def test_vision_adapter_from_hub_config_detection(self, temp_dir: Path):
        """Test loading vision adapter config from HuggingFace Hub."""
        from model_garden.training.merge import merge_vision_lora_adapter

        # Create temp config file that hf_hub_download will "return"
        config_path = temp_dir / "adapter_config.json"
        config = {"base_model_name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct"}
        config_path.write_text(json.dumps(config))

        with (
            patch("model_garden.training.merge.hf_hub_download") as mock_download,
            patch("model_garden.training.merge.AutoModelForVision2Seq") as mock_model,
            patch("model_garden.training.merge.get_hf_token") as mock_token,
        ):
            mock_token.return_value = "test_token"
            mock_download.return_value = str(config_path)
            mock_model.from_pretrained.side_effect = Exception("Expected failure")

            with pytest.raises(Exception):
                merge_vision_lora_adapter(
                    adapter_path="user/vision-adapter",  # Hub ID, not local path
                    output_dir=str(temp_dir / "output"),
                    base_model=None,
                )

            # Verify hf_hub_download was called with correct arguments
            mock_download.assert_called_once()
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs["repo_id"] == "user/vision-adapter"
            assert call_kwargs["filename"] == "adapter_config.json"

    def test_text_adapter_from_hub_config_detection(self, temp_dir: Path):
        """Test loading text adapter config from HuggingFace Hub."""
        from model_garden.training.merge import merge_text_lora_adapter

        config_path = temp_dir / "adapter_config.json"
        config = {"base_model_name_or_path": "unsloth/llama-3.2-3b"}
        config_path.write_text(json.dumps(config))

        with (
            patch("model_garden.training.merge.hf_hub_download") as mock_download,
            patch("transformers.AutoModelForCausalLM") as mock_model,
            patch("model_garden.training.merge.get_hf_token") as mock_token,
        ):
            mock_token.return_value = "test_token"
            mock_download.return_value = str(config_path)
            mock_model.from_pretrained.side_effect = Exception("Expected failure")

            with pytest.raises(Exception):
                merge_text_lora_adapter(
                    adapter_path="user/text-adapter",
                    output_dir=str(temp_dir / "output"),
                    base_model=None,
                )

            mock_download.assert_called_once()
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs["repo_id"] == "user/text-adapter"
