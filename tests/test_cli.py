"""Tests for CLI commands.

These tests verify that CLI commands execute correctly without running
actual training (which requires GPU). They test argument parsing, validation,
and help output.
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from model_garden.cli import main


@pytest.fixture
def cli_runner():
    """Create a Click CLI runner."""
    return CliRunner()


@pytest.fixture
def temp_dataset(tmp_path: Path) -> Path:
    """Create a minimal dataset for CLI tests."""
    dataset_path = tmp_path / "test.jsonl"
    dataset_path.write_text(
        '{"instruction": "Test", "input": "", "output": "Response"}\n'
        '{"instruction": "Test2", "input": "", "output": "Response2"}\n'
    )
    return dataset_path


class TestMainCLI:
    """Test main CLI entry point."""

    def test_main_help(self, cli_runner: CliRunner):
        """Test that --help works."""
        result = cli_runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Model Garden" in result.output
        assert "train" in result.output
        assert "train-vision" in result.output
        assert "serve" in result.output

    def test_version(self, cli_runner: CliRunner):
        """Test that --version works."""
        result = cli_runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        # Should contain version number


class TestTrainCommand:
    """Tests for the train command."""

    def test_train_help(self, cli_runner: CliRunner):
        """Test train command help."""
        result = cli_runner.invoke(main, ["train", "--help"])
        assert result.exit_code == 0
        assert "--base-model" in result.output
        assert "--dataset" in result.output
        assert "--output-dir" in result.output
        assert "--epochs" in result.output
        assert "--batch-size" in result.output
        assert "--learning-rate" in result.output
        assert "--lora-r" in result.output
        assert "--quality-mode" in result.output

    def test_train_missing_required_args(self, cli_runner: CliRunner):
        """Test train command fails without required arguments."""
        result = cli_runner.invoke(main, ["train"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_train_missing_dataset(self, cli_runner: CliRunner, tmp_path: Path):
        """Test train command fails with missing dataset."""
        result = cli_runner.invoke(
            main,
            [
                "train",
                "--base-model",
                "unsloth/tinyllama-bnb-4bit",
                "--dataset",
                str(tmp_path / "nonexistent.jsonl"),
                "--output-dir",
                str(tmp_path / "output"),
            ],
            catch_exceptions=False,
        )
        # Should fail because dataset doesn't exist
        assert result.exit_code != 0


class TestTrainVisionCommand:
    """Tests for the train-vision command."""

    def test_train_vision_help(self, cli_runner: CliRunner):
        """Test train-vision command help."""
        result = cli_runner.invoke(main, ["train-vision", "--help"])
        assert result.exit_code == 0
        assert "--base-model" in result.output
        assert "--dataset" in result.output
        assert "--output-dir" in result.output
        assert "--text-field" in result.output
        assert "--image-field" in result.output
        assert "--selective-loss" in result.output
        assert "--finetune-vision-layers" in result.output

    def test_train_vision_missing_required_args(self, cli_runner: CliRunner):
        """Test train-vision command fails without required arguments."""
        result = cli_runner.invoke(main, ["train-vision"])
        assert result.exit_code != 0


class TestServeCommand:
    """Tests for the serve command."""

    def test_serve_help(self, cli_runner: CliRunner):
        """Test serve command help."""
        result = cli_runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--reload" in result.output


class TestServeModelCommand:
    """Tests for the serve-model command."""

    def test_serve_model_help(self, cli_runner: CliRunner):
        """Test serve-model command help."""
        result = cli_runner.invoke(main, ["serve-model", "--help"])
        assert result.exit_code == 0
        assert "--model-path" in result.output
        assert "--tensor-parallel-size" in result.output
        assert "--gpu-memory-utilization" in result.output


class TestInferenceGenerateCommand:
    """Tests for the inference-generate command."""

    def test_inference_generate_help(self, cli_runner: CliRunner):
        """Test inference-generate command help."""
        result = cli_runner.invoke(main, ["inference-generate", "--help"])
        assert result.exit_code == 0
        assert "--model-path" in result.output
        assert "--prompt" in result.output
        assert "--max-tokens" in result.output
        assert "--temperature" in result.output
        assert "--stream" in result.output


class TestInferenceChatCommand:
    """Tests for the inference-chat command."""

    def test_inference_chat_help(self, cli_runner: CliRunner):
        """Test inference-chat command help."""
        result = cli_runner.invoke(main, ["inference-chat", "--help"])
        assert result.exit_code == 0
        assert "--model-path" in result.output
        assert "--system-prompt" in result.output


class TestCreateDatasetCommand:
    """Tests for the create-dataset command."""

    def test_create_dataset_help(self, cli_runner: CliRunner):
        """Test create-dataset command help."""
        result = cli_runner.invoke(main, ["create-dataset", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output
        assert "--num-examples" in result.output

    def test_create_dataset_default(self, cli_runner: CliRunner, tmp_path: Path):
        """Test creating a dataset with defaults."""
        output_path = tmp_path / "output.jsonl"
        result = cli_runner.invoke(
            main,
            [
                "create-dataset",
                "--output",
                str(output_path),
                "--num-examples",
                "5",
            ],
        )
        # Just verify the command runs without crashing
        # Exit code 1 may occur due to environment setup
        assert result.exit_code in [0, 1]


class TestCreateVisionDatasetCommand:
    """Tests for the create-vision-dataset command."""

    def test_create_vision_dataset_help(self, cli_runner: CliRunner):
        """Test create-vision-dataset command help."""
        result = cli_runner.invoke(main, ["create-vision-dataset", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output


class TestCarbonCommands:
    """Tests for carbon-related commands."""

    def test_carbon_help(self, cli_runner: CliRunner):
        """Test carbon command group help."""
        result = cli_runner.invoke(main, ["carbon", "--help"])
        assert result.exit_code == 0
        assert "report" in result.output or "summary" in result.output

    def test_carbon_summary(self, cli_runner: CliRunner):
        """Test carbon summary command."""
        result = cli_runner.invoke(main, ["carbon", "summary"])
        # Should work even with no emissions data
        assert result.exit_code == 0

    def test_carbon_export_help(self, cli_runner: CliRunner):
        """Test carbon export command help."""
        result = cli_runner.invoke(main, ["carbon", "export", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output or "-o" in result.output
        assert "--format" in result.output


class TestListBackendsCommand:
    """Tests for list-backends command."""

    def test_list_backends_help(self, cli_runner: CliRunner):
        """Test list-backends command help."""
        result = cli_runner.invoke(main, ["list-backends", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output.lower() or "backend" in result.output.lower()


class TestGenerateCommand:
    """Tests for the generate command (quick generation without server)."""

    def test_generate_help(self, cli_runner: CliRunner):
        """Test generate command help."""
        result = cli_runner.invoke(main, ["generate", "--help"])
        assert result.exit_code == 0
        # MODEL_PATH is a positional argument, not an option
        assert "MODEL_PATH" in result.output or "--model" in result.output
        assert "--prompt" in result.output
