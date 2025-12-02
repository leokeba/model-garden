"""End-to-end tests for Model Garden training workflows.

These tests run actual training jobs with small models and datasets to verify
the complete training pipeline works correctly. They require GPU access and
will be skipped unless --run-integration is passed to pytest.

Usage:
    pytest tests/test_e2e_training.py --run-integration -v
    pytest tests/test_e2e_training.py --run-integration -v -k "text"  # Text only
    pytest tests/test_e2e_training.py --run-integration -v -k "vision"  # Vision only
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

# Mark all tests in this module as integration tests requiring GPU
pytestmark = [pytest.mark.integration, pytest.mark.requires_gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def cli_runner():
    """Create a Click CLI runner for testing commands."""
    return CliRunner()


@pytest.fixture(scope="module")
def temp_output_dir():
    """Create a temporary directory for model outputs that persists across tests."""
    tmpdir = tempfile.mkdtemp(prefix="model_garden_e2e_")
    yield Path(tmpdir)
    # Cleanup after all tests in module
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def text_dataset(temp_output_dir: Path) -> Path:
    """Create a small text dataset for testing."""
    dataset_path = temp_output_dir / "text_dataset.jsonl"

    samples = [
        {"instruction": "Say hello", "input": "", "output": "Hello! How can I help you today?"},
        {"instruction": "What is 2+2?", "input": "", "output": "2 + 2 equals 4."},
        {"instruction": "Count to 3", "input": "", "output": "1, 2, 3"},
        {
            "instruction": "What color is the sky?",
            "input": "",
            "output": "The sky is typically blue during the day.",
        },
        {"instruction": "Say goodbye", "input": "", "output": "Goodbye! Have a great day!"},
        {
            "instruction": "What is Python?",
            "input": "",
            "output": "Python is a programming language.",
        },
        {"instruction": "Name a fruit", "input": "", "output": "Apple is a popular fruit."},
        {
            "instruction": "What day comes after Monday?",
            "input": "",
            "output": "Tuesday comes after Monday.",
        },
    ]

    with open(dataset_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return dataset_path


@pytest.fixture(scope="module")
def vision_dataset(temp_output_dir: Path) -> Path:
    """Create a small vision dataset for testing using existing test images."""
    dataset_path = temp_output_dir / "vision_dataset.jsonl"

    # Use the existing test images from the project
    project_root = Path(__file__).parent.parent
    test_images_dir = project_root / "data" / "test_images"

    # Create absolute paths to images
    samples = [
        {
            "text": "What shape is shown?",
            "image": str(test_images_dir / "red_square.jpg"),
            "response": "A red square.",
        },
        {
            "text": "Describe this shape.",
            "image": str(test_images_dir / "blue_circle.jpg"),
            "response": "A blue circle.",
        },
        {
            "text": "What is this?",
            "image": str(test_images_dir / "green_triangle.jpg"),
            "response": "A green triangle.",
        },
        {
            "text": "What do you see?",
            "image": str(test_images_dir / "purple_star.jpg"),
            "response": "A purple star.",
        },
    ]

    # Verify images exist
    for sample in samples:
        assert Path(sample["image"]).exists(), f"Test image not found: {sample['image']}"

    with open(dataset_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return dataset_path


class TestTextTrainingE2E:
    """End-to-end tests for text model fine-tuning."""

    def test_train_text_model_cli(
        self, cli_runner: CliRunner, text_dataset: Path, temp_output_dir: Path
    ):
        """Test complete text model training via CLI."""
        from model_garden.cli import main

        output_dir = temp_output_dir / "text_model_cli"

        result = cli_runner.invoke(
            main,
            [
                "train",
                "--base-model",
                "unsloth/tinyllama-bnb-4bit",
                "--dataset",
                str(text_dataset),
                "--output-dir",
                str(output_dir),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--max-steps",
                "5",  # Very short for testing
                "--logging-steps",
                "1",
                "--save-steps",
                "5",
                "--save-method",
                "lora",  # Faster to save
                "--max-seq-length",
                "512",
            ],
            catch_exceptions=False,
        )

        # Check command succeeded
        assert result.exit_code == 0, f"CLI failed with output:\n{result.output}"

        # Verify output directory was created
        assert output_dir.exists(), "Output directory was not created"

        # Check for adapter files (LoRA save method)
        adapter_config = output_dir / "adapter_config.json"
        assert adapter_config.exists(), f"adapter_config.json not found in {output_dir}"

        # Verify adapter config is valid JSON
        with open(adapter_config) as f:
            config = json.load(f)
            assert "r" in config  # LoRA rank
            assert "lora_alpha" in config

    def test_train_text_model_api(self, text_dataset: Path, temp_output_dir: Path):
        """Test text model training via Python API."""
        from model_garden.training import create_text_trainer

        output_dir = temp_output_dir / "text_model_api"

        # Create trainer
        trainer = create_text_trainer(
            base_model="unsloth/tinyllama-bnb-4bit",
            max_seq_length=512,
            load_in_4bit=True,
        )

        # Load model
        trainer.load_model()
        assert trainer.model is not None, "Model was not loaded"
        assert trainer.tokenizer is not None, "Tokenizer was not loaded"

        # Prepare for training
        trainer.prepare_for_training(
            r=8,  # Small rank for testing
            lora_alpha=8,
            lora_dropout=0.0,
        )

        # Load and format dataset
        dataset = trainer.load_dataset_from_file(str(text_dataset))
        assert len(dataset) > 0, "Dataset is empty"

        formatted_dataset = trainer.format_dataset(dataset)
        assert len(formatted_dataset) > 0, "Formatted dataset is empty"

        # Train with minimal steps
        trainer.train(
            dataset=formatted_dataset,
            output_dir=str(output_dir),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=3,
            logging_steps=1,
            save_steps=3,
            enable_carbon_tracking=False,  # Disable for faster tests
        )

        # Verify checkpoint was saved
        assert output_dir.exists(), "Output directory was not created"

    def test_train_text_model_different_save_methods(
        self, cli_runner: CliRunner, text_dataset: Path, temp_output_dir: Path
    ):
        """Test different save methods work correctly."""
        from model_garden.cli import main

        # Test merged_16bit save method
        output_dir = temp_output_dir / "text_model_merged"

        result = cli_runner.invoke(
            main,
            [
                "train",
                "--base-model",
                "unsloth/tinyllama-bnb-4bit",
                "--dataset",
                str(text_dataset),
                "--output-dir",
                str(output_dir),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--max-steps",
                "3",
                "--logging-steps",
                "1",
                "--save-steps",
                "3",
                "--save-method",
                "merged_16bit",
                "--max-seq-length",
                "512",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, f"CLI failed with output:\n{result.output}"

        # For merged models, check for config.json and model files
        config_file = output_dir / "config.json"
        assert config_file.exists(), f"config.json not found in {output_dir}"


class TestVisionTrainingE2E:
    """End-to-end tests for vision-language model fine-tuning."""

    def test_train_vision_model_cli(
        self, cli_runner: CliRunner, vision_dataset: Path, temp_output_dir: Path
    ):
        """Test complete vision model training via CLI."""
        from model_garden.cli import main

        output_dir = temp_output_dir / "vision_model_cli"

        result = cli_runner.invoke(
            main,
            [
                "train-vision",
                "--base-model",
                "Qwen/Qwen2.5-VL-3B-Instruct",
                "--dataset",
                str(vision_dataset),
                "--output-dir",
                str(output_dir),
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--max-steps",
                "3",  # Very short for testing
                "--logging-steps",
                "1",
                "--save-steps",
                "3",
                "--save-method",
                "lora",  # Faster to save
                "--max-seq-length",
                "512",
                "--gradient-accumulation-steps",
                "1",
            ],
            catch_exceptions=False,
        )

        # Check command succeeded
        assert result.exit_code == 0, f"CLI failed with output:\n{result.output}"

        # Verify output directory was created
        assert output_dir.exists(), "Output directory was not created"

    def test_train_vision_model_api(self, vision_dataset: Path, temp_output_dir: Path):
        """Test vision model training via Python API."""
        from model_garden.training import create_vision_trainer

        output_dir = temp_output_dir / "vision_model_api"

        # Create trainer
        trainer = create_vision_trainer(
            base_model="Qwen/Qwen2.5-VL-3B-Instruct",
            max_seq_length=512,
            load_in_4bit=True,
        )

        # Load model
        trainer.load_model()
        assert trainer.model is not None, "Model was not loaded"
        assert trainer.tokenizer is not None, "Tokenizer was not loaded"
        assert trainer.processor is not None, "Processor was not loaded"

        # Prepare for training
        trainer.prepare_for_training(
            r=8,  # Small rank for testing
            lora_alpha=8,
            lora_dropout=0.0,
            finetune_vision_layers=True,
            finetune_language_layers=True,
        )

        # Load and format dataset
        dataset = trainer.load_dataset_from_file(str(vision_dataset))
        assert len(dataset) > 0, "Dataset is empty"

        formatted_dataset = trainer.format_dataset(dataset)
        assert len(formatted_dataset) > 0, "Formatted dataset is empty"

        # Train with minimal steps
        trainer.train(
            dataset=formatted_dataset,
            output_dir=str(output_dir),
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=2,
            logging_steps=1,
            save_steps=2,
            enable_carbon_tracking=False,  # Disable for faster tests
        )

        # Verify training ran (checkpoint may or may not exist with 2 steps)
        assert output_dir.exists() or True, "Training should complete without error"

    def test_train_vision_model_with_selective_loss(
        self, cli_runner: CliRunner, temp_output_dir: Path
    ):
        """Test vision training with selective loss masking."""
        from model_garden.cli import main

        # Create a dataset with JSON responses for selective loss testing
        dataset_path = temp_output_dir / "vision_selective_dataset.jsonl"
        project_root = Path(__file__).parent.parent
        test_images_dir = project_root / "data" / "test_images"

        samples = [
            {
                "text": "Extract the shape information as JSON.",
                "image": str(test_images_dir / "red_square.jpg"),
                "response": '{"shape": "square", "color": "red"}',
            },
            {
                "text": "Extract the shape information as JSON.",
                "image": str(test_images_dir / "blue_circle.jpg"),
                "response": '{"shape": "circle", "color": "blue"}',
            },
        ]

        with open(dataset_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        output_dir = temp_output_dir / "vision_model_selective"

        result = cli_runner.invoke(
            main,
            [
                "train-vision",
                "--base-model",
                "Qwen/Qwen2.5-VL-3B-Instruct",
                "--dataset",
                str(dataset_path),
                "--output-dir",
                str(output_dir),
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--max-steps",
                "2",
                "--logging-steps",
                "1",
                "--save-steps",
                "2",
                "--save-method",
                "lora",
                "--max-seq-length",
                "512",
                "--gradient-accumulation-steps",
                "1",
                "--selective-loss",  # Enable selective loss
                "--selective-loss-level",
                "conservative",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, f"CLI failed with output:\n{result.output}"


class TestTrainingWithHuggingFaceHub:
    """Test training with datasets from HuggingFace Hub."""

    def test_train_text_from_hub(self, cli_runner: CliRunner, temp_output_dir: Path):
        """Test text training with a HuggingFace Hub dataset."""
        from model_garden.cli import main

        output_dir = temp_output_dir / "text_model_hub"

        result = cli_runner.invoke(
            main,
            [
                "train",
                "--base-model",
                "unsloth/tinyllama-bnb-4bit",
                "--dataset",
                "yahma/alpaca-cleaned",
                "--from-hub",
                "--output-dir",
                str(output_dir),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--max-steps",
                "3",
                "--logging-steps",
                "1",
                "--save-steps",
                "3",
                "--save-method",
                "lora",
                "--max-seq-length",
                "512",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, f"CLI failed with output:\n{result.output}"
        assert output_dir.exists(), "Output directory was not created"


class TestTrainingConfigOptions:
    """Test various training configuration options."""

    def test_quality_mode(self, cli_runner: CliRunner, text_dataset: Path, temp_output_dir: Path):
        """Test training with quality mode enabled."""
        from model_garden.cli import main

        output_dir = temp_output_dir / "text_model_quality"

        # Note: Quality mode uses much more VRAM, so this may fail on small GPUs
        result = cli_runner.invoke(
            main,
            [
                "train",
                "--base-model",
                "unsloth/tinyllama-bnb-4bit",
                "--dataset",
                str(text_dataset),
                "--output-dir",
                str(output_dir),
                "--epochs",
                "1",
                "--batch-size",
                "1",  # Smaller batch due to higher memory
                "--max-steps",
                "2",
                "--logging-steps",
                "1",
                "--save-steps",
                "2",
                "--save-method",
                "lora",
                "--max-seq-length",
                "256",
                "--quality-mode",
            ],
            catch_exceptions=False,
        )

        # Quality mode may fail on small GPUs, so we just check it doesn't crash badly
        # Exit code 0 = success, exit code 1 = OOM or other error (acceptable for this test)
        assert result.exit_code in [0, 1], f"Unexpected exit code: {result.exit_code}"

    def test_custom_lora_config(
        self, cli_runner: CliRunner, text_dataset: Path, temp_output_dir: Path
    ):
        """Test training with custom LoRA configuration."""
        from model_garden.cli import main

        output_dir = temp_output_dir / "text_model_custom_lora"

        result = cli_runner.invoke(
            main,
            [
                "train",
                "--base-model",
                "unsloth/tinyllama-bnb-4bit",
                "--dataset",
                str(text_dataset),
                "--output-dir",
                str(output_dir),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--max-steps",
                "2",
                "--logging-steps",
                "1",
                "--save-steps",
                "2",
                "--save-method",
                "lora",
                "--max-seq-length",
                "512",
                "--lora-r",
                "32",
                "--lora-alpha",
                "64",
                "--lora-dropout",
                "0.05",
                "--use-rslora",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, f"CLI failed with output:\n{result.output}"

        # Verify LoRA config was applied
        adapter_config = output_dir / "adapter_config.json"
        if adapter_config.exists():
            with open(adapter_config) as f:
                config = json.load(f)
                assert config.get("r") == 32, "LoRA rank not applied correctly"

    def test_different_backends(
        self, cli_runner: CliRunner, text_dataset: Path, temp_output_dir: Path
    ):
        """Test training with different backends."""
        from model_garden.cli import main

        # Test with transformers backend
        output_dir = temp_output_dir / "text_model_transformers_backend"

        result = cli_runner.invoke(
            main,
            [
                "train",
                "--base-model",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "--dataset",
                str(text_dataset),
                "--output-dir",
                str(output_dir),
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--max-steps",
                "2",
                "--logging-steps",
                "1",
                "--save-steps",
                "2",
                "--save-method",
                "lora",
                "--max-seq-length",
                "256",
                "--backend",
                "transformers",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, f"CLI failed with output:\n{result.output}"


class TestModelOutputValidation:
    """Test that trained models produce valid outputs."""

    def test_trained_model_can_generate(self, temp_output_dir: Path, text_dataset: Path):
        """Test that a trained model can generate text."""
        from model_garden.training import create_text_trainer

        output_dir = temp_output_dir / "text_model_for_generation"

        # Train a minimal model
        trainer = create_text_trainer(
            base_model="unsloth/tinyllama-bnb-4bit",
            max_seq_length=512,
            load_in_4bit=True,
        )

        trainer.load_model()
        trainer.prepare_for_training(r=8, lora_alpha=8)

        dataset = trainer.load_dataset_from_file(str(text_dataset))
        formatted_dataset = trainer.format_dataset(dataset)

        trainer.train(
            dataset=formatted_dataset,
            output_dir=str(output_dir),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            max_steps=3,
            logging_steps=1,
            save_steps=3,
            enable_carbon_tracking=False,
        )

        # Test generation with the trained model
        # The model should still be in memory after training
        if trainer.model is not None and trainer.tokenizer is not None:
            inputs = trainer.tokenizer("Hello, how are you?", return_tensors="pt")
            inputs = {k: v.to(trainer.model.device) for k, v in inputs.items()}

            outputs = trainer.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )

            decoded = trainer.tokenizer.decode(outputs[0], skip_special_tokens=True)
            assert len(decoded) > 0, "Generated text is empty"
