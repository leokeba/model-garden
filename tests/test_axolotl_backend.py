"""Unit tests for the Axolotl training backend.

These tests validate config translation and dataset materialization without
requiring the Axolotl package to be installed.
"""

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from model_garden.training.backends import axolotl_backend as axb
from model_garden.training.config import TrainingConfig, VisionTrainingConfig


@pytest.fixture(autouse=True)
def _patch_require_axolotl(monkeypatch):
    """Skip the real dependency check during tests."""

    monkeypatch.setattr(axb, "require_axolotl", lambda *_args, **_kwargs: None)


def test_common_precision_flags():
    # 4-bit prefers bf16
    four_bit = axb._build_common_precision(load_in_4bit=True, load_in_8bit=False, dtype=None)
    assert four_bit["load_in_4bit"] is True
    assert four_bit["bf16"] is True
    assert four_bit["fp16"] is False

    # 8-bit disables fp16/bf16 flags
    eight_bit = axb._build_common_precision(load_in_4bit=False, load_in_8bit=True, dtype=None)
    assert eight_bit["load_in_8bit"] is True
    assert eight_bit["bf16"] is False
    assert eight_bit["fp16"] is False

    # 16-bit bf16 vs fp16
    bf16 = axb._build_common_precision(load_in_4bit=False, load_in_8bit=False, dtype="bf16")
    assert bf16["bf16"] is True and bf16["fp16"] is False

    fp16 = axb._build_common_precision(load_in_4bit=False, load_in_8bit=False, dtype="fp16")
    assert fp16["bf16"] is False and fp16["fp16"] is True


def test_text_train_materializes_and_builds_config(monkeypatch, tmp_path: Path):
    trainer = axb.AxolotlTextTrainer(
        base_model="test/model",
        load_in_4bit=False,
        load_in_8bit=False,
        dtype="bf16",
    )
    trainer.prepare_for_training(r=8, lora_alpha=32, use_gradient_checkpointing="off")

    captured: dict[str, Any] = {}

    def fake_run(cfg: dict, job_id: str | None, work_dir: Path | None):
        captured["cfg"] = cfg
        captured["work_dir"] = work_dir
        dataset_path = Path(cfg["datasets"][0]["path"])
        captured["dataset_lines"] = dataset_path.read_text().strip().splitlines()

    monkeypatch.setattr(axb, "_run_axolotl", fake_run)

    raw_dataset = [
        {"instruction": "Hi", "input": "x", "output": "y"},
        {"instruction": "Bye", "input": "", "output": "z"},
    ]
    formatted = trainer.format_dataset(raw_dataset)

    config = TrainingConfig(output_dir=str(tmp_path / "out"), batch_size=1, num_epochs=1)
    # Selective loss (text) – ensure it is carried into the config
    config.selective_loss = True  # type: ignore[attr-defined]
    config.selective_loss_level = "aggressive"  # type: ignore[attr-defined]

    eval_raw = [{"instruction": "Eval", "input": "", "output": "ok"}]
    eval_formatted = trainer.format_dataset(eval_raw)

    trainer.train(formatted, config, job_id="job-text", eval_dataset=eval_formatted)

    assert "cfg" in captured and "work_dir" in captured

    lines = captured["dataset_lines"]
    assert json.loads(lines[0])["instruction"] == "Hi"

    datasets_cfg = captured["cfg"]["datasets"]
    assert len(datasets_cfg) == 2
    assert datasets_cfg[1]["type"] == "validation"

    assert captured["cfg"]["selective_loss"]["enabled"] is True

    # Precision propagated
    assert captured["cfg"]["bf16"] is True
    assert captured["cfg"]["fp16"] is False
    assert captured["cfg"]["gradient_checkpointing"] is False


def test_vision_train_builds_openchat_and_messages(monkeypatch, tmp_path: Path):
    trainer = axb.AxolotlVisionTrainer(base_model="test/model", load_in_4bit=True)

    captured: dict[str, Any] = {}

    def fake_run(cfg: dict, job_id: str | None, work_dir: Path | None):
        captured["cfg"] = cfg
        captured["work_dir"] = work_dir
        dataset_path = Path(cfg["datasets"][0]["path"])
        captured["dataset_lines"] = dataset_path.read_text().strip().splitlines()

    monkeypatch.setattr(axb, "_run_axolotl", fake_run)

    raw_dataset = [
        {"text": "describe", "image": "img.png", "response": "desc"},
    ]
    formatted = trainer.format_dataset(raw_dataset, system_message="You are helpful")

    config = VisionTrainingConfig(output_dir=str(tmp_path / "out"), batch_size=1, num_epochs=1)
    config.selective_loss = True
    config.selective_loss_level = "moderate"

    eval_formatted = trainer.format_dataset(
        [{"text": "eval describe", "image": "eval.png", "response": "desc"}],
        system_message="You are helpful",
    )

    trainer.train(formatted, config, job_id="job-vis", eval_dataset=eval_formatted)

    record = json.loads(captured["dataset_lines"][0])

    assert record["messages"][0]["role"] == "system"
    assert record["messages"][1]["role"] == "user"
    assert captured["cfg"]["datasets"][0]["format"] == "openchat"
    assert captured["cfg"]["vision"] is True
    assert captured["cfg"]["datasets"][1]["type"] == "validation"
    assert captured["cfg"]["selective_loss"]["level"] == "moderate"


def test_run_axolotl_serializes_yaml_and_invokes_cli(monkeypatch, tmp_path: Path):
    calls: dict[str, Any] = {}

    def fake_run(cmd, cwd=None, capture_output=None, text=None):
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(axb.subprocess, "run", fake_run)

    cfg = {"base_model": "test/model", "output_dir": str(tmp_path / "out")}

    axb._run_axolotl(cfg, job_id="job-123", work_dir=tmp_path)

    config_path = tmp_path / "config.yaml"
    assert config_path.exists()

    loaded = yaml.safe_load(config_path.read_text())
    assert loaded["base_model"] == "test/model"

    assert calls["cmd"][0:4] == ["python", "-m", "axolotl.cli.train", "-c"]
    assert Path(calls["cmd"][4]) == config_path
    assert calls["cwd"] == tmp_path
