"""Axolotl training backend (CLI-driven).

Registered only when the `axolotl` package is importable. Uses Axolotl's
CLI to launch training jobs with generated YAML configs for text and vision
models.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from model_garden.training.backends.base import TextTrainer, TrainingBackend, VisionTrainer
from model_garden.training.mixins import TrainerMixin
from model_garden.utils.console import console
from model_garden.utils.optional_deps import require_axolotl


class AxolotlTextTrainer(TrainerMixin, TextTrainer):
    """Text trainer that shells out to Axolotl CLI with a generated config."""

    def __init__(
        self,
        base_model: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: str | None = None,
    ):
        super().__init__(base_model, max_seq_length, load_in_4bit, load_in_8bit, dtype)
        self._lora_params: dict[str, Any] = {}
        self._hf_dataset: Any = None

    def load_model(self) -> None:
        require_axolotl("Axolotl text training")
        console.print(f"[cyan]Axolotl backend (text) using base model: {self.base_model}[/cyan]")
        # Axolotl loads models inside its training CLI; nothing to load here.

    def prepare_for_training(
        self,
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: list[str] | None = None,
        use_rslora: bool = False,
        lora_bias: str = "none",
        task_type: str = "CAUSAL_LM",
        use_gradient_checkpointing: str | bool = "unsloth",
        random_state: int = 42,
        loftq_config: dict | None = None,
    ) -> None:
        # Store LoRA/GC params to map into axolotl config at train time
        self._lora_params = {
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "use_rslora": use_rslora,
            "lora_bias": lora_bias,
            "task_type": task_type,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "random_state": random_state,
            "loftq_config": loftq_config,
        }

    def load_dataset_from_file(self, dataset_path: str):  # type: ignore[override]
        if not dataset_path:
            raise ValueError("dataset_path must be a non-empty string")

        return {"path": str(dataset_path), "source": "file"}

    def load_dataset_from_hub(self, dataset_name: str, split: str = "train", **kwargs):  # type: ignore[override]
        return super().load_dataset_from_hub(dataset_name, split=split, **kwargs)

    def format_dataset(  # type: ignore[override]
        self,
        dataset,
        instruction_field: str = "instruction",
        input_field: str = "input",
        output_field: str = "output",
        prompt_template: str | None = None,
    ):
        descriptor = _ensure_dataset_descriptor(
            dataset,
            {
                "instruction_field": instruction_field,
                "input_field": input_field,
                "output_field": output_field,
                "prompt_template": prompt_template,
            },
        )
        return descriptor

    def train(
        self,
        dataset: Any,
        config,
        job_id: str | None = None,
        enable_carbon_tracking: bool = True,
        callbacks: list | None = None,
        eval_dataset=None,
    ) -> None:
        require_axolotl("Axolotl text training")
        descriptor = _ensure_dataset_descriptor(
            dataset,
            {
                "instruction_field": "instruction",
                "input_field": "input",
                "output_field": "output",
            },
        )

        eval_descriptor = (
            _ensure_dataset_descriptor(
                eval_dataset,
                {
                    "instruction_field": descriptor.get("instruction_field", "instruction"),
                    "input_field": descriptor.get("input_field", "input"),
                    "output_field": descriptor.get("output_field", "output"),
                },
            )
            if eval_dataset is not None
            else None
        )

        with tempfile.TemporaryDirectory(prefix="axolotl-mg-text-") as tmpdir:
            work_path = Path(tmpdir)
            local_path = _materialize_dataset(descriptor, work_path, is_vision=False, name="train")
            eval_local_path = (
                _materialize_dataset(eval_descriptor, work_path, is_vision=False, name="eval")
                if eval_descriptor is not None
                else None
            )

            yaml_config = _build_axolotl_text_config(
                base_model=self.base_model,
                dataset_descriptor={**descriptor, "path": local_path},
                training_config=config,
                lora_params=self._lora_params,
                max_seq_length=self.max_seq_length,
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                dtype=self.dtype,
                output_dir=config.output_dir,
                eval_dataset_path=eval_local_path,
            )

            _run_axolotl(yaml_config, job_id=job_id, work_dir=Path(tmpdir))

    def save_model(
        self,
        output_dir: str,
        save_method: str = "merged_16bit",
        maximum_memory_usage: float = 0.75,
        max_shard_size: str = "5GB",
    ) -> None:
        # Axolotl writes into output_dir during train; no-op
        console.print(f"[cyan]Axolotl output already in {output_dir}[/cyan]")


class AxolotlVisionTrainer(TrainerMixin, VisionTrainer):
    """Vision trainer that shells out to Axolotl CLI with a generated config."""

    def __init__(
        self,
        base_model: str,
        max_seq_length: int = 16384,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: Any | None = None,
    ):
        super().__init__(base_model, max_seq_length, load_in_4bit, load_in_8bit, dtype)
        self._lora_params: dict[str, Any] = {}

    def load_model(self) -> None:
        require_axolotl("Axolotl vision training")
        console.print(f"[cyan]Axolotl backend (vision) base model: {self.base_model}[/cyan]")

    def prepare_for_training(
        self,
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: list[str] | None = None,
        use_rslora: bool = False,
        lora_bias: str = "none",
        task_type: str = "CAUSAL_LM",
        use_gradient_checkpointing: str | bool = "unsloth",
        random_state: int = 42,
        loftq_config: dict | None = None,
        finetune_vision_layers: bool = True,
        finetune_language_layers: bool = True,
        finetune_attention_modules: bool = True,
        finetune_mlp_modules: bool = True,
    ) -> None:
        self._lora_params = {
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "use_rslora": use_rslora,
            "lora_bias": lora_bias,
            "task_type": task_type,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "random_state": random_state,
            "loftq_config": loftq_config,
            "finetune_vision_layers": finetune_vision_layers,
            "finetune_language_layers": finetune_language_layers,
            "finetune_attention_modules": finetune_attention_modules,
            "finetune_mlp_modules": finetune_mlp_modules,
        }

    def load_dataset_from_file(self, dataset_path: str):  # type: ignore[override]
        if not dataset_path:
            raise ValueError("dataset_path must be a non-empty string")

        return {"path": str(dataset_path), "source": "file"}

    def load_dataset_from_hub(self, dataset_name: str, split: str = "train", **kwargs):  # type: ignore[override]
        return super().load_dataset_from_hub(dataset_name, split=split, **kwargs)

    def format_dataset(  # type: ignore[override]
        self,
        dataset,
        text_field: str = "text",
        image_field: str = "image",
        system_message: str | None = None,
        messages_field: str | None = None,
        lazy_loading: bool = False,
        *,
        output_field: str = "response",
        image_list_field: str = "images",
    ):
        descriptor = _ensure_dataset_descriptor(
            dataset,
            {
                "text_field": text_field,
                "image_field": image_field,
                "output_field": output_field,
                "image_list_field": image_list_field,
                "system_message": system_message,
                "messages_field": messages_field,
                "lazy_loading": lazy_loading,
            },
        )
        return descriptor

    def train(
        self,
        dataset: Any,
        config,
        job_id: str | None = None,
        enable_carbon_tracking: bool = True,
        callbacks: list | None = None,
        eval_dataset=None,
    ) -> None:
        require_axolotl("Axolotl vision training")
        descriptor = _ensure_dataset_descriptor(
            dataset,
            {
                "text_field": "text",
                "image_field": "image",
                "output_field": "response",
                "image_list_field": "images",
                "messages_field": "messages",
            },
        )

        eval_descriptor = (
            _ensure_dataset_descriptor(
                eval_dataset,
                {
                    "text_field": descriptor.get("text_field", "text"),
                    "image_field": descriptor.get("image_field", "image"),
                    "output_field": descriptor.get("output_field", "response"),
                    "image_list_field": descriptor.get("image_list_field", "images"),
                    "messages_field": descriptor.get("messages_field", "messages"),
                },
            )
            if eval_dataset is not None
            else None
        )

        with tempfile.TemporaryDirectory(prefix="axolotl-mg-vis-") as tmpdir:
            work_path = Path(tmpdir)
            local_path = _materialize_dataset(descriptor, work_path, is_vision=True, name="train")
            eval_local_path = (
                _materialize_dataset(eval_descriptor, work_path, is_vision=True, name="eval")
                if eval_descriptor is not None
                else None
            )

            yaml_config = _build_axolotl_vision_config(
                base_model=self.base_model,
                dataset_descriptor={**descriptor, "path": local_path},
                training_config=config,
                lora_params=self._lora_params,
                max_seq_length=self.max_seq_length,
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                dtype=self.dtype,
                output_dir=config.output_dir,
                eval_dataset_path=eval_local_path,
            )

            _run_axolotl(yaml_config, job_id=job_id, work_dir=Path(tmpdir))

    def save_model(
        self,
        output_dir: str,
        save_method: str = "merged_16bit",
        maximum_memory_usage: float = 0.75,
        max_shard_size: str = "5GB",
    ) -> None:
        console.print(f"[cyan]Axolotl output already in {output_dir}[/cyan]")


class AxolotlBackend(TrainingBackend):
    """Axolotl training backend (experimental, subprocess-driven)."""

    @property
    def name(self) -> str:
        return "axolotl"

    @property
    def description(self) -> str:
        return "Axolotl training (experimental, subprocess CLI)"

    def supports_text_training(self) -> bool:
        return True

    def supports_vision_training(self) -> bool:
        # Vision is planned; currently stubbed but advertised so UI can show capability
        return True

    def create_text_trainer(
        self,
        base_model: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: str | None = None,
    ) -> TextTrainer:
        require_axolotl("Axolotl text training")
        return AxolotlTextTrainer(
            base_model=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            dtype=dtype,
        )

    def create_vision_trainer(
        self,
        base_model: str,
        max_seq_length: int = 16384,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: Any | None = None,
    ) -> VisionTrainer:
        require_axolotl("Axolotl vision training")
        return AxolotlVisionTrainer(
            base_model=base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            dtype=dtype,
        )


def _build_common_precision(load_in_4bit: bool, load_in_8bit: bool, dtype: Any | None) -> dict:
    # 8-bit takes priority
    if load_in_8bit:
        return {"load_in_8bit": True, "load_in_4bit": False, "bf16": False, "fp16": False}
    if load_in_4bit:
        return {"load_in_4bit": True, "load_in_8bit": False, "bf16": True, "fp16": False}
    wants_bf16 = dtype in {"bfloat16", "bf16"}
    return {
        "load_in_4bit": False,
        "load_in_8bit": False,
        "bf16": wants_bf16,
        "fp16": not wants_bf16,
    }


def _normalize_gradient_checkpointing(value: Any) -> bool:
    """Normalize gradient checkpointing flag from various truthy/falsey inputs."""

    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered not in {"false", "0", "off", "none", "no"}
    return bool(value)


def _ensure_dataset_descriptor(dataset: Any, defaults: dict[str, Any]) -> dict:
    """Ensure a dataset descriptor dict with default field names.

    Accepts either a dict that may already include a `path` or `dataset`, or a
    HuggingFace Dataset / list of rows. This keeps typing noise out of the
    trainer methods.
    """

    if dataset is None:
        raise ValueError("dataset must be provided")

    if isinstance(dataset, dict) and ("path" in dataset or "dataset" in dataset):
        merged = {**defaults, **dataset}
    else:
        merged = {**defaults, "dataset": dataset, "path": None}

    return merged


def _attach_selective_loss(cfg: dict[str, Any], training_config: Any) -> None:
    """Attach selective loss settings if enabled.

    This mirrors the config surface from VisionTrainingConfig/SelectiveLossConfig.
    Axolotl will ignore unknown fields; we surface them so downstream tooling or
    custom Axolotl hooks can consume them.
    """

    enabled = getattr(training_config, "selective_loss", False)
    if not enabled:
        return

    cfg["selective_loss"] = {
        "enabled": True,
        "level": getattr(training_config, "selective_loss_level", None),
        "schema_keys": getattr(training_config, "selective_loss_schema_keys", None),
        "masking_strategy": getattr(training_config, "selective_loss_masking_strategy", None),
        "masking_start_epoch": getattr(training_config, "selective_loss_masking_start_epoch", None),
        "mask_every_n_steps": getattr(training_config, "selective_loss_mask_every_n_steps", None),
        "mask_for_n_steps": getattr(training_config, "selective_loss_mask_for_n_steps", None),
        "structural_weight": getattr(training_config, "selective_loss_structural_weight", None),
        "verbose": getattr(training_config, "selective_loss_verbose", None),
    }


def _build_axolotl_text_config(
    base_model: str,
    dataset_descriptor: dict,
    training_config: Any,
    lora_params: dict[str, Any],
    max_seq_length: int,
    load_in_4bit: bool,
    load_in_8bit: bool,
    dtype: Any | None,
    output_dir: str,
    eval_dataset_path: str | None = None,
) -> dict:
    precision = _build_common_precision(load_in_4bit, load_in_8bit, dtype)
    grad_ckpt = _normalize_gradient_checkpointing(lora_params.get("use_gradient_checkpointing"))

    cfg = {
        "base_model": base_model,
        "output_dir": output_dir,
        "cutoff_len": max_seq_length,
        "micro_batch_size": training_config.batch_size,
        "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
        "num_epochs": training_config.num_epochs,
        "learning_rate": training_config.learning_rate,
        "lr_scheduler": training_config.lr_scheduler_type,
        "warmup_steps": training_config.warmup_steps,
        "max_steps": training_config.max_steps,
        "logging_steps": training_config.logging_steps,
        "save_steps": training_config.save_steps,
        "eval_steps": training_config.eval_steps,
        "weight_decay": training_config.weight_decay,
        "lora": True,
        "lora_r": lora_params.get("r", 16),
        "lora_alpha": lora_params.get("lora_alpha", 16),
        "lora_dropout": lora_params.get("lora_dropout", 0.0),
        "target_modules": lora_params.get("target_modules"),
        "lora_bias": lora_params.get("lora_bias", "none"),
        "use_rslora": lora_params.get("use_rslora", False),
        "gradient_checkpointing": grad_ckpt,
        "seed": lora_params.get("random_state", 42),
        **precision,
    }

    dataset_path = (
        dataset_descriptor.get("path")
        if isinstance(dataset_descriptor, dict)
        else dataset_descriptor
    )
    if dataset_path is None:
        raise ValueError("Axolotl backend requires a dataset path")
    dataset_path = str(Path(dataset_path))

    cfg["datasets"] = [
        {
            "path": dataset_path,
            "format": "alpaca",
            "field_instruction": dataset_descriptor.get("instruction_field", "instruction"),
            "field_input": dataset_descriptor.get("input_field", "input"),
            "field_output": dataset_descriptor.get("output_field", "output"),
        }
    ]

    if eval_dataset_path:
        cfg["datasets"].append(
            {
                "path": eval_dataset_path,
                "format": "alpaca",
                "type": "validation",
                "field_instruction": dataset_descriptor.get("instruction_field", "instruction"),
                "field_input": dataset_descriptor.get("input_field", "input"),
                "field_output": dataset_descriptor.get("output_field", "output"),
            }
        )

    _attach_selective_loss(cfg, training_config)

    return cfg


def _build_axolotl_vision_config(
    base_model: str,
    dataset_descriptor: dict,
    training_config: Any,
    lora_params: dict[str, Any],
    max_seq_length: int,
    load_in_4bit: bool,
    load_in_8bit: bool,
    dtype: Any | None,
    output_dir: str,
    eval_dataset_path: str | None = None,
) -> dict:
    precision = _build_common_precision(load_in_4bit, load_in_8bit, dtype)
    grad_ckpt = _normalize_gradient_checkpointing(lora_params.get("use_gradient_checkpointing"))

    cfg = {
        "base_model": base_model,
        "output_dir": output_dir,
        "cutoff_len": max_seq_length,
        "micro_batch_size": training_config.batch_size,
        "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
        "num_epochs": training_config.num_epochs,
        "learning_rate": training_config.learning_rate,
        "lr_scheduler": training_config.lr_scheduler_type,
        "warmup_steps": training_config.warmup_steps,
        "max_steps": training_config.max_steps,
        "logging_steps": training_config.logging_steps,
        "save_steps": training_config.save_steps,
        "eval_steps": training_config.eval_steps,
        "weight_decay": training_config.weight_decay,
        "vision": True,
        "lora": True,
        "lora_r": lora_params.get("r", 16),
        "lora_alpha": lora_params.get("lora_alpha", 16),
        "lora_dropout": lora_params.get("lora_dropout", 0.0),
        "target_modules": lora_params.get("target_modules"),
        "lora_bias": lora_params.get("lora_bias", "none"),
        "use_rslora": lora_params.get("use_rslora", False),
        "gradient_checkpointing": grad_ckpt,
        "seed": lora_params.get("random_state", 42),
        **precision,
    }

    dataset_path = (
        dataset_descriptor.get("path")
        if isinstance(dataset_descriptor, dict)
        else dataset_descriptor
    )
    if dataset_path is None:
        raise ValueError("Axolotl vision backend requires a dataset path")
    dataset_path = str(Path(dataset_path))

    cfg["datasets"] = [
        {
            "path": dataset_path,
            "format": "openchat",
            "field_messages": dataset_descriptor.get("messages_field", "messages"),
            "field_images": dataset_descriptor.get("image_list_field", "images"),
        }
    ]

    if eval_dataset_path:
        cfg["datasets"].append(
            {
                "path": eval_dataset_path,
                "format": "openchat",
                "type": "validation",
                "field_messages": dataset_descriptor.get("messages_field", "messages"),
                "field_images": dataset_descriptor.get("image_list_field", "images"),
            }
        )

    _attach_selective_loss(cfg, training_config)

    return cfg


def _materialize_dataset(
    dataset_descriptor: Any, work_dir: Path, is_vision: bool, *, name: str | None = None
) -> str:
    """Materialize dataset to a JSONL file that Axolotl can read."""

    if not isinstance(dataset_descriptor, dict):
        raise ValueError("dataset_descriptor must be a dict")

    if dataset_descriptor.get("path"):
        return str(dataset_descriptor["path"])

    ds = dataset_descriptor.get("dataset")
    if ds is None:
        raise ValueError("Dataset descriptor must include either 'path' or 'dataset'")

    base_name = name or ("vision" if is_vision else "text")
    jsonl_path = work_dir / f"{base_name}.jsonl"

    with jsonl_path.open("w") as f:
        for row in ds:
            if is_vision:
                text_field = dataset_descriptor.get("text_field", "text")
                image_field = dataset_descriptor.get("image_field", "image")
                output_field = dataset_descriptor.get("output_field", "response")
                messages_field = dataset_descriptor.get("messages_field", "messages")
                image_list_field = dataset_descriptor.get("image_list_field", "images")
                system_message = dataset_descriptor.get("system_message")

                messages = row.get(messages_field)
                if messages is None:
                    messages = [
                        {"role": "user", "content": row.get(text_field, "")},
                        {"role": "assistant", "content": row.get(output_field, "")},
                    ]

                if system_message and not any(m.get("role") == "system" for m in messages):
                    messages = [{"role": "system", "content": system_message}, *messages]

                images_value = row.get(image_list_field)
                if images_value is None:
                    images_value = [row.get(image_field, "")]
                elif isinstance(images_value, str):
                    images_value = [images_value]
                elif isinstance(images_value, list):
                    images_value = [img for img in images_value if img]
                else:
                    raise ValueError("images field must be a string or list of strings")

                rec = {
                    "messages": messages,
                    image_list_field: images_value,
                }
            else:
                rec = {
                    dataset_descriptor.get("instruction_field", "instruction"): row.get(
                        dataset_descriptor.get("instruction_field", "instruction"), ""
                    ),
                    dataset_descriptor.get("input_field", "input"): row.get(
                        dataset_descriptor.get("input_field", "input"), ""
                    ),
                    dataset_descriptor.get("output_field", "output"): row.get(
                        dataset_descriptor.get("output_field", "output"), ""
                    ),
                }
            f.write(json.dumps(rec) + "\n")

    return str(jsonl_path)


def _run_axolotl(
    yaml_config: dict, job_id: str | None = None, work_dir: Path | None = None
) -> None:
    """Serialize config to temp YAML and invoke Axolotl CLI."""

    if work_dir is None:
        tmp_context = tempfile.TemporaryDirectory(prefix="axolotl-mg-")
        work_dir_path = Path(tmp_context.name)
    else:
        tmp_context = None
        work_dir_path = work_dir

    try:
        config_path = work_dir_path / "config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(yaml_config, f, sort_keys=False)

        console.print(
            "[cyan]Launching Axolotl subprocess[/cyan]"
            + (f" [dim](job {job_id})[/dim]" if job_id else "")
        )

        cmd = ["python", "-m", "axolotl.cli.train", "-c", str(config_path)]

        result = subprocess.run(cmd, cwd=work_dir_path, capture_output=True, text=True)

        if result.returncode != 0:
            console.print("[red]❌ Axolotl training failed[/red]")
            console.print(result.stderr)
            raise RuntimeError(f"Axolotl training failed (exit {result.returncode})")

        if result.stdout:
            console.print(result.stdout)
    finally:
        if tmp_context is not None:
            tmp_context.cleanup()
