"""BoAmps report generator for standardized emissions reporting.

Implements the BoAmps v1.1.0 specification from Boavizta for standardized
reporting of AI/ML energy consumption and carbon emissions.

Reference: https://github.com/Boavizta/BoAmps
Schema: https://raw.githubusercontent.com/Boavizta/BoAmps/main/model/report_schema.json
"""

import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .hardware_detection import get_hardware_detector

# Settings persistence (shared with system settings)
PROJECT_ROOT = Path(__file__).parent.parent.parent
SYSTEM_SETTINGS_FILE = PROJECT_ROOT / "storage" / "system_settings.json"


def _load_report_settings() -> dict[str, Any]:
    """Load report-related defaults from system settings if available."""
    defaults = {
        "publisher_name": "Model Garden",
        "division": None,
        "default_project_name": "Model Garden",
        "infra_type": None,
        "location_country": None,
        "location_region": None,
    }

    try:
        if SYSTEM_SETTINGS_FILE.exists():
            with open(SYSTEM_SETTINGS_FILE) as f:
                data = json.load(f)
            if isinstance(data, dict):
                report = data.get("report", {}) if isinstance(data.get("report"), dict) else {}
                defaults.update(
                    {
                        "publisher_name": report.get("publisher_name", defaults["publisher_name"]),
                        "division": report.get("division", defaults["division"]),
                        "default_project_name": report.get(
                            "default_project_name", defaults["default_project_name"]
                        ),
                        "infra_type": report.get("infra_type", defaults["infra_type"]),
                        "location_country": report.get(
                            "location_country", defaults["location_country"]
                        ),
                        "location_region": report.get(
                            "location_region", defaults["location_region"]
                        ),
                    }
                )
    except Exception:
        # If settings can't be read, fall back to defaults silently
        pass

    return defaults


def _as_number(value: Any, default: float = 0.0) -> float:
    """Convert arbitrary input to a float, falling back safely."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return float(default)


def _extract_model_params_from_name(model_name: str) -> float | None:
    """Extract parameter count (in billions) from model name.

    Examples:
        "Qwen2.5-VL-7B-Instruct" -> 7.0
        "llama-3.1-8b" -> 8.0
        "mistral-7b-v0.1" -> 7.0
    """
    if not model_name:
        return None

    # Common patterns for parameter counts
    patterns = [
        r"(\d+(?:\.\d+)?)[bB](?:-|_|$|\s)",  # 7B, 8B, 3.5B
        r"-(\d+(?:\.\d+)?)(?:-|_|$|\s)",  # -7-, -8-
    ]

    for pattern in patterns:
        match = re.search(pattern, model_name)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return None


def _get_huggingface_model_uri(model_name: str) -> str | None:
    """Generate HuggingFace URI for a model if it looks like a HF model."""
    if not model_name:
        return None

    # If it already looks like a HF path (contains /)
    if "/" in model_name:
        return f"https://huggingface.co/{model_name}"

    return None


class BoAmpsReportGenerator:
    """Generate BoAmps-compliant emissions reports from CodeCarbon data.

    Follows the BoAmps v1.1.0 specification for standardized AI energy
    consumption reporting. Reports can be validated against the official
    BoAmps schema and contributed to the Boavizta open dataset.
    """

    BOAMPS_VERSION = "1.1.0"
    BOAMPS_SPEC_URI = (
        "https://raw.githubusercontent.com/Boavizta/BoAmps/main/model/report_schema.json"
    )
    LICENSING = "Creative Commons 4.0"

    def __init__(
        self,
        publisher_name: str = "Model Garden",
        publisher_division: str | None = None,
        default_project_name: str | None = None,
        infra_type_default: str | None = None,
        location_country_default: str | None = None,
        location_region_default: str | None = None,
        confidentiality_level: str = "public",
    ):
        """
        Initialize BoAmps report generator.

        Args:
            publisher_name: Name of the organization
            publisher_division: Division or team name
            default_project_name: Default project name for reports
            infra_type_default: Default infrastructure type (onPremise|publicCloud|privateCloud)
            location_country_default: Default country code or name
            location_region_default: Default region/city
            confidentiality_level: public|internal|confidential|secret
        """
        self.publisher_name = publisher_name
        self.publisher_division = publisher_division
        self.default_project_name = default_project_name or "Model Garden"
        self.infra_type_default = infra_type_default
        self.location_country_default = location_country_default
        self.location_region_default = location_region_default
        self.confidentiality_level = confidentiality_level

    def generate_report(
        self,
        emissions_data: dict[str, Any],
        job_config: dict[str, Any] | None = None,
        report_status: str = "final",
    ) -> dict[str, Any]:
        """
        Generate complete BoAmps report from emissions data.

        Args:
            emissions_data: Emissions data from CodeCarbon or EmissionsDatabase
            job_config: Training/inference job configuration
            report_status: final|draft|corrective

        Returns:
            BoAmps-compliant JSON report
        """
        job_config = job_config or {}

        return {
            "header": self._generate_header(emissions_data, report_status),
            "task": self._generate_task(emissions_data, job_config),
            "measures": self._generate_measures(emissions_data),
            "infrastructure": self._generate_infrastructure(emissions_data),
            "system": self._generate_system(emissions_data),
            "software": self._generate_software(emissions_data),
            "environment": self._generate_environment(emissions_data),
            "quality": self._estimate_quality(emissions_data),
        }

    def _generate_header(
        self, emissions_data: dict[str, Any], report_status: str
    ) -> dict[str, Any]:
        """Generate header section."""
        # Convert timestamp to BoAmps format: YYYY-MM-DD HH:MM:SS
        timestamp_str = emissions_data.get("timestamp", "")
        if timestamp_str:
            try:
                if isinstance(timestamp_str, str):
                    # Parse ISO format and convert to BoAmps format
                    dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                else:
                    dt = datetime.now(UTC)
                report_datetime = dt.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, AttributeError):
                report_datetime = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        else:
            report_datetime = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        return {
            "licensing": self.LICENSING,
            "formatVersion": self.BOAMPS_VERSION,
            "formatVersionSpecificationUri": self.BOAMPS_SPEC_URI,
            "reportId": emissions_data.get("job_id", str(uuid.uuid4())),
            "reportDatetime": report_datetime,
            "reportStatus": report_status,
            "publisher": {
                "name": self.publisher_name,
                "division": self.publisher_division,
                "projectName": self.default_project_name,
                "confidentialityLevel": self.confidentiality_level,
            },
        }

    def _generate_task(
        self, emissions_data: dict[str, Any], job_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate task section with comprehensive job configuration.

        Per BoAmps schema, the task section must include:
        - taskStage (required): training, finetuning, inference, etc.
        - taskFamily (required): textGeneration, imageClassification, etc.
        - algorithms (required): list of algorithm descriptions
        - dataset (required): list of dataset descriptions
        - nbRequest (optional): number of inference requests
        - measuredAccuracy (optional): 0-1 accuracy value
        - taskDescription (optional): free text description
        """
        job_type = emissions_data.get("job_type", "training")

        # Determine task stage - BoAmps uses "finetuning" for fine-tuning scenarios
        # Training with a pre-trained base model is typically "finetuning"
        if job_type == "training":
            # If we have a base_model and lora_config, it's fine-tuning
            if job_config.get("base_model") or job_config.get("lora_config"):
                task_stage = "finetuning"
            else:
                task_stage = "training"
        elif job_type == "inference":
            task_stage = "inference"
        else:
            task_stage = job_type

        # Determine if vision model - check config, then job_id/output_dir
        is_vision = job_config.get("is_vision", False)
        model_type = job_config.get("model_type", "")

        # Also infer from emissions data if not in config
        if not is_vision:
            job_id = emissions_data.get("job_id", "")
            output_dir = emissions_data.get("output_dir", "")
            model_name = emissions_data.get("model_name", "") or job_config.get("base_model", "")

            # Check for vision indicators
            vision_hints = [job_id, output_dir, model_name]
            for hint in vision_hints:
                hint_lower = hint.lower()
                if any(
                    v in hint_lower
                    for v in ["vision", "-vl-", "-vl.", "qwen3-vl", "qwen2.5-vl", "qwen-vl"]
                ):
                    is_vision = True
                    break

        # Determine task family based on model capabilities
        if is_vision or model_type == "vision":
            task_family = "multiModalTextGeneration"
        else:
            task_family = "textGeneration"

        # Build algorithms section (per algorithm_schema.json)
        algorithms = self._build_algorithms(emissions_data, job_config, task_stage)

        # Build dataset section (per dataset_schema.json)
        datasets = self._build_datasets(emissions_data, job_config, is_vision, task_stage)

        # Build result
        result = {
            "taskStage": task_stage,
            "taskFamily": task_family,
            "algorithms": algorithms,
            "dataset": datasets,
        }

        # Add inference request count if applicable
        if task_stage == "inference":
            result["nbRequest"] = job_config.get("num_requests", 1)

        # Add task description for context
        task_description = self._build_task_description(
            emissions_data, job_config, task_stage, is_vision
        )
        if task_description:
            result["taskDescription"] = task_description

        # Add measured accuracy if available from metrics
        if "final_loss" in job_config:
            # Note: loss is not accuracy, but we can include it in description
            pass

        return result

    def _build_algorithms(
        self,
        emissions_data: dict[str, Any],
        job_config: dict[str, Any],
        task_stage: str,
    ) -> list[dict[str, Any]]:
        """Build algorithms section per BoAmps algorithm_schema.json.

        Includes:
        - trainingType: supervisedLearning, transferLearning, etc.
        - algorithmType: llm, transformer, etc.
        - algorithmName: name of the algorithm/architecture
        - foundationModelName: name of the base model
        - foundationModelUri: URI to model on HuggingFace etc.
        - parametersNumber: billions of parameters
        - framework: PyTorch, TensorFlow, etc.
        - frameworkVersion: version string
        - epochsNumber: number of training epochs
        - optimizer: adam, sgd, lora, etc.
        - quantization: fp32, fp16, int8, etc.
        """
        model_name = emissions_data.get("model_name") or job_config.get("base_model", "")

        # Try to infer model name from output_dir or job_id if not set
        if not model_name or model_name == "unknown" or model_name == "Unknown":
            # Check output_dir for clues
            output_dir = emissions_data.get("output_dir", "")
            job_id = emissions_data.get("job_id", "")

            # Try to extract model hints from paths
            path_hints = output_dir + " " + job_id
            path_lower = path_hints.lower()

            # Look for known model patterns
            if "qwen" in path_lower:
                # Try to extract full model name
                if "qwen3-vl" in path_lower:
                    model_name = "Qwen/Qwen3-VL"
                elif "qwen2.5-vl" in path_lower or "qwen-2.5-vl" in path_lower:
                    model_name = "Qwen/Qwen2.5-VL"
                elif "qwen-vl" in path_lower:
                    model_name = "Qwen/Qwen-VL"
                else:
                    model_name = "Qwen"
            elif "llama" in path_lower:
                model_name = "Meta-Llama"
            elif "mistral" in path_lower:
                model_name = "Mistral"
            elif "nanonets" in path_lower:
                model_name = "nanonets/Nanonets-OCR"

            # Extract size hints
            for size in ["3b", "7b", "8b", "14b", "32b", "70b", "72b"]:
                if size in path_lower:
                    model_name = f"{model_name}-{size.upper()}"
                    break

        # Extract framework version
        framework_version = "2.x"
        try:
            import torch

            framework_version = torch.__version__.split("+")[0]
        except Exception:
            pass

        # Build algorithm entry per schema
        algorithm: dict[str, Any] = {}

        # Training type (for training/finetuning stages)
        if task_stage in ["training", "finetuning"]:
            if job_config.get("lora_config"):
                algorithm["trainingType"] = "transferLearning"
            else:
                algorithm["trainingType"] = "supervisedLearning"

        # Algorithm type - for LLMs/VLMs
        is_vision = job_config.get("is_vision", False)

        # Also check if it's a vision task from job_id hints
        job_id = emissions_data.get("job_id", "")
        output_dir = emissions_data.get("output_dir", "")
        if (
            "vision" in job_id.lower()
            or "vision" in output_dir.lower()
            or "vl" in model_name.lower()
        ):
            is_vision = True

        if is_vision:
            algorithm["algorithmType"] = "vlm"  # Vision Language Model
        else:
            algorithm["algorithmType"] = "llm"  # Large Language Model

        # Algorithm name (architecture type)
        # Try to extract architecture from model name
        model_lower = (model_name or "").lower()
        if "qwen" in model_lower:
            algorithm["algorithmName"] = "transformer"
        elif "llama" in model_lower:
            algorithm["algorithmName"] = "transformer"
        elif "mistral" in model_lower:
            algorithm["algorithmName"] = "transformer"
        else:
            algorithm["algorithmName"] = "transformer"  # Default for LLMs

        # Foundation model info
        if model_name and model_name != "unknown" and model_name != "Unknown":
            algorithm["foundationModelName"] = model_name

            # Add HuggingFace URI
            model_uri = _get_huggingface_model_uri(model_name)
            if model_uri:
                algorithm["foundationModelUri"] = model_uri

        # Extract parameter count from model name
        params = _extract_model_params_from_name(model_name or "")
        if params:
            algorithm["parametersNumber"] = params

        # Framework info
        algorithm["framework"] = "PyTorch"
        algorithm["frameworkVersion"] = framework_version

        # Training-specific fields
        if task_stage in ["training", "finetuning"]:
            job_hyperparams = job_config.get("hyperparameters", {})

            # Number of epochs
            epochs = job_hyperparams.get("num_epochs", job_hyperparams.get("epochs"))
            if epochs:
                algorithm["epochsNumber"] = epochs

            # Optimizer
            optimizer = job_hyperparams.get("optim", job_hyperparams.get("optimizer"))
            if optimizer:
                algorithm["optimizer"] = optimizer
            elif job_config.get("lora_config"):
                algorithm["optimizer"] = "lora"  # LoRA is an optimization technique

        # Quantization info
        if job_config.get("lora_config") or job_config.get("load_in_4bit"):
            algorithm["quantization"] = "int4"
        elif job_config.get("load_in_8bit"):
            algorithm["quantization"] = "int8"
        else:
            algorithm["quantization"] = "fp16"  # Default for modern training

        return [algorithm]

    def _detect_file_type(self, path: str, source_type: str) -> str:
        """Detect detailed file type per BoAmps enum."""
        path = path.lower()
        if path.endswith(".json"):
            return "json"
        if path.endswith(".jsonl"):
            return "json"  # BoAmps doesn't have jsonl, map to json
        if path.endswith(".csv"):
            return "csv"
        if path.endswith(".parquet"):
            return "parquet"
        if path.endswith(".txt"):
            return "txt"
        if path.endswith(".jpg") or path.endswith(".jpeg"):
            return "jpg"
        if path.endswith(".png"):
            return "png"
        if path.endswith(".webp"):
            return "webp"

        return "other"

    def _compute_dataset_stats(self, dataset_path: str) -> tuple[float | None, int | None]:
        """Estimate dataset size (bytes) and item count for local paths."""
        try:
            path = Path(dataset_path).expanduser()
        except Exception:
            return None, None

        try:
            if path.is_file():
                size_bytes = float(path.stat().st_size)
                num_items = None

                # Avoid expensive scans on very large files
                max_scan_bytes = 256 * 1024 * 1024  # 256 MB safety guard
                suffix = path.suffix.lower()

                if size_bytes <= max_scan_bytes:
                    if suffix in {".jsonl", ".txt"}:
                        with path.open("r", encoding="utf-8", errors="ignore") as f:
                            num_items = sum(1 for _ in f)
                    elif suffix == ".json":
                        with path.open("r", encoding="utf-8", errors="ignore") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                num_items = len(data)
                            elif isinstance(data, dict):
                                for key in ["data", "samples", "records"]:
                                    if isinstance(data.get(key), list):
                                        num_items = len(data[key])
                                        break
                    elif suffix == ".csv":
                        with path.open("r", encoding="utf-8", errors="ignore") as f:
                            rows = sum(1 for _ in f)
                            if rows > 1:
                                num_items = rows - 1  # Drop header
                            elif rows > 0:
                                num_items = rows

                return size_bytes, num_items

            if path.is_dir():
                size_bytes = 0.0
                num_items = 0
                image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}

                for file in path.rglob("*"):
                    if not file.is_file():
                        continue
                    try:
                        size_bytes += float(file.stat().st_size)
                    except Exception:
                        continue

                    if file.suffix.lower() in image_exts:
                        num_items += 1

                return (size_bytes or None), (num_items or None)

        except Exception:
            return None, None

        return None, None

    def _infer_file_type_from_name(self, name: str) -> tuple[str | None, str | None]:
        """Infer BoAmps fileType and dataFormat from a filename or path."""
        lower = name.lower()
        mappings = {
            ".jsonl": ("json", "json"),
            ".json": ("json", "json"),
            ".csv": ("csv", "csv"),
            ".tsv": ("csv", "csv"),
            ".parquet": ("parquet", "parquet"),
            ".txt": ("txt", "txt"),
            ".jpg": ("jpg", None),
            ".jpeg": ("jpg", None),
            ".png": ("png", None),
            ".webp": ("webp", None),
        }
        for ext, val in mappings.items():
            if lower.endswith(ext):
                return val
        return None, None

    def _infer_hf_file_type(self, dataset_path: str) -> tuple[str | None, str | None]:
        """Infer fileType and dataFormat for a Hub dataset by inspecting files."""
        if not dataset_path or "/" not in dataset_path:
            return None, None

        if os.getenv("BOAMPS_SKIP_HF_FETCH", "0") == "1":
            return None, None

        try:
            from model_garden.utils.hf_cache import configure_hf_cache

            configure_hf_cache()
            from huggingface_hub import HfApi

            api = HfApi()
            files = api.list_files_info(dataset_path, repo_type="dataset")

            # Count extensions to pick the dominant type
            counts: dict[str, int] = {}
            formats: dict[str, str | None] = {}
            for file_info in files:
                candidate = getattr(file_info, "path", "") or getattr(file_info, "rfilename", "")
                ftype, dformat = self._infer_file_type_from_name(candidate)
                if ftype:
                    counts[ftype] = counts.get(ftype, 0) + 1
                    if ftype not in formats and dformat:
                        formats[ftype] = dformat

            if not counts:
                return None, None

            # Priority: json > jsonl (maps to json) when present
            for preferred in ["json", "csv", "parquet", "txt", "jpg", "png", "webp"]:
                if preferred in counts:
                    return preferred, formats.get(preferred)

            # Otherwise pick the most frequent fileType
            file_type = max(counts.items(), key=lambda kv: kv[1])[0]
            data_format = formats.get(file_type)
            return file_type, data_format
        except Exception:
            return None, None

    def _fetch_hf_dataset_metadata(
        self, dataset_path: str
    ) -> tuple[float | None, int | None, dict[str, dict[str, float | int | None]]]:
        """Fetch dataset size/items and split stats from HuggingFace Hub metadata.

        Returns (size_bytes, num_items, split_stats) when available, otherwise (None, None, {}).
        split_stats maps split name -> {"size": bytes|None, "items": int|None}.
        Respects HF_HUB_OFFLINE and can be disabled via BOAMPS_SKIP_HF_FETCH=1.
        """
        if not dataset_path or "/" not in dataset_path:
            return None, None, {}

        if os.getenv("BOAMPS_SKIP_HF_FETCH", "0") == "1":
            return None, None, {}

        try:
            # Ensure HF caches are configured before importing hub client
            from model_garden.utils.hf_cache import configure_hf_cache

            configure_hf_cache()

            from huggingface_hub import HfApi

            api = HfApi()
            # files_metadata=True ensures sizes are returned for siblings
            info = api.dataset_info(dataset_path, files_metadata=True)

            size_bytes = None
            num_items = None
            split_stats: dict[str, dict[str, float | int | None]] = {}

            # Prefer explicit dataset_size/download_size fields when present
            try:
                for key in ["dataset_size", "download_size", "size"]:
                    if getattr(info, key, None):
                        size_bytes = float(getattr(info, key))
                        break
            except Exception:
                size_bytes = None

            # Sum file sizes from siblings when provided (fallback)
            try:
                if size_bytes is None and getattr(info, "siblings", None):
                    total = 0.0
                    for sibling in info.siblings:
                        sibling_size = getattr(sibling, "size", None)
                        if sibling_size:
                            total += float(sibling_size)
                    if total > 0:
                        size_bytes = total
            except Exception:
                size_bytes = None

            # List files to accumulate sizes (may require auth for private datasets) and infer split sizes
            try:
                split_size_heuristics: dict[str, float] = {"train": 0.0, "validation": 0.0, "test": 0.0}
                files = api.list_files_info(dataset_path, repo_type="dataset")
                total = 0.0
                for file_info in files:
                    if getattr(file_info, "size", None):
                        total += float(file_info.size)

                    # Heuristic split bucketing by filename
                    try:
                        name = getattr(file_info, "rfilename", None) or getattr(file_info, "path", "")
                        name_l = str(name).lower()
                        if "train" in name_l:
                            split_size_heuristics["train"] += float(file_info.size or 0.0)
                        elif any(token in name_l for token in ["validation", "valid", "val", "eval"]):
                            split_size_heuristics["validation"] += float(file_info.size or 0.0)
                        elif "test" in name_l:
                            split_size_heuristics["test"] += float(file_info.size or 0.0)
                    except Exception:
                        pass

                if size_bytes is None and total > 0:
                    size_bytes = total

                # If we inferred split sizes heuristically, record them
                for split_name, split_size in split_size_heuristics.items():
                    if split_size > 0:
                        existing = split_stats.get(split_name, {})
                        existing.setdefault("items", None)
                        existing["size"] = split_size
                        split_stats[split_name] = existing
            except Exception:
                size_bytes = size_bytes

            # Extract counts from cardData hints
            try:
                card = getattr(info, "cardData", None) or {}
                if isinstance(card, dict):
                    for key in [
                        "num_rows",
                        "num_examples",
                        "samples",
                        "dataset_num_samples",
                        "num_items",
                    ]:
                        if isinstance(card.get(key), (int, float)):
                            num_items = int(card[key])
                            break

                    # Aggregate split counts if present and capture per-split stats
                    splits = card.get("splits")
                    if isinstance(splits, list):
                        total_rows = 0
                        for split in splits:
                            if not isinstance(split, dict):
                                continue
                            name = str(split.get("name", "")).lower()
                            rows = split.get("num_examples") or split.get("num_rows")
                            size_split = split.get("num_bytes") or split.get("size_bytes")
                            if isinstance(rows, (int, float)):
                                total_rows += int(rows)
                            if name:
                                split_stats[name] = {
                                    "items": int(rows) if isinstance(rows, (int, float)) else None,
                                    "size": float(size_split)
                                    if isinstance(size_split, (int, float))
                                    else None,
                                }
                        if total_rows > 0:
                            num_items = total_rows
            except Exception:
                num_items = num_items

            # Extract split stats from info.splits when available
            try:
                hf_splits = getattr(info, "splits", None)
                if hf_splits:
                    total_rows = 0
                    total_bytes = 0.0
                    for split_name, split_obj in hf_splits.items():
                        name = str(split_name).lower()
                        split_items = None
                        split_size = None
                        try:
                            split_items = getattr(split_obj, "num_examples", None)
                        except Exception:
                            pass
                        try:
                            split_size = getattr(split_obj, "num_bytes", None)
                        except Exception:
                            pass

                        if isinstance(split_items, (int, float)):
                            total_rows += int(split_items)
                        if isinstance(split_size, (int, float)):
                            total_bytes += float(split_size)

                        if name:
                            existing = split_stats.get(name, {})
                            if isinstance(split_items, (int, float)):
                                existing["items"] = int(split_items)
                            if isinstance(split_size, (int, float)):
                                existing["size"] = float(split_size)
                            split_stats[name] = existing

                    if num_items is None and total_rows > 0:
                        num_items = total_rows
                    if size_bytes is None and total_bytes > 0:
                        size_bytes = total_bytes
            except Exception:
                pass

            return size_bytes, num_items, split_stats
        except Exception:
            return None, None, {}

    def _build_datasets(
        self,
        emissions_data: dict[str, Any],
        job_config: dict[str, Any],
        is_vision: bool,
        task_stage: str,
    ) -> list[dict[str, Any]]:
        """Build dataset section per BoAmps dataset_schema.json.

        Includes:
        - dataUsage (required): input or output
        - dataType (required): text, image, tabular, etc.
        - dataFormat: json, csv, parquet, etc.
        - dataSize: size in GB
        - dataQuantity: number of samples
        - shape: shape of data (e.g., for images)
        - source: public, private, other
        - sourceUri: URI to dataset
        - owner: dataset owner
        """
        datasets = []

        # Infer vision from emissions data if not set
        if not is_vision:
            job_id = emissions_data.get("job_id", "")
            output_dir = emissions_data.get("output_dir", "")
            if "vision" in job_id.lower() or "vision" in output_dir.lower():
                is_vision = True
            # Also check for VL (Vision-Language) models
            model_name = emissions_data.get("model_name", "") or job_config.get("base_model", "")
            if "-vl-" in model_name.lower() or "-vl." in model_name.lower():
                is_vision = True

        # Determine data type - vision models use multimodal data
        if is_vision:
            # Vision-language models process both text and images
            primary_data_type = "image"  # Primary modality
        else:
            primary_data_type = "text"

        # Training/input dataset
        if "dataset_path" in job_config:
            source_type = "public" if job_config.get("from_hub", False) else "private"
            dataset_path = job_config["dataset_path"]

            # Used for later fallback enrichment
            computed_size_bytes = None
            computed_items = None

            # Try fetching Hub metadata for public datasets when explicit stats are missing
            hub_split_stats: dict[str, dict[str, float | int | None]] = {}
            hub_total_size = None
            hub_total_items = None
            if source_type == "public":
                hub_size, hub_items, hub_split_stats = self._fetch_hf_dataset_metadata(dataset_path)
                hub_total_size = hub_size
                hub_total_items = hub_items
                if hub_total_items is None and hub_split_stats:
                    # Derive total items from split stats if missing
                    total_items_from_splits = sum(
                        int(v["items"]) for v in hub_split_stats.values() if v.get("items")
                    )
                    if total_items_from_splits > 0:
                        hub_total_items = total_items_from_splits
                if hub_total_size is None and hub_split_stats:
                    # Derive total size from split stats if sizes exist
                    total_size_from_splits = sum(
                        float(v["size"]) for v in hub_split_stats.values() if v.get("size")
                    )
                    if total_size_from_splits > 0:
                        hub_total_size = total_size_from_splits
                if hub_size:
                    computed_size_bytes = hub_size
                if hub_items:
                    computed_items = hub_items
                # Prefer train split-specific stats when available
                for name in ["train", "training", "train_split", "train_split_0"]:
                    split_stat = hub_split_stats.get(name)
                    if split_stat:
                        split_items = split_stat.get("items")
                        split_size = split_stat.get("size")
                        if split_size:
                            computed_size_bytes = float(split_size)
                        elif split_items and hub_total_size and hub_total_items:
                            # Approximate split size proportionally to item counts
                            computed_size_bytes = (
                                float(hub_total_size) * float(split_items) / float(hub_total_items)
                            )
                        if split_items:
                            computed_items = int(split_items)
                        break

            dataset_entry: dict[str, Any] = {
                "dataUsage": "input",
                "dataType": primary_data_type,
                "source": source_type,
                "subset": "train",
            }

            # Add source URI
            if source_type == "public" and "/" in dataset_path:
                # HuggingFace dataset
                dataset_entry["sourceUri"] = f"https://huggingface.co/datasets/{dataset_path}"
            else:
                dataset_entry["sourceUri"] = dataset_path

            # Determine data format and file type
            # Determine data format and file type
            file_type = None
            data_format = None

            # 1) From explicit path extension
            file_type, data_format = self._infer_file_type_from_name(dataset_path)

            # 2) If public and still unknown, infer from Hub file list
            if source_type == "public" and not file_type:
                inferred_type, inferred_format = self._infer_hf_file_type(dataset_path)
                file_type = inferred_type or file_type
                data_format = inferred_format or data_format

            # 3) Fallback to generic detect, then "other"
            if not file_type:
                file_type = self._detect_file_type(dataset_path, source_type)
            if not data_format:
                # Map file_type to a sensible dataFormat when possible
                mapping = {
                    "json": "json",
                    "csv": "csv",
                    "parquet": "parquet",
                    "txt": "txt",
                }
                data_format = mapping.get(file_type)

            dataset_entry["fileType"] = file_type or "other"
            if data_format:
                dataset_entry["dataFormat"] = data_format

            # If still unknown for public HF datasets, default to json/jsonl
            if source_type == "public" and dataset_entry["fileType"] == "other":
                dataset_entry["fileType"] = "json"
                dataset_entry.setdefault("dataFormat", "json")

            # Add dataset size info if available
            size_bytes = 0.0
            if "dataset_size" in job_config:
                size_bytes = _as_number(job_config["dataset_size"], 0.0)
            elif source_type == "private":
                try:
                    p = Path(dataset_path)
                    if p.exists() and p.is_file():
                        size_bytes = float(p.stat().st_size)
                except Exception:
                    pass

            if size_bytes <= 0:
                # Prefer Hub-derived size if available; otherwise scan local
                if computed_size_bytes:
                    size_bytes = computed_size_bytes
                else:
                    computed_size_bytes, computed_items = self._compute_dataset_stats(dataset_path)
                    if computed_size_bytes:
                        size_bytes = computed_size_bytes

            if size_bytes > 0:
                dataset_entry["dataSize"] = round(size_bytes / (1024**3), 4)
                dataset_entry["volume"] = size_bytes
                dataset_entry["volumeUnit"] = "byte"

            # Add number of samples if available (dataQuantity per schema)
            if "dataset_num_samples" in job_config:
                samples = _as_number(job_config["dataset_num_samples"], 0)
                if samples > 0:
                    dataset_entry["dataQuantity"] = int(samples)
                    dataset_entry["items"] = int(samples)
            elif "num_samples" in job_config:
                samples = _as_number(job_config["num_samples"], 0)
                if samples > 0:
                    dataset_entry["dataQuantity"] = int(samples)
                    dataset_entry["items"] = int(samples)
            elif computed_items:
                dataset_entry["dataQuantity"] = int(computed_items)
                dataset_entry["items"] = int(computed_items)

            # Add shape info for vision datasets
            if is_vision and "image_size" in job_config:
                img_size = job_config["image_size"]
                if isinstance(img_size, (list, tuple)) and len(img_size) >= 2:
                    dataset_entry["shape"] = f"({img_size[0]}, {img_size[1]})"
                elif isinstance(img_size, int):
                    dataset_entry["shape"] = f"({img_size}, {img_size})"

            # Add owner if from Hub (extract from path)
            if source_type == "public" and "/" in dataset_path:
                owner = dataset_path.split("/")[0]
                dataset_entry["owner"] = owner

            datasets.append(dataset_entry)

        # Validation dataset if present
        val_path = job_config.get("validation_dataset_path")
        if val_path:
            source_type = "public" if job_config.get("validation_from_hub", False) else "private"

            computed_size_bytes = None
            computed_items = None
            hub_split_stats: dict[str, dict[str, float | int | None]] = {}
            hub_total_size = None
            hub_total_items = None

            if source_type == "public":
                hub_size, hub_items, hub_split_stats = self._fetch_hf_dataset_metadata(val_path)
                hub_total_size = hub_size
                hub_total_items = hub_items
                if hub_total_items is None and hub_split_stats:
                    total_items_from_splits = sum(
                        int(v["items"]) for v in hub_split_stats.values() if v.get("items")
                    )
                    if total_items_from_splits > 0:
                        hub_total_items = total_items_from_splits
                if hub_total_size is None and hub_split_stats:
                    total_size_from_splits = sum(
                        float(v["size"]) for v in hub_split_stats.values() if v.get("size")
                    )
                    if total_size_from_splits > 0:
                        hub_total_size = total_size_from_splits
                if hub_size:
                    computed_size_bytes = hub_size
                if hub_items:
                    computed_items = hub_items
                # Prefer validation split-specific stats when available
                for name in [
                    "validation",
                    "valid",
                    "val",
                    "eval",
                    "validation_split",
                    "validation_split_0",
                ]:
                    split_stat = hub_split_stats.get(name)
                    if split_stat:
                        split_items = split_stat.get("items")
                        split_size = split_stat.get("size")
                        if split_size:
                            computed_size_bytes = float(split_size)
                        elif split_items and hub_total_size and hub_total_items:
                            computed_size_bytes = (
                                float(hub_total_size) * float(split_items) / float(hub_total_items)
                            )
                        if split_items:
                            computed_items = int(split_items)
                        break

            val_entry: dict[str, Any] = {
                "dataUsage": "input",  # Validation is input data
                "dataType": primary_data_type,
                "source": source_type,
                "subset": "validation",
            }

            if source_type == "public" and "/" in val_path:
                val_entry["sourceUri"] = f"https://huggingface.co/datasets/{val_path}"
            else:
                val_entry["sourceUri"] = val_path

            # Determine file type/format for validation set
            file_type, data_format = self._infer_file_type_from_name(val_path)
            if source_type == "public" and not file_type:
                inferred_type, inferred_format = self._infer_hf_file_type(val_path)
                file_type = inferred_type or file_type
                data_format = inferred_format or data_format

            if not file_type:
                file_type = self._detect_file_type(val_path, source_type)
            if not data_format:
                mapping = {"json": "json", "csv": "csv", "parquet": "parquet", "txt": "txt"}
                data_format = mapping.get(file_type)

            val_entry["fileType"] = file_type or "other"
            if data_format:
                val_entry["dataFormat"] = data_format
            if source_type == "public" and val_entry["fileType"] == "other":
                val_entry["fileType"] = "json"
                val_entry.setdefault("dataFormat", "json")

            # Size and sample counts
            size_bytes = 0.0
            try:
                p = Path(val_path)
                if p.exists() and p.is_file():
                    size_bytes = float(p.stat().st_size)
            except Exception:
                pass

            if size_bytes <= 0 and computed_size_bytes:
                size_bytes = computed_size_bytes
            if size_bytes <= 0:
                computed_size_bytes, computed_items = self._compute_dataset_stats(val_path)
                if computed_size_bytes:
                    size_bytes = computed_size_bytes

            if size_bytes > 0:
                val_entry["dataSize"] = round(size_bytes / (1024**3), 4)
                val_entry["volume"] = size_bytes
                val_entry["volumeUnit"] = "byte"

            if computed_items:
                val_entry["dataQuantity"] = int(computed_items)
                val_entry["items"] = int(computed_items)

            if source_type == "public" and "/" in val_path:
                val_entry["owner"] = val_path.split("/")[0]

            datasets.append(val_entry)

        # For inference, add output dataset description
        if task_stage == "inference":
            # Input dataset (prompts)
            input_entry: dict[str, Any] = {
                "dataUsage": "input",
                "dataType": "text",
                "source": "private",
            }

            inference_props = {}
            if "prompt_tokens" in job_config:
                inference_props["queryTokens"] = job_config["prompt_tokens"]

            # If we have num_requests, we can estimate queryLength if we assume tokens ~= chars/4
            # But better to just include what we know.

            if inference_props:
                input_entry["inferenceProperties"] = [inference_props]

            if "num_requests" in job_config:
                input_entry["dataQuantity"] = job_config["num_requests"]
                input_entry["items"] = job_config["num_requests"]

            datasets.append(input_entry)

            # Output dataset (completions)
            output_entry: dict[str, Any] = {
                "dataUsage": "output",
                "dataType": "text",  # LLM/VLM outputs are text
            }
            if "num_requests" in job_config:
                output_entry["dataQuantity"] = job_config["num_requests"]
                output_entry["items"] = job_config["num_requests"]
            datasets.append(output_entry)

        # Ensure at least one dataset entry (required by schema)
        if not datasets:
            datasets.append(
                {
                    "dataUsage": "input",
                    "dataType": primary_data_type,
                }
            )

        # Enrich the primary dataset entry with size and count metadata when available
        primary_entry = datasets[0]

        dataset_path = job_config.get("dataset_path")
        if dataset_path:
            source_type = "public" if job_config.get("from_hub", False) else "private"
            primary_entry.setdefault("dataUsage", "input")
            primary_entry.setdefault("dataType", primary_data_type)
            primary_entry.setdefault("source", source_type)

            if source_type == "public" and "/" in dataset_path:
                primary_entry.setdefault(
                    "sourceUri", f"https://huggingface.co/datasets/{dataset_path}"
                )
                primary_entry.setdefault("owner", dataset_path.split("/")[0])
            else:
                primary_entry.setdefault("sourceUri", dataset_path)

            primary_entry.setdefault("fileType", self._detect_file_type(dataset_path, source_type))

            # Keep dataFormat for backward compatibility
            if dataset_path.endswith(".jsonl") or dataset_path.endswith(".json"):
                primary_entry.setdefault("dataFormat", "json")
            elif dataset_path.endswith(".csv"):
                primary_entry.setdefault("dataFormat", "csv")
            elif dataset_path.endswith(".parquet"):
                primary_entry.setdefault("dataFormat", "parquet")
            elif source_type == "public":
                primary_entry.setdefault("dataFormat", "parquet")

        size_bytes = _as_number(job_config.get("dataset_size", 0.0), 0.0)
        if size_bytes > 0 and "dataSize" not in primary_entry:
            primary_entry["dataSize"] = round(size_bytes / (1024**3), 4)
            primary_entry["volume"] = size_bytes
            primary_entry["volumeUnit"] = "byte"

        samples = _as_number(
            job_config.get("dataset_num_samples") or job_config.get("num_samples"), 0
        )
        if samples > 0 and "dataQuantity" not in primary_entry:
            primary_entry["dataQuantity"] = int(samples)
            primary_entry["items"] = int(samples)

        return datasets

    def _build_task_description(
        self,
        emissions_data: dict[str, Any],
        job_config: dict[str, Any],
        task_stage: str,
        is_vision: bool,
    ) -> str:
        """Build a descriptive task description for the report."""
        parts = []

        model_name = emissions_data.get("model_name") or job_config.get("base_model", "")

        # Try to infer model name from paths if not available
        if not model_name or model_name == "Unknown":
            output_dir = emissions_data.get("output_dir", "")
            job_id = emissions_data.get("job_id", "")
            path_hints = (output_dir + " " + job_id).lower()

            if "qwen" in path_hints:
                if "3b" in path_hints:
                    model_name = "Qwen-3B"
                elif "7b" in path_hints:
                    model_name = "Qwen-7B"
                elif "8b" in path_hints:
                    model_name = "Qwen-8B"
                else:
                    model_name = "Qwen"
            elif "llama" in path_hints:
                model_name = "LLaMA"
            elif "nanonets" in path_hints:
                model_name = "Nanonets-OCR"

        if task_stage == "finetuning":
            if is_vision:
                parts.append("Vision-language model fine-tuning")
            else:
                parts.append("Language model fine-tuning")

            if model_name:
                parts.append(f"using {model_name}")

            if job_config.get("lora_config"):
                lora = job_config["lora_config"]
                r = lora.get("r", "")
                alpha = lora.get("lora_alpha", "")
                if r and alpha:
                    parts.append(f"with LoRA (r={r}, alpha={alpha})")
                else:
                    parts.append("with LoRA adapter")

            dataset_path = job_config.get("dataset_path", "")
            if dataset_path:
                parts.append(f"on dataset: {dataset_path}")

        elif task_stage == "inference":
            if is_vision:
                parts.append("Vision-language model inference")
            else:
                parts.append("Language model inference")

            if model_name:
                parts.append(f"using {model_name}")

        elif task_stage == "training":
            if is_vision:
                parts.append("Vision-language model training")
            else:
                parts.append("Language model training")

            if model_name:
                parts.append(f"({model_name})")

        return " ".join(parts) if parts else ""

    def _generate_measures(self, emissions_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate measures section with real hardware utilization data."""
        hardware = get_hardware_detector()

        # Calculate actual utilization from power consumption
        cpu_util = None
        gpu_util = None

        cpu_power = _as_number(emissions_data.get("cpu_power_watts", 0), 0.0)
        gpu_power = _as_number(emissions_data.get("gpu_power_watts", 0), 0.0)

        # Get actual hardware max power specs for accurate utilization
        cpu_info = hardware.get_cpu_info()
        gpu_info = hardware.get_gpu_info()

        # Estimate max power based on hardware (more accurate than fixed values)
        if cpu_power > 0:
            # Typical CPU TDP ranges: 65-125W for consumer, 150-280W for server
            cpu_max_power = 200.0  # Conservative default
            if "Xeon" in cpu_info.get("family", "") or "EPYC" in cpu_info.get("family", ""):
                cpu_max_power = 280.0  # Server CPUs
            elif "i9" in cpu_info.get("family", "") or "Threadripper" in cpu_info.get("family", ""):
                cpu_max_power = 250.0  # High-end desktop
            cpu_util = min(cpu_power / cpu_max_power, 1.0)

        if gpu_power > 0 and gpu_info:
            # Get GPU-specific max power from model name
            gpu_max_power = 300.0  # Default
            primary_gpu = gpu_info.get("primary", {})
            gpu_model = primary_gpu.get("model", "").upper()

            # Known TDP values for common GPUs
            if "A100" in gpu_model:
                gpu_max_power = 400.0  # A100 PCIe/SXM
            elif "H100" in gpu_model:
                gpu_max_power = 700.0  # H100
            elif "V100" in gpu_model:
                gpu_max_power = 350.0  # V100
            elif "RTX 4090" in gpu_model:
                gpu_max_power = 450.0
            elif "RTX 4080" in gpu_model:
                gpu_max_power = 320.0
            elif "RTX 3090" in gpu_model or "RTX 4070" in gpu_model:
                gpu_max_power = 350.0
            elif "RTX 3080" in gpu_model or "RTX 3070" in gpu_model:
                gpu_max_power = 320.0

            gpu_util = min(gpu_power / gpu_max_power, 1.0)

        # Parse timestamp to BoAmps format: YYYY-MM-DD HH:MM:SS
        timestamp_str = emissions_data.get("timestamp", "")
        try:
            if isinstance(timestamp_str, str) and timestamp_str:
                dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                measurement_datetime = dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                measurement_datetime = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, AttributeError):
            measurement_datetime = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        # Determine tracking mode from emissions data
        tracking_mode = emissions_data.get("tracking_mode", "machine")

        measure = {
            "measurementMethod": "codecarbon",
            "version": "2.5.0",
            "cpuTrackingMode": tracking_mode,
            "gpuTrackingMode": "nvml" if gpu_power > 0 else "none",
            "powerConsumption": round(
                _as_number(emissions_data.get("energy_consumed_kwh", 0.0), 0.0), 6
            ),
            "measurementDuration": round(
                _as_number(emissions_data.get("duration_seconds", 0.0), 0.0), 2
            ),
            "measurementDateTime": measurement_datetime,
        }

        # Add utilization if calculated (as decimal 0-1 per BoAmps schema)
        if cpu_util is not None:
            measure["averageUtilizationCpu"] = round(cpu_util, 4)

        if gpu_util is not None:
            measure["averageUtilizationGpu"] = round(gpu_util, 4)

        return [measure]

    def _generate_infrastructure(self, emissions_data: dict[str, Any]) -> dict[str, Any]:
        """Generate infrastructure section with real component data."""
        components = []
        hardware = get_hardware_detector()

        total_energy = _as_number(emissions_data.get("energy_consumed_kwh", 0.0), 0.0)

        # Add GPU if GPU energy is present
        gpu_energy = _as_number(emissions_data.get("gpu_energy_kwh", 0.0), 0.0)
        if gpu_energy > 0:
            gpu_share = gpu_energy / total_energy if total_energy > 0 else 0
            gpu_info = hardware.get_gpu_info()

            component = {
                "componentType": "gpu",  # Required field per BoAmps schema
                "nbComponent": 1,
                "share": round(gpu_share, 4),
            }

            if gpu_info and gpu_info.get("primary"):
                primary_gpu = gpu_info["primary"]
                gpu_model = primary_gpu.get("model", "Unknown")
                component.update(
                    {
                        "componentName": f"1 x {gpu_model}",
                        "manufacturer": primary_gpu.get("manufacturer", "NVIDIA"),
                        "series": gpu_model,
                        "family": primary_gpu.get("family", "Unknown"),
                    }
                )

                # Parse memory string to integer GB (e.g., "24564 MiB" -> 24, "24 GB" -> 24)
                memory_str = primary_gpu.get("memory", "0")
                try:
                    match = re.search(r"([\d.]+)\s*(MiB|MB|GiB|GB)?", memory_str)
                    if match:
                        value = float(match.group(1))
                        unit = match.group(2) if match.group(2) else "GB"

                        # Convert to GB
                        if unit in ["MiB", "MB"]:
                            memory_gb = int(value / 1024)
                        else:
                            memory_gb = int(value)

                        if memory_gb > 0:
                            component["memorySize"] = memory_gb
                except (ValueError, TypeError, AttributeError):
                    pass
            else:
                component.update(
                    {
                        "componentName": "1 x NVIDIA GPU",
                        "manufacturer": "NVIDIA",
                        "family": "Unknown",
                        "series": "Unknown",
                    }
                )

            components.append(component)

        # Add CPU
        cpu_energy = _as_number(emissions_data.get("cpu_energy_kwh", 0.0), 0.0)
        if cpu_energy > 0:
            cpu_share = cpu_energy / total_energy if total_energy > 0 else 0
            cpu_info = hardware.get_cpu_info()

            component = {
                "componentType": "cpu",  # Required field per BoAmps schema
                "nbComponent": 1,
                "share": round(cpu_share, 4),
            }

            cpu_manufacturer = cpu_info.get("manufacturer") or "Unknown"

            if cpu_manufacturer != "Unknown":
                cpu_model = cpu_info.get("model", "Unknown")
                component.update(
                    {
                        "componentName": cpu_model,
                        "manufacturer": cpu_manufacturer,
                        "series": cpu_model,
                        "family": cpu_info.get("family", "Unknown"),
                    }
                )
            else:
                component["componentName"] = "Unknown CPU"
                component["manufacturer"] = "Unknown"

            components.append(component)

        # Add RAM
        ram_energy = _as_number(emissions_data.get("ram_energy_kwh", 0.0), 0.0)
        if ram_energy > 0:
            ram_share = ram_energy / total_energy if total_energy > 0 else 0
            ram_info = hardware.get_ram_info()

            component = {
                "componentType": "ram",  # Required field per BoAmps schema
                "nbComponent": 1,
                "share": round(ram_share, 4),
            }

            # Add memory size as integer in GB
            ram_total = _as_number(ram_info.get("total_gb", 0), 0)
            if ram_total > 0:
                component["memorySize"] = int(ram_total)

            components.append(component)

        # Ensure at least one component (required by schema)
        if not components:
            components.append(
                {
                    "componentType": "cpu",
                    "nbComponent": 1,
                }
            )

        # Note: Removed custom fields not in BoAmps schema
        # (energyConsumption, unit, totalEnergyConsumption, totalEnergyUnit)

        infra_type = self.infra_type_default or "onPremise"
        return {"infraType": infra_type, "components": components}

    def _generate_system(self, emissions_data: dict[str, Any]) -> dict[str, Any]:
        """Generate system section with real OS information."""
        hardware = get_hardware_detector()
        system_info = hardware.get_system_info()

        # BoAmps v1.1.0 compliant field names
        system_data = {
            "os": system_info.get("os_name", "Linux"),  # Fixed: was "osName"
            "distributionVersion": system_info.get(
                "os_version", "Unknown"
            ),  # Fixed: was "osVersion"
        }

        # Add distribution info if available (Linux)
        if "os_distribution" in system_info:
            system_data["distribution"] = system_info[
                "os_distribution"
            ]  # Fixed: was "osDistribution"

        # Note: Removed architecture and pythonVersion (not in BoAmps schema)
        # Python version should go in Software section

        return system_data

    def _generate_software(self, emissions_data: dict[str, Any]) -> dict[str, Any]:
        """Generate software section."""
        hardware = get_hardware_detector()
        system_info = hardware.get_system_info()

        # BoAmps v1.1.0 compliant field names
        software_data = {
            "language": "Python",  # Fixed: was "programmingLanguage"
        }

        # Add Python version if available
        if "python_version" in system_info:
            software_data["version"] = system_info["python_version"]

        # Note: Removed framework and library (not in BoAmps schema)
        # These should be documented in the algorithm section instead

        return software_data

    def _generate_environment(self, emissions_data: dict[str, Any]) -> dict[str, Any]:
        """Generate environment section with real location and carbon intensity data."""
        # Use actual data from CodeCarbon
        country_name = emissions_data.get("country_name", self.location_country_default or "USA")
        region = emissions_data.get("region", self.location_region_default or "Unknown")
        carbon_intensity = _as_number(emissions_data.get("carbon_intensity_g_per_kwh", 0.0), 0.0)

        # If carbon intensity is 0, try to calculate it from emissions and energy
        if carbon_intensity == 0.0:
            emissions_kg = _as_number(emissions_data.get("emissions_kg_co2", 0.0), 0.0)
            energy_kwh = _as_number(emissions_data.get("energy_consumed_kwh", 0.0), 0.0)
            if energy_kwh > 0 and emissions_kg > 0:
                # carbon_intensity (g/kWh) = emissions (kg) * 1000 / energy (kWh)
                carbon_intensity = (emissions_kg * 1000) / energy_kwh

        # If still 0, use default values based on country
        if carbon_intensity == 0.0:
            # Default carbon intensities by country (g CO2/kWh) - approximate 2024 values
            country_defaults = {
                "France": 56.0,  # Mostly nuclear
                "USA": 380.0,
                "United States": 380.0,
                "Germany": 350.0,
                "United Kingdom": 200.0,
                "Canada": 130.0,
                "China": 540.0,
                "Japan": 470.0,
                "Australia": 510.0,
            }
            carbon_intensity = country_defaults.get(country_name, 240.0)  # World average fallback

        # Convert country name to ISO code (simple mapping for common ones)
        country_code_map = {
            "USA": "US",
            "United States": "US",
            "France": "FR",
            "Germany": "DE",
            "United Kingdom": "GB",
            "Canada": "CA",
            "China": "CN",
            "Japan": "JP",
            "Australia": "AU",
        }
        country_code = country_code_map.get(country_name, country_name[:2].upper())

        return {
            "country": country_code,
            "location": region if region != "Unknown" else country_code,
            "powerSupplierType": "public",
            "powerSourceCarbonIntensity": round(carbon_intensity, 2),
        }

    def _estimate_quality(self, emissions_data: dict[str, Any]) -> str:
        """
        Estimate the quality of the report based on available data.

        Per BoAmps spec:
        - high: percentage error +/-10%
        - medium: percentage error +/-25%
        - low: percentage error +/-50%

        Quality is determined by:
        - Tracking method (hardware-based vs constant)
        - Completeness of data (CPU, GPU, RAM energy breakdown)
        - Duration of measurement
        """
        # Check what data we have available
        has_gpu_data = _as_number(emissions_data.get("gpu_energy_kwh", 0), 0) > 0
        has_cpu_data = _as_number(emissions_data.get("cpu_energy_kwh", 0), 0) > 0
        has_ram_data = _as_number(emissions_data.get("ram_energy_kwh", 0), 0) > 0
        has_duration = _as_number(emissions_data.get("duration_seconds", 0), 0) > 0
        has_power_data = (
            _as_number(emissions_data.get("gpu_power_watts", 0), 0) > 0
            or _as_number(emissions_data.get("cpu_power_watts", 0), 0) > 0
        )
        tracking_mode = emissions_data.get("tracking_mode", "constant")

        # Determine if we have accurate hardware-based tracking
        # "process" mode with NVML/RAPL data is accurate
        # "machine" mode is also accurate
        is_accurate_tracking = tracking_mode in ["process", "machine", "nvml", "rapl"]

        if is_accurate_tracking:
            # High quality: Have GPU+CPU+RAM breakdown with power data
            if has_gpu_data and has_cpu_data and has_ram_data and has_duration and has_power_data:
                return "high"
            # Medium quality: Have GPU and CPU data with duration
            elif has_duration and has_gpu_data and has_cpu_data:
                return "medium"
            # Medium quality: Have GPU or CPU data with accurate tracking
            elif has_duration and (has_gpu_data or has_cpu_data):
                return "medium"

        # Low quality: Constant mode or missing key data
        return "low"

    def save_report(self, report: dict[str, Any], output_path: Path) -> None:
        """Save report to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)


def get_boamps_generator() -> BoAmpsReportGenerator:
    """Get a configured BoAmps report generator."""
    settings = _load_report_settings()
    return BoAmpsReportGenerator(
        publisher_name=settings.get("publisher_name") or "Model Garden",
        publisher_division=settings.get("division"),
        default_project_name=settings.get("default_project_name"),
        infra_type_default=settings.get("infra_type"),
        location_country_default=settings.get("location_country"),
        location_region_default=settings.get("location_region"),
        confidentiality_level="public",
    )


def _normalize_path(path_like: Any) -> Path | None:
    """Normalize a filesystem path, returning None if invalid."""
    if not path_like:
        return None
    try:
        return Path(path_like).expanduser().resolve()
    except Exception:
        return None


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse ISO timestamp to datetime, returning None on failure."""
    if not value:
        return None
    try:
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    return None


def _select_training_job(
    emissions_data: dict[str, Any] | None, training_jobs: dict[str, Any]
) -> dict[str, Any] | None:
    """Select a training job deterministically using strong signals only.

    Matching order (all are exact matches, no fuzzy heuristics):
    1) Exact job_id match against keys or embedded id
    2) Exact resolved output_dir match
    3) Combined strong identity signals (name/base_model) when unique

    A match is accepted only if its score clears a strict threshold and is unique
    at that score to avoid accidental pairings.
    """

    if not training_jobs:
        return None

    job_id = emissions_data.get("job_id") if emissions_data else None
    job_type = (emissions_data.get("job_type") or "").lower() if emissions_data else ""
    emitted_model_name = (emissions_data.get("model_name") or "").strip()
    emitted_base_model = (emissions_data.get("base_model") or "").strip()

    # Strongest: direct key match
    if job_id and job_id in training_jobs:
        return training_jobs[job_id]

    # Strongest: embedded id field match
    if job_id:
        for candidate in training_jobs.values():
            if candidate.get("id") == job_id:
                return candidate

    # Prepare emitted hints
    emitted_output_raw = emissions_data.get("output_dir") if emissions_data else None
    emitted_output = _normalize_path(emitted_output_raw) if emissions_data else None

    # Heuristic: derive model name hint from raw output_dir (without resolving ..)
    # Example: /.../models/qwen-7b/../logs/<job> -> hint = qwen-7b
    model_name_hint = None
    if emitted_output_raw:
        try:
            raw_parts = Path(emitted_output_raw).parts
            if "models" in raw_parts:
                idx = raw_parts.index("models")
                if len(raw_parts) > idx + 1 and raw_parts[idx + 1] not in ["..", "."]:
                    model_name_hint = raw_parts[idx + 1]
        except Exception:
            model_name_hint = None
    # Derive a likely training output dir from log paths (e.g., .../logs/<job> -> .../<model_name>)
    derived_output = None
    if emitted_output and emitted_output.parent.name == "logs" and emitted_model_name:
        derived_output = emitted_output.parent.parent / emitted_model_name

    candidates: list[tuple[int, datetime, dict[str, Any]]] = []

    for candidate in training_jobs.values():
        score = 0

        cand_output = _normalize_path(candidate.get("output_dir"))
        if emitted_output and cand_output:
            if emitted_output == cand_output:
                score += 90  # exact output dir match
            else:
                try:
                    if emitted_output.is_relative_to(cand_output) or cand_output.is_relative_to(
                        emitted_output
                    ):
                        score += 60  # nested relationship
                except Exception:
                    pass

        if derived_output and cand_output and derived_output == cand_output:
            score += 85  # derived training dir match from logs path

        if emitted_model_name and candidate.get("name") == emitted_model_name:
            score += 40

            # Vision-specific boost to rescue VL training jobs lacking job_id
            if job_type.startswith("vision") or candidate.get("is_vision"):
                score += 25

        if model_name_hint and candidate.get("name") == model_name_hint:
            score += 70

        if emitted_base_model and candidate.get("base_model") == emitted_base_model:
            score += 30

        # Match on model name appearing inside emitted output path
        candidate_name = candidate.get("name")
        if emitted_output and candidate_name:
            if candidate_name in emitted_output.as_posix():
                score += 20

        if score > 0:
            ts_hint = (
                _parse_timestamp(candidate.get("completed_at"))
                or _parse_timestamp(candidate.get("started_at"))
                or _parse_timestamp(candidate.get("created_at"))
                or datetime.min.replace(tzinfo=UTC)
            )
            candidates.append((score, ts_hint, candidate))

    if not candidates:
        return None

    # Choose highest score, breaking ties by most recent timestamp
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_score, _, best_job = candidates[0]

    if best_score < 50:
        return None  # Require reasonable evidence

    return best_job


def build_boamps_job_config(
    job_id: str | None,
    storage: Any | None = None,
    emissions_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build BoAmps job_config by enriching from stored training jobs.

    This centralizes how dataset and hyperparameter metadata are surfaced in
    BoAmps reports for both the API (WebUI) and CLI paths. If training job
    metadata is missing or storage is unavailable, returns an empty config.
    """

    if not job_id:
        return {}

    try:
        # Prefer provided storage (lets API tests inject a mock)
        if storage is None:
            from model_garden.api.storage import get_storage_manager

            storage = get_storage_manager()

        training_jobs = storage.load_training_jobs()
    except Exception:
        return {}

    job = _select_training_job(emissions_data, training_jobs)

    if not job:
        return {}

    job_config: dict[str, Any] = {
        # Core model info
        "base_model": job.get("base_model"),
        "model_type": job.get("model_type"),
        "is_vision": job.get("is_vision", False),
        # Dataset info
        "dataset_path": job.get("dataset_path"),
        "from_hub": job.get("from_hub", False),
        "validation_dataset_path": job.get("validation_dataset_path"),
        "validation_from_hub": job.get("validation_from_hub", False),
        "dataset_size": job.get("dataset_size"),
        "dataset_num_samples": job.get("dataset_num_samples"),
        # Training config
        "hyperparameters": job.get("hyperparameters", {}),
        "lora_config": job.get("lora_config"),
        "selective_loss": job.get("selective_loss", False),
        "max_seq_length": job.get("max_seq_length"),
        "save_method": job.get("save_method"),
        # Progress/metrics
        "current_step": job.get("current_step"),
        "total_steps": job.get("total_steps"),
        "current_epoch": job.get("current_epoch"),
    }

    # Extract dataset stats from metrics if present
    metrics = job.get("metrics", {})
    if metrics:
        training_metrics = metrics.get("training", [])
        if training_metrics:
            hyperparams = job.get("hyperparameters", {})
            batch_size = hyperparams.get("batch_size") or hyperparams.get(
                "per_device_train_batch_size", 1
            )
            grad_accum = hyperparams.get("gradient_accumulation_steps", 1)
            total_steps = job.get("total_steps", 0)
            epochs = hyperparams.get("num_epochs") or hyperparams.get("num_train_epochs", 1)

            if total_steps and epochs:
                estimated_samples = int((total_steps * batch_size * grad_accum) / epochs)
                if estimated_samples > 0:
                    job_config["dataset_num_samples"] = (
                        job_config.get("dataset_num_samples") or estimated_samples
                    )

        # Capture final loss if available for context
        if metrics.get("training"):
            last_metric = metrics["training"][-1]
            if "loss" in last_metric:
                job_config["final_loss"] = last_metric["loss"]

    return job_config
