"""BoAmps report generator for standardized emissions reporting.

Implements the BoAmps v1.1.0 specification from Boavizta for standardized
reporting of AI/ML energy consumption and carbon emissions.

Reference: https://github.com/Boavizta/BoAmps
Schema: https://raw.githubusercontent.com/Boavizta/BoAmps/main/model/report_schema.json
"""

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .hardware_detection import get_hardware_detector


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
        confidentiality_level: str = "public",
    ):
        """
        Initialize BoAmps report generator.

        Args:
            publisher_name: Name of the organization
            publisher_division: Division or team name
            confidentiality_level: public|internal|confidential|secret
        """
        self.publisher_name = publisher_name
        self.publisher_division = publisher_division
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
                "projectName": "Model Garden",
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

        if source_type == "public":
            # HuggingFace datasets typically use parquet internally
            return "parquet"

        return "other"

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

            dataset_entry: dict[str, Any] = {
                "dataUsage": "input",
                "dataType": primary_data_type,
                "source": source_type,
            }

            # Add source URI
            if source_type == "public" and "/" in dataset_path:
                # HuggingFace dataset
                dataset_entry["sourceUri"] = f"https://huggingface.co/datasets/{dataset_path}"
            else:
                dataset_entry["sourceUri"] = dataset_path

            # Determine data format and file type
            dataset_entry["fileType"] = self._detect_file_type(dataset_path, source_type)

            # Keep dataFormat for backward compatibility if needed, or map from fileType
            if dataset_path.endswith(".jsonl") or dataset_path.endswith(".json"):
                dataset_entry["dataFormat"] = "json"
            elif dataset_path.endswith(".csv"):
                dataset_entry["dataFormat"] = "csv"
            elif dataset_path.endswith(".parquet"):
                dataset_entry["dataFormat"] = "parquet"
            elif source_type == "public":
                dataset_entry["dataFormat"] = "parquet"

            # Add dataset size info if available
            size_bytes = 0
            if "dataset_size" in job_config:
                size_bytes = job_config["dataset_size"]
            elif source_type == "private":
                try:
                    p = Path(dataset_path)
                    if p.exists() and p.is_file():
                        size_bytes = p.stat().st_size
                except Exception:
                    pass

            if size_bytes > 0:
                # dataSize is in GB per schema
                dataset_entry["dataSize"] = round(size_bytes / (1024**3), 4)
                # volume is in bytes
                dataset_entry["volume"] = size_bytes
                dataset_entry["volumeUnit"] = "byte"

            # Add number of samples if available (dataQuantity per schema)
            if "dataset_num_samples" in job_config:
                dataset_entry["dataQuantity"] = job_config["dataset_num_samples"]
                dataset_entry["items"] = job_config["dataset_num_samples"]
            elif "num_samples" in job_config:
                dataset_entry["dataQuantity"] = job_config["num_samples"]
                dataset_entry["items"] = job_config["num_samples"]

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
        if "validation_dataset_path" in job_config:
            source_type = "public" if job_config.get("validation_from_hub", False) else "private"
            val_path = job_config["validation_dataset_path"]

            val_entry: dict[str, Any] = {
                "dataUsage": "input",  # Validation is input data
                "dataType": primary_data_type,
                "source": source_type,
            }

            if source_type == "public" and "/" in val_path:
                val_entry["sourceUri"] = f"https://huggingface.co/datasets/{val_path}"
            else:
                val_entry["sourceUri"] = val_path

            val_entry["fileType"] = self._detect_file_type(val_path, source_type)

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

        cpu_power = emissions_data.get("cpu_power_watts", 0)
        gpu_power = emissions_data.get("gpu_power_watts", 0)

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
            "powerConsumption": round(emissions_data.get("energy_consumed_kwh", 0.0), 6),
            "measurementDuration": round(emissions_data.get("duration_seconds", 0.0), 2),
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

        total_energy = emissions_data.get("energy_consumed_kwh", 0.0)

        # Add GPU if GPU energy is present
        gpu_energy = emissions_data.get("gpu_energy_kwh", 0)
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
        cpu_energy = emissions_data.get("cpu_energy_kwh", 0)
        if cpu_energy > 0:
            cpu_share = cpu_energy / total_energy if total_energy > 0 else 0
            cpu_info = hardware.get_cpu_info()

            component = {
                "componentType": "cpu",  # Required field per BoAmps schema
                "nbComponent": 1,
                "share": round(cpu_share, 4),
            }

            if cpu_info.get("manufacturer") != "Unknown":
                cpu_model = cpu_info.get("model", "Unknown")
                component.update(
                    {
                        "componentName": cpu_model,
                        "manufacturer": cpu_info["manufacturer"],
                        "series": cpu_model,
                        "family": cpu_info.get("family", "Unknown"),
                    }
                )
            else:
                component["componentName"] = "Unknown CPU"
                component["manufacturer"] = "Unknown"

            components.append(component)

        # Add RAM
        ram_energy = emissions_data.get("ram_energy_kwh", 0)
        if ram_energy > 0:
            ram_share = ram_energy / total_energy if total_energy > 0 else 0
            ram_info = hardware.get_ram_info()

            component = {
                "componentType": "ram",  # Required field per BoAmps schema
                "nbComponent": 1,
                "share": round(ram_share, 4),
            }

            # Add memory size as integer in GB
            if ram_info.get("total_gb", 0) > 0:
                component["memorySize"] = int(ram_info["total_gb"])

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

        return {"infraType": "onPremise", "components": components}

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
        country_name = emissions_data.get("country_name", "USA")
        region = emissions_data.get("region", "Unknown")
        carbon_intensity = emissions_data.get("carbon_intensity_g_per_kwh", 0.0)

        # If carbon intensity is 0, try to calculate it from emissions and energy
        if carbon_intensity == 0.0:
            emissions_kg = emissions_data.get("emissions_kg_co2", 0.0)
            energy_kwh = emissions_data.get("energy_consumed_kwh", 0.0)
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
        has_gpu_data = emissions_data.get("gpu_energy_kwh", 0) > 0
        has_cpu_data = emissions_data.get("cpu_energy_kwh", 0) > 0
        has_ram_data = emissions_data.get("ram_energy_kwh", 0) > 0
        has_duration = emissions_data.get("duration_seconds", 0) > 0
        has_power_data = (
            emissions_data.get("gpu_power_watts", 0) > 0
            or emissions_data.get("cpu_power_watts", 0) > 0
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
    return BoAmpsReportGenerator(
        publisher_name="Model Garden",
        publisher_division="AI Research",
        confidentiality_level="public",
    )
