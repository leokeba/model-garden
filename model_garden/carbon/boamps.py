"""BoAmps report generator for standardized emissions reporting."""

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .hardware_detection import get_hardware_detector


class BoAmpsReportGenerator:
    """Generate BoAmps-compliant emissions reports from CodeCarbon data."""

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
        """Generate task section with comprehensive job configuration."""
        job_type = emissions_data.get("job_type", "training")

        # Determine task stage and family
        task_stage = {"training": "training", "inference": "inference"}.get(job_type, "finetuning")

        # Determine if vision model
        is_vision = job_config.get("is_vision", False)
        task_family = "multiModalTextGeneration" if is_vision else "textGeneration"

        # Build algorithms section
        algorithms = []
        model_name = emissions_data.get("model_name") or job_config.get("base_model", "unknown")

        # Extract framework version if available
        framework_version = "2.x"  # Default
        try:
            import torch

            framework_version = torch.__version__.split("+")[0]  # Remove CUDA suffix
        except Exception:
            pass

        # Base algorithm structure (BoAmps v1.1.0 compliant)
        algorithm = {
            "algorithmName": model_name,
            "framework": "PyTorch",
            "frameworkVersion": framework_version,
        }

        # Build hyperparameters object (BoAmps compliant structure)
        hyperparameters_list = []

        # Add training-specific hyperparameters
        if task_stage in ["training", "finetuning"]:
            job_hyperparams = job_config.get("hyperparameters", {})
            lora_config = job_config.get("lora_config", {})

            # Epochs
            epochs = job_hyperparams.get("num_epochs", job_hyperparams.get("epochs", 3))
            hyperparameters_list.append(
                {"hyperparameterName": "epochs", "hyperparameterValue": str(epochs)}
            )

            # Batch size
            if "batch_size" in job_config:
                hyperparameters_list.append(
                    {
                        "hyperparameterName": "batch_size",
                        "hyperparameterValue": str(job_config["batch_size"]),
                    }
                )

            # Optimizer
            optimizer = job_hyperparams.get(
                "optim", job_hyperparams.get("optimizer", "adamw_torch")
            )
            hyperparameters_list.append(
                {"hyperparameterName": "optimizer", "hyperparameterValue": optimizer}
            )

            # Learning rate
            if "learning_rate" in job_hyperparams:
                hyperparameters_list.append(
                    {
                        "hyperparameterName": "learning_rate",
                        "hyperparameterValue": str(job_hyperparams["learning_rate"]),
                    }
                )

            # LoRA-specific parameters
            if lora_config:
                if "r" in lora_config:
                    hyperparameters_list.append(
                        {
                            "hyperparameterName": "lora_r",
                            "hyperparameterValue": str(lora_config["r"]),
                        }
                    )
                if "lora_alpha" in lora_config:
                    hyperparameters_list.append(
                        {
                            "hyperparameterName": "lora_alpha",
                            "hyperparameterValue": str(lora_config["lora_alpha"]),
                        }
                    )
                if "lora_dropout" in lora_config:
                    hyperparameters_list.append(
                        {
                            "hyperparameterName": "lora_dropout",
                            "hyperparameterValue": str(lora_config["lora_dropout"]),
                        }
                    )

            # Max sequence length
            if "max_seq_length" in job_config:
                hyperparameters_list.append(
                    {
                        "hyperparameterName": "max_seq_length",
                        "hyperparameterValue": str(job_config["max_seq_length"]),
                    }
                )

        # Add hyperparameters if any exist
        if hyperparameters_list:
            tuning_method = "standard"
            if "lora_config" in job_config:
                tuning_method = "lora"
            elif job_config.get("selective_loss", False):
                tuning_method = "selective_loss"

            algorithm["hyperparameters"] = {
                "tuningMethod": tuning_method,
                "values": hyperparameters_list,
            }

        # Add quantization info if available (must be string like "fp16", "int8", "q4")
        if "lora_config" in job_config or "load_in_4bit" in job_config:
            algorithm["quantization"] = "q4"  # 4-bit quantization
        elif "load_in_8bit" in job_config:
            algorithm["quantization"] = "int8"  # 8-bit quantization

        algorithms.append(algorithm)

        # Build dataset section (note: singular "dataset" per BoAmps schema)
        dataset = []

        # Training dataset
        if "dataset_path" in job_config:
            # Map source type to BoAmps enum: public, private, other
            source_type = "public" if job_config.get("from_hub", False) else "private"

            dataset_entry = {
                "dataUsage": "input",
                "dataType": "image" if is_vision else "text",
                "source": source_type,
                "sourceUri": job_config["dataset_path"],
            }
            # Add dataFormat if we can determine it
            dataset_path = job_config["dataset_path"]
            if dataset_path.endswith(".jsonl") or dataset_path.endswith(".json"):
                dataset_entry["dataFormat"] = "json"
            elif dataset_path.endswith(".csv"):
                dataset_entry["dataFormat"] = "csv"
            elif dataset_path.endswith(".parquet"):
                dataset_entry["dataFormat"] = "parquet"

            dataset.append(dataset_entry)

        # Validation dataset if present
        if "validation_dataset_path" in job_config:
            source_type = "public" if job_config.get("validation_from_hub", False) else "private"

            val_dataset_entry = {
                "dataUsage": "input",  # Validation is also input data per BoAmps
                "dataType": "image" if is_vision else "text",
                "source": source_type,
                "sourceUri": job_config["validation_dataset_path"],
            }
            dataset.append(val_dataset_entry)

        # Ensure at least one dataset entry (required by schema)
        if not dataset:
            dataset.append(
                {
                    "dataUsage": "input",
                    "dataType": "image" if is_vision else "text",
                }
            )

        return {
            "taskFamily": task_family,
            "taskStage": task_stage,
            "algorithms": algorithms,
            "dataset": dataset,  # Singular "dataset" per BoAmps schema
        }

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
        """
        # Check how much data we have to assess quality
        has_gpu_data = emissions_data.get("gpu_energy_kwh", 0) > 0
        has_cpu_data = emissions_data.get("cpu_energy_kwh", 0) > 0
        has_duration = emissions_data.get("duration_seconds", 0) > 0
        tracking_mode = emissions_data.get("tracking_mode", "constant")

        # NVML/RAPL tracking is more accurate than constant mode
        if tracking_mode in ["nvml", "rapl", "machine"]:
            if has_gpu_data and has_cpu_data and has_duration:
                return "high"
            elif has_duration and (has_gpu_data or has_cpu_data):
                return "medium"

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
