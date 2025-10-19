"""BoAmps report generator for standardized emissions reporting."""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from .hardware_detection import get_hardware_detector


class BoAmpsReportGenerator:
    """Generate BoAmps-compliant emissions reports from CodeCarbon data."""
    
    BOAMPS_VERSION = "1.1.0"
    LICENSING = "Creative Commons 4.0"
    
    def __init__(
        self,
        publisher_name: str = "Model Garden",
        publisher_division: Optional[str] = None,
        confidentiality_level: str = "public"
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
        emissions_data: Dict[str, Any],
        job_config: Optional[Dict[str, Any]] = None,
        report_status: str = "final"
    ) -> Dict[str, Any]:
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
            "environment": self._generate_environment(emissions_data)
        }
    
    def _generate_header(
        self,
        emissions_data: Dict[str, Any],
        report_status: str
    ) -> Dict[str, Any]:
        """Generate header section."""
        return {
            "licensing": self.LICENSING,
            "formatVersion": self.BOAMPS_VERSION,
            "reportId": emissions_data.get("job_id", str(uuid.uuid4())),
            "reportDatetime": emissions_data.get(
                "timestamp",
                datetime.utcnow().isoformat() + "Z"
            ),
            "reportStatus": report_status,
            "publisher": {
                "name": self.publisher_name,
                "division": self.publisher_division,
                "projectName": "Model Garden",
                "confidentialityLevel": self.confidentiality_level
            }
        }
    
    def _generate_task(
        self,
        emissions_data: Dict[str, Any],
        job_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate task section with comprehensive job configuration."""
        job_type = emissions_data.get("job_type", "training")
        
        # Determine task stage and family
        task_stage = {
            "training": "training",
            "inference": "inference"
        }.get(job_type, "finetuning")
        
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
            framework_version = torch.__version__.split('+')[0]  # Remove CUDA suffix
        except Exception:
            pass
        
        # Base algorithm structure (BoAmps v1.1.0 compliant)
        algorithm = {
            "algorithmName": model_name,
            "framework": "PyTorch",
            "frameworkVersion": framework_version
        }
        
        # Build hyperparameters object (BoAmps compliant structure)
        hyperparameters_list = []
        
        # Add training-specific hyperparameters
        if task_stage in ["training", "finetuning"]:
            job_hyperparams = job_config.get("hyperparameters", {})
            lora_config = job_config.get("lora_config", {})
            
            # Epochs
            epochs = job_hyperparams.get("num_epochs", job_hyperparams.get("epochs", 3))
            hyperparameters_list.append({
                "hyperparameterName": "epochs",
                "hyperparameterValue": str(epochs)
            })
            
            # Batch size
            if "batch_size" in job_config:
                hyperparameters_list.append({
                    "hyperparameterName": "batch_size",
                    "hyperparameterValue": str(job_config["batch_size"])
                })
            
            # Optimizer
            optimizer = job_hyperparams.get("optim", job_hyperparams.get("optimizer", "adamw_torch"))
            hyperparameters_list.append({
                "hyperparameterName": "optimizer",
                "hyperparameterValue": optimizer
            })
            
            # Learning rate
            if "learning_rate" in job_hyperparams:
                hyperparameters_list.append({
                    "hyperparameterName": "learning_rate",
                    "hyperparameterValue": str(job_hyperparams["learning_rate"])
                })
            
            # LoRA-specific parameters
            if lora_config:
                if "r" in lora_config:
                    hyperparameters_list.append({
                        "hyperparameterName": "lora_r",
                        "hyperparameterValue": str(lora_config["r"])
                    })
                if "lora_alpha" in lora_config:
                    hyperparameters_list.append({
                        "hyperparameterName": "lora_alpha",
                        "hyperparameterValue": str(lora_config["lora_alpha"])
                    })
                if "lora_dropout" in lora_config:
                    hyperparameters_list.append({
                        "hyperparameterName": "lora_dropout",
                        "hyperparameterValue": str(lora_config["lora_dropout"])
                    })
            
            # Max sequence length
            if "max_seq_length" in job_config:
                hyperparameters_list.append({
                    "hyperparameterName": "max_seq_length",
                    "hyperparameterValue": str(job_config["max_seq_length"])
                })
        
        # Add hyperparameters if any exist
        if hyperparameters_list:
            tuning_method = "standard"
            if "lora_config" in job_config:
                tuning_method = "lora"
            elif job_config.get("selective_loss", False):
                tuning_method = "selective_loss"
            
            algorithm["hyperparameters"] = {
                "tuningMethod": tuning_method,
                "values": hyperparameters_list
            }
        
        # Add quantization info if available (must be integer)
        if "lora_config" in job_config or "load_in_4bit" in job_config:
            algorithm["quantization"] = 4
        elif "load_in_8bit" in job_config:
            algorithm["quantization"] = 8
        
        algorithms.append(algorithm)
        
        # Build datasets section (note: plural "datasets")
        datasets = []
        
        # Training dataset
        if "dataset_path" in job_config:
            dataset_entry = {
                "dataUsage": "input",
                "dataType": "multiModal" if is_vision else "text",
                "fileType": "jsonl",
                "source": "huggingface" if job_config.get("from_hub", False) else "local",
                "sourceUri": job_config["dataset_path"]
            }
            datasets.append(dataset_entry)
        
        # Validation dataset if present
        if "validation_dataset_path" in job_config:
            val_dataset_entry = {
                "dataUsage": "validation",
                "dataType": "multiModal" if is_vision else "text",
                "fileType": "jsonl",
                "source": "huggingface" if job_config.get("validation_from_hub", False) else "local",
                "sourceUri": job_config["validation_dataset_path"]
            }
            datasets.append(val_dataset_entry)
        
        return {
            "taskFamily": task_family,
            "taskStage": task_stage,
            "algorithms": algorithms,
            "datasets": datasets
        }
    
    def _generate_measures(
        self,
        emissions_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
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
        
        # Parse timestamp to Unix epoch integer (BoAmps requirement)
        timestamp_str = emissions_data.get("timestamp", datetime.utcnow().isoformat() + "Z")
        try:
            if isinstance(timestamp_str, str):
                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                measurement_datetime = int(dt.timestamp())
            else:
                measurement_datetime = int(timestamp_str)
        except (ValueError, AttributeError):
            measurement_datetime = int(datetime.utcnow().timestamp())
        
        # Determine tracking mode from emissions data
        tracking_mode = emissions_data.get("tracking_mode", "machine")
        
        measure = {
            "measurementMethod": "codecarbon",
            "manufacturer": "CodeCarbon",
            "version": "2.5.0",
            "cpuTrackingMode": tracking_mode,
            "gpuTrackingMode": "nvml" if gpu_power > 0 else "none",
            "unit": "kWh",
            "powerConsumption": round(emissions_data.get("energy_consumed_kwh", 0.0), 6),
            "measurementDuration": round(emissions_data.get("duration_seconds", 0.0), 2),
            "measurementDateTime": measurement_datetime,
        }
        
        # Add utilization if calculated (as percentages 0-100)
        if cpu_util is not None:
            measure["averageUtilizationCpu"] = round(cpu_util * 100, 1)
        
        if gpu_util is not None:
            measure["averageUtilizationGpu"] = round(gpu_util * 100, 1)
        
        return [measure]
    
    def _generate_infrastructure(
        self,
        emissions_data: Dict[str, Any]
    ) -> Dict[str, Any]:
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
                "componentName": "GPU",
                "nbComponent": 1,
                "share": round(gpu_share, 4)
            }
            
            if gpu_info and gpu_info.get("primary"):
                primary_gpu = gpu_info["primary"]
                component.update({
                    "manufacturer": primary_gpu.get("manufacturer", "NVIDIA"),
                    "series": primary_gpu.get("model", "Unknown"),  # Fixed: was "model", now "series"
                    "family": primary_gpu.get("family", "Unknown")
                })
                
                # Parse memory string to integer GB (e.g., "24564 MiB" -> 24, "24 GB" -> 24)
                memory_str = primary_gpu.get("memory", "0")
                try:
                    # Extract number and unit
                    import re
                    match = re.search(r'([\d.]+)\s*(MiB|MB|GiB|GB)?', memory_str)
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
                component.update({
                    "manufacturer": "NVIDIA",
                    "family": "Unknown",
                    "series": "Unknown"  # Fixed: was "model"
                })
            
            components.append(component)
        
        # Add CPU
        cpu_energy = emissions_data.get("cpu_energy_kwh", 0)
        if cpu_energy > 0:
            cpu_share = cpu_energy / total_energy if total_energy > 0 else 0
            cpu_info = hardware.get_cpu_info()
            
            component = {
                "componentName": "CPU",
                "nbComponent": 1,
                "share": round(cpu_share, 4)
            }
            
            if cpu_info.get("manufacturer") != "Unknown":
                component.update({
                    "manufacturer": cpu_info["manufacturer"],
                    "series": cpu_info.get("model", "Unknown"),  # Fixed: was "model", now "series"
                    "family": cpu_info.get("family", "Unknown")
                })
            else:
                component["manufacturer"] = "Unknown"
            
            components.append(component)
        
        # Add RAM
        ram_energy = emissions_data.get("ram_energy_kwh", 0)
        if ram_energy > 0:
            ram_share = ram_energy / total_energy if total_energy > 0 else 0
            ram_info = hardware.get_ram_info()
            
            component = {
                "componentName": "RAM",
                "manufacturer": "Unknown",
                "nbComponent": 1,
                "share": round(ram_share, 4)
            }
            
            # Add memory size as integer in GB
            if ram_info.get("total_gb", 0) > 0:
                component["memorySize"] = int(ram_info['total_gb'])  # Fixed: now integer
            
            components.append(component)
        
        # Note: Removed custom fields not in BoAmps schema
        # (energyConsumption, unit, totalEnergyConsumption, totalEnergyUnit)
        
        return {
            "infraType": "onPremise",
            "components": components
        }
    
    def _generate_system(
        self,
        emissions_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate system section with real OS information."""
        hardware = get_hardware_detector()
        system_info = hardware.get_system_info()
        
        # BoAmps v1.1.0 compliant field names
        system_data = {
            "os": system_info.get("os_name", "Linux"),  # Fixed: was "osName"
            "distributionVersion": system_info.get("os_version", "Unknown")  # Fixed: was "osVersion"
        }
        
        # Add distribution info if available (Linux)
        if "os_distribution" in system_info:
            system_data["distribution"] = system_info["os_distribution"]  # Fixed: was "osDistribution"
        
        # Note: Removed architecture and pythonVersion (not in BoAmps schema)
        # Python version should go in Software section
        
        return system_data
    
    def _generate_software(
        self,
        emissions_data: Dict[str, Any]
    ) -> Dict[str, Any]:
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
    
    def _generate_environment(
        self,
        emissions_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate environment section with real location and carbon intensity data."""
        # Use actual data from CodeCarbon
        country_name = emissions_data.get("country_name", "USA")
        region = emissions_data.get("region", "Unknown")
        carbon_intensity = emissions_data.get("carbon_intensity_g_per_kwh", 240.0)
        
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
            "Australia": "AU"
        }
        country_code = country_code_map.get(country_name, country_name[:2].upper())
        
        return {
            "country": country_code,
            "location": region if region != "Unknown" else country_code,
            "powerSupplierType": "public",
            "powerSourceCarbonIntensity": round(carbon_intensity, 2),
            "powerSourceCarbonIntensityUnit": "gCO2eq/kWh"
        }
    
    def save_report(
        self,
        report: Dict[str, Any],
        output_path: Path
    ) -> None:
        """Save report to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)


def get_boamps_generator() -> BoAmpsReportGenerator:
    """Get a configured BoAmps report generator."""
    return BoAmpsReportGenerator(
        publisher_name="Model Garden",
        publisher_division="AI Research",
        confidentiality_level="public"
    )
