# BoAmps Integration - Standardized Emissions Reporting

## Overview

**BoAmps** (Boavizta AI/ML Power Measures Sharing) is an open standard for reporting energy consumption and carbon emissions of AI/ML models. Model Garden integrates BoAmps to provide standardized, interoperable emissions data that can be shared with open data repositories.

### Why BoAmps?

- **Standardization**: Common format for AI energy consumption across platforms
- **Interoperability**: Easy comparison and aggregation of emissions data
- **Open Data**: Contribute to Boavizta's open dataset on HuggingFace
- **Transparency**: Comprehensive metadata about algorithms, datasets, and hardware
- **Research**: Enable research into AI efficiency and carbon reduction

### Links

- **GitHub Repository**: https://github.com/Boavizta/BoAmps
- **Open Data Repository**: https://huggingface.co/datasets/boavizta/open_data_boamps
- **Report Creation Tool**: https://huggingface.co/spaces/boavizta/BoAmps_report_creation
- **License**: Creative Commons 4.0

---

## BoAmps Datamodel

### Report Structure

A BoAmps report is a JSON document with the following main sections:

```
BoAmps Report
├── header          # Publisher info, report metadata
├── task            # ML task description
│   ├── algorithms  # Models and algorithms used
│   └── dataset     # Input/output data descriptions
├── measures        # Energy measurements
├── infrastructure  # Hardware components
├── system          # Operating system info
├── software        # Programming environment
└── environment     # Location and power source
```

### Key Schemas

#### 1. Header Schema
```json
{
  "header": {
    "licensing": "Creative Commons 4.0",
    "formatVersion": "1.1.0",
    "reportId": "uuid4-generated",
    "reportDatetime": "2025-10-16T10:00:00",
    "reportStatus": "final|draft|corrective",
    "publisher": {
      "name": "Organization Name",
      "division": "Team/Department",
      "projectName": "Model Garden",
      "confidentialityLevel": "public|internal|confidential|secret"
    }
  }
}
```

#### 2. Task Schema
```json
{
  "task": {
    "taskFamily": "textGeneration|imageClassification|chatbot|...",
    "taskStage": "training|finetuning|inference",
    "nbRequest": 1,
    "algorithms": [
      {
        "trainingType": "supervisedlearning|transferlearning|...",
        "algorithmType": "llm|cnn|transformer|...",
        "foundationModelName": "llama-3.1-8b",
        "foundationModelUri": "https://huggingface.co/...",
        "parametersNumber": 8,
        "layersNumber": 32,
        "epochsNumber": 3,
        "quantization": "fp16|int8|...",
        "optimizer": "adam|sgd|..."
      }
    ],
    "dataset": [
      {
        "dataUsage": "input|output",
        "dataType": "text|image|audio|video|...",
        "fileType": "json|csv|parquet|...",
        "volume": 1024,
        "volumeUnit": "megabyte|gigabyte|...",
        "items": 10000,
        "shape": [512, 512],
        "source": "public|private",
        "sourceUri": "https://..."
      }
    ]
  }
}
```

#### 3. Measures Schema
```json
{
  "measures": [
    {
      "measurementMethod": "codecarbon|wattmeter|...",
      "manufacturer": "Nvidia|Intel|...",
      "version": "2.5.0",
      "cpuTrackingMode": "machine|process|rapl",
      "gpuTrackingMode": "machine|nvml|...",
      "unit": "kWh|Wh|MWh|...",
      "powerConsumption": 1.234,
      "measurementDuration": 3600.0,
      "measurementDateTime": 1234567890,
      "powerCalibrationMeasurement": 0.1,
      "durationCalibrationMeasurement": 300.0,
      "averageUtilizationCpu": 0.75,
      "averageUtilizationGpu": 0.95
    }
  ]
}
```

#### 4. Infrastructure Schema
```json
{
  "infrastructure": {
    "infraType": "onPremise|publicCloud|privateCloud",
    "cloudProvider": "aws|azure|gcp|ovh|...",
    "cloudInstance": "p3.2xlarge|Standard_NC6|...",
    "components": [
      {
        "componentName": "GPU|CPU|RAM",
        "manufacturer": "NVIDIA|Intel|AMD|...",
        "family": "A100|H100|Xeon|...",
        "series": "SXM4|PCIe|...",
        "nbComponent": 1,
        "memorySize": 80,
        "share": 1.0
      }
    ]
  }
}
```

#### 5. Environment Schema
```json
{
  "environment": {
    "country": "France",
    "latitude": 48.8566,
    "longitude": 2.3522,
    "location": "Paris",
    "powerSupplierType": "public|private|internal",
    "powerSource": "nuclear|solar|wind|coal|gas|...",
    "powerSourceCarbonIntensity": 60.0
  }
}
```

---

## Model Garden Integration

### Architecture

```
Training/Inference Job
        ↓
CodeCarbon Tracker
        ↓
    Emissions Data
        ↓
BoAmpsReportGenerator ←─ Job Config
        ↓
  BoAmps JSON Report
        ↓
    Storage + API
```

### Implementation

#### 1. Report Generator Class

```python
# model_garden/carbon/boamps_generator.py

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

class BoAmpsReportGenerator:
    """Generate BoAmps-compliant emissions reports"""
    
    BOAMPS_VERSION = "1.1.0"
    LICENSING = "Creative Commons 4.0"
    
    def __init__(
        self,
        job_id: str,
        job_config: Dict[str, Any],
        codecarbon_data: Dict[str, Any],
        hardware_info: Dict[str, Any],
        publisher_info: Optional[Dict[str, Any]] = None
    ):
        self.job_id = job_id
        self.job_config = job_config
        self.codecarbon_data = codecarbon_data
        self.hardware_info = hardware_info
        self.publisher_info = publisher_info or {}
        self.report_id = str(uuid.uuid4())
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate complete BoAmps report"""
        return {
            "header": self._build_header(),
            "task": self._build_task(),
            "measures": self._build_measures(),
            "infrastructure": self._build_infrastructure(),
            "system": self._build_system(),
            "software": self._build_software(),
            "environment": self._build_environment()
        }
    
    def _build_header(self) -> Dict[str, Any]:
        """Build report header section"""
        return {
            "licensing": self.LICENSING,
            "formatVersion": self.BOAMPS_VERSION,
            "reportId": self.report_id,
            "reportDatetime": datetime.utcnow().isoformat(),
            "reportStatus": "final",
            "publisher": {
                "name": self.publisher_info.get("name", "Model Garden"),
                "projectName": self.publisher_info.get("project", "model-garden"),
                "confidentialityLevel": self.publisher_info.get("confidentiality", "public")
            }
        }
    
    def _build_task(self) -> Dict[str, Any]:
        """Build task description section"""
        is_inference = self.job_config.get("stage") == "inference"
        
        task = {
            "taskFamily": self.job_config.get("task_family", "textGeneration"),
            "taskStage": self.job_config.get("stage", "training"),
            "algorithms": self._build_algorithms(),
            "dataset": self._build_datasets()
        }
        
        if is_inference:
            task["nbRequest"] = self.job_config.get("num_requests", 1)
        
        return task
    
    def _build_algorithms(self) -> List[Dict[str, Any]]:
        """Build algorithms section"""
        model_name = self.job_config.get("model_name", "")
        
        algo = {
            "algorithmType": "llm",
            "foundationModelName": model_name,
        }
        
        # Add optional fields if available
        if uri := self.job_config.get("model_uri"):
            algo["foundationModelUri"] = uri
        
        if params := self.job_config.get("num_parameters"):
            algo["parametersNumber"] = params
        
        if quant := self.job_config.get("quantization"):
            algo["quantization"] = quant
        
        if stage := self.job_config.get("stage"):
            if stage in ["training", "finetuning"]:
                algo["trainingType"] = self.job_config.get("training_type", "transferlearning")
                if epochs := self.job_config.get("num_epochs"):
                    algo["epochsNumber"] = epochs
                if opt := self.job_config.get("optimizer"):
                    algo["optimizer"] = opt
        
        return [algo]
    
    def _build_datasets(self) -> List[Dict[str, Any]]:
        """Build dataset section"""
        datasets = []
        
        # Input dataset
        if input_data := self.job_config.get("dataset"):
            datasets.append({
                "dataUsage": "input",
                "dataType": input_data.get("type", "text"),
                "items": input_data.get("num_examples", 0),
                "volume": input_data.get("size_mb", 0),
                "volumeUnit": "megabyte"
            })
        
        # Output dataset (for inference)
        if self.job_config.get("stage") == "inference":
            datasets.append({
                "dataUsage": "output",
                "dataType": "text",
                "items": self.job_config.get("num_requests", 1)
            })
        
        return datasets
    
    def _build_measures(self) -> List[Dict[str, Any]]:
        """Build measurements section from CodeCarbon data"""
        cc = self.codecarbon_data
        
        measure = {
            "measurementMethod": "codecarbon",
            "version": cc.get("codecarbon_version", "2.x.x"),
            "unit": "kWh",
            "powerConsumption": cc.get("energy_consumed", 0.0),
            "measurementDuration": cc.get("duration", 0.0),
            "measurementDateTime": int(datetime.utcnow().timestamp())
        }
        
        # Add tracking modes if available
        if cpu_mode := cc.get("cpu_tracking_mode"):
            measure["cpuTrackingMode"] = cpu_mode
        
        if gpu_mode := cc.get("gpu_tracking_mode"):
            measure["gpuTrackingMode"] = gpu_mode
        
        # Add utilization if available
        if cpu_util := cc.get("cpu_energy"):
            measure["averageUtilizationCpu"] = cpu_util
        
        if gpu_util := cc.get("gpu_energy"):
            measure["averageUtilizationGpu"] = gpu_util
        
        # Calibration data
        if calib := cc.get("calibration"):
            measure["powerCalibrationMeasurement"] = calib.get("power", 0.0)
            measure["durationCalibrationMeasurement"] = calib.get("duration", 0.0)
        
        return [measure]
    
    def _build_infrastructure(self) -> Dict[str, Any]:
        """Build infrastructure section"""
        hw = self.hardware_info
        
        infra = {
            "infraType": hw.get("infra_type", "onPremise"),
            "components": []
        }
        
        # Add cloud info if applicable
        if cloud_provider := hw.get("cloud_provider"):
            infra["cloudProvider"] = cloud_provider
            infra["cloudInstance"] = hw.get("cloud_instance", "")
        
        # GPU components
        if gpus := hw.get("gpus"):
            for gpu in gpus:
                infra["components"].append({
                    "componentName": "GPU",
                    "manufacturer": gpu.get("manufacturer", "NVIDIA"),
                    "family": gpu.get("family", ""),
                    "nbComponent": gpu.get("count", 1),
                    "memorySize": gpu.get("memory_gb", 0),
                    "share": 1.0
                })
        
        # CPU components
        if cpu := hw.get("cpu"):
            infra["components"].append({
                "componentName": "CPU",
                "manufacturer": cpu.get("manufacturer", ""),
                "family": cpu.get("family", ""),
                "nbComponent": cpu.get("count", 1),
                "share": 1.0
            })
        
        return infra
    
    def _build_system(self) -> Dict[str, Any]:
        """Build system info section"""
        import platform
        
        return {
            "os": platform.system(),
            "distribution": platform.release(),
            "distributionVersion": platform.version()
        }
    
    def _build_software(self) -> Dict[str, Any]:
        """Build software environment section"""
        import sys
        
        return {
            "language": "python",
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }
    
    def _build_environment(self) -> Dict[str, Any]:
        """Build environment section from CodeCarbon data"""
        cc = self.codecarbon_data
        
        env = {}
        
        if country := cc.get("country_name"):
            env["country"] = country
        
        if location := cc.get("region"):
            env["location"] = location
        
        if carbon_int := cc.get("carbon_intensity"):
            env["powerSourceCarbonIntensity"] = carbon_int
        
        return env
    
    def export_json(self, output_path: Path) -> Path:
        """Export report to JSON file"""
        report = self.generate_report()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_path
    
    def validate_schema(self) -> bool:
        """Validate report against BoAmps schema"""
        # TODO: Implement schema validation using jsonschema
        # and BoAmps schema files from GitHub
        return True
```

#### 2. Integration with Training

```python
# model_garden/training.py

from model_garden.carbon.boamps_generator import BoAmpsReportGenerator

def train_model(config: dict):
    # ... existing training code ...
    
    # After training completes
    if tracker:
        tracker.stop()
        
        # Generate BoAmps report
        boamps_gen = BoAmpsReportGenerator(
            job_id=job_id,
            job_config={
                "stage": "training",
                "model_name": config.model_name,
                "num_parameters": get_model_params(model),
                "num_epochs": config.num_epochs,
                "optimizer": config.optimizer,
                "dataset": {
                    "type": "text",
                    "num_examples": len(dataset),
                    "size_mb": dataset.size_in_bytes / 1024 / 1024
                }
            },
            codecarbon_data=tracker.final_emissions_data,
            hardware_info=get_hardware_info()
        )
        
        # Export to file
        boamps_path = f"storage/logs/{job_id}/emissions_boamps.json"
        boamps_gen.export_json(Path(boamps_path))
        
        logger.info(f"BoAmps report saved to {boamps_path}")
```

#### 3. Integration with Inference

```python
# model_garden/inference.py

def track_inference_emissions(model_id: str, num_requests: int):
    """Track emissions for inference requests"""
    
    tracker = EmissionsTracker()
    tracker.start()
    
    # ... inference code ...
    
    tracker.stop()
    
    # Generate BoAmps report
    boamps_gen = BoAmpsReportGenerator(
        job_id=f"inference-{model_id}-{timestamp}",
        job_config={
            "stage": "inference",
            "model_name": model_id,
            "num_requests": num_requests,
            "task_family": "textGeneration"
        },
        codecarbon_data=tracker.final_emissions_data,
        hardware_info=get_hardware_info()
    )
    
    boamps_path = f"storage/logs/inference/{model_id}/emissions_boamps_{timestamp}.json"
    boamps_gen.export_json(Path(boamps_path))
```

#### 4. API Endpoints

```python
# model_garden/api.py

from fastapi import APIRouter, HTTPException
from pathlib import Path

router = APIRouter(prefix="/api/v1/carbon/boamps", tags=["BoAmps"])

@router.get("/{job_id}")
async def get_boamps_report(job_id: str):
    """Get BoAmps report for a specific job"""
    boamps_path = Path(f"storage/logs/{job_id}/emissions_boamps.json")
    
    if not boamps_path.exists():
        raise HTTPException(status_code=404, detail="BoAmps report not found")
    
    with open(boamps_path) as f:
        return json.load(f)

@router.get("/")
async def list_boamps_reports():
    """List all available BoAmps reports"""
    reports = []
    logs_dir = Path("storage/logs")
    
    for boamps_file in logs_dir.rglob("emissions_boamps.json"):
        with open(boamps_file) as f:
            report = json.load(f)
            reports.append({
                "job_id": boamps_file.parent.name,
                "report_id": report["header"]["reportId"],
                "datetime": report["header"]["reportDatetime"],
                "task_stage": report["task"]["taskStage"],
                "path": str(boamps_file)
            })
    
    return reports

@router.post("/validate")
async def validate_boamps_report(report: dict):
    """Validate a BoAmps report against the schema"""
    # TODO: Implement validation using jsonschema
    return {"valid": True, "errors": []}

@router.post("/export")
async def export_to_repository(job_id: str, repository_url: str):
    """Export BoAmps report to open data repository"""
    # TODO: Implement export to Boavizta/HuggingFace
    pass
```

---

## Best Practices

### 1. Calibration Measurements

Always include calibration measurements to separate baseline power consumption from task-specific consumption:

```python
# Measure baseline before task
calibration_tracker = EmissionsTracker()
calibration_tracker.start()
time.sleep(300)  # 5 minutes baseline
calibration_tracker.stop()

# Then run actual task
task_tracker = EmissionsTracker()
task_tracker.start()
# ... run training/inference ...
task_tracker.stop()

# Include both in BoAmps report
codecarbon_data = {
    **task_tracker.final_emissions_data,
    "calibration": {
        "power": calibration_tracker.final_emissions_data["energy_consumed"],
        "duration": 300.0
    }
}
```

### 2. Hardware Details

Provide as much hardware detail as possible:

```python
def get_hardware_info() -> dict:
    """Collect detailed hardware information"""
    import torch
    import psutil
    
    hw_info = {
        "infra_type": "onPremise",
        "gpus": [],
        "cpu": {}
    }
    
    # GPU info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            hw_info["gpus"].append({
                "manufacturer": "NVIDIA",
                "family": props.name.split()[0],
                "model": props.name,
                "count": 1,
                "memory_gb": props.total_memory / (1024**3)
            })
    
    # CPU info
    hw_info["cpu"] = {
        "manufacturer": "Intel" if "Intel" in psutil.cpu_info() else "AMD",
        "count": psutil.cpu_count(logical=False)
    }
    
    return hw_info
```

### 3. Dataset Metadata

Include comprehensive dataset information:

```python
def analyze_dataset(dataset_path: Path) -> dict:
    """Extract dataset metadata for BoAmps report"""
    import datasets
    
    ds = datasets.load_dataset("json", data_files=str(dataset_path))
    
    return {
        "type": "text",
        "file_type": "json",
        "num_examples": len(ds["train"]),
        "size_mb": dataset_path.stat().st_size / (1024 * 1024),
        "source": "private",
        "sourceUri": f"file://{dataset_path}"
    }
```

### 4. Model Information

Capture complete model specifications:

```python
def get_model_info(model) -> dict:
    """Extract model information for BoAmps report"""
    
    num_params = sum(p.numel() for p in model.parameters())
    num_params_billions = num_params / 1e9
    
    return {
        "model_name": model.config._name_or_path,
        "num_parameters": round(num_params_billions, 2),
        "num_layers": model.config.num_hidden_layers,
        "quantization": get_quantization_type(model)
    }
```

---

## Schema Validation

### Installing Schema Validator

```bash
# Clone BoAmps repository
git clone https://github.com/Boavizta/BoAmps.git

# Install dependencies
cd BoAmps/tools/schema_validator
pip install -r requirements.txt
```

### Validating Reports

```python
import subprocess
import json
from pathlib import Path

def validate_boamps_report(report_path: Path) -> tuple[bool, list]:
    """Validate BoAmps report using official validator"""
    
    validator_script = Path("BoAmps/tools/schema_validator/validate-schema.py")
    
    result = subprocess.run(
        ["python", str(validator_script), str(report_path)],
        capture_output=True,
        text=True
    )
    
    is_valid = result.returncode == 0
    errors = [] if is_valid else result.stdout.split("\n")
    
    return is_valid, errors
```

---

## Contributing to Open Data

### Exporting to HuggingFace

```python
from huggingface_hub import HfApi, upload_file

def export_to_huggingface(report_path: Path, hf_token: str):
    """Upload BoAmps report to Boavizta open dataset"""
    
    api = HfApi()
    
    # Upload to Boavizta dataset
    upload_file(
        path_or_fileobj=str(report_path),
        path_in_repo=f"reports/{report_path.name}",
        repo_id="boavizta/open_data_boamps",
        repo_type="dataset",
        token=hf_token
    )
    
    print(f"Report uploaded to HuggingFace: {report_path.name}")
```

### Privacy Considerations

Before contributing to open data:

1. **Review confidentiality level** in report header
2. **Remove sensitive information** (internal paths, private datasets)
3. **Anonymize if needed** (organization names, project details)
4. **Set appropriate license** (default: Creative Commons 4.0)

```python
def anonymize_report(report: dict) -> dict:
    """Remove sensitive information from report"""
    
    # Anonymize publisher
    if "publisher" in report["header"]:
        report["header"]["publisher"]["name"] = "Anonymous"
        report["header"]["publisher"].pop("division", None)
        report["header"]["publisher"].pop("projectName", None)
    
    # Remove private dataset URIs
    for dataset in report["task"].get("dataset", []):
        if dataset.get("source") == "private":
            dataset.pop("sourceUri", None)
            dataset.pop("owner", None)
    
    # Set confidentiality to public
    report["header"]["publisher"]["confidentialityLevel"] = "public"
    
    return report
```

---

## CLI Commands

### Generate BoAmps Report

```bash
# Generate BoAmps report for a training job
model-garden boamps generate --job-id my-training-job-123

# Generate for inference tracking
model-garden boamps generate --job-id inference-llama-20251016

# Validate report
model-garden boamps validate --report-path storage/logs/job-123/emissions_boamps.json

# Export to HuggingFace
model-garden boamps export --job-id my-job --hf-token $HF_TOKEN
```

### Implementation

```python
# model_garden/cli.py

@cli.group()
def boamps():
    """BoAmps emissions reporting commands"""
    pass

@boamps.command()
@click.option("--job-id", required=True, help="Training/inference job ID")
def generate(job_id: str):
    """Generate BoAmps report for a job"""
    # Load job data, codecarbon data, generate report
    click.echo(f"Generating BoAmps report for job {job_id}...")

@boamps.command()
@click.option("--report-path", required=True, type=click.Path(exists=True))
def validate(report_path: str):
    """Validate BoAmps report"""
    is_valid, errors = validate_boamps_report(Path(report_path))
    if is_valid:
        click.echo("✓ Report is valid!")
    else:
        click.echo("✗ Report validation failed:")
        for error in errors:
            click.echo(f"  - {error}")

@boamps.command()
@click.option("--job-id", required=True)
@click.option("--hf-token", envvar="HF_TOKEN")
def export(job_id: str, hf_token: str):
    """Export report to HuggingFace open dataset"""
    report_path = Path(f"storage/logs/{job_id}/emissions_boamps.json")
    export_to_huggingface(report_path, hf_token)
    click.echo("✓ Report exported to HuggingFace!")
```

---

## Future Enhancements

### Phase 2 (Current)
- [ ] Implement `BoAmpsReportGenerator` class
- [ ] Add CodeCarbon to BoAmps conversion
- [ ] Integrate with training pipeline
- [ ] Add API endpoints for BoAmps reports
- [ ] Implement schema validation

### Phase 3
- [ ] Automatic BoAmps generation for all jobs
- [ ] Dashboard UI for viewing BoAmps reports
- [ ] Bulk export to HuggingFace
- [ ] Comparison tools for efficiency analysis
- [ ] Integration with Boavizta API

### Phase 4
- [ ] Real-time BoAmps report generation
- [ ] Custom report templates
- [ ] Multi-model aggregated reports
- [ ] Advanced anonymization tools
- [ ] Carbon efficiency recommendations based on BoAmps data

---

## References

- **BoAmps GitHub**: https://github.com/Boavizta/BoAmps
- **BoAmps Documentation**: See README in repository
- **Schema Files**: https://github.com/Boavizta/BoAmps/tree/main/model
- **Examples**: https://github.com/Boavizta/BoAmps/tree/main/examples
- **Open Dataset**: https://huggingface.co/datasets/boavizta/open_data_boamps
- **Boavizta**: https://www.boavizta.org/
- **CodeCarbon**: https://mlco2.github.io/codecarbon/
