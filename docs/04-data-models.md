# Data Models & Schemas

## Overview
This document defines all data models and schemas used throughout Model Garden, using Pydantic for validation.

---

## 1. Model Schemas

### ModelConfig
Configuration for a fine-tuned model.

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class ModelConfig(BaseModel):
    """Model configuration details"""
    max_seq_length: int = Field(2048, ge=128, le=32768)
    quantization: str = Field("4bit", pattern="^(4bit|8bit|16bit|none)$")
    lora_r: Optional[int] = Field(None, ge=1, le=256)
    lora_alpha: Optional[int] = Field(None, ge=1, le=256)
    lora_dropout: Optional[float] = Field(None, ge=0.0, le=0.5)
    target_modules: Optional[List[str]] = None
```

### CarbonEmissions
Carbon emissions data for a model.

```python
class CarbonEmissions(BaseModel):
    """Carbon emissions tracking"""
    training_kg: float = Field(0.0, ge=0)
    training_duration_hours: float = Field(0.0, ge=0)
    training_energy_kwh: float = Field(0.0, ge=0)
    inference_kg: float = Field(0.0, ge=0)
    inference_requests: int = Field(0, ge=0)
```

### ModelMetrics
Training and evaluation metrics.

```python
class ModelMetrics(BaseModel):
    """Model performance metrics"""
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    perplexity: Optional[float] = None
    training_time_seconds: Optional[float] = None
    gpu_memory_peak_gb: Optional[float] = None
```

### Model
Complete model representation.

```python
class ModelStatus(str, Enum):
    AVAILABLE = "available"
    TRAINING = "training"
    ERROR = "error"
    LOADING = "loading"

class Model(BaseModel):
    """Model entity"""
    id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., min_length=1, max_length=200)
    base_model: str = Field(..., description="Base model identifier")
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    size_bytes: int = Field(ge=0)
    path: Optional[str] = None
    training_job_id: Optional[str] = None
    config: ModelConfig
    carbon_emissions: CarbonEmissions
    metrics: ModelMetrics
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "model-abc123",
                "name": "finance-qa-llama",
                "base_model": "meta-llama/Llama-3.1-8B-Instruct",
                "status": "available",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T14:20:00Z",
                "size_bytes": 17179869184,
                "path": "/storage/models/finetuned/job-456",
                "training_job_id": "job-456",
                "config": {
                    "max_seq_length": 2048,
                    "quantization": "4bit",
                    "lora_r": 16,
                    "lora_alpha": 16
                },
                "carbon_emissions": {
                    "training_kg": 0.45,
                    "training_duration_hours": 2.5,
                    "training_energy_kwh": 0.8
                },
                "metrics": {
                    "train_loss": 0.189,
                    "eval_loss": 0.234,
                    "perplexity": 12.5
                }
            }
        }

class ModelList(BaseModel):
    """Paginated model list"""
    items: List[Model]
    total: int
    page: int = 1
    page_size: int = 20
    pages: int
```

---

## 2. Training Schemas

### LoRAConfig
LoRA adapter configuration.

```python
class LoRAConfig(BaseModel):
    """LoRA configuration"""
    r: int = Field(16, ge=1, le=256, description="LoRA rank")
    lora_alpha: int = Field(16, ge=1, le=256)
    lora_dropout: float = Field(0.0, ge=0.0, le=0.5)
    target_modules: List[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        description="Modules to apply LoRA"
    )
    use_rslora: bool = Field(False, description="Use rank-stabilized LoRA")
```

### QuantizationConfig
Quantization settings.

```python
class QuantizationConfig(BaseModel):
    """Quantization configuration"""
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = Field("bfloat16", pattern="^(float16|bfloat16|float32)$")
    bnb_4bit_quant_type: str = Field("nf4", pattern="^(nf4|fp4)$")
```

### TrainingHyperparameters
Training hyperparameters.

```python
class TrainingHyperparameters(BaseModel):
    """Training hyperparameters"""
    learning_rate: float = Field(2e-4, gt=0, le=1e-2)
    num_epochs: int = Field(3, ge=1, le=100)
    batch_size: int = Field(2, ge=1, le=128)
    gradient_accumulation_steps: int = Field(4, ge=1, le=256)
    max_seq_length: int = Field(2048, ge=128, le=32768)
    warmup_steps: int = Field(10, ge=0)
    weight_decay: float = Field(0.01, ge=0, le=1)
    max_grad_norm: float = Field(1.0, gt=0)
    optimizer: str = Field("adamw_8bit", pattern="^(adamw|adamw_8bit|sgd)$")
    lr_scheduler_type: str = Field("linear", pattern="^(linear|cosine|constant)$")
    logging_steps: int = Field(10, ge=1)
    eval_steps: Optional[int] = Field(None, ge=1)
    save_steps: Optional[int] = Field(None, ge=1)
    save_total_limit: Optional[int] = Field(3, ge=1)
```

### TrainingJobCreate
Request to create a training job.

```python
class TrainingJobCreate(BaseModel):
    """Create training job request"""
    name: str = Field(..., min_length=1, max_length=200)
    base_model: str = Field(..., description="HuggingFace model ID or local path")
    dataset_id: str = Field(..., description="Dataset identifier")
    hyperparameters: TrainingHyperparameters = TrainingHyperparameters()
    lora_config: LoRAConfig = LoRAConfig()
    quantization: QuantizationConfig = QuantizationConfig()
    output_dir: str = Field(..., min_length=1)
    eval_dataset_id: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "finance-qa-model",
                "base_model": "meta-llama/Llama-3.1-8B-Instruct",
                "dataset_id": "dataset-789",
                "hyperparameters": {
                    "learning_rate": 2e-4,
                    "num_epochs": 3,
                    "batch_size": 2
                },
                "output_dir": "my-finance-model"
            }
        }
```

### TrainingJobStatus
Job status enumeration.

```python
class TrainingJobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

### TrainingMetrics
Real-time training metrics.

```python
class TrainingMetrics(BaseModel):
    """Training metrics"""
    train_loss: float
    eval_loss: Optional[float] = None
    learning_rate: float
    epoch: float
    step: int
    gpu_memory_used_gb: Optional[float] = None
    tokens_per_second: Optional[float] = None
    timestamp: datetime
```

### CarbonMetrics
Carbon tracking metrics.

```python
class CarbonMetrics(BaseModel):
    """Carbon emissions metrics"""
    emissions_kg: float = Field(ge=0)
    energy_kwh: float = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    carbon_intensity: float = Field(ge=0, description="g CO2/kWh")
    location: Optional[str] = None
    breakdown: Optional[dict] = None
```

### TrainingCheckpoint
Training checkpoint information.

```python
class TrainingCheckpoint(BaseModel):
    """Training checkpoint"""
    step: int
    path: str
    loss: Optional[float] = None
    created_at: datetime
    size_bytes: int = Field(ge=0)
```

### TrainingJob
Complete training job representation.

```python
class TrainingJob(BaseModel):
    """Training job entity"""
    job_id: str
    name: str
    status: TrainingJobStatus
    progress: float = Field(0.0, ge=0.0, le=1.0)
    base_model: str
    dataset_id: str
    eval_dataset_id: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    hyperparameters: TrainingHyperparameters
    lora_config: LoRAConfig
    quantization: QuantizationConfig
    output_dir: str
    
    current_epoch: int = 0
    total_epochs: int
    current_step: int = 0
    total_steps: int = 0
    
    metrics: Optional[TrainingMetrics] = None
    carbon: Optional[CarbonMetrics] = None
    checkpoints: List[TrainingCheckpoint] = []
    
    model_id: Optional[str] = None  # Set when completed

class TrainingJobList(BaseModel):
    """Paginated training job list"""
    items: List[TrainingJob]
    total: int
    page: int = 1
    page_size: int = 20
    pages: int
```

---

## 3. Inference Schemas

### SamplingParameters
Text generation parameters.

```python
class SamplingParameters(BaseModel):
    """Sampling parameters for generation"""
    max_tokens: int = Field(512, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0, le=200)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    stop: Optional[List[str]] = Field(None, max_items=10)
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
```

### GenerateRequest
Text generation request.

```python
class GenerateRequest(BaseModel):
    """Generate text request"""
    model_id: str = Field(..., description="Model identifier")
    prompt: str = Field(..., min_length=1, max_length=8192)
    parameters: SamplingParameters = SamplingParameters()
    stream: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "model-123",
                "prompt": "What is machine learning?",
                "parameters": {
                    "max_tokens": 512,
                    "temperature": 0.7
                }
            }
        }
```

### TokenUsage
Token usage statistics.

```python
class TokenUsage(BaseModel):
    """Token usage statistics"""
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
```

### GenerateResponse
Text generation response.

```python
class GenerateResponse(BaseModel):
    """Generate text response"""
    generated_text: str
    usage: TokenUsage
    carbon_emissions_g: float = Field(ge=0)
    latency_ms: float = Field(ge=0)
    finish_reason: str = Field("stop", pattern="^(stop|length|error)$")
```

### ChatMessage
Chat message format.

```python
class ChatMessage(BaseModel):
    """Chat message"""
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str = Field(..., min_length=1)
```

### ChatCompletionRequest
Chat completion request (OpenAI-compatible).

```python
class ChatCompletionRequest(BaseModel):
    """Chat completion request"""
    model: str = Field(..., description="Model identifier")
    messages: List[ChatMessage] = Field(..., min_items=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(512, ge=1, le=4096)
    stream: bool = False
    stop: Optional[List[str]] = None
```

### ChatCompletionChoice
Chat completion choice.

```python
class ChatCompletionChoice(BaseModel):
    """Chat completion choice"""
    index: int
    message: ChatMessage
    finish_reason: str
```

### ChatCompletionResponse
Chat completion response (OpenAI-compatible).

```python
class ChatCompletionResponse(BaseModel):
    """Chat completion response"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: TokenUsage
    carbon_emissions_g: float = Field(ge=0)
```

### BatchInferenceRequest
Batch inference request.

```python
class BatchInferenceRequest(BaseModel):
    """Batch inference request"""
    model_id: str
    prompts: List[str] = Field(..., min_items=1, max_items=100)
    parameters: SamplingParameters = SamplingParameters()
```

### BatchInferenceResult
Single batch result.

```python
class BatchInferenceResult(BaseModel):
    """Batch inference single result"""
    prompt: str
    generated_text: str
    usage: TokenUsage
    error: Optional[str] = None
```

### BatchInferenceResponse
Batch inference response.

```python
class BatchInferenceResponse(BaseModel):
    """Batch inference response"""
    results: List[BatchInferenceResult]
    total_carbon_emissions_g: float = Field(ge=0)
    total_latency_ms: float = Field(ge=0)
```

---

## 4. Dataset Schemas

### DatasetFormat
Supported dataset formats.

```python
class DatasetFormat(str, Enum):
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    PARQUET = "parquet"
```

### DatasetSchema
Dataset schema information.

```python
class DatasetSchema(BaseModel):
    """Dataset schema"""
    columns: List[str]
    sample: dict
    required_columns: List[str] = ["instruction", "input", "output"]
```

### DatasetStatistics
Dataset statistics.

```python
class DatasetStatistics(BaseModel):
    """Dataset statistics"""
    avg_input_length: float = Field(ge=0)
    avg_output_length: float = Field(ge=0)
    max_length: int = Field(ge=0)
    min_length: int = Field(ge=0)
    total_tokens: Optional[int] = None
```

### Dataset
Dataset representation.

```python
class Dataset(BaseModel):
    """Dataset entity"""
    dataset_id: str
    name: str
    description: Optional[str] = None
    size_bytes: int = Field(ge=0)
    format: DatasetFormat
    num_samples: int = Field(ge=0)
    path: str
    created_at: datetime
    schema: Optional[DatasetSchema] = None
    statistics: Optional[DatasetStatistics] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "dataset_id": "dataset-789",
                "name": "finance-qa",
                "description": "Finance Q&A dataset",
                "size_bytes": 5242880,
                "format": "jsonl",
                "num_samples": 1000,
                "path": "/storage/datasets/raw/dataset-789.jsonl",
                "created_at": "2024-01-15T10:00:00Z"
            }
        }

class DatasetList(BaseModel):
    """Paginated dataset list"""
    items: List[Dataset]
    total: int
    page: int = 1
    page_size: int = 20
    pages: int
```

---

## 5. Carbon Tracking Schemas

### CarbonSummary
Aggregated carbon metrics.

```python
class CarbonByModel(BaseModel):
    """Carbon emissions by model"""
    model_id: str
    model_name: str
    emissions_kg: float = Field(ge=0)
    percentage: float = Field(ge=0, le=100)

class CarbonSummary(BaseModel):
    """Carbon metrics summary"""
    total_emissions_kg: float = Field(ge=0)
    training_emissions_kg: float = Field(ge=0)
    inference_emissions_kg: float = Field(ge=0)
    total_energy_kwh: float = Field(ge=0)
    num_training_jobs: int = Field(ge=0)
    num_inference_requests: int = Field(ge=0)
    period: dict
    by_model: List[CarbonByModel]
```

### CarbonBreakdown
Detailed power breakdown.

```python
class CarbonBreakdown(BaseModel):
    """Detailed carbon breakdown"""
    gpu_power_w: float = Field(ge=0)
    cpu_power_w: float = Field(ge=0)
    ram_power_w: float = Field(ge=0)
    gpu_energy_kwh: float = Field(ge=0)
    cpu_energy_kwh: float = Field(ge=0)
    ram_energy_kwh: float = Field(ge=0)
```

### CarbonTimepoint
Carbon at specific time.

```python
class CarbonTimepoint(BaseModel):
    """Carbon measurement at timepoint"""
    timestamp: datetime
    emissions_kg: float = Field(ge=0)
    power_w: float = Field(ge=0)
```

### JobCarbonDetails
Detailed carbon for a job.

```python
class JobCarbonDetails(BaseModel):
    """Job carbon details"""
    job_id: str
    total_emissions_kg: float = Field(ge=0)
    total_energy_kwh: float = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    carbon_intensity: float = Field(ge=0)
    location: str
    breakdown: CarbonBreakdown
    timeline: List[CarbonTimepoint]
```

---

## 6. System Schemas

### GPUInfo
GPU information.

```python
class GPUInfo(BaseModel):
    """GPU information"""
    available: bool
    count: int = Field(ge=0)
    name: Optional[str] = None
    memory_free_gb: float = Field(ge=0)
    memory_total_gb: float = Field(ge=0)
    memory_used_gb: float = Field(ge=0)
    compute_capability: Optional[str] = None
```

### StorageInfo
Storage information.

```python
class StorageInfo(BaseModel):
    """Storage information"""
    free_gb: float = Field(ge=0)
    total_gb: float = Field(ge=0)
    used_gb: float = Field(ge=0)
    models_count: int = Field(ge=0)
    datasets_count: int = Field(ge=0)
    total_size_gb: float = Field(ge=0)
```

### HealthStatus
Health check status.

```python
class ComponentHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthComponents(BaseModel):
    """Health of components"""
    api: ComponentHealth
    gpu: ComponentHealth
    storage: ComponentHealth
    vllm: ComponentHealth

class HealthStatus(BaseModel):
    """Health check response"""
    status: ComponentHealth
    version: str
    components: HealthComponents
    gpu: Optional[GPUInfo] = None
    storage: Optional[StorageInfo] = None
```

### SystemInfo
System information.

```python
class SystemInfo(BaseModel):
    """System information"""
    version: str
    python_version: str
    dependencies: dict
    hardware: dict
    storage: StorageInfo
```

---

## 7. Common Schemas

### ErrorDetail
Error details.

```python
class ErrorDetail(BaseModel):
    """Error details"""
    code: str
    message: str
    details: Optional[dict] = None
```

### ErrorResponse
Error response format.

```python
class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = False
    error: ErrorDetail
```

### SuccessResponse
Generic success response.

```python
class SuccessResponse(BaseModel):
    """Success response"""
    success: bool = True
    message: str
    data: Optional[dict] = None
```

---

## 8. WebSocket Event Schemas

### LogEvent
Log event from training.

```python
class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class LogEvent(BaseModel):
    """Log event"""
    type: str = "log"
    timestamp: datetime
    level: LogLevel
    message: str
```

### MetricsEvent
Metrics update event.

```python
class MetricsEvent(BaseModel):
    """Metrics event"""
    type: str = "metrics"
    step: int
    metrics: TrainingMetrics
```

### StatusChangeEvent
Status change event.

```python
class StatusChangeEvent(BaseModel):
    """Status change event"""
    type: str = "status_change"
    old_status: TrainingJobStatus
    new_status: TrainingJobStatus
    timestamp: datetime
```

---

## 9. Configuration Schemas

### AppConfig
Application configuration.

```python
class AppConfig(BaseModel):
    """Application configuration"""
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    
    # Storage paths
    storage_root: str = "./storage"
    models_dir: str = "./storage/models"
    datasets_dir: str = "./storage/datasets"
    logs_dir: str = "./storage/logs"
    
    # Training settings
    max_concurrent_jobs: int = 1
    checkpoint_interval: int = 100
    
    # Inference settings
    vllm_gpu_memory_utilization: float = 0.9
    vllm_max_num_seqs: int = 256
    
    # Carbon tracking
    carbon_tracking_enabled: bool = True
    carbon_tracking_interval: int = 15
    
    # Security (Phase 2+)
    enable_auth: bool = False
    jwt_secret_key: Optional[str] = None
```

---

## Usage Examples

### Creating a Training Job
```python
from app.models.training import TrainingJobCreate

job_request = TrainingJobCreate(
    name="finance-qa",
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    dataset_id="dataset-789",
    hyperparameters={
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "batch_size": 2
    },
    output_dir="finance-qa-model"
)
```

### Inference Request
```python
from app.models.inference import GenerateRequest, SamplingParameters

request = GenerateRequest(
    model_id="model-123",
    prompt="What is AI?",
    parameters=SamplingParameters(
        max_tokens=256,
        temperature=0.7
    )
)
```

### Handling Errors
```python
from fastapi import HTTPException
from app.models.common import ErrorResponse, ErrorDetail

raise HTTPException(
    status_code=422,
    detail=ErrorDetail(
        code="VALIDATION_ERROR",
        message="Invalid model configuration",
        details={"field": "lora_r", "error": "Must be between 1 and 256"}
    ).dict()
)
```
