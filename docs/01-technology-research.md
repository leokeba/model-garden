# Technology Research

## Overview
This document provides detailed research on the core technologies that will power Model Garden.

---

## 1. Unsloth - Fine-tuning Backend

### What is Unsloth?
Unsloth is an optimization library that accelerates LLM fine-tuning by 2-5x while reducing memory usage by up to 70%. It achieves this through custom CUDA kernels written in OpenAI's Triton language and manual backpropagation engines.

### Key Features
- **Performance**: 2x faster training, 70% less VRAM usage
- **Compatibility**: Works with HuggingFace Transformers, TRL, Pytorch
- **Model Support**: Llama, Mistral, Gemma, Qwen, DeepSeek, gpt-oss, and more
- **Methods**: LoRA, QLoRA, full fine-tuning, 4-bit, 8-bit, 16-bit training
- **Zero Accuracy Loss**: No approximation methods, all exact calculations
- **Hardware**: NVIDIA GPUs (V100, T4, A100, H100, RTX series), AMD, Intel
- **Easy Export**: GGUF, Ollama, vLLM, HuggingFace format support

### Integration Points
```python
from unsloth import FastLanguageModel, FastModel

# Load model with Unsloth
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/llama-3.1-8b",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
)

# Train with HF Trainer
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=SFTConfig(
        per_device_train_batch_size=2,
        max_steps=60,
    ),
)
trainer.train()
```

### Best Practices
- Use 4-bit quantization for memory efficiency
- Set `use_gradient_checkpointing="unsloth"` for long contexts
- Use rank 16-32 for LoRA for best balance
- Target all linear layers for best results
- Save models with `model.save_pretrained_merged()` for production

### Limitations
- Requires CUDA 7.0+ compatible GPUs
- Windows support requires special setup
- Some advanced features require specific PyTorch versions

---

## 2. vLLM - Inference Engine

### What is vLLM?
vLLM is a high-throughput, memory-efficient inference engine for LLMs. It uses PagedAttention for efficient key-value cache management and achieves state-of-the-art serving performance.

### Key Features
- **High Throughput**: State-of-the-art serving performance
- **PagedAttention**: Efficient memory management for attention
- **Continuous Batching**: Dynamic batching of incoming requests
- **Fast Execution**: CUDA/HIP graph optimization
- **Quantization**: GPTQ, AWQ, INT4, INT8, FP8 support
- **Optimized Kernels**: FlashAttention and FlashInfer integration
- **Speculative Decoding**: Faster generation with draft models
- **OpenAI Compatible**: Drop-in replacement for OpenAI API

### Architecture
```
┌─────────────────┐
│  Client Request │
└────────┬────────┘
         │
    ┌────▼─────┐
    │  Engine  │
    │  Async   │
    └────┬─────┘
         │
    ┌────▼─────────┐
    │  Scheduler   │ ← Continuous batching
    │  PagedAttn   │ ← Memory management
    └────┬─────────┘
         │
    ┌────▼─────────┐
    │  Worker      │
    │  (GPU)       │
    └────┬─────────┘
         │
    ┌────▼─────────┐
    │  Response    │
    └──────────────┘
```

### Integration Points

#### 1. Python API (Offline Inference)
```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)
```

#### 2. OpenAI-Compatible Server
```bash
# Start server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 8000 \
  --tensor-parallel-size 1

# Client usage
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

#### 3. Async API (for FastAPI integration)
```python
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

engine_args = AsyncEngineArgs(model="meta-llama/Llama-3.1-8B-Instruct")
engine = AsyncLLMEngine.from_engine_args(engine_args)

async def generate(prompt: str):
    sampling_params = SamplingParams(temperature=0.8)
    request_id = f"request-{id(prompt)}"
    
    results_generator = engine.generate(prompt, sampling_params, request_id)
    
    async for result in results_generator:
        yield result
```

### Configuration Options
- **Parallelism**: Tensor, pipeline, data parallelism
- **Memory**: KV cache size, GPU memory utilization
- **Batching**: Max batch size, scheduling policy
- **Quantization**: AWQ, GPTQ, FP8 options
- **Engine**: Max model length, trust remote code

### Performance Tuning
- Use `--gpu-memory-utilization 0.9` for max throughput
- Enable `--enable-prefix-caching` for repeated prompts
- Use tensor parallelism for large models
- Adjust `--max-num-seqs` based on hardware

### Monitoring
vLLM exposes Prometheus metrics:
- `vllm:num_requests_running`
- `vllm:num_requests_waiting`
- `vllm:gpu_cache_usage_perc`
- `vllm:time_to_first_token_seconds`
- `vllm:time_per_output_token_seconds`

---

## 3. FastAPI - REST API Framework

### What is FastAPI?
FastAPI is a modern, high-performance web framework for building APIs with Python. It's based on standard Python type hints and provides automatic API documentation.

### Key Features
- **Fast**: Very high performance (NodeJS/Go level)
- **Type Safety**: Automatic validation with Pydantic
- **Auto Docs**: Interactive API docs (Swagger/ReDoc)
- **Async**: Full async/await support
- **Standards**: OpenAPI & JSON Schema compliant
- **DI**: Built-in dependency injection
- **Easy**: Minimal boilerplate

### Core Concepts

#### 1. Path Operations
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class FineTuneRequest(BaseModel):
    model_name: str
    dataset_name: str
    learning_rate: float = 2e-4
    num_epochs: int = 3

@app.post("/api/v1/finetune")
async def start_finetune(request: FineTuneRequest):
    # FastAPI automatically validates request
    job_id = await launch_training_job(request)
    return {"job_id": job_id, "status": "started"}
```

#### 2. Dependencies
```python
from fastapi import Depends

def get_current_user(token: str = Header(...)):
    user = verify_token(token)
    if not user:
        raise HTTPException(status_code=401)
    return user

@app.get("/api/v1/jobs")
async def list_jobs(user = Depends(get_current_user)):
    return await get_user_jobs(user)
```

#### 3. Background Tasks
```python
from fastapi import BackgroundTasks

def cleanup_temp_files(job_id: str):
    # Long-running cleanup
    pass

@app.post("/api/v1/jobs/{job_id}/complete")
async def complete_job(
    job_id: str,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(cleanup_temp_files, job_id)
    return {"status": "completed"}
```

#### 4. WebSocket Support
```python
from fastapi import WebSocket

@app.websocket("/ws/training/{job_id}")
async def training_logs(websocket: WebSocket, job_id: str):
    await websocket.accept()
    async for log_line in get_training_logs(job_id):
        await websocket.send_json({"log": log_line})
```

### Project Structure for Model Garden
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── config.py            # Settings
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── finetune.py  # Fine-tuning endpoints
│   │   │   ├── inference.py # Inference endpoints
│   │   │   ├── models.py    # Model management
│   │   │   └── datasets.py  # Dataset endpoints
│   ├── models/              # Pydantic models
│   ├── services/            # Business logic
│   │   ├── training_service.py
│   │   ├── inference_service.py
│   │   └── carbon_service.py
│   └── core/
│       ├── unsloth_wrapper.py
│       ├── vllm_wrapper.py
│       └── carbon_tracker.py
```

### Best Practices
- Use Pydantic models for validation
- Implement proper error handling
- Use dependency injection for shared resources
- Add request/response examples in schemas
- Enable CORS for frontend integration
- Use background tasks for long operations
- Implement rate limiting for production

---

## 4. CodeCarbon - Carbon Tracking

### What is CodeCarbon?
CodeCarbon is a Python package that estimates the carbon emissions from computing based on electricity consumption and regional carbon intensity.

### How It Works
```
Emissions (kg CO2) = Energy Consumed (kWh) × Carbon Intensity (kg CO2/kWh)

Energy = Power (W) × Time (h) / 1000
Power = GPU Power + CPU Power + RAM Power
```

### Key Features
- **Automatic Tracking**: Monitors GPU, CPU, RAM power
- **Regional Awareness**: Uses location-based carbon intensity
- **Multiple Modes**: Offline, online dashboard, API mode
- **Framework Agnostic**: Works with any Python code
- **Low Overhead**: Minimal performance impact
- **Detailed Reports**: CSV exports and visualizations

### Integration Methods

#### 1. Decorator (Simplest)
```python
from codecarbon import track_emissions

@track_emissions()
def train_model():
    # Training code
    pass
```

#### 2. Context Manager
```python
from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()

# Training code

emissions = tracker.stop()
print(f"Emissions: {emissions} kg CO2")
```

#### 3. Manual Control
```python
from codecarbon import OfflineEmissionsTracker

tracker = OfflineEmissionsTracker(
    project_name="model-garden",
    output_dir="./emissions",
    country_iso_code="USA",
    region="california"
)

tracker.start()
# ... training ...
tracker.stop()
```

#### 4. With Online Dashboard
```python
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(
    save_to_api=True,
    experiment_id="experiment-123",
    api_endpoint="https://api.codecarbon.io"
)
```

### Configuration Options
```python
tracker = EmissionsTracker(
    project_name="fine-tuning-llama",
    measure_power_secs=15,          # Measurement interval
    tracking_mode="machine",         # or "process"
    log_level="info",
    save_to_file=True,
    output_dir="./emissions",
    output_file="emissions.csv",
    on_csv_write="append",          # or "update"
    country_iso_code="USA",
    region="california",
    cloud_provider="aws",
    cloud_region="us-west-1",
)
```

### Output Data
CodeCarbon generates CSV files with:
- `timestamp`: When measurement was taken
- `project_name`: Project identifier
- `duration`: Duration in seconds
- `emissions`: CO2 emissions in kg
- `energy_consumed`: Energy in kWh
- `cpu_power`, `gpu_power`, `ram_power`: Power usage
- `cpu_energy`, `gpu_energy`, `ram_energy`: Energy consumed
- `country_iso_code`, `region`: Location
- `carbon_intensity`: g CO2/kWh

### Best Practices
- Start tracking before model loading
- Stop tracking after all cleanup
- Use project names consistently
- Store emissions data persistently
- Aggregate data for reports
- Compare emissions across experiments

### Limitations
- Accuracy depends on hardware support
- May not capture all power sources
- Cloud environments need special config
- Overhead for very short operations

---

## 5. Click - CLI Framework

### What is Click?
Click is a Python package for creating command-line interfaces with minimal code.

### Key Features
- **Decorators**: Simple function decorators for CLI
- **Type Safety**: Automatic argument validation
- **Help Pages**: Auto-generated help text
- **Nested Commands**: Command groups and subcommands
- **Testing**: Easy CLI testing support

### Structure for Model Garden CLI
```python
import click
from typing import Optional

@click.group()
def cli():
    """Model Garden CLI - Fine-tune and serve LLMs"""
    pass

@cli.group()
def train():
    """Training commands"""
    pass

@train.command()
@click.option('--model', required=True, help='Model name')
@click.option('--dataset', required=True, help='Dataset path')
@click.option('--output', default='./output', help='Output directory')
@click.option('--epochs', default=3, type=int, help='Number of epochs')
@click.option('--lr', default=2e-4, type=float, help='Learning rate')
def start(model: str, dataset: str, output: str, epochs: int, lr: float):
    """Start a fine-tuning job"""
    click.echo(f"Starting training: {model}")
    # Call training service
    
@cli.group()
def serve():
    """Serving commands"""
    pass

@serve.command()
@click.option('--model', required=True)
@click.option('--port', default=8000, type=int)
def start(model: str, port: int):
    """Start model server"""
    click.echo(f"Starting server on port {port}")
    # Start vLLM server

if __name__ == '__main__':
    cli()
```

---

## 6. uv - Package Management

### What is uv?
uv is a modern, fast Python package installer and resolver written in Rust. It's 10-100x faster than pip.

### Key Features
- **Speed**: 10-100x faster than pip
- **Compatibility**: Drop-in pip replacement
- **Modern**: Better dependency resolution
- **Lock Files**: Reproducible installs
- **Virtual Envs**: Built-in venv management

### Commands for Model Garden
```bash
# Initialize project
uv init

# Add dependencies
uv add fastapi uvicorn pydantic
uv add --dev pytest black ruff

# Install all dependencies
uv sync

# Run commands in venv
uv run python app/main.py
uv run pytest

# Update dependencies
uv update

# Create lock file
uv lock
```

### Project Configuration (pyproject.toml)
```toml
[project]
name = "model-garden"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.5.0",
    "unsloth @ git+https://github.com/unslothai/unsloth.git",
    "vllm>=0.2.7",
    "codecarbon>=2.3.0",
    "click>=8.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.12.0",
    "ruff>=0.1.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=7.4.0",
]
```

---

## Integration Strategy

### 1. Backend Architecture
```
FastAPI App
├── Unsloth Integration (Fine-tuning)
├── vLLM Integration (Inference)
├── CodeCarbon Tracking (Emissions)
└── Storage Layer (Models, Datasets, Logs)
```

### 2. Request Flow

#### Fine-tuning Flow
```
Client → FastAPI → Queue Job → Unsloth Training
                             ↓
                        CodeCarbon Tracking
                             ↓
                        Save Model → vLLM Loading
```

#### Inference Flow
```
Client → FastAPI → vLLM Engine → Response
                      ↓
                 CodeCarbon Tracking
```

### 3. Data Flow
```
Datasets → Preprocessing → Training → Model Artifacts
                              ↓
                         Emissions Data → Dashboard
```

---

## Potential Challenges

### 1. GPU Memory Management
- **Challenge**: Multiple models competing for GPU memory
- **Solution**: Queue system, dynamic model loading/unloading

### 2. Long-running Training Jobs
- **Challenge**: HTTP timeouts, connection drops
- **Solution**: Background tasks, WebSocket updates, job queue

### 3. Model Storage
- **Challenge**: Large model files (GBs)
- **Solution**: Efficient storage, compression, dedupe base models

### 4. Concurrent Inference
- **Challenge**: Multiple inference requests
- **Solution**: vLLM's built-in batching and scheduling

### 5. Carbon Tracking Accuracy
- **Challenge**: Cloud environments, shared resources
- **Solution**: Best-effort tracking, clear disclaimers

---

## Next Steps
1. Set up development environment with uv
2. Create FastAPI project structure
3. Implement basic Unsloth wrapper
4. Integrate CodeCarbon
5. Add vLLM inference endpoint
6. Build CLI with Click
7. Design data models and API contracts
