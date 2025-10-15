# API Specification

## Overview
This document defines the REST API endpoints for Model Garden. The API follows RESTful conventions and returns JSON responses.

**Base URL**: `http://localhost:8000/api/v1`

**Content-Type**: `application/json`

---

## Authentication
*Phase 1 (MVP)*: No authentication
*Phase 2+*: JWT Bearer token

```
Authorization: Bearer <token>
```

---

## Common Response Format

### Success Response
```json
{
  "success": true,
  "data": { ... },
  "message": "Operation successful"
}
```

### Error Response
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": { ... }
  }
}
```

### Pagination
```json
{
  "items": [...],
  "total": 100,
  "page": 1,
  "page_size": 20,
  "pages": 5
}
```

---

## Endpoints

### 1. Models

#### List Models
```http
GET /models
```

**Query Parameters:**
- `page` (integer, default: 1): Page number
- `page_size` (integer, default: 20): Items per page
- `status` (string, optional): Filter by status (available, training, error)
- `base_model` (string, optional): Filter by base model

**Response:**
```json
{
  "items": [
    {
      "id": "model-123",
      "name": "my-llama-3.1-finance",
      "base_model": "meta-llama/Llama-3.1-8B-Instruct",
      "status": "available",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T14:20:00Z",
      "size_bytes": 17179869184,
      "training_job_id": "job-456",
      "carbon_emissions": {
        "training_kg": 0.45,
        "inference_kg": 0.02
      },
      "metrics": {
        "eval_loss": 0.234,
        "perplexity": 12.5
      }
    }
  ],
  "total": 15,
  "page": 1,
  "page_size": 20
}
```

#### Get Model Details
```http
GET /models/{model_id}
```

**Response:**
```json
{
  "id": "model-123",
  "name": "my-llama-3.1-finance",
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
    "training_energy_kwh": 0.8,
    "inference_kg": 0.02,
    "inference_requests": 1000
  },
  "metrics": {
    "train_loss": 0.189,
    "eval_loss": 0.234,
    "perplexity": 12.5,
    "training_time_seconds": 9000
  }
}
```

#### Download Model
```http
GET /models/{model_id}/download
```

**Query Parameters:**
- `format` (string): Export format (huggingface, gguf, ollama)

**Response:** File download (binary)

#### Delete Model
```http
DELETE /models/{model_id}
```

**Response:**
```json
{
  "success": true,
  "message": "Model deleted successfully"
}
```

---

### 2. Training

#### Create Training Job
```http
POST /training/jobs
```

**Request Body:**
```json
{
  "name": "finance-qa-model",
  "base_model": "meta-llama/Llama-3.1-8B-Instruct",
  "dataset_id": "dataset-789",
  "hyperparameters": {
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 2048,
    "warmup_steps": 10,
    "optimizer": "adamw_8bit"
  },
  "lora_config": {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "use_rslora": false
  },
  "quantization": {
    "load_in_4bit": true,
    "load_in_8bit": false
  },
  "output_dir": "my-finance-model"
}
```

**Response:**
```json
{
  "job_id": "job-456",
  "status": "pending",
  "created_at": "2024-01-15T10:30:00Z",
  "estimated_duration_minutes": 180
}
```

#### List Training Jobs
```http
GET /training/jobs
```

**Query Parameters:**
- `page`, `page_size`: Pagination
- `status`: Filter by status

**Response:**
```json
{
  "items": [
    {
      "job_id": "job-456",
      "name": "finance-qa-model",
      "status": "running",
      "progress": 0.45,
      "base_model": "meta-llama/Llama-3.1-8B-Instruct",
      "dataset_id": "dataset-789",
      "created_at": "2024-01-15T10:30:00Z",
      "started_at": "2024-01-15T10:32:00Z",
      "estimated_completion": "2024-01-15T13:30:00Z",
      "current_epoch": 2,
      "total_epochs": 3,
      "current_step": 450,
      "total_steps": 1000,
      "metrics": {
        "train_loss": 0.234,
        "learning_rate": 1.8e-4
      },
      "carbon_emissions_kg": 0.23
    }
  ],
  "total": 10,
  "page": 1
}
```

#### Get Training Job Details
```http
GET /training/jobs/{job_id}
```

**Response:**
```json
{
  "job_id": "job-456",
  "name": "finance-qa-model",
  "status": "running",
  "progress": 0.45,
  "base_model": "meta-llama/Llama-3.1-8B-Instruct",
  "dataset_id": "dataset-789",
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:32:00Z",
  "completed_at": null,
  "hyperparameters": {
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "batch_size": 2
  },
  "current_epoch": 2,
  "total_epochs": 3,
  "current_step": 450,
  "total_steps": 1000,
  "metrics": {
    "train_loss": 0.234,
    "eval_loss": 0.267,
    "learning_rate": 1.8e-4,
    "gpu_memory_used_gb": 14.5,
    "tokens_per_second": 1250
  },
  "carbon": {
    "emissions_kg": 0.23,
    "energy_kwh": 0.5,
    "duration_seconds": 5400,
    "carbon_intensity": 460
  },
  "checkpoints": [
    {
      "step": 100,
      "path": "/storage/checkpoints/job-456/checkpoint-100",
      "loss": 0.456
    }
  ]
}
```

#### Cancel Training Job
```http
POST /training/jobs/{job_id}/cancel
```

**Response:**
```json
{
  "success": true,
  "message": "Training job cancelled"
}
```

#### Get Training Logs
```http
GET /training/jobs/{job_id}/logs
```

**Query Parameters:**
- `tail` (integer, optional): Return last N lines
- `follow` (boolean, optional): Stream logs (SSE)

**Response (non-streaming):**
```json
{
  "logs": [
    {"timestamp": "2024-01-15T10:32:00Z", "level": "INFO", "message": "Starting training..."},
    {"timestamp": "2024-01-15T10:32:15Z", "level": "INFO", "message": "Epoch 1/3"}
  ]
}
```

**Response (streaming via SSE):**
```
data: {"timestamp": "2024-01-15T10:32:00Z", "level": "INFO", "message": "Starting training..."}

data: {"timestamp": "2024-01-15T10:32:15Z", "level": "INFO", "message": "Epoch 1/3"}
```

#### WebSocket: Real-time Training Updates
```
WS /training/jobs/{job_id}/ws
```

**Messages from Server:**
```json
{
  "type": "log",
  "timestamp": "2024-01-15T10:32:00Z",
  "level": "INFO",
  "message": "Epoch 1/3 - Step 100/1000 - Loss: 0.456"
}

{
  "type": "metrics",
  "step": 100,
  "metrics": {
    "train_loss": 0.456,
    "learning_rate": 1.9e-4,
    "tokens_per_second": 1250
  }
}

{
  "type": "status",
  "status": "completed",
  "final_metrics": {
    "train_loss": 0.189,
    "eval_loss": 0.234
  }
}
```

---

### 3. Inference

#### Generate Text
```http
POST /inference/generate
```

**Request Body:**
```json
{
  "model_id": "model-123",
  "prompt": "What is machine learning?",
  "parameters": {
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "stop": ["</s>", "\n\n"]
  }
}
```

**Response:**
```json
{
  "generated_text": "Machine learning is a subset of artificial intelligence...",
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 156,
    "total_tokens": 168
  },
  "carbon_emissions_g": 0.00045,
  "latency_ms": 1234
}
```

#### Streaming Generation
```http
POST /inference/generate/stream
```

**Request:** Same as above

**Response (SSE):**
```
data: {"token": "Machine", "finished": false}

data: {"token": " learning", "finished": false}

data: {"token": " is", "finished": false}

data: {"generated_text": "Machine learning is...", "usage": {...}, "finished": true}
```

#### Chat Completion (OpenAI Compatible)
```http
POST /inference/chat/completions
```

**Request Body:**
```json
{
  "model": "model-123",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"}
  ],
  "temperature": 0.7,
  "max_tokens": 512,
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1705318200,
  "model": "model-123",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Machine learning is..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 156,
    "total_tokens": 181
  },
  "carbon_emissions_g": 0.00052
}
```

#### Batch Inference
```http
POST /inference/batch
```

**Request Body:**
```json
{
  "model_id": "model-123",
  "prompts": [
    "What is AI?",
    "Explain neural networks.",
    "What is deep learning?"
  ],
  "parameters": {
    "max_tokens": 256,
    "temperature": 0.7
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "prompt": "What is AI?",
      "generated_text": "AI stands for...",
      "usage": {"prompt_tokens": 5, "completion_tokens": 78}
    },
    {
      "prompt": "Explain neural networks.",
      "generated_text": "Neural networks are...",
      "usage": {"prompt_tokens": 4, "completion_tokens": 102}
    }
  ],
  "total_carbon_emissions_g": 0.0012,
  "total_latency_ms": 3456
}
```

#### Load Model (Admin)
```http
POST /inference/models/{model_id}/load
```

**Response:**
```json
{
  "success": true,
  "message": "Model loaded successfully",
  "memory_used_gb": 14.5
}
```

#### Unload Model (Admin)
```http
POST /inference/models/{model_id}/unload
```

---

### 4. Datasets

#### Upload Dataset
```http
POST /datasets/upload
```

**Request:** Multipart form data
- `file`: Dataset file (CSV, JSON, JSONL)
- `name`: Dataset name
- `description`: Optional description

**Response:**
```json
{
  "dataset_id": "dataset-789",
  "name": "finance-qa",
  "size_bytes": 5242880,
  "format": "jsonl",
  "num_samples": 1000,
  "created_at": "2024-01-15T10:00:00Z"
}
```

#### List Datasets
```http
GET /datasets
```

**Response:**
```json
{
  "items": [
    {
      "dataset_id": "dataset-789",
      "name": "finance-qa",
      "description": "Finance Q&A dataset",
      "size_bytes": 5242880,
      "format": "jsonl",
      "num_samples": 1000,
      "created_at": "2024-01-15T10:00:00Z"
    }
  ],
  "total": 5
}
```

#### Get Dataset Details
```http
GET /datasets/{dataset_id}
```

**Response:**
```json
{
  "dataset_id": "dataset-789",
  "name": "finance-qa",
  "description": "Finance Q&A dataset",
  "size_bytes": 5242880,
  "format": "jsonl",
  "num_samples": 1000,
  "path": "/storage/datasets/raw/dataset-789.jsonl",
  "created_at": "2024-01-15T10:00:00Z",
  "schema": {
    "columns": ["instruction", "input", "output"],
    "sample": {
      "instruction": "Answer the question",
      "input": "What is ROI?",
      "output": "ROI stands for Return on Investment..."
    }
  },
  "statistics": {
    "avg_input_length": 45,
    "avg_output_length": 120,
    "max_length": 512
  }
}
```

#### Preview Dataset
```http
GET /datasets/{dataset_id}/preview
```

**Query Parameters:**
- `limit` (integer, default: 10): Number of samples

**Response:**
```json
{
  "samples": [
    {
      "instruction": "Answer the question",
      "input": "What is ROI?",
      "output": "ROI stands for..."
    }
  ],
  "total_samples": 1000
}
```

#### Delete Dataset
```http
DELETE /datasets/{dataset_id}
```

---

### 5. Carbon Tracking

#### Get Carbon Metrics Summary
```http
GET /carbon/summary
```

**Query Parameters:**
- `start_date`, `end_date`: Date range
- `type`: Filter by operation type (training, inference)

**Response:**
```json
{
  "total_emissions_kg": 12.45,
  "training_emissions_kg": 10.2,
  "inference_emissions_kg": 2.25,
  "total_energy_kwh": 28.5,
  "num_training_jobs": 15,
  "num_inference_requests": 50000,
  "period": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-31T23:59:59Z"
  },
  "by_model": [
    {
      "model_id": "model-123",
      "model_name": "finance-qa",
      "emissions_kg": 5.6,
      "percentage": 45
    }
  ]
}
```

#### Get Job Carbon Details
```http
GET /carbon/jobs/{job_id}
```

**Response:**
```json
{
  "job_id": "job-456",
  "total_emissions_kg": 0.45,
  "total_energy_kwh": 0.8,
  "duration_seconds": 9000,
  "carbon_intensity": 460,
  "location": "US-CA",
  "breakdown": {
    "gpu_power_w": 250,
    "cpu_power_w": 85,
    "ram_power_w": 15,
    "gpu_energy_kwh": 0.625,
    "cpu_energy_kwh": 0.15,
    "ram_energy_kwh": 0.025
  },
  "timeline": [
    {
      "timestamp": "2024-01-15T10:30:00Z",
      "emissions_kg": 0.05,
      "power_w": 350
    }
  ]
}
```

#### Export Carbon Report
```http
GET /carbon/export
```

**Query Parameters:**
- `start_date`, `end_date`: Date range
- `format`: Report format (csv, json, pdf)

**Response:** File download

---

### 6. System

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "components": {
    "api": "healthy",
    "gpu": {
      "available": true,
      "count": 1,
      "memory_free_gb": 24,
      "memory_total_gb": 40
    },
    "storage": {
      "free_gb": 500,
      "total_gb": 1000
    },
    "vllm": "healthy"
  }
}
```

#### Get System Info
```http
GET /system/info
```

**Response:**
```json
{
  "version": "0.1.0",
  "python_version": "3.11.5",
  "dependencies": {
    "unsloth": "2024.1",
    "vllm": "0.2.7",
    "fastapi": "0.109.0"
  },
  "hardware": {
    "gpu": {
      "name": "NVIDIA A100",
      "memory_total_gb": 40,
      "compute_capability": "8.0"
    },
    "cpu": {
      "cores": 32,
      "model": "Intel Xeon"
    },
    "memory_total_gb": 256
  },
  "storage": {
    "models_count": 15,
    "datasets_count": 5,
    "total_size_gb": 450
  }
}
```

---

## WebSocket Events

### Training Updates
**Endpoint:** `WS /ws/training/{job_id}`

**Client → Server:** None (read-only)

**Server → Client:**
```json
// Log event
{
  "type": "log",
  "timestamp": "2024-01-15T10:32:00Z",
  "level": "INFO",
  "message": "Epoch 1/3"
}

// Metrics event
{
  "type": "metrics",
  "step": 100,
  "metrics": {
    "loss": 0.456,
    "learning_rate": 1.9e-4
  }
}

// Status change
{
  "type": "status_change",
  "old_status": "running",
  "new_status": "completed"
}
```

---

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 422 | Invalid input data |
| `NOT_FOUND` | 404 | Resource not found |
| `ALREADY_EXISTS` | 409 | Resource already exists |
| `GPU_OOM` | 507 | Insufficient GPU memory |
| `DISK_FULL` | 507 | Insufficient disk space |
| `MODEL_LOAD_ERROR` | 500 | Failed to load model |
| `TRAINING_ERROR` | 500 | Training failed |
| `INFERENCE_ERROR` | 500 | Inference failed |
| `UNAUTHORIZED` | 401 | Authentication required (Phase 2+) |
| `FORBIDDEN` | 403 | Insufficient permissions (Phase 2+) |
| `RATE_LIMIT` | 429 | Too many requests (Phase 2+) |

---

## Rate Limiting (Phase 2+)

Headers returned with rate-limited responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705318200
```

---

## Versioning

API versions are specified in the URL path: `/api/v1/...`

Breaking changes will require a new version number.
