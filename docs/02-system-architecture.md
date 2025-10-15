# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Svelte)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │Dashboard │  │  Models  │  │ Training │  │  Datasets    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST API / WebSocket
┌────────────────────────────┴────────────────────────────────────┐
│                    Backend API (FastAPI)                         │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐ │
│  │  Auth/Session  │  │   Job Queue    │  │  File Storage    │ │
│  └────────────────┘  └────────────────┘  └──────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  Core Services Layer                        │ │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐  │ │
│  │  │   Training   │ │  Inference   │ │ Carbon Tracking  │  │ │
│  │  │   Service    │ │   Service    │ │    Service       │  │ │
│  │  └──────────────┘ └──────────────┘ └──────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                     Compute Layer                                │
│  ┌────────────────┐         ┌────────────────┐                 │
│  │    Unsloth     │         │      vLLM      │                 │
│  │  Fine-tuning   │         │   Inference    │                 │
│  └────────────────┘         └────────────────┘                 │
│           │                          │                           │
│           └──────────┬───────────────┘                          │
│                      │                                           │
│              ┌───────▼────────┐                                 │
│              │  CodeCarbon    │                                 │
│              │  Monitoring    │                                 │
│              └────────────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
                      │
              ┌───────▼────────┐
              │   GPU/Hardware │
              └────────────────┘
```

---

## Component Architecture

### 1. Frontend Layer (Svelte + TailwindCSS)

#### Architecture Pattern: Component-Based SPA

```
frontend/
├── src/
│   ├── routes/              # SvelteKit routes
│   │   ├── +layout.svelte   # Root layout
│   │   ├── +page.svelte     # Home/Dashboard
│   │   ├── models/
│   │   │   ├── +page.svelte # Model list
│   │   │   └── [id]/
│   │   │       └── +page.svelte # Model detail
│   │   ├── training/
│   │   │   ├── +page.svelte # Training jobs
│   │   │   └── new/
│   │   │       └── +page.svelte # New training
│   │   └── datasets/
│   │       └── +page.svelte # Dataset management
│   ├── lib/
│   │   ├── components/      # Reusable components
│   │   │   ├── ModelCard.svelte
│   │   │   ├── TrainingStatus.svelte
│   │   │   ├── CarbonChart.svelte
│   │   │   └── DatasetUpload.svelte
│   │   ├── stores/          # Svelte stores (state)
│   │   │   ├── models.ts
│   │   │   ├── training.ts
│   │   │   └── auth.ts
│   │   ├── api/             # API client
│   │   │   └── client.ts
│   │   └── utils/
│   └── app.html
└── static/
```

#### Key Features
- **Reactive State**: Svelte stores for global state
- **Real-time Updates**: WebSocket for training logs
- **Responsive Design**: TailwindCSS utilities
- **API Integration**: Typed API client

---

### 2. Backend API Layer (FastAPI)

#### Architecture Pattern: Layered Architecture

```
backend/
├── app/
│   ├── main.py                      # FastAPI app entry
│   ├── config.py                    # Settings & environment
│   │
│   ├── api/                         # API routes
│   │   ├── __init__.py
│   │   ├── deps.py                  # Dependencies
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── models.py            # Model mgmt endpoints
│   │       ├── training.py          # Training endpoints
│   │       ├── inference.py         # Inference endpoints
│   │       ├── datasets.py          # Dataset endpoints
│   │       ├── jobs.py              # Job status endpoints
│   │       └── carbon.py            # Carbon metrics endpoints
│   │
│   ├── models/                      # Pydantic models
│   │   ├── __init__.py
│   │   ├── model.py                 # Model schemas
│   │   ├── training.py              # Training request/response
│   │   ├── inference.py             # Inference schemas
│   │   ├── dataset.py               # Dataset schemas
│   │   └── carbon.py                # Carbon tracking schemas
│   │
│   ├── services/                    # Business logic
│   │   ├── __init__.py
│   │   ├── training_service.py      # Training orchestration
│   │   ├── inference_service.py     # Inference management
│   │   ├── model_service.py         # Model operations
│   │   ├── dataset_service.py       # Dataset operations
│   │   └── carbon_service.py        # Carbon tracking
│   │
│   ├── core/                        # Core integrations
│   │   ├── __init__.py
│   │   ├── unsloth_wrapper.py       # Unsloth integration
│   │   ├── vllm_wrapper.py          # vLLM integration
│   │   ├── carbon_tracker.py        # CodeCarbon wrapper
│   │   └── storage.py               # File storage utilities
│   │
│   ├── db/                          # Database (optional Phase 2)
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── session.py
│   │   └── models.py
│   │
│   └── utils/                       # Utilities
│       ├── __init__.py
│       ├── logging.py
│       └── exceptions.py
│
└── tests/                           # Tests
    ├── api/
    ├── services/
    └── core/
```

#### Request Flow
```
HTTP Request
    ↓
FastAPI Router
    ↓
Pydantic Validation
    ↓
Dependency Injection
    ↓
Service Layer (Business Logic)
    ↓
Core Layer (Unsloth/vLLM/CodeCarbon)
    ↓
Response
```

---

### 3. Training Service Architecture

#### Components
```
Training Service
├── Job Queue Manager
│   ├── Add jobs to queue
│   ├── Priority handling
│   └── Job status tracking
│
├── Training Executor
│   ├── Unsloth Model Loader
│   ├── Dataset Preprocessor
│   ├── Training Loop Manager
│   └── Checkpoint Handler
│
├── Carbon Tracker Integration
│   ├── Start tracking
│   ├── Periodic measurements
│   └── Final report generation
│
└── Model Saver
    ├── Save checkpoints
    ├── Export formats (HF, GGUF)
    └── Upload to model registry
```

#### Training Job Lifecycle
```
1. Job Creation
   ↓
2. Validation (model, dataset, params)
   ↓
3. Queue Addition
   ↓
4. Resource Allocation
   ↓
5. Model Loading (with Unsloth)
   ↓
6. Dataset Loading & Preprocessing
   ↓
7. Training Execution
   │  ├── CodeCarbon Start
   │  ├── Training Loop
   │  ├── Periodic Checkpointing
   │  └── Log Streaming (WebSocket)
   ↓
8. Training Completion
   ↓
9. Model Export
   ↓
10. Carbon Report Generation
    ↓
11. Cleanup
```

#### State Machine
```
PENDING → QUEUED → RUNNING → COMPLETED
             ↓         ↓
          FAILED    CANCELLED
```

---

### 4. Inference Service Architecture

#### Components
```
Inference Service
├── vLLM Engine Manager
│   ├── Engine initialization
│   ├── Model loading
│   ├── Dynamic batching
│   └── Memory management
│
├── Request Handler
│   ├── Input validation
│   ├── Prompt preprocessing
│   ├── Sampling params
│   └── Response formatting
│
└── Carbon Tracker
    ├── Per-request tracking
    └── Aggregated metrics
```

#### Inference Modes

##### 1. Synchronous (REST API)
```
POST /v1/inference
├── Load model (if not loaded)
├── Process request
├── Track carbon
└── Return response
```

##### 2. Streaming (SSE/WebSocket)
```
POST /v1/inference/stream
├── Load model
├── Start generation
├── Stream tokens as generated
├── Track carbon
└── Close stream
```

##### 3. Batch Processing
```
POST /v1/inference/batch
├── Load model
├── Process all prompts
├── Use vLLM batching
├── Track total carbon
└── Return all results
```

---

### 5. Data Storage Architecture

#### Storage Structure
```
storage/
├── models/                  # Model artifacts
│   ├── base/                # Base models (downloaded)
│   │   └── llama-3.1-8b/
│   └── finetuned/           # Fine-tuned models
│       └── job-{id}/
│           ├── model/       # Model weights
│           ├── tokenizer/
│           ├── config.json
│           └── adapter_config.json
│
├── datasets/                # Training datasets
│   ├── raw/                 # Original uploads
│   └── processed/           # Preprocessed
│
├── checkpoints/             # Training checkpoints
│   └── job-{id}/
│       └── checkpoint-{step}/
│
├── logs/                    # Training logs
│   └── job-{id}/
│       ├── training.log
│       └── emissions.csv
│
└── temp/                    # Temporary files
```

#### Metadata Storage (JSON/SQLite for MVP)
```json
{
  "models": {
    "model-id": {
      "name": "string",
      "base_model": "string",
      "created_at": "timestamp",
      "size_bytes": "number",
      "status": "ready|training|error",
      "carbon_emissions_kg": "number"
    }
  },
  "jobs": {
    "job-id": {
      "model_id": "string",
      "dataset_id": "string",
      "status": "pending|running|completed|failed",
      "started_at": "timestamp",
      "completed_at": "timestamp",
      "hyperparameters": {},
      "metrics": {},
      "carbon_emissions_kg": "number"
    }
  }
}
```

---

### 6. Carbon Tracking Architecture

#### Integration Points
```
Training Flow:
├── Pre-training: Initialize tracker
├── During training: Continuous monitoring
└── Post-training: Generate report

Inference Flow:
├── Per-request: Track single inference
└── Aggregated: Daily/weekly reports
```

#### Data Collection
```python
class CarbonMetrics:
    timestamp: datetime
    duration_seconds: float
    emissions_kg: float
    energy_kwh: float
    cpu_power_w: float
    gpu_power_w: float
    ram_power_w: float
    carbon_intensity: float  # g CO2/kWh
    location: str
```

#### Reporting
- Real-time dashboard updates
- CSV export for analysis
- Comparison across jobs
- Recommendations for optimization

---

## Communication Patterns

### 1. REST API
- Standard CRUD operations
- Model management
- Dataset upload/download
- Job submission

### 2. WebSocket
- Real-time training logs
- Training metrics streaming
- Live carbon emissions updates

### 3. Server-Sent Events (SSE)
- Inference streaming
- Progress updates

### 4. Background Tasks
- Long-running training jobs
- Model export operations
- Cleanup tasks

---

## Scalability Considerations

### Phase 1 (MVP) - Single Machine
```
- One GPU for training/inference
- Sequential job processing
- Local file storage
- JSON-based metadata
```

### Phase 2 - Queue System
```
- Job queue (Redis/Celery)
- Multiple workers
- Concurrent jobs
- Database for metadata
```

### Phase 3 - Multi-GPU
```
- GPU resource pooling
- Distributed training
- Model parallelism
- Shared storage
```

### Phase 4 - Cloud/Distributed
```
- Kubernetes deployment
- Auto-scaling
- S3/Object storage
- Multi-node training
```

---

## Security Considerations

### MVP (Phase 1)
- Local deployment only
- No authentication
- File-based access control

### Future Phases
- JWT-based authentication
- Role-based access control (RBAC)
- API rate limiting
- Model access permissions
- Dataset privacy controls
- Secure model storage
- Audit logging

---

## Error Handling Strategy

### Levels
1. **API Layer**: HTTP exceptions, input validation
2. **Service Layer**: Business logic errors
3. **Core Layer**: Integration failures (GPU OOM, model loading)
4. **System Layer**: Hardware/OS errors

### Error Categories
- **ValidationError**: Invalid inputs
- **ResourceError**: Insufficient GPU memory, disk space
- **ModelError**: Model loading/training failures
- **InferenceError**: Generation failures
- **StorageError**: File I/O errors

### Recovery Strategies
- Automatic retries (with backoff)
- Checkpoint recovery for training
- Graceful degradation
- Clear error messages to users

---

## Monitoring & Observability

### Metrics to Track
- Training job success/failure rate
- Average training time per model size
- GPU utilization
- Memory usage
- Carbon emissions per job
- Inference latency (p50, p95, p99)
- API response times
- Error rates

### Logging Strategy
- Structured logging (JSON)
- Log levels: DEBUG, INFO, WARNING, ERROR
- Separate logs per component
- Training logs stored per job
- Centralized error tracking

### Health Checks
```
GET /health
├── API status
├── GPU availability
├── Disk space
├── vLLM engine status
└── Model registry status
```

---

## Technology Choices Summary

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **API Framework** | FastAPI | High performance, async, auto docs, type safety |
| **Fine-tuning** | Unsloth | 2x faster, 70% less VRAM, easy integration |
| **Inference** | vLLM | State-of-art throughput, OpenAI compatible |
| **Carbon Tracking** | CodeCarbon | Comprehensive, accurate, Python-native |
| **Frontend** | Svelte | Lightweight, reactive, fast |
| **Styling** | TailwindCSS | Utility-first, rapid development |
| **Package Manager** | uv | 10-100x faster than pip, modern |
| **CLI** | Click | Simple, powerful, well-documented |
| **Storage (MVP)** | File System + JSON | Simple, no extra dependencies |
| **Job Queue (Phase 2)** | Redis + Celery | Proven, scalable |
| **Database (Phase 2)** | PostgreSQL | Reliable, feature-rich |

---

## Next Steps
1. Set up project structure following this architecture
2. Implement core wrappers (Unsloth, vLLM, CodeCarbon)
3. Build FastAPI endpoints
4. Create CLI commands
5. Implement storage layer
6. Add error handling and logging
7. Build frontend components
