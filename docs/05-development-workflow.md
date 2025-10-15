# Development Workflow & Tooling

## Overview
This document outlines the development setup, workflows, testing strategies, and deployment procedures for Model Garden.

---

## 1. Development Environment Setup

### Prerequisites
- Python 3.10 or 3.11 (not 3.14)
- NVIDIA GPU with CUDA 11.8+ or 12.1+
- Node.js 18+ and npm (for frontend)
- Git
- 50GB+ free disk space

### Initial Setup

#### 1. Clone Repository
```bash
git clone https://github.com/your-org/model-garden.git
cd model-garden
```

#### 2. Install uv (Package Manager)
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (if needed)
export PATH="$HOME/.cargo/bin:$PATH"
```

#### 3. Setup Backend

```bash
cd backend

# Initialize uv project (first time only)
uv init

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

#### 4. Setup Frontend
```bash
cd ../frontend

# Install dependencies
npm install

# or use pnpm (recommended)
npm install -g pnpm
pnpm install
```

#### 5. Create Configuration
```bash
# Backend
cp backend/.env.example backend/.env
# Edit .env with your settings

# Frontend
cp frontend/.env.example frontend/.env
```

#### 6. Initialize Storage
```bash
mkdir -p storage/{models/{base,finetuned},datasets/{raw,processed},logs,checkpoints,temp}
```

---

## 2. Project Structure

```
model-garden/
├── README.md
├── LICENSE
├── .gitignore
├── .pre-commit-config.yaml
├── docker-compose.yml
│
├── docs/                          # Documentation
│   ├── 00-project-overview.md
│   ├── 01-technology-research.md
│   ├── 02-system-architecture.md
│   ├── 03-api-specification.md
│   ├── 04-data-models.md
│   └── 05-development-workflow.md
│
├── backend/                       # Python backend
│   ├── pyproject.toml             # uv/Python config
│   ├── uv.lock                    # Dependency lock file
│   ├── .env                       # Environment variables
│   ├── .env.example
│   ├── README.md
│   │
│   ├── app/                       # Application code
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI app
│   │   ├── config.py              # Settings
│   │   ├── api/                   # API routes
│   │   ├── models/                # Pydantic models
│   │   ├── services/              # Business logic
│   │   ├── core/                  # Core integrations
│   │   └── utils/                 # Utilities
│   │
│   ├── tests/                     # Tests
│   │   ├── conftest.py
│   │   ├── test_api/
│   │   ├── test_services/
│   │   └── test_core/
│   │
│   └── scripts/                   # Utility scripts
│       ├── seed_data.py
│       └── cleanup.py
│
├── frontend/                      # Svelte frontend
│   ├── package.json
│   ├── pnpm-lock.yaml
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   ├── .env
│   ├── .env.example
│   │
│   ├── src/
│   │   ├── routes/                # SvelteKit routes
│   │   ├── lib/                   # Components & utilities
│   │   ├── app.html
│   │   └── app.css
│   │
│   └── static/                    # Static assets
│
├── cli/                           # CLI tool
│   ├── __init__.py
│   ├── main.py                    # Click commands
│   └── commands/
│
├── storage/                       # Data storage (gitignored)
│   ├── models/
│   ├── datasets/
│   ├── logs/
│   └── checkpoints/
│
└── scripts/                       # Project scripts
    ├── setup.sh
    ├── start-dev.sh
    └── deploy.sh
```

---

## 3. pyproject.toml Configuration

```toml
[project]
name = "model-garden"
version = "0.1.0"
description = "Fine-tune and serve LLMs with carbon tracking"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
readme = "README.md"
requires-python = ">=3.10,<3.14"
license = {text = "MIT"}

dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "python-multipart>=0.0.6",
    "python-jose[cryptography]>=3.3.0",
    "passlib[bcrypt]>=1.7.4",
    "aiofiles>=23.2.1",
    "httpx>=0.26.0",
    "websockets>=12.0",
    "click>=8.1.0",
    "rich>=13.7.0",
    "unsloth @ git+https://github.com/unslothai/unsloth.git",
    "vllm>=0.2.7",
    "codecarbon>=2.3.0",
    "transformers>=4.36.0",
    "datasets>=2.16.0",
    "accelerate>=0.25.0",
    "torch>=2.1.0",
    "trl>=0.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "httpx>=0.26.0",
    "black>=23.12.0",
    "ruff>=0.1.0",
    "mypy>=1.8.0",
    "pre-commit>=3.6.0",
]

[project.scripts]
model-garden = "cli.main:cli"

[tool.uv]
dev-dependencies = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=app --cov-report=html --cov-report=term"

[tool.black]
line-length = 100
target-version = ['py310']
include = '\.pyi?$'

[tool.ruff]
line-length = 100
target-version = "py310"
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

---

## 4. Development Commands

### Backend Development

```bash
# Activate environment
uv venv
source .venv/bin/activate

# Install dependencies
uv sync

# Install dev dependencies
uv sync --dev

# Add new dependency
uv add fastapi
uv add --dev pytest

# Update dependencies
uv lock --upgrade

# Run backend server
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
uv run pytest

# Run specific test
uv run pytest tests/test_api/test_training.py -v

# Run with coverage
uv run pytest --cov=app --cov-report=html

# Format code
uv run black app tests
uv run ruff check app tests --fix

# Type checking
uv run mypy app

# Run CLI
uv run model-garden --help
```

### Frontend Development

```bash
cd frontend

# Install dependencies
pnpm install

# Development server
pnpm dev

# Build for production
pnpm build

# Preview production build
pnpm preview

# Run tests
pnpm test

# Lint
pnpm lint

# Format
pnpm format
```

### Full Stack Development

Create `scripts/start-dev.sh`:
```bash
#!/bin/bash

# Start backend
cd backend
uv run uvicorn app.main:app --reload &
BACKEND_PID=$!

# Start frontend
cd ../frontend
pnpm dev &
FRONTEND_PID=$!

# Trap Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID" EXIT

# Wait
wait
```

---

## 5. Testing Strategy

### Backend Testing

#### Unit Tests
Test individual functions and classes.

```python
# tests/test_core/test_unsloth_wrapper.py
import pytest
from app.core.unsloth_wrapper import UnslothWrapper

def test_model_loading():
    wrapper = UnslothWrapper()
    model, tokenizer = wrapper.load_model("test-model")
    assert model is not None
    assert tokenizer is not None
```

#### Integration Tests
Test API endpoints.

```python
# tests/test_api/test_training.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_create_training_job():
    response = client.post("/api/v1/training/jobs", json={
        "name": "test-job",
        "base_model": "test-model",
        "dataset_id": "test-dataset",
        "output_dir": "test-output"
    })
    assert response.status_code == 200
    assert "job_id" in response.json()
```

#### Test Configuration
```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import get_settings

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def test_settings():
    settings = get_settings()
    settings.storage_root = "./test_storage"
    return settings
```

### Frontend Testing

```javascript
// src/lib/components/__tests__/ModelCard.test.ts
import { render, screen } from '@testing-library/svelte'
import ModelCard from '../ModelCard.svelte'

test('displays model name', () => {
  render(ModelCard, { 
    props: { 
      model: { name: 'Test Model', status: 'available' } 
    } 
  })
  expect(screen.getByText('Test Model')).toBeInTheDocument()
})
```

### Running Tests

```bash
# Backend tests
cd backend
uv run pytest                    # All tests
uv run pytest -v                 # Verbose
uv run pytest -k "test_model"    # Specific tests
uv run pytest --cov              # With coverage

# Frontend tests
cd frontend
pnpm test                        # All tests
pnpm test:watch                  # Watch mode
pnpm test:coverage               # With coverage
```

---

## 6. Code Quality Tools

### Pre-commit Hooks

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.10
        
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
```

Setup:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Linting & Formatting

```bash
# Backend
black app tests                  # Format
ruff check app tests --fix       # Lint and fix
mypy app                         # Type check

# Frontend
pnpm lint                        # ESLint
pnpm format                      # Prettier
```

---

## 7. Environment Variables

### Backend `.env`
```bash
# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
LOG_LEVEL=INFO

# Storage
STORAGE_ROOT=./storage
MODELS_DIR=./storage/models
DATASETS_DIR=./storage/datasets
LOGS_DIR=./storage/logs

# Training
MAX_CONCURRENT_JOBS=1
CHECKPOINT_INTERVAL=100

# vLLM
VLLM_GPU_MEMORY_UTILIZATION=0.9
VLLM_MAX_NUM_SEQS=256

# Carbon Tracking
CARBON_TRACKING_ENABLED=true
CARBON_TRACKING_INTERVAL=15
CARBON_COUNTRY_CODE=USA
CARBON_REGION=california

# Security (Phase 2+)
ENABLE_AUTH=false
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=30
```

### Frontend `.env`
```bash
VITE_API_BASE_URL=http://localhost:8000/api/v1
VITE_WS_BASE_URL=ws://localhost:8000/ws
VITE_APP_NAME=Model Garden
```

---

## 8. Docker Setup

### Dockerfile (Backend)
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --no-dev

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run app
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./storage:/app/storage
      - ./backend:/app
    environment:
      - STORAGE_ROOT=/app/storage
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
  frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    environment:
      - VITE_API_BASE_URL=http://backend:8000/api/v1
    depends_on:
      - backend
```

---

## 9. Deployment

### Local Deployment

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Production Deployment (Future)

1. **Build Production Images**
```bash
docker build -t model-garden-backend:latest ./backend
docker build -t model-garden-frontend:latest ./frontend
```

2. **Deploy to Kubernetes** (example)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-garden-backend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: model-garden-backend
  template:
    metadata:
      labels:
        app: model-garden-backend
    spec:
      containers:
      - name: backend
        image: model-garden-backend:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
```

---

## 10. CI/CD Pipeline (Future)

### GitHub Actions Example

`.github/workflows/test.yml`:
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install uv
      run: pip install uv
    
    - name: Install dependencies
      run: |
        cd backend
        uv sync
    
    - name: Run tests
      run: |
        cd backend
        uv run pytest --cov
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

---

## 11. Monitoring & Logging

### Logging Configuration
```python
# app/utils/logging.py
import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )
```

### Metrics Collection (Future)
- Prometheus for metrics
- Grafana for visualization
- Application metrics: request latency, error rates
- Training metrics: job duration, success rate
- Carbon metrics: emissions per job

---

## 12. Documentation

### API Documentation
- Auto-generated with FastAPI: `http://localhost:8000/docs`
- ReDoc alternative: `http://localhost:8000/redoc`

### Code Documentation
```python
def train_model(config: TrainingConfig) -> Model:
    """
    Train a model with the given configuration.
    
    Args:
        config: Training configuration including model, dataset, and hyperparameters
        
    Returns:
        Trained model instance
        
    Raises:
        GPUOutOfMemoryError: If GPU memory is insufficient
        ModelLoadError: If base model cannot be loaded
    """
    pass
```

### User Documentation
- Keep docs/ folder up to date
- Use Markdown for all documentation
- Include code examples
- Keep README.md concise with links to detailed docs

---

## 13. Best Practices

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/training-api

# Make changes and commit
git add .
git commit -m "feat: add training job API"

# Push to remote
git push origin feature/training-api

# Create pull request on GitHub
```

### Commit Messages
Follow conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

### Code Review Checklist
- [ ] Tests pass
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] No security vulnerabilities
- [ ] Performance acceptable
- [ ] Error handling implemented

---

## 14. Troubleshooting

### Common Issues

#### GPU Out of Memory
```bash
# Check GPU memory
nvidia-smi

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Reduce batch size or use 4-bit quantization
```

#### Import Errors
```bash
# Reinstall dependencies
uv sync --reinstall

# Clear cache
uv cache clean
```

#### Port Already in Use
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>
```

---

## Next Steps
1. Follow setup instructions
2. Run tests to verify setup
3. Start implementing features
4. Refer to architecture docs for structure
5. Use API spec for endpoint implementation
