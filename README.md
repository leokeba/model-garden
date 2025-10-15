# 🌱 Model Garden

**Fine-tune and serve LLMs with carbon footprint tracking**

Model Garden is a comprehensive platform for fine-tuning, deploying, and serving large language models (LLMs) while tracking their environmental impact. Built with performance and sustainability in mind.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## ✨ Features

### 🚀 High-Performance Training
- **2x faster fine-tuning** with Unsloth's optimized CUDA kernels
- **70% less VRAM usage** - train larger models on consumer GPUs
- Support for LoRA, QLoRA, and full fine-tuning
- 4-bit, 8-bit, and 16-bit quantization

### ⚡ Efficient Inference
- **High-throughput serving** powered by vLLM
- PagedAttention for optimized memory usage
- OpenAI-compatible API endpoints
- Continuous batching for better GPU utilization

### 🌍 Carbon Footprint Tracking
- Real-time emissions monitoring with CodeCarbon
- Track GPU, CPU, and RAM power consumption
- Historical carbon data and analytics
- Per-model and per-job carbon reports

### 🎯 Developer-Friendly
- Simple REST API and CLI interface
- Real-time training progress via WebSocket
- Comprehensive model management
- Beautiful web dashboard (coming soon)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (Phase 2)                    │
│                  Svelte + TailwindCSS + Vite                 │
└────────────────────────┬────────────────────────────────────┘
                         │ REST API / WebSocket
┌────────────────────────┴────────────────────────────────────┐
│                      Backend (FastAPI)                       │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │  Training Engine│  │   Inference  │  │ Carbon Tracker │ │
│  │    (Unsloth)    │  │    (vLLM)    │  │  (CodeCarbon)  │ │
│  └─────────────────┘  └──────────────┘  └────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                    Storage & Data                            │
│     Models | Datasets | Configs | Logs | Carbon Reports     │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/model-garden.git
cd model-garden

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### Start the API Server

```bash
# Start FastAPI server
model-garden serve --host 0.0.0.0 --port 8000

# Or using uvicorn directly
uvicorn model_garden.api.main:app --reload
```

### CLI Usage

```bash
# List available models
model-garden models list

# Start a training job
model-garden train \
  --base-model unsloth/llama-3-8b-bnb-4bit \
  --dataset my-dataset.jsonl \
  --output-dir ./models/my-model \
  --learning-rate 2e-4 \
  --epochs 3

# Check training status
model-garden jobs status <job-id>

# Generate text
model-garden generate \
  --model ./models/my-model \
  --prompt "Explain quantum computing" \
  --max-tokens 256

# View carbon footprint
model-garden carbon report <model-id>
```

### API Usage

```python
import requests

# Start a training job
response = requests.post("http://localhost:8000/api/v1/training/jobs", json={
    "name": "finance-model-v1",
    "base_model": "unsloth/llama-3-8b-bnb-4bit",
    "dataset_id": "dataset-123",
    "hyperparameters": {
        "learning_rate": 2e-4,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2
    }
})

job_id = response.json()["job_id"]
print(f"Training job started: {job_id}")

# Generate text (OpenAI-compatible)
response = requests.post("http://localhost:8000/api/v1/inference/generate", json={
    "model": "my-model",
    "prompt": "Write a Python function to calculate fibonacci numbers",
    "max_tokens": 256,
    "temperature": 0.7
})

print(response.json()["generated_text"])
```

---

## 📚 Documentation

Comprehensive design documentation is available in the [`docs/`](./docs) directory:

- [**Project Overview**](./docs/00-project-overview.md) - Vision, objectives, and roadmap
- [**Technology Research**](./docs/01-technology-research.md) - Deep dive into core technologies
- [**System Architecture**](./docs/02-system-architecture.md) - Component design and data flow
- [**API Specification**](./docs/03-api-specification.md) - Complete REST API reference
- [**Data Models**](./docs/04-data-models.md) - Database and API schemas
- [**Development Workflow**](./docs/05-development-workflow.md) - Setup, testing, and deployment
- [**Frontend Design**](./docs/06-frontend-design.md) - UI/UX guidelines and components

---

## 🛠️ Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/model-garden.git
cd model-garden

# Create virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all dependencies
uv pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=model_garden --cov-report=html

# Run specific test file
pytest tests/test_training.py

# Run with verbose output
pytest -v
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy model_garden

# Run all checks (formatting, linting, type checking)
pre-commit run --all-files
```

---

## 🗺️ Roadmap

### Phase 1: Foundation & MVP ✅ (In Progress)
- [x] Research and design documentation
- [ ] Core training engine with Unsloth
- [ ] Basic inference with vLLM
- [ ] REST API with FastAPI
- [ ] CLI interface with Click
- [ ] Carbon tracking integration
- [ ] Model management system

### Phase 2: Core Features 🚧
- [ ] Web dashboard (Svelte + TailwindCSS)
- [ ] Real-time training monitoring
- [ ] Dataset management
- [ ] Advanced carbon analytics
- [ ] Model versioning
- [ ] Job scheduling and queuing

### Phase 3: Production Features 📋
- [ ] User authentication and authorization
- [ ] Multi-tenancy support
- [ ] Model registry and marketplace
- [ ] Advanced deployment options
- [ ] A/B testing framework
- [ ] Cost optimization tools

### Phase 4: Enterprise & Scale 🔮
- [ ] Distributed training
- [ ] Multi-GPU and multi-node support
- [ ] Advanced observability
- [ ] Integration with cloud providers
- [ ] Enterprise security features
- [ ] Custom model architectures

---

## 🌍 Carbon Footprint

Model Garden takes sustainability seriously. Every training job and inference request is tracked for carbon emissions:

- **Real-time monitoring** of GPU, CPU, and RAM power consumption
- **Historical data** to track improvements over time
- **Regional carbon intensity** factored into calculations
- **Detailed reports** per model, job, and time period

Example carbon report:
```json
{
  "model_id": "finance-model-v1",
  "training": {
    "emissions_kg": 0.234,
    "energy_kwh": 0.456,
    "duration_hours": 2.5
  },
  "inference": {
    "emissions_kg": 0.012,
    "requests": 1000,
    "avg_per_request_g": 0.012
  }
}
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](./CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgments

Model Garden is built on top of incredible open-source projects:

- [**Unsloth**](https://github.com/unslothai/unsloth) - 2x faster LLM fine-tuning
- [**vLLM**](https://github.com/vllm-project/vllm) - High-throughput inference engine
- [**FastAPI**](https://fastapi.tiangolo.com/) - Modern web framework
- [**CodeCarbon**](https://github.com/mlco2/codecarbon) - Carbon emissions tracking
- [**HuggingFace Transformers**](https://github.com/huggingface/transformers) - State-of-the-art NLP

---

## 📞 Support

- **Documentation**: [docs/](./docs)
- **Issues**: [GitHub Issues](https://github.com/yourusername/model-garden/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/model-garden/discussions)

---

<div align="center">
  <strong>Built with ❤️ for sustainable AI</strong>
</div>
