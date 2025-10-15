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
- CUDA-capable GPU (minimum 6GB VRAM)
- [uv](https://github.com/astral-sh/uv) package manager (recommended for faster dependency resolution)

### Installation

```bash
# Clone the repository
git clone https://github.com/leokeba/model-garden.git
cd model-garden

# Install with uv (recommended)
uv sync

# Or create a virtual environment manually
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .

# Or with pip
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

**📖 Full installation guide: [INSTALL.md](./INSTALL.md)**

### CLI Usage

**🚀 Quick start: [QUICKSTART.md](./QUICKSTART.md)**

```bash
# Create a sample dataset for testing
uv run model-garden create-dataset --output ./data/sample.jsonl --num-examples 100

# Fine-tune a small model (TinyLlama 1.1B)
uv run model-garden train \
  --base-model unsloth/tinyllama-bnb-4bit \
  --dataset ./data/sample.jsonl \
  --output-dir ./models/my-model \
  --epochs 3 \
  --batch-size 2

# Or use a real dataset from HuggingFace Hub
uv run model-garden train \
  --base-model unsloth/tinyllama-bnb-4bit \
  --dataset yahma/alpaca-cleaned \
  --output-dir ./models/alpaca-model \
  --from-hub \
  --epochs 3

# Check training status
uv run model-garden jobs status <job-id>

# Generate text from your fine-tuned model
uv run model-garden generate ./models/my-model \
  --prompt "Explain quantum computing in simple terms" \
  --max-tokens 256

# View carbon footprint
uv run model-garden carbon report <model-id>
```

### API Usage

**⚠️ API server coming in Phase 2**

The REST API and web dashboard are planned for Phase 2. Currently, only the CLI is available.

For now, you can use the CLI for all operations:

```bash
# See all available commands
uv run model-garden --help

# Get help for a specific command
uv run model-garden train --help
```

---

## 📚 Documentation

### Getting Started
- **[Installation Guide](./INSTALL.md)** - Detailed installation and setup instructions
- **[Quick Start](./QUICKSTART.md)** - Get up and running in minutes
- **[GitHub Setup](./GITHUB_SETUP.md)** - Guide for creating the repository

### Design Documentation
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
uv sync

# Or manually
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
- [x] Core training engine with Unsloth
- [x] CLI interface with Click
- [x] Basic model management
- [ ] ~~Basic inference with vLLM~~ (deferred to Phase 2)
- [ ] ~~REST API with FastAPI~~ (deferred to Phase 2)
- [ ] ~~Carbon tracking integration~~ (deferred to Phase 2)

**Current Focus**: Testing and refining the fine-tuning CLI

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
