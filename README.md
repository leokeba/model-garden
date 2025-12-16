# 🌱 Model Garden

**Fine-tune and serve LLMs and Vision-Language Models with carbon footprint tracking**

Model Garden is a comprehensive platform for fine-tuning, deploying, and serving large language models (LLMs) and vision-language models (VLMs) while tracking their environmental impact. Built with performance and sustainability in mind.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## 🎉 What's New

### Recent Achievements (October 2025)
- ✅ **Full vLLM Integration** - High-throughput inference with OpenAI-compatible API
- ✅ **Structured Outputs** - Generate JSON following strict schemas (docs/10-structured-outputs.md)
- ✅ **Vision Model Inference** - Serve multimodal models with image understanding
- ✅ **Complete CLI Suite** - Commands for serving, generation, and interactive chat
- ✅ **Web Dashboard** - Manage models and training jobs via intuitive UI
- ✅ **HuggingFace Hub Integration** - Direct dataset and model loading

---

## ✨ Features

### 🚀 High-Performance Training
- **2x faster fine-tuning** with Unsloth's optimized CUDA kernels
- **70% less VRAM usage** - train larger models on consumer GPUs
- Support for LoRA, QLoRA, and full fine-tuning
- 4-bit, 8-bit, and 16-bit quantization
- **🆕 Vision-Language Models** - Fine-tune Qwen2.5-VL (3B/7B/72B) for image + text tasks
- **🎨 Multimodal Training** - Unified interface for text-only and vision-language models
- **🧪 Axolotl backend** (optional) - drive Axolotl CLI for text + vision fine-tuning when installed
- **📊 Dataset Validation** - Auto-detect schemas, validate formats, generate statistics

### ⚡ Efficient Inference (NEW!)
- **High-throughput serving** powered by vLLM
- **OpenAI-compatible API** - Drop-in replacement for OpenAI client
- **🎯 Structured Outputs** - Generate JSON that strictly follows schemas
- **🔧 LoRA Adapter Loading** - Load adapters directly from HuggingFace Hub ([guide](./LORA_ADAPTER_LOADING.md))
- **🤗 Model Upload** - Publish models to HuggingFace Hub with one click ([guide](./HUGGINGFACE_MODEL_UPLOAD.md))
- **Streaming support** for real-time responses
- **PagedAttention** for optimized memory usage
- **Continuous batching** for better GPU utilization
- **Multi-GPU support** via tensor parallelism
- **Quantization support** (AWQ, GPTQ, FP8) for memory efficiency

### 🌍 Carbon Footprint Tracking
- Real-time emissions monitoring with CodeCarbon
- Track GPU, CPU, and RAM power consumption
- Historical carbon data and analytics
- Per-model and per-job carbon reports
- **BoAmps v1.1.0 compliance** - Standardized carbon reporting with real hardware data

### 🎯 Developer-Friendly
- Simple REST API and CLI interface
- Real-time training progress and job management
- Comprehensive model management
- **Integrated web dashboard** - Manage text and vision models via intuitive UI
- **Priority job queue** - Background processing with queue management endpoints

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
# Note: Models are stored in the models/ directory. You can use either:
#   --output-dir my-model (recommended, simpler)
#   --output-dir ./models/my-model (also works for backward compatibility)
uv run model-garden train \
  --base-model unsloth/tinyllama-bnb-4bit \
  --dataset ./data/sample.jsonl \
  --output-dir my-model \
  --epochs 3 \
  --batch-size 2

# Or use a real dataset from HuggingFace Hub
uv run model-garden train \
  --base-model unsloth/tinyllama-bnb-4bit \
  --dataset yahma/alpaca-cleaned \
  --output-dir alpaca-model \
  --from-hub \
  --epochs 3

# Check training status
uv run model-garden jobs status <job-id>

# Generate text from your fine-tuned model
# When loading models, you can use just the model name
uv run model-garden generate my-model \
  --prompt "Explain quantum computing in simple terms" \
  --max-tokens 256

# View carbon footprint
uv run model-garden carbon report <model-id>
```

### 🆕 Vision-Language Model Training

Fine-tune models that understand both images and text:

```bash
# Create a sample vision-language dataset
uv run model-garden create-vision-dataset \
  --output ./data/vision_sample.jsonl

# Fine-tune Qwen2.5-VL for image understanding
uv run model-garden train-vision \
  --base-model Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset ./data/vision_dataset.jsonl \
  --output-dir ./models/my-vision-model \
  --epochs 3 \
  --batch-size 1
```

**Supported vision-language models:**
- Qwen/Qwen2.5-VL-3B-Instruct (3B params, ~6GB VRAM)
- Qwen/Qwen2.5-VL-7B-Instruct (7B params, ~10GB VRAM)

**Dataset format** (JSONL):
```json
{"text": "What is in this image?", "image": "/path/to/img.jpg", "response": "A cat"}
```

📖 **Full guide**: [Vision-Language Training](./docs/08-vision-language-training.md)

### 🧪 Axolotl Backend (optional)

Leverage Axolotl's training stack while keeping Model Garden's CLI and job tracking.

```bash
# Install optional extra
uv pip install 'model-garden[axolotl]'

# Text fine-tuning via Axolotl CLI
uv run model-garden train \
  --backend axolotl \
  --base-model <model> \
  --dataset ./data/sample.jsonl \
  --output-dir ./models/my-axolotl-model

# Vision fine-tuning (OpenChat-style messages are generated automatically)
uv run model-garden train-vision \
  --backend axolotl \
  --base-model Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset ./data/vision_dataset.jsonl \
  --output-dir ./models/my-axolotl-vlm
```

Notes:
- Hugging Face Hub datasets are downloaded then materialized to JSONL for Axolotl.
- Eval datasets are not yet wired into the Axolotl config; run eval separately.

📖 **Backend details**: [Axolotl Backend](./docs/19-axolotl-backend.md)

### 🆕 Model Inference & Serving

Serve your fine-tuned models with high-throughput inference:

```bash
# Start an inference server
uv run model-garden serve-model --model-path ./models/my-model

# 🔧 NEW: Load LoRA adapters directly from HuggingFace Hub
uv run model-garden serve-model \
  --model-path Barth371/Qwen2.5-VL-72B-Instruct-bnb-4bit-2025-10-21_16-26_batch_size_4_cmr-block-2_adapters_4bit

# One-off text generation
uv run model-garden inference-generate \
  --model-path ./models/my-model \
  --prompt "Explain machine learning" \
  --max-tokens 512 \
  --stream

# Interactive chat interface
uv run model-garden inference-chat \
  --model-path ./models/my-model \
  --system-prompt "You are a helpful AI assistant"

# With multi-GPU and quantization
uv run model-garden serve-model \
  --model-path ./models/my-model \
  --tensor-parallel-size 2 \
  --quantization awq \
  --gpu-memory-utilization 0.8
```

**Key features:**
- 🔧 **LoRA adapter loading** - Load adapters directly from HuggingFace Hub without merging
- OpenAI-compatible API (use OpenAI Python client!)
- Streaming responses for real-time output
- Multi-GPU tensor parallelism
- Quantization support (AWQ, GPTQ, FP8)
- Interactive CLI chat interface

📖 **LoRA Adapter Guide**: [Loading LoRA Adapters](./LORA_ADAPTER_LOADING.md)

**Use with OpenAI Python client:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/api/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="my-model",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

📖 **Full guide**: [Inference Serving](./docs/09-inference-serving.md)

### Web UI & API

**🎉 NEW: Web UI and REST API now available!**

```bash
# Start the server (serves both UI and API)
uv run model-garden serve

# Or with custom settings
uv run model-garden serve --host 127.0.0.1 --port 3000 --reload
```

Once started, access:
- **🌐 Web Dashboard**: `http://localhost:8000` - Interactive web interface
- **📚 API Documentation**: `http://localhost:8000/docs` - Swagger UI
- **📖 ReDoc Documentation**: `http://localhost:8000/redoc` - Alternative API docs

**Key features:**
- Browse and manage your fine-tuned models
- Monitor training jobs in real-time
- Create new training jobs with a simple form
- View system status and resource usage
- RESTful API for programmatic access

**Key API endpoints:**
- `GET /api/v1/models` - List all models
- `POST /api/v1/training/jobs` - Create training job  
- `GET /api/v1/training/jobs/{job_id}` - Get job status
- `GET /api/v1/system/status` - System information

### CLI Usage

**📖 Full CLI guide: [QUICKSTART.md](./QUICKSTART.md)**

```bash
# See all available CLI commands
uv run model-garden --help

# Text-only training
uv run model-garden train --help

# Vision-language training
uv run model-garden train-vision --help

# Start the API server
uv run model-garden serve --help

# Carbon emissions tracking
uv run model-garden carbon --help
```

### 🌱 Carbon Emissions CLI

Track and analyze the environmental impact of your ML workloads:

```bash
# View summary of all carbon emissions
uv run model-garden carbon summary

# View recent emissions records
uv run model-garden carbon report

# View detailed report for a specific job
uv run model-garden carbon report <job-id>

# Export data in various formats
uv run model-garden carbon export -o emissions.json
uv run model-garden carbon export --format csv -o emissions.csv
uv run model-garden carbon export --format boamps -o report.json

# Clear old emissions data
uv run model-garden carbon clear --older-than 30
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
- [**Vision-Language Training**](./docs/08-vision-language-training.md) - Fine-tune multimodal models
- [**Inference Serving**](./docs/09-inference-serving.md) - vLLM inference and serving guide
- [**Structured Outputs**](./docs/10-structured-outputs.md) - Constrained generation and JSON mode
- [**BoAmps Integration**](./docs/11-boamps-integration.md) - Standardized emissions reporting

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

# Type checking (using ty from Astral)
uv run ty check model_garden/

# Run all checks (formatting, linting, type checking)
pre-commit run --all-files
```

---

## 🗺️ Roadmap

### Phase 1: Foundation & MVP ✅ (Completed)
- [x] Research and design documentation
- [x] Core training engine with Unsloth
- [x] CLI interface with Click
- [x] Basic model management
- [x] **FastAPI backend with REST endpoints** 
- [x] **Model and training job management API**
- [x] **System status and monitoring endpoints**
- [x] **Web UI with SvelteKit (dashboard, models, training jobs)**
- [x] **Static frontend served by FastAPI backend**

**Current Status**: Phase 1 complete! Ready for production testing.

### Phase 2: Core Features ✅ (Complete!)
- [x] **vLLM inference integration with streaming**
- [x] **OpenAI-compatible inference API endpoints**
- [x] **CLI commands for inference (serve, generate, chat)**
- [x] **Structured outputs** (JSON mode, schema validation) 🆕
- [x] **Multi-modal inference** (vision + text) 🆕
- [x] **HuggingFace Hub integration** 🆕
- [x] **Real carbon tracking with CodeCarbon** ✨
- [x] **Carbon tracking for training & inference** ✨
- [x] **Persistent emissions database** ✨
- [x] **Inference UI in web dashboard** (chat + completion modes)
- [x] **Carbon emissions in training job details** ✨
- [x] **Emissions page with BoAmps reports** ✨
- [x] **BoAmps real data integration** ✨ - Real hardware metrics in all report sections
- [x] **Job queue for training & model loading** ✨ - Priority-based queue with API endpoints
- [x] **Dataset validation and statistics** ✨ - Schema detection, format support, quality checks
- [x] **Automatic Chat Template Detection** ✨ - Uses `tokenizer.apply_chat_template()` for any model
- [x] **Real-time training monitoring via WebSocket** ✨ - Live loss/accuracy charts with reconnection

### Phase 2.5: Polish & CLI Enhancement ✅ (Complete!)
- [x] **Carbon CLI commands** ✨ - `carbon report`, `carbon summary`, `carbon export`, `carbon clear`
- [x] **Advanced carbon analytics dashboard** ✨ - Trends, comparisons, recommendations with Chart.js visualizations

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
  - [Installation Guide](./INSTALL.md)
  - [Quick Start](./QUICKSTART.md)
  - [Vision-Language Models](./VISION_SUPPORT.md)
  - [Inference Serving](./docs/09-inference-serving.md)
  - [**Structured Outputs**](./docs/10-structured-outputs.md) 🆕
  - [HuggingFace Hub Integration](./HUGGINGFACE_HUB_SUPPORT.md)
- **Issues**: [GitHub Issues](https://github.com/yourusername/model-garden/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/model-garden/discussions)

---

<div align="center">
  <strong>Built with ❤️ for sustainable AI</strong>
</div>
