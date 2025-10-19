# Model Garden - Project Overview

## Vision
Model Garden is a comprehensive platform for fine-tuning and serving Large Language Models (LLMs) and AI models with carbon footprint monitoring. It aims to make AI model training and deployment accessible, efficient, and environmentally conscious.

## Key Objectives
1. **Simplify LLM Fine-tuning**: Provide an intuitive interface for fine-tuning open-source models
2. **Efficient Inference**: Deploy models with high-performance inference capabilities
3. **Carbon Awareness**: Track and report carbon emissions during training and inference
4. **User-Friendly**: Offer both web UI and CLI for different use cases

## Core Features

### 1. Fine-Tuning
- Support for popular open-source models (Llama, Mistral, Qwen, etc.)
- Unsloth-powered efficient training (2x faster, 70% less VRAM)
- Multiple fine-tuning methods: LoRA, QLoRA, full fine-tuning
- Dataset management and preprocessing
- Training monitoring and visualization
- Hyperparameter configuration

### 2. Inference/Serving
- vLLM-powered high-throughput inference
- OpenAI-compatible API endpoints
- Model versioning and management
- Batch and streaming inference
- Multiple model deployment strategies

### 3. Carbon Tracking
- Real-time carbon emissions monitoring during training
- Historical emissions tracking per model/experiment
- Regional carbon intensity awareness
- Emissions reporting and visualization
- Recommendations for reducing carbon footprint
- **BoAmps datamodel integration**: Standardized JSON format for emissions reporting
  - Compliance with Boavizta's open standard for AI energy consumption reporting
  - Interoperable emissions data for open data repositories
  - Comprehensive metadata: algorithms, datasets, hardware, infrastructure details
  - Support for both training and inference emissions tracking

### 4. Web Interface
- Dashboard for model management
- Training job monitoring
- Dataset upload and management
- Carbon footprint visualization
- Model deployment controls
- Experiment tracking

### 5. CLI Interface
- Quick fine-tuning commands
- Model serving controls
- Dataset preprocessing
- Export and conversion utilities
- Configuration management

## Target Users
1. **ML Researchers**: Need to fine-tune models efficiently for research
2. **ML Engineers**: Deploy and serve models in production
3. **Data Scientists**: Experiment with different models and datasets
4. **Organizations**: Monitor and reduce AI carbon footprint

## Success Metrics
- Training speed improvements (target: 2x faster than baseline)
- Memory efficiency (target: 70% less VRAM usage)
- Carbon emissions visibility (100% of operations tracked)
- User satisfaction (ease of use, documentation quality)
- Model serving throughput (requests/second)

## Technology Stack Summary
- **Backend**: Python, FastAPI, Pydantic
- **Fine-tuning**: Unsloth, HuggingFace Transformers
- **Inference**: vLLM
- **Carbon Tracking**: CodeCarbon, BoAmps (Boavizta)
- **Frontend**: Svelte, TailwindCSS
- **Package Management**: uv
- **CLI**: Click

## Project Phases

### Phase 1: Foundation (MVP) ✅ COMPLETE
- ✅ FastAPI backend with REST API
- ✅ Model fine-tuning with Unsloth (text + vision)
- ✅ vLLM inference with streaming
- ✅ CodeCarbon integration
- ✅ CLI for training, inference, and serving
- ✅ Web UI with Svelte + TailwindCSS
- ✅ Model registry and job management

### Phase 2: Core Features 🚧 90% COMPLETE
**Completed:**
- ✅ Multiple model support (text + vision-language)
- ✅ HuggingFace Hub integration
- ✅ OpenAI-compatible inference API
- ✅ Structured outputs (JSON mode)
- ✅ Real carbon tracking (training + inference)
- ✅ Persistent emissions database
- ✅ Emissions dashboard with BoAmps reports
- ✅ Model versioning
- ✅ Inference UI (chat + completion)

**In Progress:**
- 🔧 BoAmps real data integration (scaffold complete)
- 🔧 Carbon CLI commands (documented, not implemented)
- 🔧 Dataset management UI
- 🔧 Training job queue
- 🔧 Real-time monitoring via WebSocket
- 🔧 Advanced carbon analytics

### Phase 3: Production Features 📋 PLANNED
- User authentication and authorization
- Multi-tenancy support
- Model registry and marketplace
- Advanced deployment options
- A/B testing framework
- Cost optimization tools
- Enhanced observability

### Phase 4: Enterprise & Scale 🔮 FUTURE
- Multi-GPU support
- Advanced deployment options
- Performance optimization
- Comprehensive documentation
- CI/CD pipeline

## Non-Goals (Initially)
- User authentication/multi-tenancy (Phase 3+)
- Cloud deployment automation (Phase 4+)
- Custom model architectures
- Distributed training across machines
- Commercial support

## Related Resources
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [CodeCarbon Documentation](https://mlco2.github.io/codecarbon/)
