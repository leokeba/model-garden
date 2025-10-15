# Troubleshooting

This document covers common issues and their solutions when using Model Garden.

## Common Warnings

### TorchAO Compatibility Warning

You may see this warning when starting the application:

```
Skipping import of cpp extensions due to incompatible torch version 2.9.0+cu128 for torchao version 0.14.0
```

**What it means**: TorchAO (a PyTorch optimization library) is skipping C++ extensions due to version compatibility.

**Impact**: This is non-critical. The Python-only APIs work fine, and all functionality is preserved.

**Solution**: You can safely ignore this warning. It doesn't affect training or inference performance significantly.

### PyTorch CUDA Allocator Warning

You may see:

```
Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead
```

**What it means**: PyTorch is transitioning to a new environment variable name for CUDA memory allocation configuration.

**Impact**: Non-critical deprecation warning.

**Solution**: This is automatically handled in the code. The warning will disappear in future PyTorch versions.

### Unsloth Import Order Warning

If you see:

```
WARNING: Unsloth should be imported before trl, transformers, peft to ensure all optimizations are applied.
```

**What it means**: The import order affects Unsloth's ability to apply optimizations.

**Impact**: May result in slower training or higher memory usage.

**Solution**: This has been fixed in the codebase by reordering imports properly.

## Performance Issues

### Slow Training

If training is slower than expected:

1. **Check GPU utilization**: Use `nvidia-smi` to monitor GPU usage
2. **Verify batch size**: Increase `--batch-size` if you have enough GPU memory
3. **Adjust gradient accumulation**: Use `--gradient-accumulation-steps` to simulate larger batches
4. **Use mixed precision**: Ensure 4-bit quantization is enabled (default)

### Out of Memory Errors

If you encounter CUDA out of memory errors:

1. **Reduce batch size**: Use `--batch-size 1` 
2. **Increase gradient accumulation**: Use `--gradient-accumulation-steps 8` or higher
3. **Reduce sequence length**: Use `--max-seq-length 1024` or lower
4. **Use gradient checkpointing**: This is enabled by default in Unsloth

## Installation Issues

### Missing Dependencies

If you get import errors:

```bash
# Reinstall with all dependencies
uv sync --all-extras

# Or manually install missing packages
uv add <missing-package>
```

### CUDA Issues

If CUDA is not detected:

1. **Check CUDA installation**: `nvidia-smi`
2. **Verify PyTorch CUDA support**: `python -c "import torch; print(torch.cuda.is_available())"`
3. **Reinstall PyTorch with CUDA**: Follow PyTorch installation guide

## Model Loading Issues

### Model Not Found

If you get "model not found" errors:

1. **Check model path**: Ensure the path is correct
2. **Verify HuggingFace access**: For private models, ensure you're logged in: `huggingface-cli login`
3. **Check internet connection**: For downloading models from HuggingFace Hub

### Incompatible Model Format

If model loading fails:

1. **Use supported models**: Model Garden works best with Llama, Mistral, and similar transformer architectures
2. **Check model compatibility**: Ensure the model is compatible with Unsloth
3. **Try different base models**: Use models like `unsloth/tinyllama-bnb-4bit` for testing

## Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Look for detailed error messages
2. **Search existing issues**: Check the GitHub repository for similar problems
3. **Create an issue**: Provide detailed error messages and system information
4. **Community support**: Join relevant communities for assistance

## System Requirements

- **GPU**: NVIDIA GPU with at least 8GB VRAM (recommended: 16GB+)
- **Python**: 3.11 or 3.12
- **CUDA**: Compatible CUDA toolkit installed
- **Memory**: At least 16GB system RAM (recommended: 32GB+)