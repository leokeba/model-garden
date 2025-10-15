# Contributing to Model Garden

Thank you for your interest in contributing to Model Garden! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Code Standards](#code-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Pull Request Process](#pull-request-process)
7. [Reporting Issues](#reporting-issues)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Expected Behavior

- Be respectful and considerate
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Accept responsibility for mistakes

### Unacceptable Behavior

- Harassment or discrimination of any kind
- Trolling, insulting, or derogatory comments
- Publishing others' private information
- Any conduct that would be inappropriate in a professional setting

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- uv package manager (recommended)
- CUDA-capable GPU (for training/inference development)

### Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/model-garden.git
cd model-garden

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/model-garden.git
```

### Set Up Development Environment

```bash
# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
uv pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

---

## Development Workflow

### Create a Feature Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a new branch
git checkout -b feature/your-feature-name
```

### Make Your Changes

1. Write your code following our [Code Standards](#code-standards)
2. Add tests for new functionality
3. Update documentation as needed
4. Run tests and code quality checks

### Commit Your Changes

```bash
# Add your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add carbon tracking visualization"
```

#### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(training): add LoRA fine-tuning support
fix(api): handle timeout errors in inference endpoint
docs(readme): update installation instructions
test(carbon): add unit tests for emission calculations
```

---

## Code Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (not 79)
- **String quotes**: Double quotes preferred
- **Imports**: Organized with `isort`
- **Type hints**: Required for all functions

### Code Formatting

```bash
# Format code with ruff
ruff format .

# Check formatting
ruff format --check .
```

### Linting

```bash
# Run linter
ruff check .

# Auto-fix issues
ruff check --fix .
```

### Type Checking

```bash
# Run type checker
mypy model_garden
```

### Example Code

```python
"""Module for training job management."""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel


class TrainingJob(BaseModel):
    """Represents a model training job.
    
    Attributes:
        job_id: Unique identifier for the job
        name: Human-readable name for the job
        status: Current status of the job
        created_at: Timestamp when job was created
    """
    
    job_id: str
    name: str
    status: str
    created_at: datetime


def create_training_job(
    name: str,
    base_model: str,
    dataset_id: str,
    learning_rate: float = 2e-4,
) -> TrainingJob:
    """Create a new training job.
    
    Args:
        name: Name for the training job
        base_model: Base model identifier
        dataset_id: Dataset to use for training
        learning_rate: Learning rate for training
        
    Returns:
        Created training job instance
        
    Raises:
        ValueError: If name is empty or invalid
    """
    if not name:
        raise ValueError("Job name cannot be empty")
    
    # Implementation here
    ...
```

---

## Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Mirror the source structure (`tests/test_training.py` for `model_garden/training.py`)
- Use descriptive test names
- Test both success and failure cases

### Test Structure

```python
"""Tests for training job management."""

import pytest
from model_garden.training import create_training_job


def test_create_training_job_success():
    """Test successful training job creation."""
    job = create_training_job(
        name="test-job",
        base_model="llama-3-8b",
        dataset_id="dataset-123"
    )
    
    assert job.name == "test-job"
    assert job.status == "pending"


def test_create_training_job_invalid_name():
    """Test that empty name raises ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        create_training_job(
            name="",
            base_model="llama-3-8b",
            dataset_id="dataset-123"
        )


@pytest.mark.asyncio
async def test_async_operation():
    """Test asynchronous operations."""
    result = await some_async_function()
    assert result is not None
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_training.py

# Run with coverage
pytest --cov=model_garden --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"
```

### Test Coverage

- Aim for >80% code coverage
- All new features must include tests
- Bug fixes should include regression tests

---

## Pull Request Process

### Before Submitting

1. **Update from upstream**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**
   ```bash
   # Format, lint, type check, test
   pre-commit run --all-files
   pytest
   ```

3. **Update documentation**
   - Update docstrings
   - Update README if needed
   - Update relevant docs in `docs/`

### Submitting a PR

1. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request on GitHub**
   - Use a clear, descriptive title
   - Reference any related issues
   - Describe what changed and why
   - Include screenshots for UI changes

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Changes Made
- List key changes
- With bullet points

## Testing
- [ ] Tests pass locally
- [ ] Added new tests
- [ ] Coverage maintained/improved

## Related Issues
Fixes #123
Related to #456

## Screenshots (if applicable)
[Add screenshots here]
```

### Review Process

- At least one maintainer must approve
- All CI checks must pass
- Address reviewer feedback
- Keep the PR focused and reasonably sized

---

## Reporting Issues

### Bug Reports

Include:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, GPU)
- Error messages and stack traces
- Minimal code example

### Feature Requests

Include:
- Clear use case
- Expected behavior
- Any alternatives considered
- Willingness to contribute

### Issue Template

```markdown
## Description
Clear description of the issue

## Steps to Reproduce
1. Step one
2. Step two
3. ...

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: macOS 14.0
- Python: 3.11.5
- Model Garden: 0.1.0
- GPU: NVIDIA RTX 4090

## Additional Context
Any other relevant information
```

---

## Development Tips

### Useful Commands

```bash
# Run development server with auto-reload
uvicorn model_garden.api.main:app --reload

# Run CLI commands
model-garden --help

# Check dependencies
uv pip list

# Update dependencies
uv pip install --upgrade -e ".[dev,test]"
```

### Debugging

```python
# Use built-in debugger
import pdb; pdb.set_trace()

# Or use ipdb (install with: uv pip install ipdb)
import ipdb; ipdb.set_trace()

# Use rich for better output
from rich import print
print(complex_object)
```

### Documentation

- Use Google-style docstrings
- Include examples for complex functions
- Update API docs when changing endpoints
- Keep README examples up-to-date

---

## Questions?

If you have questions:
- Check the [documentation](./docs)
- Search [existing issues](https://github.com/OWNER/model-garden/issues)
- Ask in [GitHub Discussions](https://github.com/OWNER/model-garden/discussions)

---

Thank you for contributing to Model Garden! 🌱
