#!/bin/bash
# Enable static type checking with mypy for Model Garden

set -e

echo "🔍 Setting up static type checking with mypy..."

# Install mypy if not already installed
if ! command -v mypy &> /dev/null; then
    echo "📦 Installing mypy..."
    uv add --dev mypy
fi

# Create mypy configuration if it doesn't exist
if [ ! -f "pyproject.toml" ] || ! grep -q "\[tool.mypy\]" pyproject.toml; then
    echo "⚙️  Adding mypy configuration to pyproject.toml..."
    cat >> pyproject.toml << 'EOF'

# Static type checking configuration
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # Start permissive, gradually enable
plugins = ["pydantic.mypy"]

# Pydantic plugin configuration
[tool.pydantic-mypy]
init_forbid_extra = true
init_typed = true
warn_required_dynamic_aliases = true
EOF
fi

echo "✅ Configuration complete!"
echo ""
echo "Running type checker on api.py..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Run mypy on the API file
uv run mypy model_garden/api.py || {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⚠️  Type errors found. These are suggestions for improvement."
    echo "💡 To ignore specific errors, add '# type: ignore' comments."
    echo ""
    exit 0  # Don't fail - type checking is optional
}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ No type errors found!"
echo ""
echo "💡 To run type checking manually:"
echo "   uv run mypy model_garden/api.py"
echo ""
echo "💡 To check the entire codebase:"
echo "   uv run mypy model_garden/"
