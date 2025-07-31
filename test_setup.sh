#!/bin/bash
# Test script to verify installation and setup

set -e

echo "==> Testing Iterative Composed Retrieval Setup"
echo "==> Checking Python environment..."

# Check Python version
PYTHON_VERSION=$(python --version 2>&1)
echo "Python version: $PYTHON_VERSION"

# Check if required packages are installed
echo "==> Checking required packages..."

PACKAGES=("torch" "transformers" "datasets" "PIL" "numpy")

for package in "${PACKAGES[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        echo "✓ $package is installed"
    else
        echo "✗ $package is NOT installed"
        exit 1
    fi
done

# Check if data directories exist (adjust paths as needed)
echo "==> Checking data directories..."

DATA_DIRS=(
    "$HOME/.cache/huggingface"
    "./configs"
    "./src"
)

for dir in "${DATA_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "✓ Directory exists: $dir"
    else
        echo "⚠ Directory not found: $dir (may need to be created)"
    fi
done

# Test basic import
echo "==> Testing basic imports..."

python -c "
import torch
import transformers
print('✓ Basic imports successful')
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
"

# Check GPU availability
echo "==> Checking GPU availability..."

python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.cuda.device_count()} GPUs')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠ CUDA not available - will use CPU')
"

echo "==> Setup test completed successfully!"
echo "==> You can now run the training scripts."
echo ""
echo "Quick start:"
echo "  ./run_iterative_training.sh cirr qwen2vl"
echo "  python examples.py --example 1"