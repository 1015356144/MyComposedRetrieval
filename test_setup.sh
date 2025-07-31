#!/bin/bash

# Quick Test Script for Iterative Training
# This script runs a fast test to verify the setup is working

set -e

echo "=========================================="
echo "Quick Test for Iterative Training Setup"
echo "=========================================="

# Check Python environment
echo "Checking Python environment..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# Check if required files exist
echo "Checking required files..."
if [ ! -f "src/trainer_iterative.py" ]; then
    echo "Error: src/trainer_iterative.py not found"
    exit 1
fi

if [ ! -f "train_iterative.py" ]; then
    echo "Error: train_iterative.py not found"
    exit 1
fi

if [ ! -f "configs/test_iterative.yaml" ]; then
    echo "Error: configs/test_iterative.yaml not found"
    exit 1
fi

echo "All required files found!"

# Test import of main modules
echo "Testing module imports..."
python -c "
import sys
import os
sys.path.append('src')
try:
    from trainer_iterative import IterativeTrainer
    print('✓ IterativeTrainer import successful')
except Exception as e:
    print(f'✗ IterativeTrainer import failed: {e}')

try:
    from data.dataset.composed_retrieval_dataset import IterativeCIRRDataset
    print('✓ IterativeCIRRDataset import successful')
except Exception as e:
    print(f'✗ IterativeCIRRDataset import failed: {e}')
"

echo ""
echo "Quick test completed!"
echo "If no errors appeared above, your setup should be working."
echo ""
echo "To run a full test training:"
echo "  ./run_iterative_training.sh --fast_mode --num_iterations 1"
echo ""
echo "To run with your own configuration:"
echo "  python train_iterative.py --config configs/test_iterative.yaml"
