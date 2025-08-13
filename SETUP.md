# Setup Guide

This guide will help you set up and run the Quantum Theory of Mind RL framework.

## Quick Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd quantum-tom-rl
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run a quick demo**
   ```bash
   python examples/basic_demo.py
   ```

## Detailed Setup

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation Options

#### Option 1: Basic Installation (Classical Only)
```bash
pip install torch numpy matplotlib pytest
```

#### Option 2: Full Installation (Including Quantum)
```bash
pip install torch numpy pennylane matplotlib pytest
```

### Verification

Run the tests to verify everything is working:
```bash
python -m pytest tests/ -v
```

## Usage Examples

### Basic Demo
```bash
python examples/basic_demo.py
```

### Full Training Run
```bash
# Classical model only
python main.py --model classical --episodes 300 --epochs 5

# All models (if PennyLane is installed)
python main.py --model all --episodes 600 --epochs 8

# With Q-learning agents
python main.py --model all --use-qlearn-agents --qlearn-iters 10000
```

### Custom Configuration
```bash
python main.py \
    --grid 11 \
    --fov 5 \
    --episodes 800 \
    --val-episodes 200 \
    --epochs 10 \
    --batch 256 \
    --lr 1e-4 \
    --n-qubits 12 \
    --model hybrid
```

## Troubleshooting

### PennyLane Installation Issues

If you encounter issues installing PennyLane:

1. **On Windows**: Try installing from conda-forge
   ```bash
   conda install -c conda-forge pennylane
   ```

2. **On Linux/Mac**: Ensure you have the latest pip
   ```bash
   pip install --upgrade pip
   pip install pennylane
   ```

3. **Alternative**: Use the classical-only version
   ```bash
   python main.py --model classical
   ```

### CUDA Issues

If you have CUDA issues with PyTorch:

1. **Check CUDA version**: `nvidia-smi`
2. **Install appropriate PyTorch version**: Visit [pytorch.org](https://pytorch.org)
3. **Use CPU only**: `python main.py --device cpu`

### Memory Issues

If you run out of memory:

1. **Reduce batch size**: `--batch 64`
2. **Reduce episodes**: `--episodes 200`
3. **Use smaller grid**: `--grid 7`

## Development Setup

For development and contributing:

1. **Install in development mode**
   ```bash
   pip install -e .
   ```

2. **Run tests with coverage**
   ```bash
   pip install pytest-cov
   python -m pytest tests/ --cov=src --cov-report=html
   ```

3. **Format code**
   ```bash
   pip install black
   black src/ examples/ tests/
   ```

## Environment Variables

You can set these environment variables for customization:

- `CUDA_VISIBLE_DEVICES`: Control which GPU to use
- `OMP_NUM_THREADS`: Control number of OpenMP threads
- `PYTORCH_CUDA_ALLOC_CONF`: Control CUDA memory allocation

## Performance Tips

1. **Use GPU**: Set `--device cuda` if available
2. **Increase batch size**: Larger batches are more efficient
3. **Use multiple workers**: Set `num_workers` in DataLoader (if memory allows)
4. **Profile memory**: Monitor with `nvidia-smi` or `htop`

## Common Issues

### Import Errors
- Ensure you're in the correct directory
- Check that all dependencies are installed
- Verify Python version (3.10+)

### Memory Errors
- Reduce batch size or grid size
- Use CPU instead of GPU
- Close other applications

### Slow Training
- Use GPU if available
- Increase batch size
- Reduce number of episodes for testing

## Getting Help

1. Check the [README.md](README.md) for detailed documentation
2. Run the examples to see working code
3. Check the test files for usage patterns
4. Open an issue on GitHub for bugs or questions
