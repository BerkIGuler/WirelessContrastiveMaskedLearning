# Wireless Contrastive Masked Learning (WiMAE)

A PyTorch implementation of Wireless Masked Autoencoder (WiMAE) and Contrastive WiMAE (ContraWiMAE) for wireless channel data analysis.

## Quick Start

### Training with Optimized Data Loading

The training pipeline uses `OptimizedPreloadedDataset` for maximum training speed:

```bash
# Train with default configuration
python examples/training_example.py --data-dir /path/to/your/npz/files

# Debug mode (small dataset for testing)
python examples/training_example.py --data-dir /path/to/your/npz/files --debug

# Use custom config
python examples/training_example.py --config your_config.yaml --data-dir /path/to/your/npz/files
```

### Data Format

The training pipeline expects NPZ files containing complex channel matrices:
- Each NPZ file should contain a `channels` array with shape `(num_samples, 1, height, width)`
- The data should be complex-valued (real + imaginary parts)
- Files should be placed in a directory specified by `data_dir` in the config

### Default Configuration

The default configuration (`configs/default_training.yaml`) includes:

- **Model**: ContraWiMAE with 128-dimensional encoder
- **Data**: OptimizedPreloadedDataset with normalization
- **Training**: AdamW optimizer with cosine learning rate scheduling
- **Batch Size**: 64 (adjust based on your GPU memory)

### Configuration Options

You can customize the training by modifying `configs/default_training.yaml`:

```yaml
model:
  type: "contramae"  # or "wimae"
  patch_size: [4, 4]
  encoder_dim: 128
  # ... other model parameters

data:
  data_dir: "data/pretrain"
  normalize: true
  val_split: 0.2
  debug_size: null  # Set to number for debugging

training:
  batch_size: 64
  epochs: 100
  # ... other training parameters
```

## Installation

```bash
pip install -r requirements.txt
```

## Testing

The project includes comprehensive unit tests to ensure code quality and correctness. All tests are located in the `tests/` directory.

### Running Tests

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run tests with coverage report
pytest --cov=wimae

# Run specific test file
pytest tests/test_models.py

# Run specific test class
pytest tests/test_model_components.py::TestWiMAE

# Run specific test method
pytest tests/test_model_components.py::TestWiMAE::test_wimae_forward

# Run tests in parallel (faster)
pytest -n auto

# Run tests and generate HTML coverage report
pytest --cov=wimae --cov-report=html
```

### Test Categories

The test suite includes:

- **Model Components** (`tests/test_model_components.py`):
  - Encoder functionality and masking
  - Decoder reconstruction capabilities
  - WiMAE model integration
  - ContraWiMAE contrastive learning

- **Model Integration** (`tests/test_models.py`):
  - Full model forward passes
  - Checkpoint saving/loading
  - Data integration with datasets

- **Data Loading** (`tests/test_data_loading.py`):
  - OptimizedPreloadedDataset functionality
  - MultiNPZDataset memory mapping
  - Data normalization utilities

- **Patching and Masking** (`tests/test_patching_and_masking.py`):
  - Complex input patching
  - Patch size handling
  - Data preservation

- **Contrastive Learning** (`tests/test_contrastive_learning.py`):
  - ContrastiveHead functionality
  - InfoNCE loss computation
  - Feature extraction

### Test Configuration

Tests use realistic configurations:
- **Batch size**: 256 (for thorough testing)
- **Patch size**: (1, 16) for 32×32 complex inputs
- **Number of patches**: 128 (64 complex patches × 2 for real/imaginary)
- **Model dimensions**: 64-dimensional encoder/decoder

### Continuous Integration

The test suite is designed to run quickly and reliably:
- All tests should pass on CPU
- No external data dependencies
- Deterministic test data generation
- Comprehensive error checking

### Debugging Tests

If tests fail, you can debug them with:

```bash
# Run with detailed output
pytest -v -s

# Run single test with debugger
pytest tests/test_models.py::TestWiMAEModel::test_wimae_forward_complex_input -s

# Run tests and stop on first failure
pytest -x

# Run tests and show local variables on failure
pytest --tb=long
```

## Project Structure

```
WirelessContrastiveMaskedLearning/
├── configs/                    # Configuration files
│   ├── default_training.yaml   # Default training config
│   ├── default_encoding.yaml   # Encoding config
│   └── default_downstream.yaml # Downstream tasks config
├── examples/                   # Example scripts
│   ├── training_example.py     # Training with optimized data loading
│   ├── encoding_example.py
│   └── downstream_example.py
├── wimae/                      # Main package
│   ├── models/                 # Model implementations
│   ├── training/               # Training utilities
│   ├── encoding/               # Encoding utilities
│   └── downstream/             # Downstream tasks
└── tests/                      # Unit tests
```

## Features

- **Optimized Data Loading**: Pre-loaded datasets for maximum training speed
- **Flexible Configuration**: YAML-based configuration system
- **Multiple Models**: Support for both WiMAE and ContraWiMAE
- **Efficient Training**: Gradient accumulation, mixed precision, and early stopping
- **Comprehensive Logging**: TensorBoard integration and checkpointing
- **Debug Support**: Easy debugging with small datasets

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your-paper-2024,
  title={Wireless Contrastive Masked Learning},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
``` 