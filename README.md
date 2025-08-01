# Wireless Contrastive Masked Learning (WiMAE)

A PyTorch implementation of Wireless Masked Autoencoder (WiMAE) and Contrastive WiMAE (ContraWiMAE) for wireless channel data analysis.

## Quick Start

### Training with Optimized Data Loading

The default training configuration now uses `OptimizedPreloadedDataset` for maximum training speed:

```bash
# Train with default configuration
python examples/training_with_optimized_data.py --data-dir /path/to/your/npz/files

# Debug mode (small dataset for testing)
python examples/training_with_optimized_data.py --data-dir /path/to/your/npz/files --debug

# Use custom config
python examples/training_with_optimized_data.py --config your_config.yaml --data-dir /path/to/your/npz/files
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

## Project Structure

```
WirelessContrastiveMaskedLearning/
├── configs/                    # Configuration files
│   ├── default_training.yaml   # Default training config
│   ├── default_encoding.yaml   # Encoding config
│   └── default_downstream.yaml # Downstream tasks config
├── examples/                   # Example scripts
│   ├── training_with_optimized_data.py
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