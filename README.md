# Wireless Contrastive Masked Learning (WiMAE & ContraWiMAE)

A comprehensive implementation of Wireless Masked Autoencoder (WiMAE) and Contrastive Wireless Masked Autoencoder (ContraWiMAE) for wireless channel modeling and downstream tasks.

## Overview

This repository provides a clean, modular implementation of:
- **WiMAE**: A masked autoencoder specifically designed for wireless channel data
- **ContraWiMAE**: An extension of WiMAE that incorporates contrastive learning for improved representations
- **Downstream Tasks**: Beam prediction and Line-of-Sight (LOS) classification modules
- **Training & Encoding**: Complete training and encoding pipelines

## Features

- **Modular Design**: Clean separation of concerns with extensible architecture
- **OOP Approach**: Well-structured object-oriented design following Python conventions
- **Flexible Configuration**: YAML-based configuration with command-line argument support
- **Dynamic Encoding**: Support for both pre-computed embeddings and on-the-fly generation
- **Fine-tuning Support**: Both frozen encoder and end-to-end fine-tuning modes
- **Comprehensive Evaluation**: Multiple metrics for downstream tasks
- **Easy Reproduction**: Pre-trained checkpoints and reproducibility scripts

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd WirelessContrastiveMaskedLearning

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### 1. Training a WiMAE Model

```python
from wimae.training import WiMAETrainer
from wimae.models import WiMAE

# Load configuration
trainer = WiMAETrainer.from_config("configs/default_training.yaml")

# Train the model
trainer.train()
```

### 2. Training a ContraWiMAE Model

```python
from wimae.training import ContraWiMAETrainer
from wimae.models import ContraWiMAE

# Load configuration
trainer = ContraWiMAETrainer.from_config("configs/default_training.yaml")

# Train the model
trainer.train()
```

### 3. Encoding Data

```python
from wimae.encoding import Encoder

# Load trained model
encoder = Encoder.from_checkpoint("checkpoints/wimae_model.pt")

# Encode data
embeddings = encoder.encode(data_loader)
```

### 4. Downstream Tasks

```python
from wimae.downstream import BeamPredictionTask, LOSClassificationTask

# Beam prediction
bp_task = BeamPredictionTask.from_config("configs/default_downstream.yaml")
results = bp_task.evaluate(model_checkpoint="checkpoints/wimae_model.pt")

# LOS classification
los_task = LOSClassificationTask.from_config("configs/default_downstream.yaml")
results = los_task.evaluate(model_checkpoint="checkpoints/wimae_model.pt")
```

## Project Structure

```
WirelessContrastiveMaskedLearning/
├── configs/                 # Configuration files
│   ├── default_training.yaml
│   ├── default_encoding.yaml
│   └── default_downstream.yaml
├── wimae/                   # Main package
│   ├── models/             # Model implementations
│   │   ├── base.py         # Base WiMAE model
│   │   ├── contramae.py    # ContraWiMAE implementation
│   │   ├── encoder.py      # Encoder module
│   │   ├── decoder.py      # Decoder module
│   │   └── modules/        # Supporting modules
│   ├── training/           # Training pipelines
│   │   ├── trainer.py      # Base trainer
│   │   ├── train_wimae.py  # WiMAE training
│   │   └── train_contramae.py # ContraWiMAE training
│   ├── encoding/           # Encoding utilities
│   │   └── encoder.py      # Encoding interface
│   └── downstream/         # Downstream tasks
│       ├── tasks/          # Task implementations
│       │   ├── beam_prediction.py
│       │   └── los_classification.py
│       └── trainer.py      # Downstream trainer
├── examples/               # Example scripts and notebooks
├── docs/                   # Documentation
├── checkpoints/            # Pre-trained models
├── tests/                  # Unit tests
└── requirements.txt        # Dependencies
```

## Configuration

The project uses YAML configuration files with command-line argument support. Default configurations are provided for all components:

- **Training**: Model architecture, training parameters, data settings
- **Encoding**: Encoding parameters, output formats
- **Downstream**: Task-specific parameters, evaluation metrics

### Example Configuration

```yaml
# configs/default_training.yaml
model:
  type: "wimae"  # or "contramae"
  encoder_dim: 256
  encoder_layers: 12
  encoder_nhead: 8
  decoder_layers: 8
  decoder_nhead: 8
  mask_ratio: 0.75

training:
  batch_size: 128
  learning_rate: 0.0001
  epochs: 100
  device: "cuda"
  val_split: 0.1

data:
  data_path: "path/to/data"
  patch_size: [4, 4]
```

## Data Format

The implementation expects data in `.npz` format with the following structure:
- Channel data as numpy arrays
- Labels for downstream tasks
- Metadata for data organization

## Pre-trained Models

Pre-trained checkpoints are available for both WiMAE and ContraWiMAE models. These can be used for:
- Encoding new data
- Fine-tuning for downstream tasks
- Reproducing published results

## Downstream Tasks

### Beam Prediction
Multi-class classification task for predicting optimal beam indices from codebooks of various sizes (16, 32, 64, 128, 256).

**Metrics**: Top-1 accuracy, Top-3 accuracy, Cross-entropy loss

### LOS Classification
Binary classification task for predicting Line-of-Sight vs Non-Line-of-Sight conditions.

**Metrics**: Accuracy, Precision, Recall, F1-score, AUC, Specificity, NPV

## Usage Examples

See the `examples/` directory for detailed usage examples and notebooks:

- `examples/training_example.py` - Complete training pipeline
- `examples/encoding_example.py` - Data encoding examples
- `examples/downstream_example.py` - Downstream task evaluation
- `examples/reproduction.ipynb` - Reproducing published results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your-paper,
  title={Wireless Contrastive Masked Learning},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support, please open an issue on GitHub or contact the maintainers. 