# Wireless Contrastive Masked Learning (WiMAE & ContraWiMAE)

A PyTorch implementation of Wireless Masked Autoencoders (WiMAE) and Contrastive Wireless Masked Autoencoders (ContraWiMAE) for wireless channel data modeling and representation learning.

## 🚀 Overview

This repository provides implementations of two transformer-based models designed specifically for wireless channel data:

- **WiMAE (Wireless Masked Autoencoder)**: A masked autoencoder that learns representations of wireless channel matrices through reconstruction tasks
- **ContraWiMAE (Contrastive WiMAE)**: Extends WiMAE with contrastive learning to create representations invariant to channel augmentations

Both models use patch-based processing of complex-valued wireless channel matrices and transformer architectures optimized for wireless data characteristics.

## 📋 Key Features

- **🔧 Modular Architecture**: Clean separation of encoder, decoder, and contrastive components
- **📊 Complex Data Support**: Native handling of complex-valued wireless channel matrices
- **🎯 Patch-Based Processing**: Efficient patching strategy for wireless channel data
- **🔄 Contrastive Learning**: Advanced contrastive learning with wireless-specific augmentations
- **⚡ Efficient Training**: Optimized data loading and training pipeline
- **🎛️ Flexible Configuration**: YAML-based configuration system
- **📈 Comprehensive Logging**: TensorBoard integration and checkpoint management
- **🧪 Well-Tested**: Extensive test suite covering all components

## 🛠️ Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

### Install from Source

```bash
git clone https://github.com/yourusername/WirelessContrastiveMaskedLearning.git
cd WirelessContrastiveMaskedLearning
pip install -e .
```

### Dependencies

```bash
pip install -r requirements.txt
```

## 🏗️ Repository Structure

```
WirelessContrastiveMaskedLearning/
├── configs/                    # Configuration files
│   ├── default_training.yaml   # Default training configuration
│   ├── scenario_split_simple.yaml
│   └── scenario_split_test.yaml
├── examples/                   # Example usage
│   └── training_demo.ipynb     # Training demonstration
├── wimae/                      # Main package
│   ├── models/                 # Model implementations
│   │   ├── base.py            # WiMAE base model
│   │   ├── contramae.py       # ContraWiMAE model
│   │   └── modules/           # Model components
│   │       ├── encoder.py     # Transformer encoder
│   │       ├── decoder.py     # Transformer decoder
│   │       ├── contrastive_head.py  # Contrastive learning head
│   │       ├── patching.py    # Patch extraction/reconstruction
│   │       ├── masking.py     # Masking strategies
│   │       ├── augmentations.py  # Data augmentations
│   │       └── pos_encodings.py  # Positional encodings
│   └── training/              # Training utilities
│       ├── trainer.py         # Base trainer class
│       ├── train_wimae.py     # WiMAE trainer
│       ├── train_contramae.py # ContraWiMAE trainer
│       └── data_utils.py      # Data loading utilities
├── tests/                     # Unit tests
├── docs/                      # Documentation
└── data/                      # Data directory (user-provided)
```

## 🚀 Quick Start

### 1. Prepare Your Data

Organize your wireless channel data as NPZ files in the data directory:

```
data/
├── pretrain/
│   ├── channels_001.npz
│   ├── channels_002.npz
│   └── ...
```

Each NPZ file should contain complex-valued channel matrices with shape `(N, H, W)` where:
- `N`: Number of channel realizations
- `H, W`: Spatial dimensions (e.g., antennas, subcarriers)

### 2. Configure Training

Edit the configuration file or use the default:

```yaml
# configs/default_training.yaml
model:
  type: "wimae"  # or "contramae"
  patch_size: [1, 16]
  encoder_dim: 64
  encoder_layers: 12
  mask_ratio: 0.6

data:
  data_dir: "data/pretrain"
  normalize: true
  val_split: 0.2

training:
  batch_size: 64
  epochs: 100
  device: "cuda:0"
```

### 3. Train Models

#### Train WiMAE (Base Model)

```python
from wimae.training import WiMAETrainer
import yaml

# Load configuration
with open("configs/default_training.yaml", "r") as f:
    config = yaml.safe_load(f)

config["model"]["type"] = "wimae"

# Create and train
trainer = WiMAETrainer(config)
trainer.train()
```

#### Train ContraWiMAE (With Contrastive Learning)

```python
from wimae.training import ContraWiMAETrainer
import yaml

# Load configuration
with open("configs/default_training.yaml", "r") as f:
    config = yaml.safe_load(f)

config["model"]["type"] = "contramae"

# Create and train
trainer = ContraWiMAETrainer(config)
trainer.train()
```

### 4. Transfer Learning: WiMAE → ContraWiMAE

Load a pretrained WiMAE model into ContraWiMAE for transfer learning:

```python
# Create ContraWiMAE trainer
config["model"]["type"] = "contramae"
contra_trainer = ContraWiMAETrainer(config)

# Load WiMAE weights (encoder/decoder), keep contrastive head random
contra_trainer.load_checkpoint("path/to/wimae_checkpoint.pth", 
                               model_only=True, strict=False)

# Continue training with contrastive learning
contra_trainer.train()
```

## 📊 Model Architectures

### WiMAE (Wireless Masked Autoencoder)

```
Input: Complex Channel Matrix [B, H, W]
    ↓
Patcher: Extract patches [B, N_patches, patch_dim]
    ↓
Encoder: Transformer with masking [B, N_visible, hidden_dim]
    ↓
Decoder: Reconstruct masked patches [B, N_patches, patch_dim]
    ↓
Output: Reconstructed Channel Matrix [B, H, W]
```

**Key Components:**
- **Patch Embedding**: Linear projection of flattened patches
- **Positional Encoding**: Learnable or sinusoidal position embeddings
- **Masking Strategy**: Random masking of input patches
- **Transformer Encoder**: Multi-head self-attention with GELU activation
- **Transformer Decoder**: Cross-attention between encoded patches and mask tokens

### ContraWiMAE (Contrastive WiMAE)

Extends WiMAE with contrastive learning:

```
Input: Complex Channel Matrix [B, H, W]
    ↓
Augmentation: Apply wireless-specific augmentations
    ↓
WiMAE Processing: Same as above
    ↓
Contrastive Head: Project to contrastive space [B, contrastive_dim]
    ↓
Contrastive Loss: InfoNCE between original and augmented views
```

**Additional Components:**
- **Contrastive Head**: MLP projection with L2 normalization
- **Augmentations**: SNR variation, frequency shifts, phase rotations
- **InfoNCE Loss**: Temperature-scaled contrastive learning

## ⚙️ Configuration Options

### Model Configuration

```yaml
model:
  type: "wimae"                    # "wimae" or "contramae"
  patch_size: [1, 16]             # Patch dimensions [height, width]
  encoder_dim: 64                 # Hidden dimension
  encoder_layers: 12              # Number of encoder layers
  encoder_nhead: 16               # Number of attention heads
  decoder_layers: 4               # Number of decoder layers
  decoder_nhead: 8                # Decoder attention heads
  mask_ratio: 0.6                 # Fraction of patches to mask
  
  # ContraWiMAE specific
  contrastive_dim: 64             # Contrastive projection dimension
  temperature: 0.1                # Contrastive loss temperature
  snr_min: 0.0                    # Minimum SNR for augmentation
  snr_max: 30.0                   # Maximum SNR for augmentation
```

### Training Configuration

```yaml
training:
  batch_size: 64
  epochs: 100
  device: "cuda:0"
  
  optimizer:
    type: "adam"
    lr: 0.0003
    weight_decay: 0.0
    
  scheduler:
    type: "cosine"
    T_max: 100
    
  # ContraWiMAE loss weights
  reconstruction_weight: 0.9      # Weight for reconstruction loss
  contrastive_weight: 0.1         # Weight for contrastive loss
```

## 📈 Training Features

### Checkpointing and Resuming

```python
# Save checkpoints automatically
trainer.train()  # Saves to runs/{model_type}_{exp_name}/

# Resume from checkpoint
trainer.load_checkpoint("path/to/checkpoint.pth")
trainer.train()

# Load only model weights (for inference)
trainer.load_checkpoint("path/to/checkpoint.pth", model_only=True)
```

### Transfer Learning

```python
# Load WiMAE into ContraWiMAE with strict=False
# Missing contrastive_head weights will be warned about
trainer.load_checkpoint("wimae_checkpoint.pth", strict=False)
```

### Monitoring and Logging

- **TensorBoard**: Automatic logging of losses, learning rates, and metrics
- **Console Output**: Progress bars with real-time metrics
- **Checkpoints**: Best and latest model checkpoints saved automatically

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_models.py              # Model tests
pytest tests/test_data_loading.py        # Data loading tests
pytest tests/test_contrastive_learning.py # Contrastive learning tests
```

## 📚 Examples and Tutorials

Check out the `examples/` directory:

- **`training_demo.ipynb`**: Complete training demonstration with both WiMAE and ContraWiMAE

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions, issues, or collaboration opportunities:

- Create an issue on GitHub
- Email: your.email@example.com

## 🙏 Acknowledgments

- Inspired by Vision Transformer (ViT) and Masked Autoencoder (MAE) architectures
- Built with PyTorch and modern transformer implementations
- Thanks to the wireless communications research community

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{wireless_contrastive_masked_learning,
  title={Wireless Contrastive Masked Learning},
  author={Research Team},
  year={2024},
  url={https://github.com/yourusername/WirelessContrastiveMaskedLearning}
}
``` 