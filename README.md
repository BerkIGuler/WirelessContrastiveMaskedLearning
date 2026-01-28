# ContraWiMAE: Wireless Contrastive Masked Learning
## Overview

This repository is the official implementation of the NeurIPS'25 best paper runner-up awardee at the [AI4NextG Workshop](https://ai4nextg.github.io/#).

["Robust Channel Representation for Wireless: A Multi-Task Masked Contrastive Approach"](https://openreview.net/forum?id=KXNDs9ZGb9) (Also [see](https://arxiv.org/abs/2505.09160))

We propose a transformer-based foundation model designed specifically for wireless channel representation learning:

- **ContraWiMAE (Wireless Masked Autoencoder)**: A transformer-based foundation model pretrained on realistic wireless channel datasets using a novel contrastive learning objective alongside MAE style reconstruction in a unified multi-task framework. ContraWiMAE uses patch-based processing of complex-valued wireless channel matrices and demonstrates an impressive performance across channel estimation, beam management, and channel characterization tasks.

## Installation

### Install from Source

```bash
git clone https://github.com/BerkIGuler/WirelessContrastiveMaskedLearning.git
cd WirelessContrastiveMaskedLearning

# Install Git LFS (required for downloading data files)
git lfs install

# Pull data files (managed with Git LFS)
git lfs pull

# Install the package
pip install -e .
```

**Note**: The repository uses [Git LFS](https://git-lfs.github.com/) to manage data files. Make sure to install Git LFS and run `git lfs pull` to download the sample data files.

### Installation Options

**Basic installation** (core dependencies only):
```bash
pip install -e .
```

**With documentation tools**:
```bash
pip install -e ".[docs]"
```

**Development installation** (includes docs + testing):
```bash
pip install -e ".[dev]"
```

**All dependencies** (alternative):
```bash
pip install -r requirements.txt
```

### Dependencies

The package has minimal core dependencies with optional extras for documentation and development.

## Repository Structure

```
WirelessContrastiveMaskedLearning/
├── configs/                    # Configuration files
│   └── default_training.yaml   # Default training configuration
├── examples/                   # Example usage
│   └── training_demo.ipynb     # Training demonstration
├── contrawimae/               # Main package
│   ├── models/                 # Model implementations
│   │   ├── wimae.py           # WiMAE model
│   │   ├── contrawimae.py     # ContraWiMAE model
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
├── docs/                      # Automated documentation (Sphinx)
├── scripts/                   # Build and utility scripts
└── data/                      # Data directory (user-provided)
```

## Quick Start

### 1. Prepare Your Data

The complete dataset used in the paper can be accessed [here](https://huggingface.co/datasets/BerkIGuler/ContraWiMAEDataset). To use your own data, organize it as NPZ files in the data directory:

```
data/
├── pretrain/
│   ├── channels_001.npz
│   ├── channels_002.npz
│   └── ...
```

Each NPZ file should contain a `'channels'` key with complex-valued channel matrices of shape `(N, 1, H, W)` where:
- `N`: Number of channel realizations  
- `H, W`: Spatial dimensions (e.g., antennas, subcarriers)

**Note on Repository Data**: The data files in the `data/` folder are managed using [Git LFS](https://git-lfs.github.com/) and contain only a simple sample dataset for demonstration purposes.

### 2. Configure Training

Edit the configuration file or use the [default configuration](configs/default_training.yaml):

### 3. Train Models

#### Train WiMAE (Base Model)

```python
from contrawimae.training.train_wimae import WiMAETrainer
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
from contrawimae.training.train_contramae import ContraWiMAETrainer
import yaml

# Load configuration
with open("configs/default_training.yaml", "r") as f:
    config = yaml.safe_load(f)

config["model"]["type"] = "contrawimae"

# Create and train
trainer = ContraWiMAETrainer(config)
trainer.train()
```

### 4. WiMAE → ContraWiMAE

Load a pretrained WiMAE model into ContraWiMAE:

```python
# Create ContraWiMAE trainer
config["model"]["type"] = "contrawimae"
contra_trainer = ContraWiMAETrainer(config)

# Load WiMAE weights (encoder/decoder), keep contrastive head random
contra_trainer.load_checkpoint("path/to/wimae_checkpoint.pth", 
                               model_only=True, strict=False)

# Continue training with contrastive learning
contra_trainer.train()
```

## Configuration Options

The configuration system uses YAML files with hierarchical organization. Here's a comprehensive breakdown of all available options:

### Model Configuration

```yaml
model:
  # Model Architecture Selection
  type: "wimae"                    # Options: "wimae", "contrawimae"
                                   # - "wimae": Base masked autoencoder model
                                   # - "contrawimae": Multi-task model with contrastive learning
  
  # Patch Processing
  patch_size: [16, 1]             # Patch dimensions [height, width]
                                   # - [16, 1]: Process in frequency domain (16 subcarriers)
                                   # - Determines input chunking for transformer
                                   # - Affects memory usage and spatial relationships
  
  # Encoder Architecture
  encoder_dim: 64                 # Hidden/embedding dimension
                                   # - Controls model capacity and memory usage
                                   # - Paper shows 64 is optimal for efficiency
                                   # - Higher values (128, 256) for complex tasks
  
  encoder_layers: 12              # Number of transformer encoder layers
                                   # - Paper shows diminishing returns beyond 12
                                   # - More layers = better representations but slower
                                   # - Range: 6-24 layers depending on complexity
  
  encoder_nhead: 16               # Number of attention heads in encoder
                                   # - Must divide encoder_dim evenly
                                   # - More heads capture diverse attention patterns
                                   # - Common: 8, 16, or encoder_dim//4
  
  # Decoder Architecture
  decoder_layers: 4               # Number of transformer decoder layers
                                   # - Lightweight decoder for reconstruction
                                   # - Paper shows deeper decoders improve quality
                                   # - Range: 2-8 layers
  
  decoder_nhead: 8                # Number of attention heads in decoder
                                   # - Usually fewer than encoder heads
                                   # - Must divide encoder_dim evenly
  
  # Masking Strategy
  mask_ratio: 0.9                 # Fraction of patches to mask (0.0-0.9)
                                   # - Higher ratios force better representations
                                   # - Too high (>0.95) makes task too difficult
                                   # - Default: 0.9 for strong pretraining
  
  # ContraWiMAE Specific Parameters
  contrastive_dim: 64             # Contrastive projection dimension
                                   # - Output dimension of contrastive head
                                   # - Usually same as encoder_dim
                                   # - Affects contrastive learning quality
  
  temperature: 0.2                # Contrastive loss temperature
                                   # - Controls hardness of negative sampling
                                   # - Lower values (0.1-0.2) = harder negatives
                                   # - Higher values (0.3-0.5) = softer training
  
  # Augmentation Parameters (ContraWiMAE)
  snr_min: 5.0                    # Minimum SNR for noise injection (dB)
  snr_max: 40.0                   # Maximum SNR for noise injection (dB)
                                   # - Controls noise level for positive pairs
                                   # - Wider range = more diverse augmentations
```

### Data Configuration

```yaml
data:
  # Data Source
  data_dir: "data/pretrain"       # Directory containing NPZ files
                                   # - All .npz files in this directory will be loaded
                                   # - Files must contain 'channels' key
  
  # Data Processing
  normalize: true                 # Enable data normalization
                                   # - Normalizes real/imaginary parts separately
                                   # - Essential for stable training
                                   # - Uses complex-valued statistics
  
  val_split: 0.1                  # Validation split ratio (0.0-0.5)
                                   # - Fraction of data for validation
                                   # - 0.1 = 90% train, 10% validation
                                   # - Applied after all data loading
  
  debug_size: null                # Limit dataset size for debugging
                                   # - null: Use full dataset
                                   # - Integer: Limit to N samples total
                                   # - Useful for testing: 1000, 5000, etc.
  
  # Statistics Handling (Two Options)
  calculate_statistics: true      # Calculate statistics from training data
                                   # - true: Compute mean/std from loaded data
                                   # - false: Use pre-computed statistics below
  
  statistics:                     # Pre-computed normalization statistics
    real_mean: 0.021             # Mean of real parts
    real_std: 30.745             # Standard deviation of real parts  
    imag_mean: -0.010            # Mean of imaginary parts
    imag_std: 30.705             # Standard deviation of imaginary parts
                                   # - Used when calculate_statistics: false
                                   # - Must match your dataset distribution
  
  # Advanced Data Loading (Optional) - Scenario-based splitting
  scenario_split_config:           # Embedded config (preferred)
    train_patterns:
      - "data_[0-6]\.npz"          # Regex patterns for training files
      - "train_.*\.npz"             # Files starting with "train_"
    val_patterns:
      - "data_[78]\.npz"           # Regex patterns for validation files
      - "val_.*\.npz"              # Files starting with "val_"
    test_patterns:
      - "data_9\.npz"              # Regex patterns for test files
      - "test_.*\.npz"             # Files starting with "test_"
```

### Training Configuration

```yaml
training:
  # Basic Training Parameters
  batch_size: 64                  # Training batch size
                                   # - Affects memory usage and gradient stability
                                   # - Larger batches (128, 256) for better gradients
                                   # - Smaller batches (32, 64) for limited memory
  
  epochs: 100                      # Number of training epochs
                                   # - Default: 100 for quick experimentation
                                   # - Paper uses 3000 for full pretraining
                                   # - Early stopping prevents overfitting
  
  num_workers: 4                  # Number of data loading workers
                                   # - Parallel data loading processes
                                   # - Usually 2-8 depending on CPU cores
                                   # - 0 = single-threaded loading
  
  device: "cuda:0"                # Training device
                                   # - "cuda:0", "cuda:1": Specific GPU
                                   # - "cpu": CPU training (slow)
                                   # - "auto": Automatic GPU detection
  
  # Optimizer Configuration
  optimizer:
    type: "adam"                  # Optimizer type
                                   # - "adam": Adaptive learning, good default
                                   # - "adamw": Adam with weight decay
                                   # - "sgd": Simple gradient descent
    
    lr: 0.0003                    # Learning rate
                                   # - 0.0003 is typical for transformers
                                   # - Lower (1e-4) for fine-tuning
                                   # - Higher (1e-3) for quick experiments
    
    weight_decay: 0.001            # L2 regularization strength
                                   # - 0.0: No regularization
                                   # - 0.001-0.01: Standard regularization
                                   # - Prevents overfitting
    
    betas: [0.9, 0.999]          # Adam momentum parameters
                                   # - [0.9, 0.999]: Standard Adam values
                                   # - First: gradient momentum
                                   # - Second: squared gradient momentum
    
    # SGD-specific (when type: "sgd")
    momentum: 0.9                 # SGD momentum (0.0-1.0)
  
  # Learning Rate Scheduler
  scheduler:
    type: "cosine"                # Scheduler type
                                   # - "cosine": Cosine annealing (smooth decay)
                                   # - "step": Step decay at intervals
                                   # - "exponential": Exponential decay
    
    # Cosine scheduler parameters
    T_max: 3000                   # Maximum epochs for cosine cycle
                                   # - Can be set to full training cycle (e.g., 3000)
                                   # - Works with early stopping (actual epochs may be less)
                                   # - Full cosine cycle over T_max epochs
    
    eta_min: 0.000003            # Minimum learning rate
                                   # - Learning rate at end of cycle
                                   # - Prevents complete learning stop
    
    # Step scheduler parameters (when type: "step")
    step_size: 1000               # Epochs between LR reductions
    gamma: 0.1                    # LR multiplication factor
    
    # Exponential scheduler parameters (when type: "exponential")
    gamma: 0.95                   # LR decay factor per epoch
  
  # Loss Function
  loss: "mse"                     # Reconstruction loss type
                                   # - "mse": Mean squared error (default)
                                   # - "l1": Mean absolute error
                                   # - "huber": Huber loss (robust to outliers)
  
  # Loss Weighting (ContraWiMAE Multi-task)
  reconstruction_weight: 0.9      # Weight for reconstruction loss
                                   # - contrastive_weight is automatically derived as (1 - reconstruction_weight)
                                   # - Higher reconstruction weight preserves base model
                                   # - Lower reconstruction weight emphasizes discriminability
  
  # Training Stability
  gradient_clip_val: 1.0          # Gradient clipping threshold
                                   # - Prevents exploding gradients
                                   # - 0.0: No clipping
                                   # - 1.0-5.0: Standard values
  
  # Early Stopping
  patience: 5                     # Epochs to wait for improvement
                                   # - Training stops if no improvement
                                   # - Prevents overfitting
                                   # - 5-20 epochs typical
  
  min_delta: 0.0001              # Minimum improvement threshold
                                   # - Smaller improvements ignored
                                   # - Prevents stopping on noise
  
  # Checkpointing
  save_checkpoint_every_n: 10     # Save frequency (epochs)
                                   # - Regular checkpoint saves
                                   # - Allows training resumption
                                   # - 5-20 epochs typical
  
  save_best_only: true           # Only save best validation model
                                   # - true: Save only when validation improves
                                   # - false: Save at every interval
```

### Logging Configuration

```yaml
logging:
  log_dir: "runs"                 # Base directory for experiment logs
                                   # - Contains tensorboard logs, checkpoints
                                   # - Organized by experiment name
  
  tensorboard: true               # Enable TensorBoard logging
                                   # - Loss curves, learning rate tracking
                                   # - View with: tensorboard --logdir runs
  
  log_every_n_steps: 100         # Logging frequency (training steps)
                                   # - How often to log metrics
                                   # - 50-200 steps typical
  
  exp_name: "training_demo"      # Experiment name
                                   # - Creates subdirectory in log_dir
                                   # - null: Auto-generate timestamp
                                   # - Custom: "wimae_experiment_1"
```

### Advanced: Scenario Split Configuration

For complex data organization, use scenario-based splitting:

```yaml
# Embedded in main config file (preferred)
data:
  scenario_split_config:
    train_patterns:
      - "data_[0-6]\.npz"           # Regex patterns for training files
      - "train_.*\.npz"             # Files starting with "train_"
    val_patterns:
      - "data_[78]\.npz"            # Regex patterns for validation files
      - "val_.*\.npz"               # Files starting with "val_"
    test_patterns:
      - "data_9\.npz"               # Regex patterns for test files
      - "test_.*\.npz"              # Files starting with "test_"
```

## Training Features

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
### Monitoring and Logging

- **TensorBoard**: Automatic logging of losses, learning rates, and metrics
- **Console Output**: Progress bars with real-time metrics
- **Checkpoints**: Best and latest model checkpoints saved automatically

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_models.py              # Model tests
pytest tests/test_data_loading.py        # Data loading tests
pytest tests/test_contrastive_learning.py # Contrastive learning tests
```

## Documentation

This package includes comprehensive, automatically-generated API documentation.

### Building Documentation

**Install documentation dependencies**:
```bash
pip install -e ".[docs]"
# OR
make docs-install
```

**Build and view documentation**:
```bash
# Build documentation
make docs

# Serve locally at http://localhost:8000
make docs-serve
```

**Available commands**:
```bash
make docs        # Build complete documentation
make docs-serve  # Serve documentation locally  
make docs-clean  # Clean build artifacts
make help        # Show all available commands
```
## Examples and Tutorials

Check out the `examples/` directory:

- **`training_demo.ipynb`**: Complete training demonstration with both WiMAE and ContraWiMAE

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions, issues, or collaboration opportunities:

- Create an issue on GitHub or email the author

## Citation

This repository implements the methods described in our paper. If you use this code in your research, please cite:

1) [Robust Channel Representation for Wireless: A Multi-Task Masked Contrastive Approach](https://openreview.net/forum?id=KXNDs9ZGb9)
```bibtex
@inproceedings{guler2025robust,
  title={Robust Channel Representation for Wireless: A Multi-Task Masked Contrastive Approach},
  author={Guler, Berkay and Geraci, Giovanni and Jafarkhani, Hamid},
  booktitle={NeurIPS 2025 Workshop on AI for Next Generation Wireless (AI4NextG)},
  year={2025},
  url={https://openreview.net/forum?id=KXNDs9ZGb9}
}
```
2) [A Multi-Task Foundation Model for Wireless Channel Representation Using Contrastive and Masked Autoencoder Learning](https://arxiv.org/abs/2505.09160)
```bibtex
@article{guler2025multitask,
  title={A Multi-Task Foundation Model for Wireless Channel Representation Using Contrastive and Masked Autoencoder Learning},
  author={Guler, Berkay and Geraci, Giovanni and Jafarkhani, Hamid},
  journal={arXiv preprint arXiv:2505.09160},
  year={2025},
  url={https://arxiv.org/abs/2505.09160}
}
```
