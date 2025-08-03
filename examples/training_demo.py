"""
Training Demonstration for WiMAE and ContraWiMAE

This notebook demonstrates how to train both WiMAE and ContraWiMAE models
using real wireless channel data from the pretrain folder.

Convert this to a Jupyter notebook for interactive exploration.
"""

# %%
# Cell 1: Imports and Setup
import sys
import torch
import numpy as np
import yaml
from pathlib import Path

# Add parent directory to path for imports
try:
    # For Python scripts
    sys.path.append(str(Path(__file__).parent.parent))
except NameError:
    # For Jupyter notebooks
    sys.path.append(str(Path().cwd().parent))

# WiMAE imports
from wimae.training.train_wimae import WiMAETrainer
from wimae.training.train_contramae import ContraWiMAETrainer
from wimae.models.base import WiMAE
from wimae.models.contramae import ContraWiMAE

print("All imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# %%
# Cell 2: Data Overview
data_path = "data/pretrain"
npz_files = list(Path(data_path).glob("*.npz"))

print(f"Available datasets: {len(npz_files)} cities")
for file in sorted(npz_files):
    file_size = file.stat().st_size / (1024*1024)  # MB
    print(f"  • {file.name}: {file_size:.1f} MB")

# Load one file to check data structure
with np.load(npz_files[0]) as sample_data:
    print(f"\nSample data structure from {npz_files[0].name}:")
    for key, value in sample_data.items():
        if hasattr(value, 'shape'):
            print(f"  • {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  • {key}: {value}")

# %%
# Cell 3: Configuration Loading
config_path = "configs/default_training.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print("Configuration loaded successfully!")
print(f"Model type: {config['model']['type']}")
print(f"Encoder dimensions: {config['model']['encoder_dim']}")
print(f"Batch size: {config['training']['batch_size']}")
print(f"Training epochs: {config['training']['epochs']}")
print(f"Learning rate: {config['training']['optimizer']['lr']}")

# Adjust config for demo (shorter training)
config['training']['epochs'] = 10
config['training']['batch_size'] = 32
config['data']['debug_size'] = 1000  # Use smaller dataset for demo
print(f"\nAdjusted for demo:")
print(f"Epochs: {config['training']['epochs']}")
print(f"Batch size: {config['training']['batch_size']}")
print(f"Debug size: {config['data']['debug_size']}")

# %%
# Cell 4: WiMAE Training Setup
print("Setting up WiMAE training...")

# Create WiMAE trainer (model will be created during initialization)
wimae_trainer = WiMAETrainer(config=config)

print("WiMAE trainer initialized")

# Get model information
wimae_info = wimae_trainer.model.get_model_info()
print(f"WiMAE model created:")
for key, value in wimae_info.items():
    print(f"  • {key}: {value}")

# %%
# Cell 5: WiMAE Training Execution
print("Starting WiMAE training...")

try:
    # Start training (dataloaders will be set up automatically)
    wimae_trainer.train()
    print("WiMAE training completed successfully!")
    
except Exception as e:
    print(f"Training failed: {e}")
    print("This is expected in a demo - check your data paths and configuration")

# %%
# Cell 6: ContraWiMAE Training Setup
print("Setting up ContraWiMAE training...")

# Create ContraWiMAE trainer (model will be created during initialization)
contrawimae_trainer = ContraWiMAETrainer(config=config)

print("ContraWiMAE trainer initialized")

# Get model information
contrawimae_info = contrawimae_trainer.model.get_model_info()
print(f"ContraWiMAE model created:")
for key, value in contrawimae_info.items():
    print(f"  • {key}: {value}")

# %%
# Cell 7: ContraWiMAE Training Execution
print("Starting ContraWiMAE training...")

try:
    # Start training (dataloaders will be set up automatically)
    contrawimae_trainer.train()
    print("ContraWiMAE training completed successfully!")
    
except Exception as e:
    print(f"Training failed: {e}")
    print("This is expected in a demo - check your data paths and configuration")

# %%
# Cell 8: Model Comparison and Analysis
print("Model Comparison Summary")
print("=" * 50)

# Get model info for comparison
wimae_info = wimae_trainer.model.get_model_info()
contrawimae_info = contrawimae_trainer.model.get_model_info()

print(f"WiMAE parameters: {wimae_info['total_parameters']:,}")
print(f"ContraWiMAE parameters: {contrawimae_info['total_parameters']:,}")
print(f"Additional parameters in ContraWiMAE: {contrawimae_info['total_parameters'] - wimae_info['total_parameters']:,}")

print("\nDetailed Model Information:")
print(f"WiMAE:")
for key, value in wimae_info.items():
    if key != 'model_type':
        print(f"  • {key}: {value}")

print(f"ContraWiMAE:")
for key, value in contrawimae_info.items():
    if key != 'model_type':
        print(f"  • {key}: {value}")

print("\nKey Differences:")
print("• WiMAE: Reconstruction-only (masked autoencoder)")
print("• ContraWiMAE: Reconstruction + contrastive learning")
print("• ContraWiMAE includes contrastive head and augmentation")
print("• ContraWiMAE has dual loss: reconstruction + contrastive")

print(f"\nTraining Configuration (from {config_path}):")
print(f"• Model type: {config['model']['type']}")
print(f"• Reconstruction weight: {config['training']['reconstruction_weight']}")
print(f"• Contrastive weight: {config['training']['contrastive_weight']}")
print(f"• Mask ratio: {config['model']['mask_ratio']}")
print(f"• SNR range: {config['model']['snr_min']}-{config['model']['snr_max']} dB")
print(f"• Temperature: {config['model']['temperature']}")
print(f"• Learning rate: {config['training']['optimizer']['lr']}")
print(f"• Patience: {config['training']['patience']}")

# %%
# Cell 9: Checkpoints and Model Loading
print("Checkpoint Management")
print("=" * 30)

# Check available checkpoints
wimae_checkpoints = list(Path("checkpoints/wimae").glob("*.pth")) if Path("checkpoints/wimae").exists() else []
contrawimae_checkpoints = list(Path("checkpoints/contrawimae").glob("*.pth")) if Path("checkpoints/contrawimae").exists() else []

print(f"WiMAE checkpoints: {len(wimae_checkpoints)}")
for ckpt in wimae_checkpoints:
    print(f"  • {ckpt.name}")

print(f"ContraWiMAE checkpoints: {len(contrawimae_checkpoints)}")
for ckpt in contrawimae_checkpoints:
    print(f"  • {ckpt.name}")

# Example of loading a checkpoint (if available)
if wimae_checkpoints:
    print(f"\nExample: Loading WiMAE checkpoint...")
    try:
        checkpoint = torch.load(wimae_checkpoints[0], map_location='cpu')
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation loss: {checkpoint.get('val_loss', 'unknown')}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")

print("\nNext Steps:")
print("• Use trained models for encoding wireless channel data")
print("• Fine-tune on downstream tasks (beam prediction, LoS classification)")
print("• Experiment with different mask ratios and loss weights")
print("• Analyze learned representations") 