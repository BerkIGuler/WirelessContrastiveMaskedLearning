#!/usr/bin/env python3
"""
Example script demonstrating WiMAE and ContraWiMAE training with different data loading approaches.
"""

import yaml
import torch
from pathlib import Path

from wimae.training.train_wimae import WiMAETrainer
from wimae.training.train_contramae import ContraWiMAETrainer
from wimae.models.wimae import WiMAE
from wimae.models.contramae import ContraWiMAE


def example_simple_data_loading():
    """Example: Train with simple data loading (all NPZ files)."""
    print("=" * 60)
    print("EXAMPLE 1: Simple Data Loading")
    print("=" * 60)
    
    # Load configuration
    config_path = "configs/default_training.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Using config: {config_path}")
    print(f"Data directory: {config['data']['data_dir']}")
    print(f"Model type: {config['model']['type']}")
    
    # Create model
    if config['model']['type'] == 'wimae':
        model = WiMAE(
            patch_size=tuple(config['model']['patch_size']),
            encoder_dim=config['model']['encoder_dim'],
            encoder_layers=config['model']['encoder_layers'],
            encoder_nhead=config['model']['encoder_nhead'],
            decoder_layers=config['model']['decoder_layers'],
            decoder_nhead=config['model']['decoder_nhead'],
            mask_ratio=config['model']['mask_ratio']
        )
        trainer_class = WiMAETrainer
    else:
        model = ContraWiMAE(
            patch_size=tuple(config['model']['patch_size']),
            encoder_dim=config['model']['encoder_dim'],
            encoder_layers=config['model']['encoder_layers'],
            encoder_nhead=config['model']['encoder_nhead'],
            decoder_layers=config['model']['decoder_layers'],
            decoder_nhead=config['model']['decoder_nhead'],
            mask_ratio=config['model']['mask_ratio'],
            contrastive_dim=config['model']['contrastive_dim'],
            temperature=config['model']['temperature']
        )
        trainer_class = ContraWiMAETrainer
    
    # Create trainer
    trainer = trainer_class(config, model)
    
    # Setup dataloaders (this will use simple approach)
    train_loader, val_loader = trainer.setup_dataloaders()
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test a few batches
    print("\nTesting data loading...")
    for i, batch in enumerate(train_loader):
        print(f"Train batch {i+1}: shape={batch.shape}, dtype={batch.dtype}")
        if i >= 2:  # Just test first 3 batches
            break
    
    print("✅ Simple data loading completed successfully!")


def example_scenario_split_data_loading():
    """Example: Train with scenario split data loading."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Scenario Split Data Loading")
    print("=" * 60)
    
    # Load configuration
    config_path = "configs/training_scenario_split.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Using config: {config_path}")
    print(f"Data directory: {config['data']['data_dir']}")
    print(f"Scenario split config: {config['data']['scenario_split_config']}")
    print(f"Model type: {config['model']['type']}")
    
    # Create model
    if config['model']['type'] == 'wimae':
        model = WiMAE(
            patch_size=tuple(config['model']['patch_size']),
            encoder_dim=config['model']['encoder_dim'],
            encoder_layers=config['model']['encoder_layers'],
            encoder_nhead=config['model']['encoder_nhead'],
            decoder_layers=config['model']['decoder_layers'],
            decoder_nhead=config['model']['decoder_nhead'],
            mask_ratio=config['model']['mask_ratio']
        )
        trainer_class = WiMAETrainer
    else:
        model = ContraWiMAE(
            patch_size=tuple(config['model']['patch_size']),
            encoder_dim=config['model']['encoder_dim'],
            encoder_layers=config['model']['encoder_layers'],
            encoder_nhead=config['model']['encoder_nhead'],
            decoder_layers=config['model']['decoder_layers'],
            decoder_nhead=config['model']['decoder_nhead'],
            mask_ratio=config['model']['mask_ratio'],
            contrastive_dim=config['model']['contrastive_dim'],
            temperature=config['model']['temperature']
        )
        trainer_class = ContraWiMAETrainer
    
    # Create trainer
    trainer = trainer_class(config, model)
    
    # Setup dataloaders (this will use scenario split approach)
    train_loader, val_loader = trainer.setup_dataloaders()
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test a few batches
    print("\nTesting data loading...")
    for i, batch in enumerate(train_loader):
        print(f"Train batch {i+1}: shape={batch.shape}, dtype={batch.dtype}")
        if i >= 2:  # Just test first 3 batches
            break
    
    print("✅ Scenario split data loading completed successfully!")


def example_contra_wimae_training():
    """Example: ContraWiMAE training with contrastive learning."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: ContraWiMAE Training")
    print("=" * 60)
    
    # Load configuration
    config_path = "configs/training_scenario_split.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure we're using ContraWiMAE
    config['model']['type'] = 'contramae'
    
    # Add contrastive learning weights
    config['training']['reconstruction_weight'] = 1.0
    config['training']['contrastive_weight'] = 0.1
    
    print(f"Using config: {config_path}")
    print(f"Model type: {config['model']['type']}")
    print(f"Reconstruction weight: {config['training']['reconstruction_weight']}")
    print(f"Contrastive weight: {config['training']['contrastive_weight']}")
    
    # Create ContraWiMAE model
    model = ContraWiMAE(
        patch_size=tuple(config['model']['patch_size']),
        encoder_dim=config['model']['encoder_dim'],
        encoder_layers=config['model']['encoder_layers'],
        encoder_nhead=config['model']['encoder_nhead'],
        decoder_layers=config['model']['decoder_layers'],
        decoder_nhead=config['model']['decoder_nhead'],
        mask_ratio=config['model']['mask_ratio'],
        contrastive_dim=config['model']['contrastive_dim'],
        temperature=config['model']['temperature']
    )
    
    # Create trainer
    trainer = ContraWiMAETrainer(config, model)
    
    # Setup dataloaders
    train_loader, val_loader = trainer.setup_dataloaders()
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test a few batches
    print("\nTesting data loading...")
    for i, batch in enumerate(train_loader):
        print(f"Train batch {i+1}: shape={batch.shape}, dtype={batch.dtype}")
        if i >= 2:  # Just test first 3 batches
            break
    
    print("✅ ContraWiMAE training setup completed successfully!")


def main():
    """Run all training examples."""
    print("Training Examples with Updated Data Loading")
    print("=" * 60)
    
    try:
        example_simple_data_loading()
    except Exception as e:
        print(f"❌ Example 1 failed: {e}")
    
    try:
        example_scenario_split_data_loading()
    except Exception as e:
        print(f"❌ Example 2 failed: {e}")
    
    try:
        example_contra_wimae_training()
    except Exception as e:
        print(f"❌ Example 3 failed: {e}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Updated trainers now support:")
    print()
    print("1. SIMPLE DATA LOADING:")
    print("   - Load all NPZ files from data_dir")
    print("   - Random train/val split")
    print("   - Use configs/default_training.yaml")
    print()
    print("2. SCENARIO SPLIT DATA LOADING:")
    print("   - Use file patterns for train/val/test splits")
    print("   - Scenario-based data organization")
    print("   - Use configs/training_scenario_split.yaml")
    print()
    print("3. BOTH WiMAE AND ContraWiMAE:")
    print("   - Same data loading interface")
    print("   - Automatic model selection")
    print("   - Contrastive learning support")
    print("=" * 60)


if __name__ == "__main__":
    main() 