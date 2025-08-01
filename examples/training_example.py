#!/usr/bin/env python3
"""
Example training script using OptimizedPreloadedDataset.

This script demonstrates how to use the optimized data loading pipeline
with the WiMAE/ContraWiMAE training framework.
"""

import yaml
import argparse
from pathlib import Path
from wimae.training.trainer import BaseTrainer


def main():
    parser = argparse.ArgumentParser(description="Train WiMAE/ContraWiMAE with optimized data loading")
    parser.add_argument("--config", type=str, default="configs/default_training.yaml", 
                       help="Path to config file (default: configs/default_training.yaml)")
    parser.add_argument("--data-dir", type=str, help="Override data directory from config")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (small dataset)")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override data directory if provided
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    
    # Enable debug mode if requested
    if args.debug:
        config["data"]["debug_size"] = 1000
        print("Debug mode enabled: using 1000 samples")
    
    # Validate configuration
    validate_config(config)
    
    # Create trainer and start training
    trainer = BaseTrainer(config)
    trainer.train()


def validate_config(config):
    """Validate the configuration file."""
    required_sections = ["model", "data", "training", "logging"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    # Validate data configuration
    data_config = config["data"]
    if "data_dir" not in data_config:
        raise ValueError("data.data_dir is required")
    
    data_dir = Path(data_config["data_dir"])
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Check for NPZ files
    npz_files = list(data_dir.glob("*.npz"))
    if not npz_files:
        raise ValueError(f"No NPZ files found in {data_dir}")
    
    print(f"Found {len(npz_files)} NPZ files in {data_dir}")
    
    # Validate model configuration
    model_config = config["model"]
    if model_config["type"] not in ["wimae", "contramae"]:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    
    # Validate training configuration
    training_config = config["training"]
    required_training_keys = ["batch_size", "epochs", "device"]
    for key in required_training_keys:
        if key not in training_config:
            raise ValueError(f"Missing required training key: {key}")


if __name__ == "__main__":
    main() 