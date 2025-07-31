#!/usr/bin/env python3
"""
Example script for training WiMAE and ContraWiMAE models.

This script demonstrates how to train both WiMAE and ContraWiMAE models
using the provided training framework.
"""

import yaml
from pathlib import Path
import argparse

from wimae.training import WiMAETrainer, ContraWiMAETrainer


def train_wimae(config_path: str):
    """
    Train a WiMAE model.
    
    Args:
        config_path: Path to configuration file
    """
    print("Training WiMAE model...")
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Ensure model type is set to wimae
    config["model"]["type"] = "wimae"
    
    # Create trainer
    trainer = WiMAETrainer(config)
    
    # Train model
    trainer.train()
    
    print("WiMAE training completed!")


def train_contramae(config_path: str):
    """
    Train a ContraWiMAE model.
    
    Args:
        config_path: Path to configuration file
    """
    print("Training ContraWiMAE model...")
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Ensure model type is set to contramae
    config["model"]["type"] = "contramae"
    
    # Create trainer
    trainer = ContraWiMAETrainer(config)
    
    # Train model
    trainer.train()
    
    print("ContraWiMAE training completed!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train WiMAE or ContraWiMAE models")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("--model", choices=["wimae", "contramae"], 
                       help="Model type (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration to check model type
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Determine model type
    model_type = args.model or config["model"]["type"]
    
    if model_type == "wimae":
        train_wimae(args.config)
    elif model_type == "contramae":
        train_contramae(args.config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    main() 