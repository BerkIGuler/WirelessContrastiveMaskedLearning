#!/usr/bin/env python3
"""
Example script for encoding data with trained WiMAE and ContraWiMAE models.

This script demonstrates how to use trained models to generate embeddings
from wireless channel data.
"""

import yaml
from pathlib import Path
import argparse

from wimae.encoding import Encoder


def encode_data(config_path: str, data_path: str, output_path: str = None):
    """
    Encode data using a trained model.
    
    Args:
        config_path: Path to configuration file
        data_path: Path to data file
        output_path: Optional output path for embeddings
    """
    print(f"Encoding data from: {data_path}")
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Update data path in config
    config["data"]["data_path"] = data_path
    
    # Create encoder
    encoder = Encoder(config)
    
    # Encode and save data
    if output_path:
        filepath = encoder.encode_and_save(data_path, filename=output_path)
    else:
        filepath = encoder.encode_and_save(data_path)
    
    print(f"Embeddings saved to: {filepath}")
    
    return filepath


def encode_from_checkpoint(checkpoint_path: str, data_path: str, output_path: str = None):
    """
    Encode data using a model checkpoint directly.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_path: Path to data file
        output_path: Optional output path for embeddings
    """
    print(f"Encoding data using checkpoint: {checkpoint_path}")
    
    # Create encoder from checkpoint
    encoder = Encoder.from_checkpoint(checkpoint_path)
    
    # Encode and save data
    if output_path:
        filepath = encoder.encode_and_save(data_path, filename=output_path)
    else:
        filepath = encoder.encode_and_save(data_path)
    
    print(f"Embeddings saved to: {filepath}")
    
    return filepath


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Encode data with trained models")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--checkpoint", help="Path to model checkpoint")
    parser.add_argument("--data", required=True, help="Path to data file")
    parser.add_argument("--output", help="Output path for embeddings")
    parser.add_argument("--data-format", default="npz", help="Data format (default: npz)")
    
    args = parser.parse_args()
    
    if args.config:
        # Use configuration file
        encode_data(args.config, args.data, args.output)
    elif args.checkpoint:
        # Use checkpoint directly
        encode_from_checkpoint(args.checkpoint, args.data, args.output)
    else:
        raise ValueError("Either --config or --checkpoint must be provided")


if __name__ == "__main__":
    main() 