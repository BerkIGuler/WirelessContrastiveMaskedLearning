#!/usr/bin/env python3
"""
Example script demonstrating different data loading approaches with statistics handling.

This script shows two approaches:
1. Calculate statistics on-the-fly from training data
2. Use pre-computed statistics from configuration
"""

import os
import glob
import yaml
from pathlib import Path

from wimae.training.data_utils import setup_simple_dataloaders, setup_dataloaders, setup_scenario_dataloaders


def example_on_the_fly_statistics():
    """Example: Calculate statistics on-the-fly from training data."""
    print("=" * 60)
    print("EXAMPLE 1: Calculate Statistics On-The-Fly")
    print("=" * 60)
    
    # Find NPZ files in data directory
    data_dir = "data/pretrain"
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    if not npz_files:
        print(f"No NPZ files found in {data_dir}")
        return
    
    print(f"Found {len(npz_files)} NPZ files")
    
    # Set up dataloaders with on-the-fly statistics calculation
    train_loader, val_loader, train_size, val_size, statistics = setup_simple_dataloaders(
        npz_files=npz_files,
        batch_size=32,
        num_workers=2,
        normalize=True,
        calculate_statistics=True,  # This will calculate statistics from the data
        val_split=0.2
    )
    
    print(f"Train samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Calculated statistics: {statistics}")
    
    # Test a few batches
    print("\nTesting data loading...")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i+1}: shape={batch.shape}, dtype={batch.dtype}")
        if i >= 2:  # Just test first 3 batches
            break
    
    print("✅ On-the-fly statistics calculation completed successfully!")


def example_pre_computed_statistics():
    """Example: Use pre-computed statistics from configuration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Use Pre-Computed Statistics")
    print("=" * 60)
    
    # Load configuration
    config_path = "configs/default_training.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_dir = config['data']['data_dir']
    statistics = config['data']['statistics']
    
    print(f"Using data directory: {data_dir}")
    print(f"Pre-computed statistics: {statistics}")
    
    # Set up dataloaders with pre-computed statistics
    train_loader, val_loader, train_size, val_size, used_statistics = setup_simple_dataloaders(
        npz_files=glob.glob(os.path.join(data_dir, "*.npz")),
        batch_size=32,
        num_workers=2,
        normalize=True,
        calculate_statistics=False,  # Use provided statistics
        statistics=statistics,  # Pre-computed statistics
        val_split=0.2
    )
    
    print(f"Train samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Used statistics: {used_statistics}")
    
    # Test a few batches
    print("\nTesting data loading...")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i+1}: shape={batch.shape}, dtype={batch.dtype}")
        if i >= 2:  # Just test first 3 batches
            break
    
    print("✅ Pre-computed statistics usage completed successfully!")


def example_scenario_split_with_statistics():
    """Example: Use scenario split with statistics calculation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Scenario Split with Statistics")
    print("=" * 60)
    
    # Test with simple scenario split config
    config_path = "configs/scenario_split_simple.yaml"
    
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found. Creating test files first...")
        return
    
    print(f"Using scenario split config: {config_path}")
    
    try:
        # Set up dataloaders with scenario split and on-the-fly statistics
        train_loader, val_loader, train_size, val_size, statistics = setup_dataloaders(
            config_path=config_path,
            data_dir="data/pretrain",
            batch_size=32,
            num_workers=2,
            normalize=True,
            calculate_statistics=True  # Calculate from training data
        )
        
        print(f"Train samples: {train_size}")
        print(f"Validation samples: {val_size}")
        print(f"Calculated statistics: {statistics}")
        
        # Test a few batches
        print("\nTesting scenario split data loading...")
        for i, batch in enumerate(train_loader):
            print(f"Train batch {i+1}: shape={batch.shape}, dtype={batch.dtype}")
            if i >= 2:  # Just test first 3 batches
                break
        
        for i, batch in enumerate(val_loader):
            print(f"Val batch {i+1}: shape={batch.shape}, dtype={batch.dtype}")
            if i >= 1:  # Just test first 2 batches
                break
        
        print("✅ Scenario split with on-the-fly statistics completed successfully!")
        
    except Exception as e:
        print(f"❌ Scenario split test failed: {e}")
        print("This might be because:")
        print("1. No NPZ files match the patterns in the config")
        print("2. The data directory doesn't exist")
        print("3. File naming doesn't match the patterns")
        print("\nTry creating some test NPZ files or adjusting the patterns in the config.")


def example_scenario_split_with_test():
    """Example: Use scenario split with train/val/test splits."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Scenario Split with Train/Val/Test")
    print("=" * 60)
    
    # Test with comprehensive scenario split config
    config_path = "configs/scenario_split_test.yaml"
    
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found.")
        return
    
    print(f"Using scenario split config: {config_path}")
    
    try:
        # Set up dataloaders with train/val/test splits
        train_loader, val_loader, test_loader, train_size, val_size, test_size, statistics = setup_scenario_dataloaders(
            config_path=config_path,
            data_dir="data/pretrain",
            batch_size=32,
            num_workers=2,
            normalize=True,
            calculate_statistics=True,  # Calculate from training data
            include_test=True  # Include test split
        )
        
        print(f"Train samples: {train_size}")
        print(f"Validation samples: {val_size}")
        print(f"Test samples: {test_size}")
        print(f"Calculated statistics: {statistics}")
        
        # Test a few batches from each split
        print("\nTesting scenario split data loading...")
        for i, batch in enumerate(train_loader):
            print(f"Train batch {i+1}: shape={batch.shape}, dtype={batch.dtype}")
            if i >= 1:  # Just test first 2 batches
                break
        
        for i, batch in enumerate(val_loader):
            print(f"Val batch {i+1}: shape={batch.shape}, dtype={batch.dtype}")
            if i >= 1:  # Just test first 2 batches
                break
        
        for i, batch in enumerate(test_loader):
            print(f"Test batch {i+1}: shape={batch.shape}, dtype={batch.dtype}")
            if i >= 1:  # Just test first 2 batches
                break
        
        print("✅ Scenario split with train/val/test completed successfully!")
        
    except Exception as e:
        print(f"❌ Scenario split test failed: {e}")
        print("This might be because:")
        print("1. No NPZ files match the patterns in the config")
        print("2. The data directory doesn't exist")
        print("3. File naming doesn't match the patterns")
        print("\nTry creating some test NPZ files or adjusting the patterns in the config.")


def main():
    """Run all examples."""
    print("Data Loading Examples with Statistics Handling")
    print("=" * 60)
    
    try:
        example_on_the_fly_statistics()
    except Exception as e:
        print(f"❌ Example 1 failed: {e}")
    
    try:
        example_pre_computed_statistics()
    except Exception as e:
        print(f"❌ Example 2 failed: {e}")
    
    try:
        example_scenario_split_with_statistics()
    except Exception as e:
        print(f"❌ Example 3 failed: {e}")
    
    try:
        example_scenario_split_with_test()
    except Exception as e:
        print(f"❌ Example 4 failed: {e}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Two approaches for statistics handling:")
    print()
    print("1. ON-THE-FLY CALCULATION:")
    print("   - Set calculate_statistics=True")
    print("   - Statistics computed from training data")
    print("   - Useful for new datasets or when statistics are unknown")
    print()
    print("2. PRE-COMPUTED STATISTICS:")
    print("   - Set calculate_statistics=False")
    print("   - Provide statistics in config file")
    print("   - Faster startup, consistent normalization")
    print()
    print("Both approaches ensure proper data normalization!")
    print("=" * 60)


if __name__ == "__main__":
    main() 