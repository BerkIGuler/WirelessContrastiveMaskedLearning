#!/usr/bin/env python3
"""
Script to create sample NPZ files for testing scenario split functionality.

This script creates test NPZ files with different naming patterns to demonstrate
the scenario split feature.
"""

import os
import numpy as np
import torch
from pathlib import Path


def create_sample_npz_file(filename: str, num_samples: int = 100, 
                          height: int = 32, width: int = 32):
    """
    Create a sample NPZ file with complex channel data.
    
    Args:
        filename: Output filename
        num_samples: Number of samples to generate
        height: Height of each sample
        width: Width of each sample
    """
    # Create complex data with some variation
    real_part = np.random.normal(0, 1, (num_samples, 1, height, width))
    imag_part = np.random.normal(0, 1, (num_samples, 1, height, width))
    
    # Add some structure to make it more realistic
    for i in range(num_samples):
        # Add some spatial correlation
        real_part[i, 0, :, :] += np.random.normal(0, 0.1)
        imag_part[i, 0, :, :] += np.random.normal(0, 0.1)
    
    # Combine into complex data
    complex_data = real_part + 1j * imag_part
    
    # Save as NPZ
    np.savez(filename, channels=complex_data)
    print(f"Created {filename} with {num_samples} samples")


def create_test_dataset():
    """Create a complete test dataset with various naming patterns."""
    
    # Create data directory
    data_dir = Path("data/pretrain")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating test NPZ files for scenario split testing...")
    
    # Create files with simple numbered pattern (for simple config)
    for i in range(10):
        filename = data_dir / f"data_{i}.npz"
        create_sample_npz_file(filename, num_samples=50)
    
    # Create files with train/val prefixes
    for i in range(5):
        filename = data_dir / f"train_{i}.npz"
        create_sample_npz_file(filename, num_samples=40)
    
    for i in range(3):
        filename = data_dir / f"val_{i}.npz"
        create_sample_npz_file(filename, num_samples=30)
    
    # Create files with scenario-based naming (for comprehensive config)
    scenarios = {
        'urban': 3,
        'city': 2,
        'dense': 2,
        'indoor': 2,
        'office': 2,
        'home': 2,
        'suburban': 2,
        'residential': 2,
        'rural': 2,
        'countryside': 2,
        'highway': 2,
        'motorway': 2,
        'mixed': 2,
        'test': 2
    }
    
    for scenario, count in scenarios.items():
        for i in range(count):
            filename = data_dir / f"{scenario}_{i}.npz"
            create_sample_npz_file(filename, num_samples=35)
    
    # Create files with SNR-based naming
    for snr in [5, 10, 15, 20, 25]:
        for i in range(2):
            filename = data_dir / f"data_snr_{snr}_{i}.npz"
            create_sample_npz_file(filename, num_samples=30)
    
    # Create files with frequency-based naming
    for freq in [24, 28, 32, 36, 40]:
        for i in range(2):
            filename = data_dir / f"data_freq_{freq}_{i}.npz"
            create_sample_npz_file(filename, num_samples=30)
    
    # Create files with timestamp-based naming
    for day in range(1, 10):
        for hour in [0, 6, 12, 18]:
            filename = data_dir / f"data_2024010{day}_{hour:02d}.npz"
            create_sample_npz_file(filename, num_samples=25)
    
    print(f"\n✅ Created test dataset in {data_dir}")
    print("Files created:")
    
    # List all created files
    npz_files = list(data_dir.glob("*.npz"))
    npz_files.sort()
    
    for file in npz_files:
        print(f"  - {file.name}")
    
    print(f"\nTotal files: {len(npz_files)}")
    
    # Show file count by pattern
    print("\nFile count by pattern:")
    patterns = {
        'data_[0-9].npz': len(list(data_dir.glob("data_[0-9].npz"))),
        'train_*.npz': len(list(data_dir.glob("train_*.npz"))),
        'val_*.npz': len(list(data_dir.glob("val_*.npz"))),
        'urban_*.npz': len(list(data_dir.glob("urban_*.npz"))),
        'rural_*.npz': len(list(data_dir.glob("rural_*.npz"))),
        'highway_*.npz': len(list(data_dir.glob("highway_*.npz"))),
        'data_snr_*.npz': len(list(data_dir.glob("data_snr_*.npz"))),
        'data_freq_*.npz': len(list(data_dir.glob("data_freq_*.npz"))),
        'data_2024010*.npz': len(list(data_dir.glob("data_2024010*.npz")))
    }
    
    for pattern, count in patterns.items():
        print(f"  {pattern}: {count} files")


def verify_npz_files():
    """Verify that the created NPZ files are valid."""
    print("\nVerifying NPZ files...")
    
    data_dir = Path("data/pretrain")
    npz_files = list(data_dir.glob("*.npz"))
    
    total_samples = 0
    for file in npz_files:
        try:
            with np.load(file) as data:
                if 'channels' in data:
                    samples = len(data['channels'])
                    shape = data['channels'].shape
                    total_samples += samples
                    print(f"  ✅ {file.name}: {samples} samples, shape {shape}")
                else:
                    print(f"  ❌ {file.name}: No 'channels' key found")
        except Exception as e:
            print(f"  ❌ {file.name}: Error loading - {e}")
    
    print(f"\nTotal samples across all files: {total_samples}")


def main():
    """Main function to create and verify test data."""
    print("Test Data Creation for Scenario Split Testing")
    print("=" * 60)
    
    # Check if data directory already exists
    data_dir = Path("data/pretrain")
    if data_dir.exists() and list(data_dir.glob("*.npz")):
        print(f"Data directory {data_dir} already exists with NPZ files.")
        response = input("Do you want to recreate the test data? (y/N): ")
        if response.lower() != 'y':
            print("Using existing data.")
            verify_npz_files()
            return
    
    # Create test dataset
    create_test_dataset()
    
    # Verify the files
    verify_npz_files()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Run the data loading examples:")
    print("   python examples/data_loading_examples.py")
    print()
    print("2. Test scenario split with simple config:")
    print("   python -c \"from wimae.training.data_utils import setup_dataloaders; "
          "train_loader, val_loader, train_size, val_size, stats = "
          "setup_dataloaders('configs/scenario_split_simple.yaml', 'data/pretrain', "
          "batch_size=32, calculate_statistics=True); "
          "print(f'Train: {train_size}, Val: {val_size}'); "
          "print(f'Stats: {stats}')\"")
    print()
    print("3. Test scenario split with comprehensive config:")
    print("   python -c \"from wimae.training.data_utils import setup_dataloaders; "
          "train_loader, val_loader, train_size, val_size, stats = "
          "setup_dataloaders('configs/scenario_split_test.yaml', 'data/pretrain', "
          "batch_size=32, calculate_statistics=True); "
          "print(f'Train: {train_size}, Val: {val_size}')\"")
    print("=" * 60)


if __name__ == "__main__":
    main() 