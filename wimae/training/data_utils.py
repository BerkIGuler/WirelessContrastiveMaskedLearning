"""
Data utilities for WiMAE training pipeline.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import os
import gc
import yaml
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


def setup_dataloaders(config_path: str, data_dir: str, batch_size: int, 
                     num_workers: int = 4, normalize: bool = False,
                     val_split: float = 0.2, debug_size: Optional[int] = None):
    """
    Set up train and validation dataloaders based on scenario split config.

    Args:
        config_path: Path to YAML config with scenario split patterns
        data_dir: Directory containing the NPZ files
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        normalize: Whether to normalize the data
        val_split: Validation split ratio
        debug_size: If provided, use only this many samples for debugging

    Returns:
        train_loader, val_loader, train_size, val_size
    """
    # Load the config file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    statistics = config.get('train_statistics', None)

    # Create datasets
    train_dataset = ScenarioSplitDataset(
        data_dir=data_dir,
        config_path=config_path,
        split='train',
        normalize=normalize,
        statistics=statistics
    )

    val_dataset = ScenarioSplitDataset(
        data_dir=data_dir,
        config_path=config_path,
        split='val',
        normalize=normalize,
        statistics=statistics
    )

    # Apply debug size if specified
    if debug_size is not None:
        train_size = min(debug_size, len(train_dataset))
        val_size = min(debug_size // 4, len(val_dataset))  # Use 1/4 for validation
        
        train_dataset, _ = random_split(
            train_dataset, 
            [train_size, len(train_dataset) - train_size],
            generator=torch.Generator().manual_seed(42)
        )
        val_dataset, _ = random_split(
            val_dataset, 
            [val_size, len(val_dataset) - val_size],
            generator=torch.Generator().manual_seed(42)
        )

    train_loader = create_efficient_dataloader(
        train_dataset, batch_size=batch_size,
        num_workers=num_workers, shuffle=True)
    val_loader = create_efficient_dataloader(
        val_dataset, batch_size=batch_size,
        num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, len(train_dataset), len(val_dataset)


def create_efficient_dataloader(dataset: Dataset, batch_size: int = 1024, 
                               num_workers: int = 4, shuffle: bool = True) -> DataLoader:
    """
    Create an efficient dataloader with multiple workers and prefetching.

    Args:
        dataset: The dataset instance
        batch_size: Batch size for training
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the data

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,  # Keep workers alive between iterations
    )


def calculate_complex_statistics(dataloader: DataLoader) -> Dict[str, float]:
    """
    Calculate mean and std of real and imaginary parts separately.

    Args:
        dataloader: PyTorch DataLoader containing complex matrices

    Returns:
        dict containing real_mean, real_std, imag_mean, imag_std
    """
    real_sum = 0
    imag_sum = 0
    real_square_sum = 0
    imag_square_sum = 0
    total_elements = 0

    for batch in dataloader:
        real_part = batch.real
        imag_part = batch.imag

        real_sum += torch.sum(real_part)
        imag_sum += torch.sum(imag_part)
        real_square_sum += torch.sum(real_part ** 2)
        imag_square_sum += torch.sum(imag_part ** 2)
        total_elements += real_part.numel()

    real_mean = real_sum / total_elements
    imag_mean = imag_sum / total_elements
    real_std = torch.sqrt(real_square_sum / total_elements - real_mean ** 2)
    imag_std = torch.sqrt(imag_square_sum / total_elements - imag_mean ** 2)

    return {
        'real_mean': real_mean.item(),
        'real_std': real_std.item(),
        'imag_mean': imag_mean.item(),
        'imag_std': imag_std.item()
    }


def normalize_complex_matrix(matrix: torch.Tensor, statistics: Dict[str, float]) -> torch.Tensor:
    """
    Normalize complex matrix using provided statistics.

    Args:
        matrix: Complex tensor
        statistics: Dict containing real_mean, real_std, imag_mean, imag_std

    Returns:
        Normalized complex tensor
    """
    real_part = (matrix.real - statistics['real_mean']) / statistics['real_std']
    imag_part = (matrix.imag - statistics['imag_mean']) / statistics['imag_std']
    return torch.complex(real_part, imag_part)


def denormalize_complex_matrix(matrix: torch.Tensor, statistics: Dict[str, float]) -> torch.Tensor:
    """
    Reverse normalization of complex matrix using provided statistics.

    Args:
        matrix: Normalized complex tensor
        statistics: Dict containing real_mean, real_std, imag_mean, imag_std

    Returns:
        Denormalized complex tensor
    """
    real_part = matrix.real * statistics['real_std'] + statistics['real_mean']
    imag_part = matrix.imag * statistics['imag_std'] + statistics['imag_mean']
    return torch.complex(real_part, imag_part)


class ScenarioSplitDataset(Dataset):
    """
    Dataset implementation that supports splitting by scenario groups using regex patterns.
    Allows for controlled train/validation split based on scenario groups.
    """

    def __init__(self, data_dir: str, config_path: str, split: str = 'train', 
                 normalize: bool = False, statistics: Optional[Dict[str, float]] = None):
        """
        Args:
            data_dir: Directory containing the NPZ files
            config_path: Path to the YAML config file with scenario split definitions
            split: Either 'train' or 'val'
            normalize: Whether to normalize the data
            statistics: Dict with normalization parameters
        """
        self.data_dir = data_dir
        self.split = split
        self.normalize = normalize
        self.statistics = statistics

        if self.normalize and statistics is None:
            raise ValueError("If normalize is True, statistics must be provided")

        # Load config file
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Get the appropriate file patterns for the requested split
        if split not in ['train', 'val']:
            raise ValueError(f"Split must be 'train' or 'val', got {split}")

        # Get all NPZ files in the data directory
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]

        # Filter files based on split patterns
        self.npz_files = []
        for pattern in self.config[f'{split}_patterns']:
            regex = re.compile(pattern)
            matching_files = [os.path.join(data_dir, f) for f in all_files if regex.search(f)]
            self.npz_files.extend(matching_files)

        if not self.npz_files:
            raise ValueError(f"No files found for {split} split with the provided patterns")

        print(f"Selected {len(self.npz_files)} files for {split} split")

        # First get total sample count and dimensions
        total_samples = 0
        for npz_file in self.npz_files:
            with np.load(npz_file) as data:
                total_samples += len(data['channels'])
                # Get dimensions from first file
                if not hasattr(self, 'M'):
                    sample_shape = data['channels'][0, 0, :, :].shape
                    self.M, self.N = sample_shape

        print(f"Total samples for {split}: {total_samples}, dimensions: {self.M}x{self.N}")

        # Pre-allocate a single contiguous tensor for all data
        # Using complex64 instead of complex128 to reduce memory usage
        pin_memory = torch.cuda.is_available()
        self.all_data = torch.empty((total_samples, self.M, self.N),
                                    dtype=torch.complex64,
                                    pin_memory=pin_memory)

        # Load all data with progress tracking
        idx = 0
        for file_idx, npz_file in enumerate(self.npz_files):
            print(f"Loading file {file_idx + 1}/{len(self.npz_files)}: {os.path.basename(npz_file)}")
            with np.load(npz_file) as data:
                file_samples = len(data['channels'])

                # Process in batches to avoid memory spikes
                batch_size = 2000  # Good balance between speed and memory usage
                for batch_start in range(0, file_samples, batch_size):
                    batch_end = min(batch_start + batch_size, file_samples)
                    batch_count = batch_end - batch_start

                    # Load a batch directly to the target tensor
                    batch_data = torch.from_numpy(
                        data['channels'][batch_start:batch_end, 0, :, :].copy()
                    ).to(torch.complex64)

                    # Apply normalization if needed
                    if self.normalize:
                        real_part = (batch_data.real - statistics['real_mean']) / statistics['real_std']
                        imag_part = (batch_data.imag - statistics['imag_mean']) / statistics['imag_std']
                        batch_data = torch.complex(real_part, imag_part)

                    # Store in pre-allocated tensor
                    self.all_data[idx:idx + batch_count] = batch_data
                    idx += batch_count

            # Force cleanup after each file
            gc.collect()

        print(f"Successfully loaded all {total_samples} samples for {split} split")

    def __len__(self) -> int:
        return self.all_data.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.all_data[idx]


class OptimizedPreloadedDataset(Dataset):
    """Optimized dataset implementation for maximum training speed"""

    def __init__(self, npz_files: List[str], normalize: bool = False, 
                 statistics: Optional[Dict[str, float]] = None):
        """
        Args:
            npz_files: List of NPZ file paths
            normalize: Whether to normalize the data
            statistics: Dict with normalization parameters
        """
        self.normalize = normalize
        self.statistics = statistics

        if self.normalize and statistics is None:
            raise ValueError("If normalize is True, statistics must be provided")

        # First get total sample count and dimensions
        total_samples = 0
        for npz_file in npz_files:
            with np.load(npz_file) as data:
                total_samples += len(data['channels'])
                # Get dimensions from first file
                if not hasattr(self, 'M'):
                    sample_shape = data['channels'][0, 0, :, :].shape
                    self.M, self.N = sample_shape

        print(f"Total samples: {total_samples}, dimensions: {self.M}x{self.N}")

        # Pre-allocate a single contiguous tensor for all data
        # Using complex64 instead of complex128 to reduce memory usage
        # Only use pin_memory if CUDA is available
        pin_memory = torch.cuda.is_available()
        self.all_data = torch.empty((total_samples, self.M, self.N),
                                    dtype=torch.complex64,
                                    pin_memory=pin_memory)

        # Load all data with progress tracking
        idx = 0
        for file_idx, npz_file in enumerate(npz_files):
            print(f"Loading file {file_idx + 1}/{len(npz_files)}: {os.path.basename(npz_file)}")
            with np.load(npz_file) as data:
                file_samples = len(data['channels'])

                # Process in batches to avoid memory spikes
                batch_size = 2000  # Good balance between speed and memory usage
                for batch_start in range(0, file_samples, batch_size):
                    batch_end = min(batch_start + batch_size, file_samples)
                    batch_count = batch_end - batch_start

                    # Load a batch directly to the target tensor
                    batch_data = torch.from_numpy(
                        data['channels'][batch_start:batch_end, 0, :, :].copy()
                    ).to(torch.complex64)

                    # Apply normalization if needed
                    if self.normalize:
                        real_part = (batch_data.real - statistics['real_mean']) / statistics['real_std']
                        imag_part = (batch_data.imag - statistics['imag_mean']) / statistics['imag_std']
                        batch_data = torch.complex(real_part, imag_part)

                    # Store in pre-allocated tensor
                    self.all_data[idx:idx + batch_count] = batch_data
                    idx += batch_count

            # Force cleanup after each file
            gc.collect()

        print(f"Successfully loaded all {total_samples} samples")

    def __len__(self) -> int:
        return self.all_data.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.all_data[idx]


class MultiNPZDataset(Dataset):
    """Memory-mapped dataset for large NPZ files."""
    
    def __init__(self, npz_files: List[str], normalize: bool = False, 
                 statistics: Optional[Dict[str, float]] = None):
        """
        Initialize the dataset.

        Args:
            npz_files: List of NPZ file paths
            normalize: Boolean indicating whether to normalize the data
            statistics: Dict containing normalization statistics
        """
        self.npz_files = npz_files
        self.normalize = normalize
        self.statistics = statistics

        if self.normalize and statistics is None:
            raise ValueError("If normalize is True, statistics must be provided")

        # Calculate cumulative sizes for efficient indexing
        self.cumulative_sizes = []
        total = 0

        for npz_file in npz_files:
            with np.load(npz_file, mmap_mode='r') as data:
                size = len(data['channels'])
                total += size
                self.cumulative_sizes.append(total)

        # Get matrix dimensions from first file
        with np.load(npz_files[0], mmap_mode='r') as data:
            sample_shape = data['channels'][0, 0, :, :].shape
            self.M, self.N = sample_shape

    def __len__(self) -> int:
        """Return total number of samples across all files."""
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Complex tensor of shape (M, N)
        """
        if idx < 0:
            raise IndexError("Negative index is not supported.")

        file_idx = next(i for i, size in enumerate(self.cumulative_sizes)
                        if idx < size)

        local_idx = idx
        if file_idx > 0:
            local_idx = idx - self.cumulative_sizes[file_idx - 1]
            
        # Use memory mapping
        with np.load(self.npz_files[file_idx], mmap_mode='r') as data:
            channel_matrix = torch.from_numpy(data['channels'][local_idx][0]).to(torch.complex64)

            if self.normalize:
                channel_matrix = normalize_complex_matrix(channel_matrix, self.statistics)

            return channel_matrix 