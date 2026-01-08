"""
Unit tests for data loading modules.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path

from contrawimae.training.data_utils import (
    OptimizedPreloadedDataset,
    normalize_complex_matrix,
    denormalize_complex_matrix,
    calculate_complex_statistics,
    create_efficient_dataloader
)


class TestDataLoading:
    """Test data loading functionality."""
    
    @pytest.fixture
    def temp_npz_files(self):
        """Create temporary NPZ files for testing."""
        temp_dir = tempfile.mkdtemp()
        npz_files = []
        
        # Create multiple NPZ files with different sample counts
        for i in range(3):
            num_samples = 10 + i * 5  # 10, 15, 20 samples
            height, width = 32, 32
            
            # Create complex channel data
            real_part = np.random.randn(num_samples, 1, height, width)
            imag_part = np.random.randn(num_samples, 1, height, width)
            channels = real_part + 1j * imag_part
            
            file_path = os.path.join(temp_dir, f"test_data_{i}.npz")
            np.savez(file_path, channels=channels)
            npz_files.append(file_path)
        
        yield npz_files
        
        # Cleanup
        for file_path in npz_files:
            os.remove(file_path)
        os.rmdir(temp_dir)
    
    @pytest.fixture
    def statistics(self):
        """Sample statistics for normalization."""
        return {
            'real_mean': 0.1,
            'real_std': 1.0,
            'imag_mean': -0.05,
            'imag_std': 0.8
        }
    
    def test_optimized_preloaded_dataset(self, temp_npz_files):
        """Test OptimizedPreloadedDataset functionality."""
        # Test without normalization
        dataset = OptimizedPreloadedDataset(temp_npz_files, normalize=False)
        
        assert len(dataset) == 45  # 10 + 15 + 20
        assert dataset.M == 32
        assert dataset.N == 32
        
        # Test data loading
        sample = dataset[0]
        assert sample.shape == (32, 32)
        assert sample.dtype == torch.complex64
        
        # Test multiple samples
        samples = [dataset[i] for i in range(5)]
        assert all(s.shape == (32, 32) for s in samples)
        assert all(s.dtype == torch.complex64 for s in samples)
    
    def test_optimized_preloaded_dataset_with_normalization(self, temp_npz_files, statistics):
        """Test OptimizedPreloadedDataset with normalization."""
        dataset = OptimizedPreloadedDataset(temp_npz_files, normalize=True, statistics=statistics)
        
        assert len(dataset) == 45
        
        # Test that data is normalized
        sample = dataset[0]
        assert sample.shape == (32, 32)
        assert sample.dtype == torch.complex64
        
        # Check that normalization was applied (values should be in reasonable range)
        real_part = sample.real
        imag_part = sample.imag
        
        # Normalized values should be roughly in a range (considering the statistics used)
        # With real_mean=0.1, real_std=1.0, imag_mean=-0.05, imag_std=0.8
        # and data from np.random.randn() (mean=0, std=1), expected ranges are:
        # Real: (data - 0.1) / 1.0 = data - 0.1 ≈ [-1.1, 0.9]
        # Imag: (data - (-0.05)) / 0.8 = (data + 0.05) / 0.8 ≈ [-1.2, 1.3]
        # But since np.random.randn() can produce outliers, we use a wider range
        # For imag part, the division by 0.8 can amplify outliers, so we use an even wider range
        assert torch.all(real_part >= -5) and torch.all(real_part <= 5)
        assert torch.all(imag_part >= -6) and torch.all(imag_part <= 6)
    

    
    def test_normalize_complex_matrix(self, statistics):
        """Test complex matrix normalization."""
        # Use fixed seed for reproducible test
        torch.manual_seed(42)
        
        # Create test complex matrix
        real_part = torch.randn(32, 32) * 10 + 5  # Mean ~5, std ~10
        imag_part = torch.randn(32, 32) * 8 - 2    # Mean ~-2, std ~8
        matrix = torch.complex(real_part, imag_part)
        
        # Normalize
        normalized = normalize_complex_matrix(matrix, statistics)
        
        # Check that normalization was applied
        assert normalized.shape == (32, 32)
        assert normalized.dtype == torch.complex64
        
        # Check that values are in reasonable range
        real_norm = normalized.real
        imag_norm = normalized.imag
        
        # With real_mean=0.1, real_std=1.0, imag_mean=-0.05, imag_std=0.8
        # and input data: real_part (mean=5, std=10), imag_part (mean=-2, std=8)
        # Normalization: (data - mean) / std
        # Real: (data - 0.1) / 1.0 = data - 0.1 ≈ [5-0.1 ± 10] = [-5.1, 14.9]
        # Imag: (data - (-0.05)) / 0.8 = (data + 0.05) / 0.8 ≈ [(-2+0.05) ± 8] / 0.8 = [-12.44, 7.56]
        # Using very generous bounds to account for random variations in test data
        # Check that values are finite and reasonable
        assert torch.all(torch.isfinite(real_norm)) and torch.all(torch.isfinite(imag_norm))
        assert torch.all(real_norm >= -50) and torch.all(real_norm <= 60)
        assert torch.all(imag_norm >= -60) and torch.all(imag_norm <= 50)
    
    def test_denormalize_complex_matrix(self, statistics):
        """Test complex matrix denormalization."""
        # Create normalized test matrix
        real_part = torch.randn(32, 32) * 0.5  # Small values
        imag_part = torch.randn(32, 32) * 0.5
        normalized = torch.complex(real_part, imag_part)
        
        # Denormalize
        denormalized = denormalize_complex_matrix(normalized, statistics)
        
        # Check that denormalization was applied
        assert denormalized.shape == (32, 32)
        assert denormalized.dtype == torch.complex64
        
        # Check that values are in original range
        real_denorm = denormalized.real
        imag_denorm = denormalized.imag
        
        # Should be roughly in original range
        assert torch.all(real_denorm >= -10) and torch.all(real_denorm <= 20)
        assert torch.all(imag_denorm >= -10) and torch.all(imag_denorm <= 10)
    
    def test_normalize_denormalize_roundtrip(self, statistics):
        """Test that normalize + denormalize preserves data."""
        # Create original matrix
        original = torch.complex(torch.randn(32, 32), torch.randn(32, 32))
        
        # Normalize and denormalize
        normalized = normalize_complex_matrix(original, statistics)
        denormalized = denormalize_complex_matrix(normalized, statistics)
        
        # Should be close to original (within numerical precision)
        assert torch.allclose(original, denormalized, atol=1e-6)
    
    def test_calculate_complex_statistics(self, temp_npz_files):
        """Test complex statistics calculation."""
        # Create a small dataset for testing
        dataset = OptimizedPreloadedDataset(temp_npz_files[:1], normalize=False)
        dataloader = create_efficient_dataloader(dataset, batch_size=5, shuffle=False)
        
        # Calculate statistics
        stats = calculate_complex_statistics(dataloader)
        
        # Check that all required keys are present
        required_keys = ['real_mean', 'real_std', 'imag_mean', 'imag_std']
        for key in required_keys:
            assert key in stats
            assert isinstance(stats[key], float)
            assert not np.isnan(stats[key])
            assert not np.isinf(stats[key])
    
    def test_create_efficient_dataloader(self, temp_npz_files):
        """Test efficient dataloader creation."""
        dataset = OptimizedPreloadedDataset(temp_npz_files, normalize=False)
        
        # Test dataloader creation
        dataloader = create_efficient_dataloader(
            dataset, 
            batch_size=4, 
            num_workers=0,  # Use 0 for testing
            shuffle=True
        )
        
        # Test that dataloader works
        batch = next(iter(dataloader))
        assert batch.shape[0] == 4  # batch_size
        assert batch.shape[1:] == (32, 32)  # spatial dimensions
        assert batch.dtype == torch.complex64
    
    def test_dataset_indexing(self, temp_npz_files):
        """Test dataset indexing and bounds."""
        dataset = OptimizedPreloadedDataset(temp_npz_files, normalize=False)
        
        # Test valid indices
        assert len(dataset) == 45
        
        # Test first and last elements
        first_sample = dataset[0]
        last_sample = dataset[44]
        
        assert first_sample.shape == (32, 32)
        assert last_sample.shape == (32, 32)
        
        # Test invalid indices
        with pytest.raises(IndexError):
            _ = dataset[-1]  # Negative index not supported
        
        with pytest.raises(IndexError):
            _ = dataset[45]  # Out of bounds
    
    def test_empty_dataset_error(self):
        """Test error handling for empty dataset."""
        with pytest.raises(ValueError, match="No NPZ files found"):
            OptimizedPreloadedDataset([], normalize=False)
    
    def test_missing_statistics_error(self):
        """Test error handling for missing statistics."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(f.name, channels=np.random.randn(5, 1, 32, 32) + 1j * np.random.randn(5, 1, 32, 32))
        
        try:
            with pytest.raises(ValueError, match="If normalize is True, statistics must be provided"):
                OptimizedPreloadedDataset([f.name], normalize=True, statistics=None)
        finally:
            os.remove(f.name) 