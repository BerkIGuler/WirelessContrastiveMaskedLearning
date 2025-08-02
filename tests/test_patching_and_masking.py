"""
Unit tests for patching and masking modules.
"""

import pytest
import torch
import numpy as np

from wimae.models.modules.patching import Patcher
from wimae.models.modules.masking import apply_masking


class TestPatching:
    """Test patching functionality."""
    
    @pytest.fixture
    def patcher_1x16(self):
        """Create a 1x16 patcher."""
        return Patcher(patch_size=(1, 16))
    
    @pytest.fixture
    def patcher_2x8(self):
        """Create a 2x8 patcher."""
        return Patcher(patch_size=(2, 8))
    
    @pytest.fixture
    def complex_input(self):
        """Create complex input tensor."""
        batch_size = 4
        height, width = 32, 32
        real = torch.randn(batch_size, height, width)
        imag = torch.randn(batch_size, height, width)
        return torch.complex(real, imag)
    
    def test_patcher_initialization(self):
        """Test patcher initialization."""
        patcher = Patcher(patch_size=(1, 16))
        assert patcher.patch_size == (1, 16)
        
        patcher = Patcher(patch_size=(2, 8))
        assert patcher.patch_size == (2, 8)
    
    def test_patcher_complex_input(self, patcher_1x16, complex_input):
        """Test patching with complex input."""
        patches = patcher_1x16(complex_input)
        
        # Check output shape
        batch_size = complex_input.shape[0]
        expected_patches = (32 // 1) * (32 // 16)  # 32 * 2 = 64 patches
        expected_dim = 1 * 16  # patch size
        
        assert patches.shape == (batch_size, 2 * expected_patches, expected_dim)
        assert patches.dtype == torch.float32  # Should be flattened to real
        
        # Check that complex data was properly flattened
        # First half should be real part, second half should be imaginary part
        real_patches = patches[:, :expected_patches, :]
        imag_patches = patches[:, expected_patches:, :]
        
        # Verify that patches are properly extracted
        assert torch.allclose(real_patches[0, 0, :], complex_input[0, :1, :16].real.flatten())
        assert torch.allclose(imag_patches[0, 0, :], complex_input[0, :1, :16].imag.flatten())
    
    def test_patcher_different_patch_sizes(self, complex_input):
        """Test patching with different patch sizes."""
        # Test 2x8 patches
        patcher_2x8 = Patcher(patch_size=(2, 8))
        patches_2x8 = patcher_2x8(complex_input)
        
        batch_size = complex_input.shape[0]
        expected_patches_2x8 = (32 // 2) * (32 // 8)  # 16 * 4 = 64 patches
        expected_dim_2x8 = 2 * 8  # patch size
        
        assert patches_2x8.shape == (batch_size, 2 * expected_patches_2x8, expected_dim_2x8)
        
        # Test 8x2 patches
        patcher_8x2 = Patcher(patch_size=(8, 2))
        patches_8x2 = patcher_8x2(complex_input)
        
        expected_patches_8x2 = (32 // 8) * (32 // 2)  # 4 * 16 = 64 patches
        expected_dim_8x2 = 8 * 2  # patch size
        
        assert patches_8x2.shape == (batch_size, 2 * expected_patches_8x2, expected_dim_8x2)
    
    def test_patcher_non_divisible_dimensions(self):
        """Test patching with dimensions that don't divide evenly."""
        # Create input with dimensions that don't divide evenly by patch size
        batch_size = 4
        height, width = 30, 30  # Not divisible by 16
        real = torch.randn(batch_size, height, width)
        imag = torch.randn(batch_size, height, width)
        complex_input = torch.complex(real, imag)
        
        patcher = Patcher(patch_size=(1, 16))
        
        # Should raise an error or handle gracefully
        with pytest.raises(ValueError):
            patches = patcher(complex_input)
    
    def test_patcher_edge_cases(self):
        """Test patching with edge cases."""
        # Test with patch size equal to input size
        batch_size = 4
        height, width = 1, 16
        real = torch.randn(batch_size, height, width)
        imag = torch.randn(batch_size, height, width)
        complex_input = torch.complex(real, imag)
        
        patcher = Patcher(patch_size=(1, 16))
        patches = patcher(complex_input)
        
        # Should have 1 patch per sample
        assert patches.shape == (batch_size, 2, 16)  # 2 for real+imag, 16 for 1x16
    
    def test_patcher_preserves_data(self, patcher_1x16, complex_input):
        """Test that patching preserves the original data."""
        patches = patcher_1x16(complex_input)
        
        # Reconstruct the first patch manually
        first_patch_real = patches[0, 0, :].reshape(1, 16)
        first_patch_imag = patches[0, 64, :].reshape(1, 16)  # 64 is the number of patches
        first_patch_complex = torch.complex(first_patch_real, first_patch_imag)
        
        # Should match the original data
        original_patch = complex_input[0, :1, :16]
        assert torch.allclose(first_patch_complex, original_patch, atol=1e-6)


class TestMasking:
    """Test masking functionality."""
    
    @pytest.fixture
    def sample_patches(self):
        """Create sample patches for testing."""
        batch_size = 4
        num_patches = 64
        patch_dim = 16
        return torch.randn(batch_size, num_patches, patch_dim)
    
    def test_apply_masking(self, sample_patches):
        """Test masking application."""
        mask_ratio = 0.75
        masked_patches, ids_keep, ids_mask = apply_masking(sample_patches, mask_ratio)
        
        # Check output shapes
        batch_size, num_patches, patch_dim = sample_patches.shape
        expected_keep = int(num_patches * (1 - mask_ratio))
        expected_mask = num_patches - expected_keep
        
        assert masked_patches.shape == (batch_size, expected_keep, patch_dim)
        assert ids_keep.shape == (batch_size, expected_keep)
        assert ids_mask.shape == (batch_size, expected_mask)
        
        # Check that kept indices are unique and in valid range
        for i in range(batch_size):
            assert len(torch.unique(ids_keep[i])) == expected_keep
            assert torch.all(ids_keep[i] >= 0) and torch.all(ids_keep[i] < num_patches)
            
            # Check that masked indices are unique and in valid range
            assert len(torch.unique(ids_mask[i])) == expected_mask
            assert torch.all(ids_mask[i] >= 0) and torch.all(ids_mask[i] < num_patches)
            
            # Check that kept and masked indices don't overlap
            kept_set = set(ids_keep[i].tolist())
            masked_set = set(ids_mask[i].tolist())
            assert len(kept_set.intersection(masked_set)) == 0
    
    def test_apply_masking_different_ratios(self, sample_patches):
        """Test masking with different ratios."""
        ratios = [0.0, 0.25, 0.5, 0.75, 0.9]
        
        for ratio in ratios:
            masked_patches, ids_keep, ids_mask = apply_masking(sample_patches, ratio)
            
            batch_size, num_patches, patch_dim = sample_patches.shape
            expected_keep = int(num_patches * (1 - ratio))
            expected_mask = num_patches - expected_keep
            
            assert masked_patches.shape == (batch_size, expected_keep, patch_dim)
            assert ids_keep.shape == (batch_size, expected_keep)
            assert ids_mask.shape == (batch_size, expected_mask)
    
    def test_apply_masking_edge_cases(self, sample_patches):
        """Test masking with edge cases."""
        # Test with mask_ratio = 0 (no masking)
        masked_patches, ids_keep, ids_mask = apply_masking(sample_patches, 0.0)
        
        batch_size, num_patches, patch_dim = sample_patches.shape
        assert masked_patches.shape == (batch_size, num_patches, patch_dim)
        assert ids_keep.shape == (batch_size, num_patches)
        assert ids_mask.shape == (batch_size, 0)  # No masked patches
        
        # Test with mask_ratio = 1.0 (all masked)
        masked_patches, ids_keep, ids_mask = apply_masking(sample_patches, 1.0)
        
        assert masked_patches.shape == (batch_size, 0, patch_dim)  # No kept patches
        assert ids_keep.shape == (batch_size, 0)
        assert ids_mask.shape == (batch_size, num_patches)  # All patches masked
    
    def test_apply_masking_preserves_data(self, sample_patches):
        """Test that masking preserves the original data."""
        mask_ratio = 0.5
        masked_patches, ids_keep, ids_mask = apply_masking(sample_patches, mask_ratio)
        
        # Check that kept patches match original data
        for i in range(sample_patches.shape[0]):
            for j, keep_idx in enumerate(ids_keep[i]):
                assert torch.allclose(masked_patches[i, j], sample_patches[i, keep_idx])
    
    def test_apply_masking_randomness(self, sample_patches):
        """Test that masking produces different results on different calls."""
        mask_ratio = 0.75
        
        # Apply masking multiple times
        results = []
        for _ in range(5):
            masked_patches, ids_keep, ids_mask = apply_masking(sample_patches, mask_ratio)
            results.append((masked_patches, ids_keep, ids_mask))
        
        # Check that at least some results are different (due to randomness)
        first_ids_keep = results[0][1]
        different_results = False
        
        for _, ids_keep, _ in results[1:]:
            if not torch.allclose(first_ids_keep, ids_keep):
                different_results = True
                break
        
        assert different_results, "Masking should produce different results due to randomness"
    
    def test_apply_masking_invalid_ratio(self, sample_patches):
        """Test masking with invalid ratios."""
        # Test negative ratio
        with pytest.raises(ValueError):
            apply_masking(sample_patches, -0.1)
        
        # Test ratio > 1
        with pytest.raises(ValueError):
            apply_masking(sample_patches, 1.1)
    
    def test_apply_masking_empty_input(self):
        """Test masking with empty input."""
        empty_patches = torch.empty(0, 64, 16)
        
        with pytest.raises(ValueError):
            apply_masking(empty_patches, 0.5) 