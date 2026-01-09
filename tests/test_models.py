"""
Comprehensive tests for WiMAE and ContraWiMAE models.
"""

import torch
import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

from contrawimae.models import WiMAE, ContraWiMAE
from contrawimae.training.data_utils import OptimizedPreloadedDataset


class TestWiMAEModel:
    """Test WiMAE model functionality."""
    
    @pytest.fixture
    def wimae_model(self):
        """Create a WiMAE model for testing."""
        return WiMAE(
            patch_size=(16, 1),
            encoder_dim=64,
            encoder_layers=12,
            encoder_nhead=16,
            decoder_layers=4,
            decoder_nhead=8,
            mask_ratio=0.6,
            device="cpu"
        )
    
    @pytest.fixture
    def complex_input(self):
        """Create complex input tensor."""
        batch_size = 4
        height, width = 32, 32
        real = torch.randn(batch_size, height, width)
        imag = torch.randn(batch_size, height, width)
        return torch.complex(real, imag)
    
    def test_wimae_model_creation(self, wimae_model):
        """Test WiMAE model creation."""
        assert wimae_model.patch_size == (16, 1)
        assert wimae_model.encoder_dim == 64
        assert wimae_model.mask_ratio == 0.6
        assert str(wimae_model.device) == "cpu"
        
        # Check that components are initialized
        assert hasattr(wimae_model, 'patcher')
        assert hasattr(wimae_model, 'encoder')
        assert hasattr(wimae_model, 'decoder')
    
    def test_wimae_forward_complex_input(self, wimae_model, complex_input):
        """Test WiMAE forward pass with complex input."""
        output = wimae_model(complex_input)
        
        # Check output structure
        assert "encoded_features" in output
        assert "reconstructed_patches" in output
        assert "ids_keep" in output
        assert "ids_mask" in output
        
        # Check shapes
        batch_size = complex_input.shape[0]
        num_complex_patches = (32 // 16) * (32 // 1)  # 2 * 32 = 64 complex patches

        num_patches = num_complex_patches * 2  # 128 patches (real + imaginary parts)
        expected_keep_complex = int(num_complex_patches * (1 - wimae_model.mask_ratio))
        expected_keep = expected_keep_complex * 2  # Double for real and imaginary parts
        
        assert output["encoded_features"].shape == (batch_size, expected_keep, wimae_model.encoder_dim)
        assert output["reconstructed_patches"].shape == (batch_size, num_patches, 16)  # 16x1 patch
    
    def test_wimae_encode(self, wimae_model, complex_input):
        """Test WiMAE encode method."""
        encoded_features = wimae_model.encode(complex_input)
        
        # Check output shape
        batch_size = complex_input.shape[0]
        num_complex_patches = (32 // 16) * (32 // 1)  # 2 * 32 = 64 complex patches
        num_patches = num_complex_patches * 2  # 128 patches (real + imaginary parts)
        
        assert encoded_features.shape == (batch_size, num_patches, wimae_model.encoder_dim)
    
    def test_wimae_reconstruct(self, wimae_model, complex_input):
        """Test WiMAE encode + decode for reconstruction."""
        # Encode with masking
        encoded_features, ids_keep, ids_mask = wimae_model.encode(complex_input, apply_mask=True)
        
        # Decode to reconstruct
        reconstructed = wimae_model.decode(encoded_features, ids_keep, ids_mask)
        
        # Check output shape
        batch_size = complex_input.shape[0]
        num_complex_patches = (32 // 16) * (32 // 1)  # 2 * 32 = 64 complex patches
        num_patches = num_complex_patches * 2  # 128 patches (real + imaginary parts)
        
        assert reconstructed.shape == (batch_size, num_patches, 16)  # 16x1 patch
    
    def test_wimae_get_embeddings(self, wimae_model, complex_input):
        """Test WiMAE get_embeddings method."""
        # Test different pooling methods
        pooling_methods = ["mean", "max"]
        
        for pooling in pooling_methods:
            embeddings = wimae_model.get_embeddings(complex_input, pooling=pooling)
            
            # Check output shape
            batch_size = complex_input.shape[0]
            assert embeddings.shape == (batch_size, wimae_model.encoder_dim)
    
    def test_wimae_get_embeddings_invalid_pooling(self, wimae_model, complex_input):
        """Test WiMAE get_embeddings with invalid pooling method."""
        with pytest.raises(ValueError):
            wimae_model.get_embeddings(complex_input, pooling="invalid")
    
    def test_wimae_save_load_checkpoint(self, wimae_model, complex_input, tmp_path):
        """Test WiMAE checkpoint saving and loading."""
        # Save checkpoint
        checkpoint_path = tmp_path / "test_wimae_checkpoint.pt"
        wimae_model.save_checkpoint(str(checkpoint_path), test_data="test")
        
        # Load checkpoint
        loaded_model = WiMAE.from_checkpoint(str(checkpoint_path), device="cpu")
        
        # Check that model parameters are the same
        for param_name, original_param in wimae_model.named_parameters():
            loaded_param = dict(loaded_model.named_parameters())[param_name]
            assert torch.allclose(original_param, loaded_param)
    
    def test_wimae_get_model_info(self, wimae_model):
        """Test WiMAE get_model_info method."""
        info = wimae_model.get_model_info()
        
        # Check that all required keys are present
        required_keys = [
            "model_type", "patch_size", "encoder_dim", "encoder_layers",
            "encoder_nhead", "decoder_layers", "decoder_nhead", "mask_ratio",
            "total_parameters", "trainable_parameters"
        ]
        
        for key in required_keys:
            assert key in info
        
        # Check that parameter counts are reasonable
        assert info["total_parameters"] > 0
        assert info["trainable_parameters"] > 0
        assert info["trainable_parameters"] <= info["total_parameters"]


class TestContraWiMAEModel:
    """Test ContraWiMAE model functionality."""
    
    @pytest.fixture
    def contramae_model(self):
        """Create a ContraWiMAE model for testing."""
        return ContraWiMAE(
            patch_size=(16, 1),
            encoder_dim=64,
            encoder_layers=12,
            encoder_nhead=16,
            decoder_layers=4,
            decoder_nhead=8,
            mask_ratio=0.6,
            contrastive_dim=64,
            temperature=0.1,
            snr_min=0.0,
            snr_max=30.0,
            device="cpu"
        )
    
    @pytest.fixture
    def complex_input(self):
        """Create complex input tensor."""
        batch_size = 4
        height, width = 32, 32
        real = torch.randn(batch_size, height, width)
        imag = torch.randn(batch_size, height, width)
        return torch.complex(real, imag)
    
    def test_contramae_model_creation(self, contramae_model):
        """Test ContraWiMAE model creation."""
        assert contramae_model.patch_size == (16, 1)
        assert contramae_model.encoder_dim == 64
        assert contramae_model.mask_ratio == 0.6
        assert contramae_model.temperature == 0.1
        assert contramae_model.snr_min == 0.0
        assert contramae_model.snr_max == 30.0
        
        # Check that contrastive head is initialized
        assert hasattr(contramae_model, 'contrastive_head')
    
    def test_contramae_forward(self, contramae_model, complex_input):
        """Test ContraWiMAE forward pass."""
        output = contramae_model(complex_input, return_contrastive=True)
        
        # Check output structure
        assert "encoded_features" in output
        assert "reconstructed_patches" in output
        assert "ids_keep" in output
        assert "ids_mask" in output
        assert "contrastive_features" in output
        
        # Check contrastive features shape
        batch_size = complex_input.shape[0]
        assert output["contrastive_features"].shape == (batch_size, contramae_model.contrastive_dim)
    
    def test_contramae_forward_without_contrastive(self, contramae_model, complex_input):
        """Test ContraWiMAE forward pass without contrastive features."""
        output = contramae_model(complex_input, return_contrastive=False)
        
        # Check output structure
        assert "encoded_features" in output
        assert "reconstructed_patches" in output
        assert "ids_keep" in output
        assert "ids_mask" in output
        assert "contrastive_features" not in output
    
    def test_contramae_forward_with_augmentation(self, contramae_model, complex_input):
        """Test ContraWiMAE forward pass with augmentation."""
        output = contramae_model.forward_with_augmentation(
            complex_input,
            noise_prob=0.5,
            freq_shift_prob=0.3,
            phase_rot_prob=0.2
        )
        
        # Check output structure
        assert "original" in output
        assert "augmented" in output
        
        # Check that both original and augmented outputs have the expected structure
        for key in ["original", "augmented"]:
            assert "encoded_features" in output[key]
            assert "reconstructed_patches" in output[key]
            assert "ids_keep" in output[key]
            assert "ids_mask" in output[key]
            assert "contrastive_features" in output[key]
    
    def test_contramae_compute_contrastive_loss(self, contramae_model, complex_input):
        """Test ContraWiMAE contrastive loss computation."""
        # Get encoded features
        encoded_features = contramae_model.encode(complex_input)
        
        # Create two sets of features
        features1 = contramae_model.contrastive_head(encoded_features)
        features2 = contramae_model.contrastive_head(encoded_features)
        
        # Compute contrastive loss
        loss = contramae_model.compute_contrastive_loss(features1, features2)
        
        # Check that loss is a scalar
        assert loss.shape == ()
        assert loss.dtype == torch.float32
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss > 0
    
    def test_contramae_get_contrastive_embeddings(self, contramae_model, complex_input):
        """Test ContraWiMAE get_contrastive_embeddings method."""
        embeddings = contramae_model.get_contrastive_embeddings(complex_input)
        
        # Check output shape
        batch_size = complex_input.shape[0]
        assert embeddings.shape == (batch_size, contramae_model.contrastive_dim)
    
    def test_contramae_save_load_checkpoint(self, contramae_model, complex_input, tmp_path):
        """Test ContraWiMAE checkpoint saving and loading."""
        # Save checkpoint
        checkpoint_path = tmp_path / "test_contramae_checkpoint.pt"
        contramae_model.save_checkpoint(str(checkpoint_path), test_data="test")
        
        # Load checkpoint
        loaded_model = ContraWiMAE.from_checkpoint(str(checkpoint_path), device="cpu")
        
        # Check that model parameters are the same
        for param_name, original_param in contramae_model.named_parameters():
            loaded_param = dict(loaded_model.named_parameters())[param_name]
            assert torch.allclose(original_param, loaded_param)
    
    def test_contramae_get_model_info(self, contramae_model):
        """Test ContraWiMAE get_model_info method."""
        info = contramae_model.get_model_info()
        
        # Check that all required keys are present
        required_keys = [
            "model_type", "patch_size", "encoder_dim", "encoder_layers",
            "encoder_nhead", "decoder_layers", "decoder_nhead", "mask_ratio",
            "contrastive_dim", "temperature", "snr_min", "snr_max",
            "total_parameters", "trainable_parameters"
        ]
        
        for key in required_keys:
            assert key in info
        
        # Check that parameter counts are reasonable
        assert info["total_parameters"] > 0
        assert info["trainable_parameters"] > 0
        assert info["trainable_parameters"] <= info["total_parameters"]


class TestDataIntegration:
    """Test integration with data loading."""
    
    @pytest.fixture
    def temp_npz_files(self):
        """Create temporary NPZ files for testing."""
        temp_dir = tempfile.mkdtemp()
        npz_files = []
        
        # Create multiple NPZ files with different sample counts
        for i in range(2):
            num_samples = 10 + i * 5  # 10, 15 samples
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
    
    def test_model_with_optimized_dataset(self, temp_npz_files):
        """Test model with OptimizedPreloadedDataset."""
        # Create dataset
        dataset = OptimizedPreloadedDataset(temp_npz_files)
        
        # Create model
        model = WiMAE(
            patch_size=(16, 1),
            encoder_dim=64,
            encoder_layers=12,
            encoder_nhead=16,
            decoder_layers=4,
            decoder_nhead=8,
            mask_ratio=0.6,
            device="cpu"
        )
        
        # Test with dataset samples
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            output = model(sample.unsqueeze(0))  # Add batch dimension
            
            # Check output structure
            assert "encoded_features" in output
            assert "reconstructed_patches" in output
            
            # Check shapes
            assert output["encoded_features"].shape[0] == 1  # batch size
            assert output["reconstructed_patches"].shape[0] == 1  # batch size


if __name__ == "__main__":
    pytest.main([__file__]) 