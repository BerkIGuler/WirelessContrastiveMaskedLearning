"""
Basic tests for WiMAE and ContraWiMAE models.
"""

import torch
import pytest
from wimae.models import WiMAE, ContraWiMAE


def test_wimae_model():
    """Test WiMAE model creation and forward pass."""
    # Create model
    model = WiMAE(
        patch_size=(1, 16),
        encoder_dim=64,
        encoder_layers=12,
        encoder_nhead=16,
        decoder_layers=4,
        decoder_nhead=8,
        mask_ratio=0.6,
        device="cpu"
    )
    
    # Create dummy complex channel input
    batch_size = 256
    height = 32
    width = 32
    real = torch.randn(batch_size, height, width)
    imag = torch.randn(batch_size, height, width)
    x = torch.complex(real, imag)
    
    # Forward pass
    output = model(x)
    
    # Check output structure
    assert "encoded_features" in output
    assert "reconstructed_patches" in output
    
    # Check shapes
    assert output["encoded_features"].shape[0] == batch_size
    assert output["reconstructed_patches"].shape[0] == batch_size


def test_contramae_model():
    """Test ContraWiMAE model creation and forward pass."""
    # Create model
    model = ContraWiMAE(
        patch_size=(1, 16),
        encoder_dim=64,
        encoder_layers=12,
        encoder_nhead=16,
        decoder_layers=4,
        decoder_nhead=8,
        mask_ratio=0.6,
        contrastive_dim=64,
        temperature=0.1,
        device="cpu"
    )
    
    # Create dummy complex channel input
    batch_size = 256
    height = 32
    width = 32
    real = torch.randn(batch_size, height, width)
    imag = torch.randn(batch_size, height, width)
    x = torch.complex(real, imag)
    
    # Forward pass
    output = model(x, return_contrastive=True)
    
    # Check output structure
    assert "encoded_features" in output
    assert "reconstructed_patches" in output
    assert "contrastive_features" in output
    
    # Check shapes
    assert output["encoded_features"].shape[0] == batch_size
    assert output["reconstructed_patches"].shape[0] == batch_size
    assert output["contrastive_features"].shape[0] == batch_size


def test_model_embeddings():
    """Test embedding generation."""
    # Create model
    model = WiMAE(
        patch_size=(1, 16),
        encoder_dim=64,
        encoder_layers=12,
        encoder_nhead=16,
        decoder_layers=4,
        decoder_nhead=8,
        mask_ratio=0.6,
        device="cpu"
    )
    
    # Create dummy complex channel input
    batch_size = 256
    height = 32
    width = 32
    real = torch.randn(batch_size, height, width)
    imag = torch.randn(batch_size, height, width)
    x = torch.complex(real, imag)
    
    # Get embeddings
    embeddings = model.get_embeddings(x, pooling="mean")
    
    # Check shape
    assert embeddings.shape[0] == batch_size
    assert embeddings.shape[1] == 64  # encoder_dim


if __name__ == "__main__":
    pytest.main([__file__]) 