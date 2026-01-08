"""
Unit tests for model components.
"""

import pytest
import torch
import torch.nn as nn

from contrawimae.models.base import WiMAE
from contrawimae.models.contramae import ContraWiMAE
from contrawimae.models.modules.encoder import Encoder
from contrawimae.models.modules.decoder import Decoder


class TestEncoder:
    """Test encoder functionality."""
    
    @pytest.fixture
    def encoder(self):
        """Create an encoder for testing."""
        return Encoder(
            input_dim=16,  # 1x16 patch
            d_model=64,
            nhead=16,
            num_layers=12,
            mask_ratio=0.6,
            device="cpu"
        )
    
    @pytest.fixture
    def sample_patches(self):
        """Create sample patches for testing."""
        batch_size = 4
        num_complex_patches = 64  # 32x2 complex patches
        num_patches = num_complex_patches * 2  # 128 patches (real + imaginary parts)
        patch_dim = 16    # 1x16 patch
        return torch.randn(batch_size, num_patches, patch_dim)
    
    def test_encoder_initialization(self, encoder):
        """Test encoder initialization."""
        assert encoder.d_model == 64
        assert encoder.mask_ratio == 0.6
        assert len(encoder.transformer.layers) == 12
    
    def test_encoder_forward_with_masking(self, encoder, sample_patches):
        """Test encoder forward pass with masking."""
        encoded_features, ids_keep, ids_mask = encoder(sample_patches, apply_mask=True)
        
        # Check output shapes
        batch_size, num_patches, patch_dim = sample_patches.shape
        num_complex_patches = num_patches // 2  # Real and imaginary parts are separated
        expected_keep_complex = int(num_complex_patches * (1 - encoder.mask_ratio))
        expected_keep = expected_keep_complex * 2  # Double for real and imaginary parts
        expected_mask = num_patches - expected_keep
        
        assert encoded_features.shape == (batch_size, expected_keep, encoder.d_model)
        assert ids_keep.shape == (batch_size, expected_keep)
        assert ids_mask.shape == (batch_size, expected_mask)
        
        # Check that kept indices are valid
        for i in range(batch_size):
            assert torch.all(ids_keep[i] >= 0) and torch.all(ids_keep[i] < num_patches)
            assert torch.all(ids_mask[i] >= 0) and torch.all(ids_mask[i] < num_patches)
    
    def test_encoder_forward_without_masking(self, encoder, sample_patches):
        """Test encoder forward pass without masking."""
        encoded_features = encoder(sample_patches, apply_mask=False)
        
        # Check output shape
        batch_size, num_patches, _ = sample_patches.shape
        
        assert encoded_features.shape == (batch_size, num_patches, encoder.d_model)
        
        # Check that all patches are encoded
        assert encoded_features.shape[1] == num_patches
    
    def test_encoder_different_mask_ratios(self, sample_patches):
        """Test encoder with different mask ratios."""
        ratios = [0.0, 0.25, 0.5, 0.6, 0.75, 0.9]
        
        for ratio in ratios:
            encoder = Encoder(
                input_dim=16,
                d_model=64,
                nhead=16,
                num_layers=12,
                mask_ratio=ratio,
                device="cpu"
            )
            
            encoded_features, ids_keep, ids_mask = encoder(sample_patches, apply_mask=True)
            
            batch_size, num_patches, _ = sample_patches.shape
            num_complex_patches = num_patches // 2  # Real and imaginary parts are separated
            expected_keep_complex = int(num_complex_patches * (1 - ratio))
            expected_keep = expected_keep_complex * 2  # Double for real and imaginary parts
            expected_mask = num_patches - expected_keep
            
            assert encoded_features.shape == (batch_size, expected_keep, encoder.d_model)
            assert ids_keep.shape == (batch_size, expected_keep)
            assert ids_mask.shape == (batch_size, expected_mask)
    
    def test_encoder_gradient_flow(self, encoder, sample_patches):
        """Test that gradients flow through the encoder."""
        encoded_features, ids_keep, ids_mask = encoder(sample_patches, apply_mask=True)
        
        # Compute a dummy loss
        dummy_target = torch.randn_like(encoded_features)
        loss = nn.MSELoss()(encoded_features, dummy_target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed
        for param in encoder.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


class TestDecoder:
    """Test decoder functionality."""
    
    @pytest.fixture
    def decoder(self):
        """Create a decoder for testing."""
        return Decoder(
            d_model=64,
            nhead=8,
            num_layers=4,
            output_dim=16,  # 1x16 patch
            device="cpu"
        )
    
    @pytest.fixture
    def sample_encoded_features(self):
        """Create sample encoded features for testing."""
        batch_size = 4
        num_patches = 128
        d_model = 64
        return torch.randn(batch_size, num_patches, d_model)
    
    @pytest.fixture
    def sample_ids_keep(self):
        """Create sample kept indices."""
        batch_size = 4
        num_kept = 128
        return torch.randint(0, 128, (batch_size, num_kept))
    
    def test_decoder_initialization(self, decoder):
        """Test decoder initialization."""
        assert decoder.d_model == 64
        assert decoder.output_dim == 16
        assert len(decoder.transformer.layers) == 4
    
    def test_decoder_forward(self, decoder, sample_encoded_features, sample_ids_keep):
        """Test decoder forward pass."""
        sequence_length = 128  # Original sequence length
        
        output = decoder(sample_encoded_features, sample_ids_keep, sequence_length)
        
        # Check output shape
        batch_size, num_kept, d_model = sample_encoded_features.shape
        
        assert output.shape == (batch_size, sequence_length, decoder.output_dim)
    
    def test_decoder_reconstruction(self, decoder, sample_encoded_features, sample_ids_keep):
        """Test that decoder can reconstruct patches."""
        sequence_length = 128
        
        output = decoder(sample_encoded_features, sample_ids_keep, sequence_length)
        
        # Check that output has reasonable values
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        # Check output range (should be reasonable)
        assert torch.all(output >= -10) and torch.all(output <= 10)
    
    def test_decoder_gradient_flow(self, decoder, sample_encoded_features, sample_ids_keep):
        """Test that gradients flow through the decoder."""
        sequence_length = 128
        
        output = decoder(sample_encoded_features, sample_ids_keep, sequence_length)
        
        # Compute a dummy loss
        dummy_target = torch.randn_like(output)
        loss = nn.MSELoss()(output, dummy_target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed
        for param in decoder.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


class TestWiMAE:
    """Test WiMAE model functionality."""
    
    @pytest.fixture
    def wimae_model(self):
        """Create a WiMAE model for testing."""
        return WiMAE(
            patch_size=(1, 16),
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
    
    def test_wimae_initialization(self, wimae_model):
        """Test WiMAE model initialization."""
        assert wimae_model.patch_size == (1, 16)
        assert wimae_model.encoder_dim == 64
        assert wimae_model.mask_ratio == 0.6
        assert str(wimae_model.device) == "cpu"
        
        # Check that components are initialized
        assert hasattr(wimae_model, 'patcher')
        assert hasattr(wimae_model, 'encoder')
        assert hasattr(wimae_model, 'decoder')
    
    def test_wimae_forward(self, wimae_model, complex_input):
        """Test WiMAE forward pass."""
        output = wimae_model(complex_input)
        
        # Check output structure
        assert "encoded_features" in output
        assert "reconstructed_patches" in output
        assert "ids_keep" in output
        assert "ids_mask" in output
        
        # Check shapes
        batch_size = complex_input.shape[0]
        num_complex_patches = (32 // 1) * (32 // 16)  # 32 * 2 = 64 complex patches
        num_patches = num_complex_patches * 2  # 128 patches (real + imaginary parts)
        expected_keep_complex = int(num_complex_patches * (1 - wimae_model.mask_ratio))
        expected_keep = expected_keep_complex * 2  # Double for real and imaginary parts
        
        assert output["encoded_features"].shape == (batch_size, expected_keep, wimae_model.encoder_dim)
        assert output["reconstructed_patches"].shape == (batch_size, num_patches, 16)  # 1x16 patch
    
    def test_wimae_forward_without_reconstruction(self, wimae_model, complex_input):
        """Test WiMAE forward pass without reconstruction."""
        output = wimae_model(complex_input, return_reconstruction=False)
        
        # Check output structure
        assert "encoded_features" in output
        assert "ids_keep" in output
        assert "ids_mask" in output
        assert "reconstructed_patches" not in output
    
    def test_wimae_encode(self, wimae_model, complex_input):
        """Test WiMAE encode method."""
        encoded_features = wimae_model.encode(complex_input)
        
        # Check output shape
        batch_size = complex_input.shape[0]
        num_complex_patches = (32 // 1) * (32 // 16)  # 32 * 2 = 64 complex patches
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
        num_complex_patches = (32 // 1) * (32 // 16)  # 32 * 2 = 64 complex patches
        num_patches = num_complex_patches * 2  # 128 patches (real + imaginary parts)
        
        assert reconstructed.shape == (batch_size, num_patches, 16)  # 1x16 patch
    
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
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        wimae_model.save_checkpoint(str(checkpoint_path), test_data="test")
        
        # Load checkpoint
        loaded_model = WiMAE.from_checkpoint(str(checkpoint_path), device="cpu")
        
        # Check that model parameters are the same
        for param_name, original_param in wimae_model.named_parameters():
            loaded_param = dict(loaded_model.named_parameters())[param_name]
            assert torch.allclose(original_param, loaded_param)
        
        # Check that model info is the same
        original_info = wimae_model.get_model_info()
        loaded_info = loaded_model.get_model_info()
        
        for key in ["model_type", "patch_size", "encoder_dim", "encoder_layers", 
                   "encoder_nhead", "decoder_layers", "decoder_nhead", "mask_ratio"]:
            assert original_info[key] == loaded_info[key]
    
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


class TestContraWiMAE:
    """Test ContraWiMAE model functionality."""
    
    @pytest.fixture
    def contramae_model(self):
        """Create a ContraWiMAE model for testing."""
        return ContraWiMAE(
            patch_size=(1, 16),
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
    
    def test_contramae_initialization(self, contramae_model):
        """Test ContraWiMAE model initialization."""
        assert contramae_model.patch_size == (1, 16)
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
        
        # Check that model info is the same
        original_info = contramae_model.get_model_info()
        loaded_info = loaded_model.get_model_info()
        
        for key in ["model_type", "patch_size", "encoder_dim", "encoder_layers", 
                   "encoder_nhead", "decoder_layers", "decoder_nhead", "mask_ratio",
                   "contrastive_dim", "temperature", "snr_min", "snr_max"]:
            assert original_info[key] == loaded_info[key]
    
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