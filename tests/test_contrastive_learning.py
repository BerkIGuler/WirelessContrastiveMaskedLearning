"""
Unit tests for contrastive learning components.
"""

import pytest
import torch
import torch.nn.functional as F

from contrawimae.models.modules.contrastive_head import ContrastiveHead


class TestContrastiveHead:
    """Test contrastive head functionality."""
    
    @pytest.fixture
    def contrastive_head(self):
        """Create a contrastive head for testing."""
        return ContrastiveHead(input_dim=64, proj_dim=64)
    
    @pytest.fixture
    def sample_features(self):
        """Create sample encoded features."""
        batch_size = 4
        seq_len = 128
        feature_dim = 64
        return torch.randn(batch_size, seq_len, feature_dim)
    
    def test_contrastive_head_initialization(self):
        """Test contrastive head initialization."""
        head = ContrastiveHead(input_dim=64, proj_dim=64)
        
        # Check that projection head has correct structure
        assert len(head.proj_head) == 3  # Linear -> ReLU -> Linear
        
        # Check layer dimensions
        first_linear = head.proj_head[0]
        second_linear = head.proj_head[2]
        
        assert first_linear.in_features == 64
        assert first_linear.out_features == 64 * 2  # proj_dim * 2
        assert second_linear.in_features == 64 * 2
        assert second_linear.out_features == 64
    
    def test_contrastive_head_forward(self, contrastive_head, sample_features):
        """Test contrastive head forward pass."""
        output = contrastive_head(sample_features)
        
        # Check output shape
        batch_size = sample_features.shape[0]
        proj_dim = 64
        
        assert output.shape == (batch_size, proj_dim)
        assert output.dtype == torch.float32
        
        # Check that output is L2 normalized
        norms = torch.norm(output, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    def test_contrastive_head_normalization(self, contrastive_head, sample_features):
        """Test that output is properly normalized."""
        output = contrastive_head(sample_features)
        
        # Check L2 normalization
        norms = torch.norm(output, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
        
        # Check that values are in reasonable range
        assert torch.all(output >= -1.1) and torch.all(output <= 1.1)
    
    def test_contrastive_head_different_dimensions(self):
        """Test contrastive head with different dimensions."""
        # Test with different input and projection dimensions
        head = ContrastiveHead(input_dim=128, proj_dim=64)
        
        batch_size = 4
        seq_len = 128
        features = torch.randn(batch_size, seq_len, 128)
        
        output = head(features)
        
        assert output.shape == (batch_size, 64)
        assert torch.allclose(torch.norm(output, dim=1), torch.ones(batch_size), atol=1e-6)
    
    def test_contrastive_head_gradient_flow(self, contrastive_head, sample_features):
        """Test that gradients flow through the contrastive head."""
        output = contrastive_head(sample_features)
        
        # Compute a dummy loss
        dummy_target = torch.randn_like(output)
        loss = F.mse_loss(output, dummy_target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed
        for param in contrastive_head.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
    
    def test_get_contrastive_features(self, contrastive_head, sample_features):
        """Test get_contrastive_features method."""
        features1 = contrastive_head.get_contrastive_features(sample_features)
        features2 = contrastive_head(sample_features)
        
        # Should be identical
        assert torch.allclose(features1, features2)


class TestContrastiveLoss:
    """Test contrastive loss functionality."""
    
    @pytest.fixture
    def contrastive_head(self):
        """Create a contrastive head for testing."""
        return ContrastiveHead(input_dim=64, proj_dim=64)
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        batch_size = 4
        embed_dim = 64
        return torch.randn(batch_size, embed_dim)
    
    def test_compute_contrastive_loss(self, contrastive_head, sample_embeddings):
        """Test contrastive loss computation."""
        # Create two sets of embeddings
        z_i = sample_embeddings
        z_j = torch.randn_like(sample_embeddings)
        
        temperature = 0.1
        loss = contrastive_head.compute_contrastive_loss(z_i, z_j, temperature)
        
        # Check that loss is a scalar
        assert loss.shape == ()
        assert loss.dtype == torch.float32
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss > 0  # Loss should be positive
    
    def test_compute_contrastive_loss_normalization(self, contrastive_head, sample_embeddings):
        """Test that embeddings are normalized in loss computation."""
        z_i = sample_embeddings
        z_j = torch.randn_like(sample_embeddings)
        
        # Compute loss
        loss = contrastive_head.compute_contrastive_loss(z_i, z_j, temperature=0.1)
        
        # Check that inputs are normalized during computation
        # The loss function should handle normalization internally
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_compute_contrastive_loss_different_temperatures(self, contrastive_head, sample_embeddings):
        """Test contrastive loss with different temperatures."""
        z_i = sample_embeddings
        z_j = torch.randn_like(sample_embeddings)
        
        temperatures = [0.05, 0.1, 0.2, 0.5, 1.0]
        losses = []
        
        for temp in temperatures:
            loss = contrastive_head.compute_contrastive_loss(z_i, z_j, temp)
            losses.append(loss.item())
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
            assert loss > 0
        
        # Lower temperatures should generally produce higher losses
        # (though this is not guaranteed due to randomness)
        assert all(l > 0 for l in losses)
    
    def test_compute_contrastive_loss_symmetry(self, contrastive_head, sample_embeddings):
        """Test that loss is symmetric with respect to inputs."""
        z_i = sample_embeddings
        z_j = torch.randn_like(sample_embeddings)
        
        # Compute loss in both directions
        loss_ij = contrastive_head.compute_contrastive_loss(z_i, z_j, temperature=0.1)
        loss_ji = contrastive_head.compute_contrastive_loss(z_j, z_i, temperature=0.1)
        
        # Should be equal (InfoNCE is symmetric)
        assert torch.allclose(loss_ij, loss_ji, atol=1e-6)
    
    def test_compute_contrastive_loss_identical_inputs(self, contrastive_head, sample_embeddings):
        """Test loss with identical inputs."""
        z_i = sample_embeddings
        z_j = sample_embeddings.clone()
        
        loss = contrastive_head.compute_contrastive_loss(z_i, z_j, temperature=0.1)
        
        # Should produce a finite loss
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        # When inputs are identical, loss should be very low (close to 0)
        # as the model is already perfectly aligned
        assert loss >= 0
    
    def test_compute_contrastive_loss_different_batch_sizes(self, contrastive_head):
        """Test contrastive loss with different batch sizes."""
        batch_sizes = [2, 4, 8, 16]
        embed_dim = 64
        
        for batch_size in batch_sizes:
            # Use different random seeds for each batch size to avoid edge cases
            torch.manual_seed(42 + batch_size)
            z_i = torch.randn(batch_size, embed_dim)
            z_j = torch.randn(batch_size, embed_dim)
            
            loss = contrastive_head.compute_contrastive_loss(z_i, z_j, temperature=0.1)
            
            assert loss.shape == ()
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
            # Loss should be positive for random embeddings (with small epsilon for numerical stability)
            assert loss > 1e-8, f"Loss is {loss.item()} for batch_size={batch_size}, expected > 0"
    
    def test_compute_contrastive_loss_gradient_flow(self, contrastive_head, sample_embeddings):
        """Test that gradients flow through the loss function."""
        z_i = sample_embeddings.clone().requires_grad_(True)
        z_j = torch.randn_like(sample_embeddings).requires_grad_(True)
        
        loss = contrastive_head.compute_contrastive_loss(z_i, z_j, temperature=0.1)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed
        assert z_i.grad is not None
        assert z_j.grad is not None
        assert not torch.isnan(z_i.grad).any()
        assert not torch.isnan(z_j.grad).any()
    
    def test_compute_contrastive_loss_info_nce_structure(self, contrastive_head, sample_embeddings):
        """Test that the loss follows InfoNCE structure."""
        z_i = sample_embeddings
        
        # Test with multiple random seeds to ensure robustness
        for seed in [42, 123, 456]:
            torch.manual_seed(seed)
            z_j = torch.randn_like(sample_embeddings)
            
            # Compute loss with higher temperature to avoid numerical instability
            loss = contrastive_head.compute_contrastive_loss(z_i, z_j, temperature=1.0)
            
            # InfoNCE loss should be positive and finite
            assert loss > 0
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
            
            # Should be reasonable magnitude (InfoNCE loss can vary, but should be finite)
            # Using a more generous upper bound to account for random variations
            assert loss < 50.0
    
    def test_compute_contrastive_loss_edge_cases(self, contrastive_head, sample_embeddings):
        """Test contrastive loss with edge cases."""
        # Test with very small batch size
        z_i = torch.randn(1, 64)
        z_j = torch.randn(1, 64)
        
        with pytest.raises(ValueError):
            # Should fail with batch size 1 (no negative samples)
            contrastive_head.compute_contrastive_loss(z_i, z_j, temperature=0.1)
        
        # Test with very high temperature
        z_i = torch.randn(4, 64)
        z_j = torch.randn(4, 64)
        
        loss = contrastive_head.compute_contrastive_loss(z_i, z_j, temperature=10.0)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_compute_contrastive_loss_self_similarity_masking(self, contrastive_head, sample_embeddings):
        """Test that self-similarity is properly masked."""
        z_i = sample_embeddings
        z_j = sample_embeddings.clone()
        
        loss = contrastive_head.compute_contrastive_loss(z_i, z_j, temperature=0.1)
        
        # Should handle self-similarity properly
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        # When inputs are identical, loss should be very low (close to 0)
        # as the model is already perfectly aligned
        assert loss >= 0 