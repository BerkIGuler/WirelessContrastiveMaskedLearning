"""
Contrastive head module for ContraWiMAE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveHead(nn.Module):
    """
    Contrastive learning head for ContraWiMAE.
    
    This module projects encoded features to a lower-dimensional space
    for contrastive learning, enabling the model to learn representations
    that are invariant to certain augmentations.
    """
    
    def __init__(self, input_dim: int = 64, proj_dim: int = 64):
        """
        Initialize the contrastive head.
        
        Args:
            input_dim: Input dimension (encoder output dimension)
            proj_dim: Projection dimension
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        
        self.proj_head = nn.Sequential(
            nn.Linear(input_dim, proj_dim * 2),
            nn.ReLU(),
            nn.Linear(proj_dim * 2, proj_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the contrastive head.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Projected features of shape (batch_size, proj_dim)
        """
        # Mean pooling over all patches
        mean_encoding = torch.mean(x, dim=1)  # [B, embed_dim]
        
        # Project to contrastive space
        z = self.proj_head(mean_encoding)
        z = F.normalize(z, dim=1)  # L2 normalize
        
        return z
    
    def compute_contrastive_loss(
        self, 
        features1: torch.Tensor, 
        features2: torch.Tensor, 
        temperature: float = 0.1
    ) -> torch.Tensor:
        """
        Compute InfoNCE/NT-Xent contrastive loss between two sets of embeddings.

        Args:
            features1: Embeddings of original samples, shape [B, D]
            features2: Embeddings of augmented samples, shape [B, D]
            temperature: Temperature scaling parameter

        Returns:
            Scalar contrastive loss
        """
        # Ensure features are normalized
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        
        # Get batch size
        batch_size = features1.shape[0]

        # Combine features from both views
        features = torch.cat([features1, features2], dim=0)  # [2B, D]

        # Compute full similarity matrix
        sim_matrix = torch.mm(features, features.t())  # [2B, 2B]

        # Create labels identifying positive pairs
        # For i-th sample in features1, the positive is the i-th sample in features2 (at index batch_size + i)
        # For i-th sample in features2, the positive is the i-th sample in features1 (at index i)
        labels = torch.zeros(2 * batch_size, device=features1.device, dtype=torch.long)
        for i in range(batch_size):
            labels[i] = batch_size + i
            labels[i + batch_size] = i

        # Mask out self-similarity (along the diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features1.device)
        sim_matrix = sim_matrix.masked_fill_(mask, -float('inf'))

        # Apply temperature scaling
        sim_matrix = sim_matrix / temperature

        # Compute InfoNCE loss
        contrastive_loss = F.cross_entropy(sim_matrix, labels)

        return contrastive_loss
    
    def get_contrastive_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get contrastive features for a given input.
        
        Args:
            x: Input tensor
            
        Returns:
            Contrastive features
        """
        return self.forward(x) 