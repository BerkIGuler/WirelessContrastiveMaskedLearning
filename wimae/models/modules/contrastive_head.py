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
    
    def get_contrastive_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get contrastive features for a given input.
        
        Args:
            x: Input tensor
            
        Returns:
            Contrastive features
        """
        return self.forward(x) 