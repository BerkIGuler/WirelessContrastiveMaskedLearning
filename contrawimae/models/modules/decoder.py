"""
Decoder module for WiMAE and ContraWiMAE models.
"""

import torch
import torch.nn as nn
from typing import Optional

from .pos_encodings import LearnablePositionalEncoding, SinusoidalPositionalEncoding


class Decoder(nn.Module):
    """
    Transformer-based decoder for reconstructing masked patches.
    
    This decoder takes encoded representations and reconstructs the original
    patches, learning to predict the masked portions.
    """
    
    def __init__(
        self,
        output_dim: int,
        d_model: int = 64,
        nhead: int = 8,
        activation: str = "gelu",
        dropout: float = 0.1,
        num_layers: int = 4,
        max_len: int = 128,
        pos_encoding_type: str = "learnable",
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the decoder.
        
        Args:
            output_dim: Output dimension (patch dimension)
            d_model: Model dimension
            nhead: Number of attention heads
            activation: Activation function for MLP
            dropout: Dropout probability for MLP
            num_layers: Number of transformer layers for decoder
            max_len: Maximum sequence length
            pos_encoding_type: Type of positional encoding ("learnable" or "sinusoidal")
            device: Device to place the model on
        """
        super().__init__()
        
        self.d_model = d_model
        self.output_dim = output_dim
        self.device = device or torch.device("cpu")
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Choose the positional encoding type
        if pos_encoding_type == "learnable":
            self.positional_encoding = LearnablePositionalEncoding(max_len=max_len, d_model=d_model)
        elif pos_encoding_type == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(max_len=max_len, d_model=d_model)
        else:
            raise ValueError("pos_encoding_type must be either 'learnable' or 'sinusoidal'")
        
        # Transformer layers
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            activation=activation,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        
        # Final projection to patch dimension
        self.linear = nn.Linear(d_model, output_dim)
        
        self.to(self.device)
    
    def forward(self, encoded_tokens: torch.Tensor, ids_keep: torch.Tensor, orig_sequence_length: int) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            encoded_tokens: Encoded visible tokens (B, num_visible, d_model)
            ids_keep: Indices to restore original sequence [B, orig_seq_len]
            orig_sequence_length: Original sequence length
            
        Returns:
            Reconstructed sequence (B, orig_seq_len, output_dim)
        """
        B = encoded_tokens.shape[0]
        
        # Create full sequence filled with mask tokens
        full_sequence = self.mask_token.expand(B, orig_sequence_length, -1)
        num_visible = encoded_tokens.shape[1]
        
        # Create indices for scatter operation
        batch_indices = torch.arange(B, device=encoded_tokens.device).view(-1, 1)
        batch_indices = batch_indices.repeat(1, num_visible)
        
        # Scatter the encoded tokens into their original positions
        full_sequence = full_sequence.clone()
        full_sequence[batch_indices, ids_keep] = encoded_tokens
        
        # Add positional encoding
        full_sequence = self.positional_encoding(full_sequence)
        
        # Transform and project
        decoded = self.transformer(full_sequence)
        output = self.linear(decoded)
        
        return output
    
    def reconstruct_patches(
        self, 
        encoded_features: torch.Tensor,
        ids_keep: torch.Tensor,
        orig_sequence_length: int,
    ) -> torch.Tensor:
        """
        Reconstruct patches from encoded features.
        
        Args:
            encoded_features: Encoded features from encoder
            ids_keep: Indices of kept patches
            orig_sequence_length: Original sequence length
            
        Returns:
            Reconstructed patches
        """
        return self.forward(encoded_features, ids_keep, orig_sequence_length) 