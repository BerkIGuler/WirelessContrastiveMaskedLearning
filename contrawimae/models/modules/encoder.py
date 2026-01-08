"""
Encoder module for WiMAE and ContraWiMAE models.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .pos_encodings import SinusoidalPositionalEncoding, LearnablePositionalEncoding
from .masking import MaskGenerator


class Encoder(nn.Module):
    """
    Transformer-based encoder for processing wireless channel patches.
    
    This encoder takes flattened patches and processes them through multiple
    transformer layers to produce encoded representations.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 16,
        activation: str = "gelu",
        mask_ratio: float = 0.6,
        dropout: float = 0.1,
        num_layers: int = 12,
        max_len: int = 128,
        pos_encoding_type: str = "learnable",
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the encoder.
        
        Args:
            input_dim: Input dimension (patch dimension)
            d_model: Model dimension
            nhead: Number of attention heads
            activation: Activation function
            mask_ratio: Ratio of patches to mask during training
            dropout: Dropout probability
            num_layers: Number of transformer layers
            max_len: Maximum sequence length
            pos_encoding_type: Type of positional encoding ("learnable" or "sinusoidal")
            device: Device to place the model on
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.device = device or torch.device("cpu")
        
        # Input projection
        self.linear_1 = nn.Linear(input_dim, d_model)
        
        # Choose the positional encoding type
        if pos_encoding_type == "learnable":
            self.positional_encoding = LearnablePositionalEncoding(max_len=max_len, d_model=d_model)
        elif pos_encoding_type == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(max_len=max_len, d_model=d_model)
        else:
            raise ValueError("pos_encoding_type must be either 'learnable' or 'sinusoidal'")
        
        # Add masking generator
        self.masking = MaskGenerator(device=self.device, mask_ratio=mask_ratio)
        
        # Transformer encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            activation=activation,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor, apply_mask: bool = True) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            apply_mask: Whether to apply masking (True for training, False for inference)
            
        Returns:
            If apply_mask is True:
                tuple: (encoded_tokens, ids_keep, ids_mask)
                    - encoded_tokens: tensor of shape (batch_size, unmasked_length, d_model)
                    - ids_keep: indices of tokens that were kept
                    - ids_mask: indices of tokens that were masked
            If apply_mask is False:
                tensor: encoded_tokens of shape (batch_size, seq_length, d_model)
        """
        x = self.linear_1(x)
        x = self.positional_encoding(x)
        
        if apply_mask:
            # Training mode with masking
            unmasked_tokens, ids_keep, ids_mask = self.masking(x)
            encoded_tokens = self.transformer(unmasked_tokens)
            return encoded_tokens, ids_keep, ids_mask
        else:
            # Inference mode without masking
            encoded_tokens = self.transformer(x)
            return encoded_tokens 