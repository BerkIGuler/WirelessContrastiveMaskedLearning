"""
Positional encoding modules for transformers.
"""

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformers.

    This class creates a positional encoding using sinusoidal functions that remain fixed during training.

    Args:
        max_len (int): Maximum length of the sequence.
        d_model (int): Dimension of the model.
    """

    def __init__(self, max_len, d_model):
        super(SinusoidalPositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)  # position * div_term is a matrix due to broadcasting
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """Forward pass to add positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: Tensor with added positional encodings of the same shape as the input.
        """
        return x + self.pe[:, :x.size(1), :]


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding for transformers.

    This class creates a positional encoding that can be learned during training.

    Args:
        max_len (int): Maximum length of the sequence.
        d_model (int): Dimension of the model.
    """

    def __init__(self, max_len, d_model):
        super(LearnablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)  # Initialize with small values

    def forward(self, x):
        """Forward pass to add positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: Tensor with added positional encodings of the same shape as the input.
        """
        return x + self.position_embeddings[:, :x.size(1), :] 