"""
ContraWiMAE model implementation.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from .base import WiMAE
from .modules import ContrastiveHead, apply_channel_augmentations


class ContraWiMAE(WiMAE):
    """
    Contrastive Wireless Masked Autoencoder (ContraWiMAE) model.
    
    This model extends WiMAE with contrastive learning capabilities,
    enabling the model to learn representations that are invariant
    to certain augmentations.
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int],
        encoder_dim: int = 256,
        encoder_layers: int = 12,
        encoder_nhead: int = 8,
        decoder_layers: int = 8,
        decoder_nhead: int = 8,
        mask_ratio: float = 0.75,
        contrastive_dim: int = 256,
        temperature: float = 0.1,
        snr_min: float = 0.0,
        snr_max: float = 30.0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the ContraWiMAE model.
        
        Args:
            patch_size: Size of patches (height, width)
            encoder_dim: Dimension of encoder
            encoder_layers: Number of encoder layers
            encoder_nhead: Number of encoder attention heads
            decoder_layers: Number of decoder layers
            decoder_nhead: Number of decoder attention heads
            mask_ratio: Ratio of patches to mask during training
            contrastive_dim: Dimension of contrastive projection
            temperature: Temperature parameter for contrastive loss
            snr_min: Minimum SNR for augmentations
            snr_max: Maximum SNR for augmentations
            device: Device to place the model on
        """
        # Initialize base WiMAE model
        super().__init__(
            patch_size=patch_size,
            encoder_dim=encoder_dim,
            encoder_layers=encoder_layers,
            encoder_nhead=encoder_nhead,
            decoder_layers=decoder_layers,
            decoder_nhead=decoder_nhead,
            mask_ratio=mask_ratio,
            device=device,
        )
        
        # Add contrastive head
        self.contrastive_head = ContrastiveHead(
            input_dim=encoder_dim,
            proj_dim=contrastive_dim,
        ).to(self.device)
        
        # Augmentation parameters
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.temperature = temperature
    
    def forward(
        self, 
        x: torch.Tensor,
        mask_ratio: Optional[float] = None,
        return_reconstruction: bool = True,
        return_contrastive: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the ContraWiMAE model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            mask_ratio: Masking ratio (uses self.mask_ratio if None)
            return_reconstruction: Whether to return reconstruction
            return_contrastive: Whether to return contrastive features
            
        Returns:
            Dictionary containing encoded features and optionally reconstruction/contrastive features
        """
        # Get base WiMAE output
        output = super().forward(x, mask_ratio, return_reconstruction)
        
        # Add contrastive features if requested
        if return_contrastive:
            encoded_features = output["encoded_features"]
            contrastive_features = self.contrastive_head(encoded_features)
            output["contrastive_features"] = contrastive_features
        
        return output
    
    def forward_with_augmentation(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None,
        return_reconstruction: bool = True,
        noise_prob: float = 1.0,
        freq_shift_prob: float = 0.0,
        phase_rot_prob: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with augmentation for contrastive learning.
        
        Args:
            x: Input tensor
            mask_ratio: Masking ratio
            return_reconstruction: Whether to return reconstruction
            noise_prob: Probability of applying noise injection
            freq_shift_prob: Probability of applying frequency shift
            phase_rot_prob: Probability of applying phase rotation
            
        Returns:
            Dictionary containing outputs for original and augmented inputs
        """
        # Create augmented version
        x_aug = apply_channel_augmentations(
            x, 
            noise_prob=noise_prob,
            freq_shift_prob=freq_shift_prob,
            phase_rot_prob=phase_rot_prob,
            snr_min=self.snr_min, 
            snr_max=self.snr_max
        )
        
        # Forward pass for original input
        output_orig = self.forward(
            x, 
            mask_ratio=mask_ratio, 
            return_reconstruction=return_reconstruction,
            return_contrastive=True
        )
        
        # Forward pass for augmented input
        output_aug = self.forward(
            x_aug, 
            mask_ratio=mask_ratio, 
            return_reconstruction=return_reconstruction,
            return_contrastive=True
        )
        
        # Combine outputs
        combined_output = {
            "original": output_orig,
            "augmented": output_aug,
        }
        
        return combined_output
    
    def compute_contrastive_loss(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive loss between two sets of features.
        
        Args:
            features1: First set of features
            features2: Second set of features
            temperature: Temperature parameter
            
        Returns:
            Contrastive loss
        """
        return self.contrastive_head.compute_contrastive_loss(
            features1, features2, temperature
        )
    
    def get_contrastive_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get contrastive embeddings from encoded features.
        
        The contrastive head already performs mean pooling over patches.
        
        Args:
            x: Input tensor
            
        Returns:
            Contrastive embeddings
        """
        encoded_features = self.encode(x)
        return self.contrastive_head(encoded_features)
    
    def save_checkpoint(self, filepath: str, **kwargs):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            **kwargs: Additional data to save
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "patch_size": self.patch_size,
            "encoder_dim": self.encoder_dim,
            "encoder_layers": self.encoder.num_layers,
            "encoder_nhead": self.encoder.layers[0].self_attn.num_heads,
            "decoder_layers": self.decoder.num_layers,
            "decoder_nhead": self.decoder.layers[0].self_attn.num_heads,
            "mask_ratio": self.mask_ratio,
            "contrastive_dim": self.contrastive_head.proj_dim,
            "temperature": self.temperature,
            "snr_min": self.snr_min,
            "snr_max": self.snr_max,
            **kwargs
        }
        
        torch.save(checkpoint, filepath)
    
    @classmethod
    def from_checkpoint(cls, filepath: str, device: Optional[torch.device] = None) -> "ContraWiMAE":
        """
        Load model from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            device: Device to load model on
            
        Returns:
            Loaded ContraWiMAE model
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        # Extract model parameters
        patch_size = checkpoint["patch_size"]
        encoder_dim = checkpoint["encoder_dim"]
        encoder_layers = checkpoint["encoder_layers"]
        encoder_nhead = checkpoint["encoder_nhead"]
        decoder_layers = checkpoint["decoder_layers"]
        decoder_nhead = checkpoint["decoder_nhead"]
        mask_ratio = checkpoint["mask_ratio"]
        contrastive_dim = checkpoint.get("contrastive_dim", encoder_dim)
        temperature = checkpoint.get("temperature", 0.1)
        snr_min = checkpoint.get("snr_min", 0.0)
        snr_max = checkpoint.get("snr_max", 30.0)
        
        # Create model
        model = cls(
            patch_size=patch_size,
            encoder_dim=encoder_dim,
            encoder_layers=encoder_layers,
            encoder_nhead=encoder_nhead,
            decoder_layers=decoder_layers,
            decoder_nhead=decoder_nhead,
            mask_ratio=mask_ratio,
            contrastive_dim=contrastive_dim,
            temperature=temperature,
            snr_min=snr_min,
            snr_max=snr_max,
            device=device,
        )
        
        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model architecture.
        
        Returns:
            Dictionary containing model information
        """
        base_info = super().get_model_info()
        base_info.update({
            "model_type": "ContraWiMAE",
            "contrastive_dim": self.contrastive_head.proj_dim,
            "temperature": self.temperature,
            "snr_min": self.snr_min,
            "snr_max": self.snr_max,
        })
        
        return base_info 