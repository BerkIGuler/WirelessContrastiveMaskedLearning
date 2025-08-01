"""
Base WiMAE model implementation.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from .modules import Encoder, Decoder, Patcher


class WiMAE(nn.Module):
    """
    Wireless Masked Autoencoder (WiMAE) model.
    
    This model implements a masked autoencoder specifically designed for
    wireless channel data, using transformer-based encoder and decoder.
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
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the WiMAE model.
        
        Args:
            patch_size: Size of patches (height, width)
            encoder_dim: Dimension of encoder
            encoder_layers: Number of encoder layers
            encoder_nhead: Number of encoder attention heads
            decoder_layers: Number of decoder layers
            decoder_nhead: Number of decoder attention heads
            mask_ratio: Ratio of patches to mask during training
            device: Device to place the model on
        """
        super().__init__()
        
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim
        self.mask_ratio = mask_ratio
        self.device = device or torch.device("cpu")
        
        # Initialize components
        self.patcher = Patcher(patch_size)
        
        # Get patch dimension (for complex channel matrices, this is patch_height * patch_width)
        patch_dim = patch_size[0] * patch_size[1]
        
        self.encoder = Encoder(
            input_dim=patch_dim,
            d_model=encoder_dim,
            nhead=encoder_nhead,
            num_layers=encoder_layers,
            mask_ratio=mask_ratio,
            device=self.device,
        )
        
        self.decoder = Decoder(
            d_model=encoder_dim,
            nhead=decoder_nhead,
            num_layers=decoder_layers,
            output_dim=patch_dim,
            device=self.device,
        )
        
        self.to(self.device)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask_ratio: Optional[float] = None,
        return_reconstruction: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the WiMAE model.
        
        Args:
            x: Input tensor (handles both complex and regular tensors)
            mask_ratio: Masking ratio (uses self.mask_ratio if None)
            return_reconstruction: Whether to return reconstruction
            
        Returns:
            Dictionary containing:
                - encoded_features: Encoded patch features
                - ids_keep: Indices of kept patches
                - ids_mask: Indices of masked patches
                - reconstructed_patches: Reconstructed patches (if return_reconstruction=True)
        """
        # Convert to patches
        patches = self.patcher(x)
        
        # Encode patches
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        encoded_features, ids_keep, ids_mask = self.encoder(patches, apply_mask=True)
        
        output = {"encoded_features": encoded_features, "ids_keep": ids_keep, "ids_mask": ids_mask}
        
        # Decode if requested
        if return_reconstruction:
            reconstructed_patches = self.decoder(
                encoded_features, 
                ids_keep, 
                patches.shape[1]  # Original sequence length
            )
            output["reconstructed_patches"] = reconstructed_patches
        
        return output
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input data without masking.
        
        Args:
            x: Input tensor
            
        Returns:
            Encoded features tensor
        """
        with torch.no_grad():
            patches = self.patcher(x)
            encoded_features = self.encoder(patches, apply_mask=False)
            return encoded_features
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input data without masking.
        
        Args:
            x: Input tensor
            
        Returns:
            Reconstructed patches tensor
        """
        with torch.no_grad():
            output = self.forward(x, mask_ratio=0.0, return_reconstruction=True)
            return output["reconstructed_patches"]
    
    def get_embeddings(self, x: torch.Tensor, pooling: str = "mean") -> torch.Tensor:
        """
        Get embeddings from encoded features using specified pooling method.
        
        Args:
            x: Input tensor
            pooling: Pooling method ("mean", "cls", "max")
            
        Returns:
            Pooled embeddings tensor
            
        Raises:
            ValueError: If pooling method is not supported
        """
        encoded_features = self.encode(x)
        
        if pooling == "mean":
            # Mean pooling over all patches
            embeddings = torch.mean(encoded_features, dim=1)
        elif pooling == "cls":
            # Use first token as CLS token
            embeddings = encoded_features[:, 0, :]
        elif pooling == "max":
            # Max pooling over patches
            embeddings = torch.max(encoded_features, dim=1)[0]
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")
        
        return embeddings
    
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
            **kwargs
        }
        
        torch.save(checkpoint, filepath)
    
    @classmethod
    def from_checkpoint(cls, filepath: str, device: Optional[torch.device] = None) -> "WiMAE":
        """
        Load model from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            device: Device to load model on
            
        Returns:
            Loaded WiMAE model
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
        
        # Create model
        model = cls(
            patch_size=patch_size,
            encoder_dim=encoder_dim,
            encoder_layers=encoder_layers,
            encoder_nhead=encoder_nhead,
            decoder_layers=decoder_layers,
            decoder_nhead=decoder_nhead,
            mask_ratio=mask_ratio,
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
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "WiMAE",
            "patch_size": self.patch_size,
            "encoder_dim": self.encoder_dim,
            "encoder_layers": len(self.encoder.layers),
            "encoder_nhead": self.encoder.layers[0].self_attn.num_heads,
            "decoder_layers": len(self.decoder.layers),
            "decoder_nhead": self.decoder.layers[0].self_attn.num_heads,
            "mask_ratio": self.mask_ratio,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        } 