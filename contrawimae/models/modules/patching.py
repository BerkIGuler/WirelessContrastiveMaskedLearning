"""
Patching module for converting wireless channel data to patches.
"""

import torch
import numpy as np


class Patcher:
    def __init__(self, patch_size=(1, 16)):
        """
        Initialize the patcher with patch dimensions.

        Args:
            patch_size: Tuple of (height, width) for patch dimensions
        """
        if not isinstance(patch_size, tuple):
            raise ValueError("patch_size must be a tuple of (height, width)")
        self.patch_size = patch_size

    def __call__(self, channel_matrix):
        """
        Create patches from batched complex channel matrices using reshape operations.

        Args:
            channel_matrix: Complex channel matrix H ∈ C^(B×M×N) where B is batch size
                          or H ∈ C^(M×N) for single input

        Returns:
            Tensor of patches with shape (B, 2*P, L) where:
                B is batch size
                P is number of patches per matrix
                L is patch length (patch_height * patch_width)
            For single input, returns shape (2*P, L)
        """
        # Convert to torch tensor if needed
        H = torch.from_numpy(channel_matrix) if isinstance(channel_matrix, np.ndarray) else channel_matrix

        # Handle non-batched input
        if len(H.shape) == 2:
            H = H.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Get dimensions
        B, M, N = H.shape
        patch_height, patch_width = self.patch_size

        # Validate matrix dimensions
        if M % patch_height != 0 or N % patch_width != 0:
            raise ValueError(f"Matrix dimensions ({M}, {N}) must be divisible by patch_size {self.patch_size}")

        # Calculate number of patches in each dimension
        num_patches_height = M // patch_height
        num_patches_width = N // patch_width
        num_patches = num_patches_height * num_patches_width

        # Reshape matrices into patches of real components
        real_patches = H.real.reshape(B, num_patches_height, patch_height, num_patches_width, patch_width)
        real_patches = real_patches.permute(0, 1, 3, 2, 4).reshape(B, num_patches, -1)

        # Reshape matrices into patches of imaginary components
        imag_patches = H.imag.reshape(B, num_patches_height, patch_height, num_patches_width, patch_width)
        imag_patches = imag_patches.permute(0, 1, 3, 2, 4).reshape(B, num_patches, -1)

        # Concatenate real and imaginary patches
        patches = torch.cat([real_patches, imag_patches], dim=1)

        return patches.squeeze(0) if squeeze_output else patches


class InversePatcher:
    def __init__(self, original_shape, patch_size=(1, 16)):
        """
        Initialize the inverse patcher.

        Args:
            original_shape: Tuple of (height, width) of the original matrix
            patch_size: Tuple of (height, width) for patch dimensions
        """
        if not isinstance(patch_size, tuple):
            raise ValueError("patch_size must be a tuple of (height, width)")
        if not isinstance(original_shape, tuple):
            raise ValueError("original_shape must be a tuple of (height, width)")

        self.original_shape = original_shape
        self.patch_size = patch_size

    def __call__(self, patches):
        """
        Reconstruct the original complex matrices from batched patches.

        Args:
            patches: Tensor of patches, either:
                    Shape (B, 2*P, L) for batched input where:
                        B is batch size
                        P is number of patches per matrix
                        L is patch length (patch_height * patch_width)
                    Shape (2*P, L) for single input

        Returns:
            Complex channel matrix either:
                H ∈ C^(B×M×N) for batched input
                H ∈ C^(M×N) for single input
        """
        # Handle non-batched input
        if len(patches.shape) == 2:
            patches = patches.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        B, P, L = patches.shape
        actual_P = P // 2
        M, N = self.original_shape
        patch_height, patch_width = self.patch_size

        # Validate dimensions
        if M % patch_height != 0 or N % patch_width != 0:
            raise ValueError(f"Original shape ({M}, {N}) must be divisible by patch_size {self.patch_size}")

        # Calculate number of patches
        num_patches_height = M // patch_height
        num_patches_width = N // patch_width
        num_patches = num_patches_height * num_patches_width

        if actual_P != num_patches:
            raise ValueError(f"Number of patches ({actual_P}) doesn't match expected ({num_patches})")

        # Separate real and imaginary patches
        real_patches = patches[:, :actual_P, :]
        imag_patches = patches[:, actual_P:, :]

        # Reshape patches back to original dimensions
        real_patches = real_patches.reshape(B, num_patches_height, num_patches_width, patch_height, patch_width)
        real_patches = real_patches.permute(0, 1, 3, 2, 4).reshape(B, M, N)

        imag_patches = imag_patches.reshape(B, num_patches_height, num_patches_width, patch_height, patch_width)
        imag_patches = imag_patches.permute(0, 1, 3, 2, 4).reshape(B, M, N)

        # Reconstruct complex matrix
        reconstructed = torch.complex(real_patches, imag_patches)

        return reconstructed.squeeze(0) if squeeze_output else reconstructed 