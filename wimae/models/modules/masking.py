"""
Masking module for MAE models.
"""

import torch


class MaskGenerator:
    def __init__(self, device, mask_ratio=0.75, random_seed=42):
        self.mask_ratio = mask_ratio
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(random_seed if random_seed is not None else torch.seed())

    def __call__(self, patches):
        """
        Generate random mask ensuring real/imaginary parts are masked together.

        Args:
            patches (torch.Tensor): Tensor of shape [B, P, L] where:
                B is batch size
                P is number of patches (includes both real and imag)
                L is patch dimension

        Returns:
            unmasked_patches: Tensor containing only the kept patches
            ids_keep: Indices of kept patches
            ids_mask: Indices of masked patches
        """
        B, P, L = patches.shape
        actual_P = P // 2

        # Generate random noise for ACTUAL patches
        noise = torch.rand(B, actual_P, device=patches.device, generator=self.generator)

        # Sort indices for actual patches
        ids_shuffle_half = torch.argsort(noise, dim=1)

        # Keep and mask tokens for actual patches
        num_keep = int(actual_P * (1 - self.mask_ratio))
        ids_keep_half = ids_shuffle_half[:, :num_keep]
        ids_mask_half = ids_shuffle_half[:, num_keep:]  # take the remaining indices as masked patches

        # Double the indices to handle real and imaginary parts
        ids_keep = torch.cat([
            ids_keep_half,  # For real parts
            ids_keep_half + actual_P  # For imaginary parts
        ], dim=1)

        ids_mask = torch.cat([
            ids_mask_half,  # For real parts
            ids_mask_half + actual_P  # For imaginary parts
        ], dim=1)

        # Gather patches using ids_keep
        unmasked_patches = torch.gather(
            patches,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, L)
        )

        return unmasked_patches, ids_keep, ids_mask 