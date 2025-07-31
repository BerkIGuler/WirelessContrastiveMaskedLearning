"""
Data augmentation functions for wireless channel data.

This module provides channel-specific augmentations for complex channel matrices,
including SNR-based noise injection, frequency shift, and phase rotation.
"""

import torch
import math


def apply_channel_augmentations(x, noise_prob=1, freq_shift_prob=0.0, phase_rot_prob=0.0, **kwargs):
    """
    Apply channel-specific augmentations to complex channel matrices.

    Args:
        x: Complex tensor of shape (B, M, N) where:
           B: batch size
           M, N: channel matrix dimensions
        noise_prob: Probability of applying noise injection
        freq_shift_prob: Probability of applying frequency shift
        phase_rot_prob: Probability of applying phase rotation
        **kwargs: Additional arguments including snr_min and snr_max for noise injection

    Returns:
        Augmented complex tensor of same shape as input
    """
    # Create a copy to avoid modifying the original
    x_aug = x.clone()
    batch_size = x.shape[0]

    # Apply noise injection (H_aug = H + σN) for unit variance channels - vectorized
    if noise_prob > 0:
        # Check if required SNR arguments are provided
        if "snr_min" not in kwargs:
            raise ValueError("snr_min must be provided in kwargs when noise_prob > 0")
        if "snr_max" not in kwargs:
            raise ValueError("snr_max must be provided in kwargs when noise_prob > 0")

        # Assign SNR values from kwargs
        snr_db_min = kwargs["snr_min"]
        snr_db_max = kwargs["snr_max"]

        # Create batch-level noise mask
        noise_mask = (torch.rand(batch_size) < noise_prob).to(x.device)

        if noise_mask.any():
            # Generate random SNR values for the entire batch
            target_snr_db = snr_db_min + torch.rand(batch_size, 1, 1).to(x.device) * (snr_db_max - snr_db_min)

            # Convert SNR from dB to linear scale
            target_snr_linear = 10 ** (target_snr_db / 10)

            # Calculate required noise power directly (for unit variance channels)
            noise_power = 1.0 / target_snr_linear

            # Generate complex Gaussian noise with correct power for the entire batch
            noise_std = torch.sqrt(noise_power / 2)  # Divide by 2 for complex noise
            noise_real = torch.randn(batch_size, x.shape[1], x.shape[2]).to(x.device) * noise_std
            noise_imag = torch.randn(batch_size, x.shape[1], x.shape[2]).to(x.device) * noise_std
            noise = torch.complex(noise_real, noise_imag)

            # Apply noise only to masked samples using broadcasting
            # Create a mask that's properly shaped for broadcasting
            mask_expanded = noise_mask.view(batch_size, 1, 1)

            # Apply the noise selectively using the mask
            x_aug = x + noise * mask_expanded

    # Apply frequency shift (H_aug[:, i] = H[:, (i + δ) % N_f])
    # Assuming the N dimension represents frequency
    if freq_shift_prob > 0:
        shift_mask = (torch.rand(batch_size) < freq_shift_prob).to(x.device)
        if shift_mask.any():
            delta = torch.randint(1, max(1, x.shape[2] // 10), (batch_size,)).to(x.device)
            for i in range(batch_size):
                if shift_mask[i]:
                    # Roll along the frequency dimension (N)
                    x_aug[i] = torch.roll(x_aug[i], shifts=delta[i].item(), dims=1)

    # Apply phase rotation (H_aug = H · e^jθ)
    if phase_rot_prob > 0:
        phase_mask = (torch.rand(batch_size) < phase_rot_prob).to(x.device)
        if phase_mask.any():
            theta = 2 * math.pi * torch.rand(batch_size, 1, 1).to(x.device)
            # Create complex rotation factor e^(jθ) = cos(θ) + j·sin(θ)
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            rotation_factor = torch.complex(cos_theta, sin_theta)

            # Apply rotation only to selected samples in batch
            for i in range(batch_size):
                if phase_mask[i]:
                    x_aug[i] = x_aug[i] * rotation_factor[i]

    return x_aug


def apply_snr_noise(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Add noise to achieve a specific SNR.
    
    Args:
        x: Input tensor
        snr_db: SNR in dB
        
    Returns:
        Noisy tensor
    """
    # Calculate signal power
    signal_power = torch.mean(x ** 2)
    
    # Calculate noise power for desired SNR
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std = torch.sqrt(noise_power)
    
    # Add noise
    noise = torch.randn_like(x) * noise_std
    return x + noise


def apply_phase_noise(x: torch.Tensor, phase_std: float) -> torch.Tensor:
    """
    Add random phase noise to the signal.
    
    Args:
        x: Input tensor
        phase_std: Standard deviation of phase noise in radians
        
    Returns:
        Phase-noisy tensor
    """
    # Generate random phase noise
    phase_noise = torch.randn_like(x) * phase_std
    
    # Apply phase shift
    x_complex = torch.complex(x, torch.zeros_like(x))
    x_complex = x_complex * torch.exp(1j * phase_noise)
    return x_complex.real


def apply_amplitude_noise(x: torch.Tensor, amplitude_std: float) -> torch.Tensor:
    """
    Add random amplitude noise to the signal.
    
    Args:
        x: Input tensor
        amplitude_std: Standard deviation of amplitude noise
        
    Returns:
        Amplitude-noisy tensor
    """
    # Generate random amplitude noise
    amplitude_noise = torch.randn_like(x) * amplitude_std + 1.0
    
    # Apply amplitude scaling
    return x * amplitude_noise 