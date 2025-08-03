"""
Contrastive loss functions for ContraWiMAE.
"""

import torch
import torch.nn.functional as F


def compute_contrastive_loss(z_i, z_j, temperature=0.1):
    """
    Compute InfoNCE/NT-Xent contrastive loss between two sets of embeddings.

    Args:
        z_i (torch.Tensor): Embeddings of original samples, shape [B, D]
        z_j (torch.Tensor): Embeddings of augmented samples, shape [B, D]
        temperature (float): Temperature scaling parameter

    Returns:
        torch.Tensor: Scalar contrastive loss
    """
    # Get batch size
    batch_size = z_i.shape[0]
    
    # Validate batch size - need at least 2 samples for contrastive learning
    if batch_size < 2:
        raise ValueError(f"Batch size must be at least 2 for contrastive learning, got {batch_size}")

    # Combine features from both views
    features = torch.cat([z_i, z_j], dim=0)  # [2B, D]

    # Compute full similarity matrix
    sim_matrix = torch.mm(features, features.t())  # [2B, 2B]

    # Create labels identifying positive pairs
    # For i-th sample in z_i, the positive is the i-th sample in z_j (at index batch_size + i)
    # For i-th sample in z_j, the positive is the i-th sample in z_i (at index i)
    labels = torch.zeros(2 * batch_size, device=z_i.device, dtype=torch.long)
    for i in range(batch_size):
        labels[i] = batch_size + i
        labels[i + batch_size] = i

    # Mask out self-similarity (along the diagonal)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
    sim_matrix = sim_matrix.masked_fill_(mask, -float('inf'))

    # Apply temperature scaling
    sim_matrix = sim_matrix / temperature

    # Compute InfoNCE loss
    contrastive_loss = F.cross_entropy(sim_matrix, labels)

    return contrastive_loss 