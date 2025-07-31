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


def compute_beam_contrastive_loss(z_i, z_j, beam_labels, temperature=0.1):
    """
    Compute supervised contrastive loss based on beam prediction indices.
    
    Args:
        z_i (torch.Tensor): Embeddings of original samples, shape [B, D]
        z_j (torch.Tensor): Embeddings of augmented samples, shape [B, D]
        beam_labels (torch.Tensor): Beam prediction labels, shape [B]
        temperature (float): Temperature scaling parameter
    
    Returns:
        torch.Tensor: Scalar beam-based contrastive loss
    """
    batch_size = z_i.shape[0]
    features = torch.cat([z_i, z_j], dim=0)
    beam_labels_expanded = torch.cat([beam_labels, beam_labels], dim=0)
    sim_matrix = torch.mm(features, features.t())
    positive_mask = (beam_labels_expanded.unsqueeze(1) == beam_labels_expanded.unsqueeze(0))
    diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
    positive_mask = positive_mask & ~diag_mask
    sim_matrix = sim_matrix / temperature
    loss = 0.0
    valid_samples = 0
    for i in range(2 * batch_size):
        positives = positive_mask[i]
        if positives.sum() > 0:
            pos_sim = sim_matrix[i][positives]
            neg_sim = sim_matrix[i][~positives & ~diag_mask[i]]
            if len(neg_sim) > 0:
                log_pos = torch.logsumexp(pos_sim, dim=0)
                all_sim = torch.cat([pos_sim, neg_sim])
                log_all = torch.logsumexp(all_sim, dim=0)
                sample_loss = log_all - log_pos
                loss += sample_loss
                valid_samples += 1
    if valid_samples > 0:
        return loss / valid_samples
    else:
        return torch.tensor(0.0, device=features.device, requires_grad=True) 