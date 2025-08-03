"""
ContraWiMAE trainer implementation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
from tqdm import tqdm

from .train_wimae import WiMAETrainer


class ContraWiMAETrainer(WiMAETrainer):
    """
    Trainer for ContraWiMAE model.
    """
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch with contrastive learning.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_recon_loss = 0.0
        total_contrastive_loss = 0.0
        total_loss_val = 0.0
        num_batches = len(train_loader)
        
        # Get loss weights
        recon_weight = self.config["training"].get("reconstruction_weight", 0.9)
        contrastive_weight = self.config["training"].get("contrastive_weight", 0.1)
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {self.current_epoch}")
        
        for batch_idx, data in enumerate(progress_bar):
            # Move data to device
            data = data.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Compute training losses using model's method
            losses = self.model.compute_training_losses(data, self.criterion)
            
            # Apply weights to losses
            total_loss = recon_weight * losses["reconstruction_loss"] + contrastive_weight * losses["contrastive_loss"]
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            gradient_clip_val = self.config["training"].get("gradient_clip_val", 0.0)
            if gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    gradient_clip_val
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_recon_loss += losses["reconstruction_loss"].item()
            total_contrastive_loss += losses["contrastive_loss"].item()
            total_loss_val += total_loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "recon_loss": f"{losses['reconstruction_loss'].item():.4f}",
                "contrastive_loss": f"{losses['contrastive_loss'].item():.4f}",
                "total_loss": f"{total_loss.item():.4f}",
                "avg_total_loss": f"{total_loss_val / (batch_idx + 1):.4f}"
            })
            
            # Log to tensorboard
            log_interval = self.config["logging"].get("log_every_n_steps", 100)
            if self.writer and batch_idx % log_interval == 0:
                self.writer.add_scalar("train/batch_recon_loss", losses["reconstruction_loss"].item(), self.global_step)
                self.writer.add_scalar("train/batch_contrastive_loss", losses["contrastive_loss"].item(), self.global_step)
                self.writer.add_scalar("train/batch_total_loss", total_loss.item(), self.global_step)
        
        return {
            "train_recon_loss": total_recon_loss / num_batches,
            "train_contrastive_loss": total_contrastive_loss / num_batches,
            "train_total_loss": total_loss_val / num_batches
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_masked_recon_loss = 0.0
        total_full_recon_loss = 0.0
        total_contrastive_loss = 0.0
        total_masked_loss = 0.0
        total_full_loss = 0.0
        num_batches = len(val_loader)
        
        # Get loss weights
        recon_weight = self.config["training"].get("reconstruction_weight", 0.9)
        contrastive_weight = self.config["training"].get("contrastive_weight", 0.1)
        
        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validation"):
                # Move data to device
                data = data.to(self.device)
                
                # Compute validation losses using model's method
                losses = self.model.compute_validation_losses(data, self.criterion)
                
                # Apply weights to losses
                masked_loss = recon_weight * losses["masked_recon_loss"] + contrastive_weight * losses["contrastive_loss"]
                full_loss = recon_weight * losses["full_recon_loss"] + contrastive_weight * losses["contrastive_loss"]
                
                # Update metrics
                total_masked_recon_loss += losses["masked_recon_loss"].item()
                total_full_recon_loss += losses["full_recon_loss"].item()
                total_contrastive_loss += losses["contrastive_loss"].item()
                total_masked_loss += masked_loss.item()
                total_full_loss += full_loss.item()
        
        return {
            "val_masked_recon_loss": total_masked_recon_loss / num_batches,  # Primary reconstruction metric
            "val_full_recon_loss": total_full_recon_loss / num_batches,      # Secondary reconstruction metric
            "val_contrastive_loss": total_contrastive_loss / num_batches,    # Contrastive metric
            "val_masked_loss": total_masked_loss / num_batches,              # Primary total loss
            "val_full_loss": total_full_loss / num_batches,                  # Secondary total loss
            "val_loss": total_masked_loss / num_batches                      # For compatibility with base trainer
        } 