"""
WiMAE trainer implementation.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm

from .trainer import BaseTrainer


class WiMAETrainer(BaseTrainer):
    """
    Trainer for WiMAE model.
    """
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {self.current_epoch}")
        
        for batch_idx, data in enumerate(progress_bar):
            # Move data to device
            data = data.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get original patches for reconstruction target
            patches = self.model.patcher(data)
            
            # Forward pass with masking
            output = self.model(data, mask_ratio=self.model.mask_ratio)
            reconstructed_patches = output["reconstructed_patches"]
            ids_mask = output["ids_mask"]  # Indices of masked patches
            
            batch_size = patches.shape[0]

            # Compute reconstruction loss only on masked patches            
            if ids_mask.shape[1] > 0:
                # Create batch indices tensor [B, M] where M is number of masked patches
                batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(-1).expand(-1, ids_mask.shape[1])
                
                # Select masked positions from both prediction and target
                recon_masked = reconstructed_patches[batch_indices, ids_mask]
                target_masked = patches[batch_indices, ids_mask]
                
                # Compute loss only on masked positions
                loss = self.criterion(recon_masked, target_masked)
            else:
                # No masked patches when mask_ratio > 0 indicates a bug in masking logic
                if self.model.mask_ratio > 0:
                    raise RuntimeError(
                        f"No patches were masked despite mask_ratio={self.model.mask_ratio}. "
                        "This indicates a bug in the masking logic."
                    )
                # If mask_ratio is 0, no masking is expected
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            gradient_clip_val = self.config["training"].get("gradient_clip_val", 0.0)
            if gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    gradient_clip_val
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}"
            })
            
            # Log to tensorboard
            log_interval = self.config["logging"].get("log_every_n_steps", 100)
            if self.writer and batch_idx % log_interval == 0:
                self.writer.add_scalar("train/batch_loss", loss.item(), self.global_step)
        
        return {
            "train_loss": total_loss / num_batches
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
        
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validation"):
                # Move data to device
                data = data.to(self.device)
                
                # Get original patches for reconstruction target
                patches = self.model.patcher(data)
                
                # Forward pass with masking (same as training)
                output = self.model(data, mask_ratio=self.model.mask_ratio)
                reconstructed_patches = output["reconstructed_patches"]
                ids_mask = output["ids_mask"]
                
                # Compute reconstruction loss only on masked patches
                if ids_mask.shape[1] > 0:
                    batch_size = patches.shape[0]
                    batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(-1).expand(-1, ids_mask.shape[1])
                    recon_masked = reconstructed_patches[batch_indices, ids_mask]
                    target_masked = patches[batch_indices, ids_mask]
                    loss = self.criterion(recon_masked, target_masked)
                else:
                    # No masked patches when mask_ratio > 0 indicates a bug in masking logic
                    if self.model.mask_ratio > 0:
                        raise RuntimeError(
                            f"No patches were masked despite mask_ratio={self.model.mask_ratio}. "
                            "This indicates a bug in the masking logic."
                        )
                    # If mask_ratio is 0, no masking is expected
                    loss = torch.tensor(0.0, device=self.device)
                
                total_loss += loss.item()
        
        return {
            "val_loss": total_loss / num_batches
        } 