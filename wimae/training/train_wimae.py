"""
WiMAE trainer implementation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
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
            
            # Compute reconstruction loss
            loss = self.criterion(reconstructed_patches, patches)
            
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
                
                # Forward pass without masking
                output = self.model(data, mask_ratio=0.0)
                reconstructed_patches = output["reconstructed_patches"]
                
                # Compute reconstruction loss
                loss = self.criterion(reconstructed_patches, patches)
                total_loss += loss.item()
        
        val_loss = total_loss / num_batches
        
        return {
            "val_loss": val_loss
        }
    
    def setup_criterion(self) -> nn.Module:
        """
        Setup loss function for WiMAE.
        
        Returns:
            Loss function
        """
        return nn.MSELoss() 