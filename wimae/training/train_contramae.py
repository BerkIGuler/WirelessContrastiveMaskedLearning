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
        total_loss = 0.0
        num_batches = len(train_loader)
        
        # Get loss weights
        recon_weight = self.config["training"]["reconstruction_weight"]
        contrastive_weight = self.config["training"]["contrastive_weight"]
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {self.current_epoch}")
        
        for batch_idx, data in enumerate(progress_bar):
            # Move data to device
            data = data.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get original patches for reconstruction target
            patches = self.model.patcher(data)
            
            # Forward pass with augmentation for contrastive learning
            output = self.model.forward_with_augmentation(
                data, 
                mask_ratio=self.model.mask_ratio
            )
            
            # Extract outputs
            orig_output = output["original"]
            aug_output = output["augmented"]
            
            # Reconstruction loss (average of original and augmented)
            orig_recon_loss = self.criterion(
                orig_output["reconstructed_patches"], 
                patches
            )
            aug_recon_loss = self.criterion(
                aug_output["reconstructed_patches"], 
                patches
            )
            recon_loss = (orig_recon_loss + aug_recon_loss) / 2
            
            # Contrastive loss
            orig_features = orig_output["contrastive_features"]
            aug_features = aug_output["contrastive_features"]
            
            # Mean pooling for contrastive features
            orig_features = torch.mean(orig_features, dim=1)  # (batch_size, contrastive_dim)
            aug_features = torch.mean(aug_features, dim=1)    # (batch_size, contrastive_dim)
            
            contrastive_loss = self.model.compute_contrastive_loss(
                orig_features, 
                aug_features,
                temperature=self.model.temperature
            )
            
            # Combined loss
            loss = recon_weight * recon_loss + contrastive_weight * contrastive_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config["training"]["gradient_clip_val"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config["training"]["gradient_clip_val"]
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_recon_loss += recon_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "recon_loss": f"{recon_loss.item():.4f}",
                "contrastive_loss": f"{contrastive_loss.item():.4f}",
                "total_loss": f"{loss.item():.4f}",
                "avg_total_loss": f"{total_loss / (batch_idx + 1):.4f}"
            })
            
            # Log to tensorboard
            if self.writer and batch_idx % self.config["logging"]["log_interval"] == 0:
                self.writer.add_scalar("train/batch_recon_loss", recon_loss.item(), self.global_step)
                self.writer.add_scalar("train/batch_contrastive_loss", contrastive_loss.item(), self.global_step)
                self.writer.add_scalar("train/batch_total_loss", loss.item(), self.global_step)
        
        return {
            "train_recon_loss": total_recon_loss / num_batches,
            "train_contrastive_loss": total_contrastive_loss / num_batches,
            "train_total_loss": total_loss / num_batches
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
        
        total_recon_loss = 0.0
        total_contrastive_loss = 0.0
        total_loss = 0.0
        num_batches = len(val_loader)
        
        # Get loss weights
        recon_weight = self.config["training"]["reconstruction_weight"]
        contrastive_weight = self.config["training"]["contrastive_weight"]
        
        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validation"):
                # Move data to device
                data = data.to(self.device)
                
                # Get original patches for reconstruction target
                patches = self.model.patcher(data)
                
                # Forward pass with augmentation
                output = self.model.forward_with_augmentation(data, mask_ratio=0.0)
                
                # Extract outputs
                orig_output = output["original"]
                aug_output = output["augmented"]
                
                # Reconstruction loss
                orig_recon_loss = self.criterion(
                    orig_output["reconstructed_patches"], 
                    patches
                )
                aug_recon_loss = self.criterion(
                    aug_output["reconstructed_patches"], 
                    patches
                )
                recon_loss = (orig_recon_loss + aug_recon_loss) / 2
                
                # Contrastive loss
                orig_features = orig_output["contrastive_features"]
                aug_features = aug_output["contrastive_features"]
                
                # Mean pooling for contrastive features
                orig_features = torch.mean(orig_features, dim=1)
                aug_features = torch.mean(aug_features, dim=1)
                
                contrastive_loss = self.model.compute_contrastive_loss(
                    orig_features, 
                    aug_features,
                    temperature=self.model.temperature
                )
                
                # Combined loss
                loss = recon_weight * recon_loss + contrastive_weight * contrastive_loss
                
                # Update metrics
                total_recon_loss += recon_loss.item()
                total_contrastive_loss += contrastive_loss.item()
                total_loss += loss.item()
        
        return {
            "val_recon_loss": total_recon_loss / num_batches,
            "val_contrastive_loss": total_contrastive_loss / num_batches,
            "val_total_loss": total_loss / num_batches,
            "val_loss": total_loss / num_batches  # For compatibility with base trainer
        } 