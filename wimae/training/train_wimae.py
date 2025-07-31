"""
WiMAE trainer implementation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
from typing import Dict, Any, Tuple
from tqdm import tqdm

from .trainer import BaseTrainer


class WirelessChannelDataset(Dataset):
    """
    Dataset for wireless channel data.
    """
    
    def __init__(self, data_path: str, data_format: str = "npz"):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to data file
            data_format: Data format ("npz", "npy", etc.)
        """
        self.data_path = data_path
        self.data_format = data_format
        
        # Load data
        if data_format == "npz":
            data = np.load(data_path)
            # Assuming the data is stored as 'channels' key
            self.data = data['channels'] if 'channels' in data else data['data']
        else:
            self.data = np.load(data_path)
        
        # Ensure data is in the right format (batch, channels, height, width)
        if len(self.data.shape) == 3:
            # Add channel dimension if missing
            self.data = self.data[:, np.newaxis, :, :]
        elif len(self.data.shape) == 4:
            # Data is already in the right format
            pass
        else:
            raise ValueError(f"Unexpected data shape: {self.data.shape}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Convert to tensor and ensure float32
        data = torch.from_numpy(self.data[idx]).float()
        return data


class WiMAETrainer(BaseTrainer):
    """
    Trainer for WiMAE model.
    """
    
    def setup_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Setup training and validation dataloaders.
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        data_config = self.config["data"]
        training_config = self.config["training"]
        
        # Create dataset
        dataset = WirelessChannelDataset(
            data_path=data_config["data_path"],
            data_format=data_config["data_format"]
        )
        
        # Split dataset
        val_split = training_config["val_split"]
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config["batch_size"],
            shuffle=True,
            num_workers=data_config["num_workers"],
            pin_memory=data_config["pin_memory"],
            persistent_workers=data_config["persistent_workers"]
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config["batch_size"],
            shuffle=False,
            num_workers=data_config["num_workers"],
            pin_memory=data_config["pin_memory"],
            persistent_workers=data_config["persistent_workers"]
        )
        
        return train_loader, val_loader
    
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
            if self.config["training"]["gradient_clip_val"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config["training"]["gradient_clip_val"]
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
            if self.writer and batch_idx % self.config["logging"]["log_interval"] == 0:
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