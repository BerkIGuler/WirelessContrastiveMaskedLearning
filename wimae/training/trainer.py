"""
Base trainer class for WiMAE and ContraWiMAE models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm
import numpy as np
from datetime import datetime

from ..models import WiMAE, ContraWiMAE


class BaseTrainer:
    """
    Base trainer class providing common training functionality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config["training"]["device"])
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.writer = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging and tensorboard."""
        log_dir = self.config["logging"]["log_dir"]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_name = f"{self.config['model']['type']}_run_{timestamp}"
        
        self.log_dir = Path(log_dir) / self.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(self.log_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # Setup tensorboard
        if self.config["logging"]["tensorboard"]:
            self.writer = SummaryWriter(str(self.log_dir))
    
    def setup_model(self) -> nn.Module:
        """
        Setup the model based on configuration.
        
        Returns:
            Initialized model
        """
        model_config = self.config["model"]
        model_type = model_config["type"]
        
        if model_type == "wimae":
            model = WiMAE(
                patch_size=tuple(model_config["patch_size"]),
                encoder_dim=model_config["encoder_dim"],
                encoder_layers=model_config["encoder_layers"],
                encoder_nhead=model_config["encoder_nhead"],
                decoder_layers=model_config["decoder_layers"],
                decoder_nhead=model_config["decoder_nhead"],
                mask_ratio=model_config["mask_ratio"],
                device=self.device,
            )
        elif model_type == "contramae":
            model = ContraWiMAE(
                patch_size=tuple(model_config["patch_size"]),
                encoder_dim=model_config["encoder_dim"],
                encoder_layers=model_config["encoder_layers"],
                encoder_nhead=model_config["encoder_nhead"],
                decoder_layers=model_config["decoder_layers"],
                decoder_nhead=model_config["decoder_nhead"],
                mask_ratio=model_config["mask_ratio"],
                contrastive_dim=model_config["contrastive_dim"],
                temperature=model_config["temperature"],
                snr_min=model_config["snr_min"],
                snr_max=model_config["snr_max"],
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    def setup_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """
        Setup optimizer based on configuration.
        
        Args:
            model: Model to optimize
            
        Returns:
            Initialized optimizer
        """
        training_config = self.config["training"]
        optimizer_name = training_config["optimizer"]
        lr = training_config["learning_rate"]
        weight_decay = training_config["weight_decay"]
        
        if optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return optimizer
    
    def setup_scheduler(self, optimizer: optim.Optimizer) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        Setup learning rate scheduler based on configuration.
        
        Args:
            optimizer: Optimizer to schedule
            
        Returns:
            Initialized scheduler or None
        """
        training_config = self.config["training"]
        scheduler_name = training_config.get("scheduler")
        
        if scheduler_name is None:
            return None
        
        epochs = training_config["epochs"]
        warmup_epochs = training_config.get("warmup_epochs", 0)
        
        if scheduler_name == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_name == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler_name == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif scheduler_name == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=10, factor=0.5
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        return scheduler
    
    def setup_criterion(self) -> nn.Module:
        """
        Setup loss function.
        
        Returns:
            Loss function
        """
        return nn.MSELoss()
    
    def setup_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Setup training and validation dataloaders.
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # This should be implemented by subclasses
        raise NotImplementedError
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        # This should be implemented by subclasses
        raise NotImplementedError
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        # This should be implemented by subclasses
        raise NotImplementedError
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        
        # Save last checkpoint
        checkpoint_path = self.log_dir / "last_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.log_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
    
    def train(self):
        """
        Main training loop.
        """
        # Setup components
        self.model = self.setup_model()
        self.optimizer = self.setup_optimizer(self.model)
        self.scheduler = self.setup_scheduler(self.optimizer)
        self.criterion = self.setup_criterion()
        
        # Setup dataloaders
        train_loader, val_loader = self.setup_dataloaders()
        
        # Training loop
        epochs = self.config["training"]["epochs"]
        patience = self.config["training"]["early_stopping_patience"]
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Model: {self.config['model']['type']}")
        print(f"Device: {self.device}")
        print(f"Log directory: {self.log_dir}")
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Logging
            self.log_metrics(train_metrics, val_metrics, epoch)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()
            
            # Checkpointing
            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if epoch % self.config["logging"]["save_interval"] == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
        
        print("Training completed!")
    
    def log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], epoch: int):
        """
        Log training and validation metrics.
        
        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
            epoch: Current epoch
        """
        # Print metrics
        print(f"Epoch {epoch}:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value:.4f}")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Log to tensorboard
        if self.writer:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"val/{key}", value, epoch)
    
    @classmethod
    def from_config(cls, config_path: str) -> "BaseTrainer":
        """
        Create trainer from configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Initialized trainer
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        return cls(config) 