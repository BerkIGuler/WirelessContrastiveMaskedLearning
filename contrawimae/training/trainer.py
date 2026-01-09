"""
Base trainer class for WiMAE and ContraWiMAE models.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from ..models.wimae import WiMAE
from ..models.contrawimae import ContraWiMAE
from .data_utils import (
    OptimizedPreloadedDataset,
    ScenarioSplitDataset,
    create_efficient_dataloader,
    calculate_complex_statistics
)


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
        
        # Initialize all components during trainer creation
        self.model = self.setup_model()
        self.optimizer = self.setup_optimizer(self.model)
        self.scheduler = self.setup_scheduler(self.optimizer)
        self.criterion = self.setup_criterion()
        self.writer = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """
        Setup logging and tensorboard writer.
        """
        # Create log directory
        log_dir = self.config["logging"]["log_dir"]
        model_type = self.config["model"]["type"]
        
        # Use exp_name if provided, otherwise use timestamp
        exp_name = self.config["logging"].get("exp_name", None)
        if exp_name:
            self.log_dir = Path(log_dir) / f"{model_type}_{exp_name}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = Path(log_dir) / f"{model_type}_{timestamp}"
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(f"{__name__}.{model_type}")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # File handler
        log_file = self.log_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Save configuration
        with open(self.log_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # Setup tensorboard writer if enabled
        if self.config["logging"].get("tensorboard", True):
            if SummaryWriter is not None:
                self.writer = SummaryWriter(str(self.log_dir))
            else:
                self.logger.warning("tensorboard not available. Install with: pip install tensorboard")
                self.writer = None
    
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
            Configured optimizer
        """
        opt_config = self.config["training"]["optimizer"]
        opt_type = opt_config["type"].lower()
        
        if opt_type == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=opt_config["lr"],
                weight_decay=opt_config.get("weight_decay", 0.0),
                betas=tuple(opt_config.get("betas", (0.9, 0.999)))
            )
        elif opt_type == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=opt_config["lr"],
                weight_decay=opt_config.get("weight_decay", 0.0),
                betas=tuple(opt_config.get("betas", (0.9, 0.999)))
            )
        elif opt_type == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=opt_config["lr"],
                momentum=opt_config.get("momentum", 0.9),
                weight_decay=opt_config.get("weight_decay", 0.0)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")
        
        return optimizer
    
    def setup_scheduler(self, optimizer: optim.Optimizer) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        Setup learning rate scheduler based on configuration.
        
        Args:
            optimizer: Optimizer to schedule
            
        Returns:
            Configured scheduler or None
        """
        if "scheduler" not in self.config["training"]:
            return None
            
        sched_config = self.config["training"]["scheduler"]
        sched_type = sched_config["type"].lower()
        
        if sched_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=sched_config["T_max"],
                eta_min=sched_config.get("eta_min", 0.000003)
            )
        elif sched_type == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_config["step_size"],
                gamma=sched_config.get("gamma", 0.5)
            )
        elif sched_type == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=sched_config.get("gamma", 0.995)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")
        
        return scheduler
    
    def setup_criterion(self) -> nn.Module:
        """Setup loss function."""
        loss_type = self.config["training"].get("loss", "mse").lower()
        
        if loss_type == "mse":
            return nn.MSELoss()
        elif loss_type == "l1":
            return nn.L1Loss()
        elif loss_type == "huber":
            return nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def setup_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Setup train and validation data loaders.
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        data_config = self.config["data"]
        training_config = self.config["training"]
        normalize = data_config.get("normalize", True)
        calculate_stats = data_config.get("calculate_statistics", True)
        debug_size = data_config.get("debug_size", None)
        
        # Step 1: Setup statistics
        statistics = self._setup_statistics(data_config, normalize, calculate_stats)
        
        # Step 2: Create datasets with proper file limiting for debug
        train_dataset, val_dataset = self._create_datasets(data_config, normalize, calculate_stats, statistics, debug_size)
        
        # Step 3: Calculate statistics if needed (using potentially limited dataset)
        if calculate_stats and statistics is None:
            statistics = self._calculate_statistics(train_dataset, training_config)
            
            # Step 4: Recreate datasets with calculated statistics (reuse debug limitation)
            train_dataset, val_dataset = self._create_datasets(data_config, normalize, False, statistics, debug_size)
        
        # Step 5: Create dataloaders
        train_loader, val_loader = self._create_dataloaders(train_dataset, val_dataset, training_config)
        
        return train_loader, val_loader
    
    def _setup_statistics(self, data_config: Dict, normalize: bool, calculate_stats: bool) -> Optional[Dict]:
        """Setup statistics configuration."""
        if normalize and not calculate_stats:
            statistics = data_config.get("statistics", {
                'real_mean': 0.021121172234416008,
                'real_std': 30.7452392578125,
                'imag_mean': -0.01027622725814581,
                'imag_std': 30.70543670654297
            })
            self.logger.info(f"Using pre-computed statistics: {statistics}")
            return statistics
        return None
    
    def _create_datasets(self, data_config: Dict, normalize: bool, calculate_stats: bool, 
                        statistics: Optional[Dict], debug_size: Optional[int]) -> Tuple[Dataset, Dataset]:
        """
        Create initial train and validation datasets.
        
        Args:
            data_config: Data configuration dictionary
            normalize: Whether to normalize the data
            calculate_stats: Whether to calculate statistics (if False, use provided statistics)
            statistics: Pre-computed statistics dict (used if normalize=True and calculate_stats=False)
            debug_size: Optional size limit for debugging (applied before train/val split)
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Determine statistics parameter for dataset creation
        stats_param = statistics if (normalize and not calculate_stats) else None
        
        if "scenario_split_config" in data_config:
            return self._create_scenario_split_datasets(data_config, stats_param, debug_size)
        else:
            return self._create_simple_split_datasets(data_config, stats_param, debug_size)
    
    def _create_scenario_split_datasets(self, data_config: Dict, statistics: Optional[Dict], 
                                       debug_size: Optional[int]) -> Tuple[Dataset, Dataset]:
        """Create datasets using scenario-based splitting."""
        scenario_config = data_config["scenario_split_config"]
        
        if not isinstance(scenario_config, dict):
            raise ValueError("scenario_split_config must be a dictionary with train_patterns, val_patterns, and test_patterns")
        
        train_dataset = ScenarioSplitDataset(
            data_dir=data_config["data_dir"],
            config_dict=scenario_config,
            split='train',
            statistics=statistics
        )
        
        val_dataset = ScenarioSplitDataset(
            data_dir=data_config["data_dir"],
            config_dict=scenario_config,
            split='val',
            statistics=statistics
        )
        
        # Apply debug size to train dataset if specified
        if debug_size is not None:
            train_dataset = self._apply_debug_size(train_dataset, debug_size)
        
        return train_dataset, val_dataset
    
    def _create_simple_split_datasets(self, data_config: Dict, statistics: Optional[Dict], 
                                     debug_size: Optional[int]) -> Tuple[Dataset, Dataset]:
        """Create datasets using simple file-based splitting."""
        data_dir = Path(data_config["data_dir"])
        npz_files = [str(data_dir / f) for f in os.listdir(data_dir) if f.endswith('.npz')]
        
        if not npz_files:
            raise ValueError(f"No NPZ files found in {data_config['data_dir']}")
        
        # Create full dataset
        dataset = OptimizedPreloadedDataset(
            npz_files=npz_files,
            statistics=statistics
        )
        
        # Apply debug size if specified (before train/val split)
        if debug_size is not None:
            dataset = self._apply_debug_size(dataset, debug_size)
        
        # Split into train and validation
        val_split = data_config.get("val_split", 0.2)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        return train_dataset, val_dataset
    
    def _apply_debug_size(self, dataset: Dataset, debug_size: int) -> Dataset:
        """
        Apply debug size limit to a dataset.
        
        Args:
            dataset: Dataset to limit
            debug_size: Maximum number of samples to keep
            
        Returns:
            Limited dataset (or original if debug_size >= dataset size)
        """
        if debug_size >= len(dataset):
            return dataset
        
        limited_dataset, _ = random_split(
            dataset,
            [debug_size, len(dataset) - debug_size],
            generator=torch.Generator().manual_seed(42)
        )
        return limited_dataset
    
    def _calculate_statistics(self, train_dataset: Dataset, training_config: Dict) -> Dict:
        """Calculate statistics from training dataset."""
        self.logger.info("Computing statistics from training dataset...")
        
        # Create temporary dataloader for statistics calculation
        temp_loader = create_efficient_dataloader(
            train_dataset,
            batch_size=min(training_config["batch_size"], 256),
            num_workers=min(training_config.get("num_workers", 4), 2),
            shuffle=False
        )
        
        # Calculate statistics
        statistics = calculate_complex_statistics(temp_loader)
        self.logger.info(f"Calculated statistics: {statistics}")
        return statistics
    
    def _create_dataloaders(self, train_dataset: Dataset, val_dataset: Dataset, 
                           training_config: Dict) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders."""
        train_loader = create_efficient_dataloader(
            train_dataset,
            batch_size=training_config["batch_size"],
            shuffle=True,
            num_workers=training_config["num_workers"],
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = create_efficient_dataloader(
            val_dataset,
            batch_size=training_config["batch_size"],
            shuffle=False,
            num_workers=training_config["num_workers"],
            pin_memory=True,
            drop_last=False
        )
        
        self.logger.info(f"Train samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
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
    
    def load_checkpoint(self, checkpoint_path: str, model_only: bool = False, strict: bool = True):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model_only: If True, only load model weights. If False, load full training state.
            strict: If False, allows partial loading with warnings for missing/unexpected keys.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model weights
        if strict:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            
            if missing_keys:
                self.logger.warning(f"Missing keys in checkpoint (will remain randomly initialized):")
                for key in missing_keys:
                    self.logger.warning(f"  - {key}")
            
            if unexpected_keys:
                self.logger.warning(f"Unexpected keys in checkpoint (will be ignored):")
                for key in unexpected_keys:
                    self.logger.warning(f"  - {key}")
            
            if not missing_keys and not unexpected_keys:
                self.logger.info("All model weights loaded successfully")
        
        if not model_only:
            # Load optimizer state if available
            if "optimizer_state_dict" in checkpoint and self.optimizer:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            # Load scheduler state if available
            if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] and self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            # Load training state if available
            if "epoch" in checkpoint:
                self.current_epoch = checkpoint["epoch"]
            if "global_step" in checkpoint:
                self.global_step = checkpoint["global_step"]
            if "best_val_loss" in checkpoint:
                self.best_val_loss = checkpoint["best_val_loss"]
            
            self.logger.info(f"Loaded full training state from epoch {self.current_epoch}")
        else:
            self.logger.info("Loaded model weights only (training state not restored)")
    
    def train(self):
        """
        Main training loop.
        """
        # Setup dataloaders
        train_loader, val_loader = self.setup_dataloaders()
        
        # Training loop
        epochs = self.config["training"]["epochs"]
        patience = self.config["training"]["patience"]
        
        self.logger.info(f"Starting training for {epochs} epochs...")
        self.logger.info(f"Model: {self.config['model']['type']}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Log directory: {self.log_dir}")
        
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
                self.scheduler.step()
            
            # Checkpointing and early stopping
            min_delta = self.config["training"]["min_delta"]
            is_best = val_metrics["val_loss"] < (self.best_val_loss - min_delta)
            
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            save_every_n = self.config["training"]["save_checkpoint_every_n"]
            if epoch % save_every_n == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= patience:
                self.logger.info(f"Early stopping after {patience} epochs without improvement")
                break
        
        self.logger.info("Training completed!")
    
    def log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float], epoch: int):
        """
        Log training and validation metrics.
        
        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
            epoch: Current epoch
        """
        # Log metrics
        self.logger.info(f"Epoch {epoch}:")
        for key, value in train_metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
        for key, value in val_metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
        
        # Log to tensorboard
        if self.writer:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"val/{key}", value, epoch) 