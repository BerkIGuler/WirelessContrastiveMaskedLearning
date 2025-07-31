"""
Encoding interface for WiMAE and ContraWiMAE models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from tqdm import tqdm
import json
from datetime import datetime

from ..models import WiMAE, ContraWiMAE
from ..training.train_wimae import WirelessChannelDataset


class Encoder:
    """
    Encoding interface for generating embeddings from trained models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the encoder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config["model"]["device"])
        
        # Load model
        self.model = self._load_model()
        
        # Setup output directory
        self.output_dir = Path(config["encoding"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_model(self) -> Union[WiMAE, ContraWiMAE]:
        """
        Load model from checkpoint.
        
        Returns:
            Loaded model
        """
        checkpoint_path = self.config["model"]["checkpoint_path"]
        
        # Load checkpoint to determine model type
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Check if it's a ContraWiMAE checkpoint
        if "contrastive_dim" in checkpoint:
            model = ContraWiMAE.from_checkpoint(checkpoint_path, device=self.device)
        else:
            model = WiMAE.from_checkpoint(checkpoint_path, device=self.device)
        
        model.eval()
        return model
    
    def encode_data(self, data_path: str, data_format: str = "npz") -> Dict[str, np.ndarray]:
        """
        Encode data and return embeddings.
        
        Args:
            data_path: Path to data file
            data_format: Data format
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        # Create dataset
        dataset = WirelessChannelDataset(data_path, data_format)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config["encoding"]["batch_size"],
            shuffle=False,
            num_workers=self.config["encoding"]["num_workers"],
            pin_memory=self.config["encoding"]["pin_memory"]
        )
        
        # Encode
        embeddings = self._encode_dataloader(dataloader)
        
        return embeddings
    
    def _encode_dataloader(self, dataloader: DataLoader) -> Dict[str, np.ndarray]:
        """
        Encode data from dataloader.
        
        Args:
            dataloader: Data loader
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        all_embeddings = []
        all_reconstructions = []
        all_contrastive_features = []
        
        pooling = self.config["encoding"]["pooling"]
        save_reconstructions = self.config["encoding"]["save_reconstructions"]
        save_contrastive_features = self.config["encoding"]["save_contrastive_features"]
        
        print(f"Encoding {len(dataloader)} batches...")
        
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(dataloader, desc="Encoding")):
                # Move data to device
                data = data.to(self.device)
                
                # Get embeddings
                if isinstance(self.model, ContraWiMAE):
                    embeddings = self.model.get_contrastive_embeddings(data, pooling=pooling)
                else:
                    embeddings = self.model.get_embeddings(data, pooling=pooling)
                
                all_embeddings.append(embeddings.cpu().numpy())
                
                # Save reconstructions if requested
                if save_reconstructions:
                    reconstructions = self.model.reconstruct(data)
                    all_reconstructions.append(reconstructions.cpu().numpy())
                
                # Save contrastive features if requested and available
                if save_contrastive_features and isinstance(self.model, ContraWiMAE):
                    contrastive_features = self.model.get_contrastive_embeddings(data, pooling=pooling)
                    all_contrastive_features.append(contrastive_features.cpu().numpy())
        
        # Concatenate results
        result = {
            "embeddings": np.concatenate(all_embeddings, axis=0)
        }
        
        if save_reconstructions:
            result["reconstructions"] = np.concatenate(all_reconstructions, axis=0)
        
        if save_contrastive_features and isinstance(self.model, ContraWiMAE):
            result["contrastive_features"] = np.concatenate(all_contrastive_features, axis=0)
        
        return result
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], filename: Optional[str] = None):
        """
        Save embeddings to file.
        
        Args:
            embeddings: Dictionary containing embeddings
            filename: Optional filename (will generate if None)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_type = "contramae" if isinstance(self.model, ContraWiMAE) else "wimae"
            filename = f"{model_type}_embeddings_{timestamp}.pt"
        
        filepath = self.output_dir / filename
        
        # Save embeddings
        torch.save(embeddings, filepath)
        
        # Save metadata if requested
        if self.config["output"]["save_metadata"]:
            metadata = self._generate_metadata(embeddings)
            metadata_path = filepath.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Embeddings saved to: {filepath}")
        return filepath
    
    def _generate_metadata(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Generate metadata for embeddings.
        
        Args:
            embeddings: Embeddings dictionary
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            "model_info": self.model.get_model_info(),
            "encoding_params": {
                "pooling": self.config["encoding"]["pooling"],
                "batch_size": self.config["encoding"]["batch_size"],
                "output_format": self.config["encoding"]["output_format"]
            },
            "data_info": {
                "num_samples": embeddings["embeddings"].shape[0],
                "embedding_dim": embeddings["embeddings"].shape[1],
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return metadata
    
    def encode_and_save(self, data_path: str, data_format: str = "npz", filename: Optional[str] = None):
        """
        Encode data and save embeddings.
        
        Args:
            data_path: Path to data file
            data_format: Data format
            filename: Optional filename
            
        Returns:
            Path to saved embeddings file
        """
        # Encode data
        embeddings = self.encode_data(data_path, data_format)
        
        # Save embeddings
        filepath = self.save_embeddings(embeddings, filename)
        
        return filepath
    
    @classmethod
    def from_config(cls, config_path: str) -> "Encoder":
        """
        Create encoder from configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Initialized encoder
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        return cls(config)
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cuda") -> "Encoder":
        """
        Create encoder from model checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use
            
        Returns:
            Initialized encoder
        """
        # Create minimal config
        config = {
            "model": {
                "checkpoint_path": checkpoint_path,
                "device": device
            },
            "encoding": {
                "batch_size": 256,
                "num_workers": 4,
                "pin_memory": True,
                "output_dir": "embeddings",
                "pooling": "mean",
                "save_embeddings": True,
                "save_reconstructions": False,
                "save_contrastive_features": False
            },
            "output": {
                "save_metadata": True
            }
        }
        
        return cls(config) 