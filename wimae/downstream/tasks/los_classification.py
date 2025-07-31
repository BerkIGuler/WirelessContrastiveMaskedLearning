"""
LOS classification downstream task.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch.nn.functional as F

from ...models import WiMAE, ContraWiMAE
from ...encoding.encoder import Encoder
from .beam_prediction import BeamPredictionDataset, LinearClassifier, FCNClassifier, ResNetClassifier, ResBlock


class LOSClassificationTask:
    """
    Line-of-Sight (LOS) classification downstream task.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LOS classification task.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config["model"]["device"])
        
        # Task parameters
        self.threshold = config["task"]["threshold"]
        
        # Setup output directory
        self.output_dir = Path(config["output"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.classifier = None
        self.optimizer = None
        self.criterion = None
    
    def setup_model(self, input_dim: int):
        """
        Setup the classifier model.
        
        Args:
            input_dim: Input dimension
        """
        arch_config = self.config["architecture"]
        arch_type = arch_config["type"]
        
        if arch_type == "linear":
            self.classifier = LinearClassifier(
                input_dim=input_dim,
                num_classes=1,  # Binary classification
                hidden_dim=arch_config.get("hidden_dim", 64)
            )
        elif arch_type == "fcn":
            self.classifier = FCNClassifier(
                input_dim=input_dim,
                num_classes=1,  # Binary classification
                layers=arch_config.get("layers", [512, 256, 128]),
                dropout=arch_config.get("dropout", 0.1)
            )
        elif arch_type == "resnet":
            self.classifier = ResNetClassifier(
                input_dim=input_dim,
                num_classes=1,  # Binary classification
                num_blocks=arch_config.get("num_blocks", 3),
                block_channels=arch_config.get("block_channels", [64, 128, 256])
            )
        else:
            raise ValueError(f"Unknown architecture type: {arch_type}")
        
        self.classifier.to(self.device)
    
    def setup_optimizer(self):
        """Setup optimizer."""
        training_config = self.config["training"]
        
        if self.config["training"]["fine_tune_encoder"] and self.model is not None:
            # Different learning rates for encoder and classifier
            encoder_params = self.model.parameters()
            classifier_params = self.classifier.parameters()
            
            encoder_lr = training_config["learning_rate"] * training_config["encoder_lr_multiplier"]
            
            self.optimizer = torch.optim.Adam([
                {"params": encoder_params, "lr": encoder_lr},
                {"params": classifier_params, "lr": training_config["learning_rate"]}
            ], weight_decay=training_config["weight_decay"])
        else:
            self.optimizer = torch.optim.Adam(
                self.classifier.parameters(),
                lr=training_config["learning_rate"],
                weight_decay=training_config["weight_decay"]
            )
    
    def setup_criterion(self):
        """Setup loss function."""
        self.criterion = nn.BCEWithLogitsLoss()
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load and prepare data.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        data_config = self.config["data"]
        
        if data_config["input_type"] == "embeddings":
            # Load pre-computed embeddings
            embeddings_data = torch.load(data_config["embeddings_path"])
            embeddings = embeddings_data["embeddings"]
        else:
            # Generate embeddings on-the-fly
            embeddings = self._generate_embeddings()
        
        # Load labels
        labels = torch.load(data_config["labels_path"])
        
        # Apply training budget
        training_budget = self.config["training"]["training_budget"]
        if training_budget < 1.0:
            num_train = int(len(embeddings) * training_budget)
            indices = torch.randperm(len(embeddings))[:num_train]
            embeddings = embeddings[indices]
            labels = labels[indices]
        
        # Create dataset
        dataset = BeamPredictionDataset(embeddings.numpy(), labels.numpy())
        
        # Split dataset
        val_split = self.config["training"]["val_split"]
        test_split = self.config["training"]["test_split"]
        
        total_size = len(dataset)
        val_size = int(total_size * val_split)
        test_size = int(total_size * test_split)
        train_size = total_size - val_size - test_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create dataloaders
        batch_size = self.config["training"]["batch_size"]
        num_workers = data_config["num_workers"]
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=data_config["pin_memory"]
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=data_config["pin_memory"]
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=data_config["pin_memory"]
        )
        
        return train_loader, val_loader, test_loader
    
    def _generate_embeddings(self) -> torch.Tensor:
        """
        Generate embeddings on-the-fly from raw data.
        
        Returns:
            Generated embeddings
        """
        data_config = self.config["data"]
        
        # Load model
        checkpoint_path = data_config["model_checkpoint_path"]
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if "contrastive_dim" in checkpoint:
            self.model = ContraWiMAE.from_checkpoint(checkpoint_path, device=self.device)
        else:
            self.model = WiMAE.from_checkpoint(checkpoint_path, device=self.device)
        
        # Create encoder
        encoder = Encoder.from_checkpoint(checkpoint_path, device=self.device)
        
        # Generate embeddings
        embeddings_data = encoder.encode_data(
            data_config["data_path"], 
            data_config["data_format"]
        )
        
        return torch.from_numpy(embeddings_data["embeddings"])
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Train the classifier.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history
        """
        training_config = self.config["training"]
        epochs = training_config["epochs"]
        patience = training_config["early_stopping_patience"]
        
        # Setup components
        if self.classifier is None:
            # Get input dimension from first batch
            sample_embeddings, _ = next(iter(train_loader))
            input_dim = sample_embeddings.shape[1]
            self.setup_model(input_dim)
        
        self.setup_optimizer()
        self.setup_criterion()
        
        # Training history
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"Training LOS classification classifier for {epochs} epochs...")
        print(f"Threshold: {self.threshold}")
        print(f"Device: {self.device}")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self._validate(val_loader)
            
            # Update history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                self._save_model("best_model.pt")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
        
        return history
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.classifier.train()
        if self.model is not None and self.config["training"]["fine_tune_encoder"]:
            self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for embeddings, labels in tqdm(train_loader, desc="Training"):
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device).float().unsqueeze(1)  # Add channel dimension
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.model is not None and self.config["training"]["fine_tune_encoder"]:
                # Fine-tune encoder
                with torch.enable_grad():
                    encoded_features = self.model.encode(embeddings)
                    outputs = self.classifier(encoded_features)
            else:
                # Frozen encoder
                outputs = self.classifier(embeddings)
            
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities >= self.threshold).float()
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
        
        return total_loss / len(train_loader), correct / total
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.classifier.eval()
        if self.model is not None:
            self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for embeddings, labels in tqdm(val_loader, desc="Validation"):
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)  # Add channel dimension
                
                if self.model is not None and self.config["training"]["fine_tune_encoder"]:
                    encoded_features = self.model.encode(embeddings)
                    outputs = self.classifier(encoded_features)
                else:
                    outputs = self.classifier(embeddings)
                
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities >= self.threshold).float()
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        
        return total_loss / len(val_loader), correct / total
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.classifier.eval()
        if self.model is not None:
            self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for embeddings, labels in tqdm(test_loader, desc="Evaluation"):
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device).float()
                
                if self.model is not None and self.config["training"]["fine_tune_encoder"]:
                    encoded_features = self.model.encode(embeddings)
                    outputs = self.classifier(encoded_features)
                else:
                    outputs = self.classifier(embeddings)
                
                probabilities = torch.sigmoid(outputs).squeeze(1)
                predictions = (probabilities >= self.threshold).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        auc = roc_auc_score(all_labels, all_probabilities)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "specificity": specificity,
            "npv": npv,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn
        }
        
        return metrics
    
    def _save_model(self, filename: str):
        """Save the model."""
        filepath = self.output_dir / filename
        torch.save({
            "classifier_state_dict": self.classifier.state_dict(),
            "model_state_dict": self.model.state_dict() if self.model else None,
            "config": self.config
        }, filepath)
    
    def evaluate_from_checkpoint(self, model_checkpoint: str) -> Dict[str, float]:
        """
        Evaluate a pre-trained model.
        
        Args:
            model_checkpoint: Path to model checkpoint
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Load model
        checkpoint = torch.load(model_checkpoint, map_location=self.device)
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        
        # Load data
        train_loader, val_loader, test_loader = self.load_data()
        
        # Evaluate
        metrics = self.evaluate(test_loader)
        
        return metrics
    
    @classmethod
    def from_config(cls, config_path: str) -> "LOSClassificationTask":
        """
        Create task from configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Initialized task
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        return cls(config) 