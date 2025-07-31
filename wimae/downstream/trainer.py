"""
Downstream trainer for unified task execution.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import sys

from .tasks import BeamPredictionTask, LOSClassificationTask


class DownstreamTrainer:
    """
    Unified trainer for downstream tasks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the downstream trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.task_type = config["task"]["type"]
        
        # Initialize task
        if self.task_type == "beam_prediction":
            self.task = BeamPredictionTask(config)
        elif self.task_type == "los_classification":
            self.task = LOSClassificationTask(config)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def run(self):
        """
        Run the downstream task.
        """
        print(f"Running {self.task_type} task...")
        
        # Load data
        train_loader, val_loader, test_loader = self.task.load_data()
        
        # Train model
        history = self.task.train(train_loader, val_loader)
        
        # Evaluate on test set
        metrics = self.task.evaluate(test_loader)
        
        # Print results
        print(f"\nTest Results for {self.task_type}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save results
        self._save_results(history, metrics)
        
        return metrics
    
    def _save_results(self, history: Dict[str, list], metrics: Dict[str, float]):
        """
        Save training history and evaluation metrics.
        
        Args:
            history: Training history
            metrics: Evaluation metrics
        """
        results = {
            "task_type": self.task_type,
            "config": self.config,
            "history": history,
            "metrics": metrics
        }
        
        # Save to file
        output_dir = Path(self.config["output"]["output_dir"])
        timestamp = self._get_timestamp()
        results_file = output_dir / f"{self.task_type}_results_{timestamp}.json"
        
        import json
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to: {results_file}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @classmethod
    def from_config(cls, config_path: str) -> "DownstreamTrainer":
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


def main():
    """
    Main function for command-line execution.
    """
    parser = argparse.ArgumentParser(description="Downstream task trainer")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("--task", choices=["beam_prediction", "los_classification"], 
                       help="Task type (overrides config)")
    parser.add_argument("--model-checkpoint", help="Model checkpoint path (overrides config)")
    parser.add_argument("--embeddings-path", help="Embeddings path (overrides config)")
    parser.add_argument("--labels-path", help="Labels path (overrides config)")
    parser.add_argument("--output-dir", help="Output directory (overrides config)")
    parser.add_argument("--device", help="Device (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override config with command-line arguments
    modified_params = []
    
    if args.task:
        config["task"]["type"] = args.task
        modified_params.append(f"task.type: {args.task}")
    
    if args.model_checkpoint:
        config["model"]["checkpoint_path"] = args.model_checkpoint
        modified_params.append(f"model.checkpoint_path: {args.model_checkpoint}")
    
    if args.embeddings_path:
        config["data"]["embeddings_path"] = args.embeddings_path
        modified_params.append(f"data.embeddings_path: {args.embeddings_path}")
    
    if args.labels_path:
        config["data"]["labels_path"] = args.labels_path
        modified_params.append(f"data.labels_path: {args.labels_path}")
    
    if args.output_dir:
        config["output"]["output_dir"] = args.output_dir
        modified_params.append(f"output.output_dir: {args.output_dir}")
    
    if args.device:
        config["model"]["device"] = args.device
        modified_params.append(f"model.device: {args.device}")
    
    # Print modified parameters
    if modified_params:
        print("Warning: The following parameters have been modified:")
        for param in modified_params:
            print(f"  - {param}")
        print()
    
    # Create and run trainer
    trainer = DownstreamTrainer(config)
    metrics = trainer.run()
    
    return metrics


if __name__ == "__main__":
    main() 