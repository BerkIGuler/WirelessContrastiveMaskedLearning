#!/usr/bin/env python3
"""
Example script for running downstream tasks with WiMAE and ContraWiMAE models.

This script demonstrates how to train and evaluate downstream tasks
(beam prediction and LOS classification) using pre-trained models.
"""

import yaml
from pathlib import Path
import argparse

from wimae.downstream import DownstreamTrainer


def run_beam_prediction(config_path: str, codebook_size: int = None):
    """
    Run beam prediction task.
    
    Args:
        config_path: Path to configuration file
        codebook_size: Optional codebook size (overrides config)
    """
    print("Running beam prediction task...")
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Set task type
    config["task"]["type"] = "beam_prediction"
    
    # Override codebook size if provided
    if codebook_size:
        config["task"]["codebook_size"] = codebook_size
        print(f"Using codebook size: {codebook_size}")
    
    # Create trainer
    trainer = DownstreamTrainer(config)
    
    # Run task
    metrics = trainer.run()
    
    print("Beam prediction completed!")
    return metrics


def run_los_classification(config_path: str, threshold: float = None):
    """
    Run LOS classification task.
    
    Args:
        config_path: Path to configuration file
        threshold: Optional decision threshold (overrides config)
    """
    print("Running LOS classification task...")
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Set task type
    config["task"]["type"] = "los_classification"
    
    # Override threshold if provided
    if threshold:
        config["task"]["threshold"] = threshold
        print(f"Using threshold: {threshold}")
    
    # Create trainer
    trainer = DownstreamTrainer(config)
    
    # Run task
    metrics = trainer.run()
    
    print("LOS classification completed!")
    return metrics


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run downstream tasks")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("--task", choices=["beam_prediction", "los_classification"], 
                       help="Task type (overrides config)")
    parser.add_argument("--codebook-size", type=int, 
                       help="Codebook size for beam prediction (overrides config)")
    parser.add_argument("--threshold", type=float, 
                       help="Decision threshold for LOS classification (overrides config)")
    parser.add_argument("--model-checkpoint", help="Model checkpoint path (overrides config)")
    parser.add_argument("--embeddings-path", help="Embeddings path (overrides config)")
    parser.add_argument("--labels-path", help="Labels path (overrides config)")
    parser.add_argument("--output-dir", help="Output directory (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration to check task type
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Determine task type
    task_type = args.task or config["task"]["type"]
    
    # Override config parameters
    if args.model_checkpoint:
        config["model"]["checkpoint_path"] = args.model_checkpoint
    
    if args.embeddings_path:
        config["data"]["embeddings_path"] = args.embeddings_path
    
    if args.labels_path:
        config["data"]["labels_path"] = args.labels_path
    
    if args.output_dir:
        config["output"]["output_dir"] = args.output_dir
    
    # Run appropriate task
    if task_type == "beam_prediction":
        metrics = run_beam_prediction(args.config, args.codebook_size)
    elif task_type == "los_classification":
        metrics = run_los_classification(args.config, args.threshold)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Print final results
    print(f"\nFinal Results for {task_type}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main() 