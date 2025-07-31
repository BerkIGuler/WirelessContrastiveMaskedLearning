"""
Wireless Contrastive Masked Learning Package

This package provides implementations of WiMAE and ContraWiMAE models
for wireless channel modeling and downstream tasks.
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from .models import WiMAE, ContraWiMAE
from .training import WiMAETrainer, ContraWiMAETrainer
from .encoding import Encoder
from .downstream import BeamPredictionTask, LOSClassificationTask

__all__ = [
    "WiMAE",
    "ContraWiMAE", 
    "WiMAETrainer",
    "ContraWiMAETrainer",
    "Encoder",
    "BeamPredictionTask",
    "LOSClassificationTask",
] 