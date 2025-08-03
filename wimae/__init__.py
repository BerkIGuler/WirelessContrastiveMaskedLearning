"""
Wireless Contrastive Masked Learning Package

This package provides implementations of WiMAE and ContraWiMAE models
for wireless channel modeling.
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from .models import WiMAE, ContraWiMAE
from .training import BaseTrainer

__all__ = [
    "WiMAE",
    "ContraWiMAE", 
    "BaseTrainer",
] 