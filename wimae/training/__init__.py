"""
Training modules for WiMAE and ContraWiMAE.
"""

from .trainer import BaseTrainer
from .train_wimae import WiMAETrainer
from .train_contramae import ContraWiMAETrainer

__all__ = ["BaseTrainer", "WiMAETrainer", "ContraWiMAETrainer"] 