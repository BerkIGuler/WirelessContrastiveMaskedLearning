"""
Downstream tasks for WiMAE and ContraWiMAE models.
"""

from .tasks import BeamPredictionTask, LOSClassificationTask
from .trainer import DownstreamTrainer

__all__ = ["BeamPredictionTask", "LOSClassificationTask", "DownstreamTrainer"] 