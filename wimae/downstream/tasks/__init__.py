"""
Downstream task implementations.
"""

from .beam_prediction import BeamPredictionTask
from .los_classification import LOSClassificationTask

__all__ = ["BeamPredictionTask", "LOSClassificationTask"] 