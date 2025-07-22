"""Trainer classes for recommendation models."""

from .advanced_trainer import AdvancedTrainer
from .base_trainer import BaseTrainer
from .simple_trainer import SimpleTrainer

__all__ = ["BaseTrainer", "SimpleTrainer", "AdvancedTrainer"]


def create_trainer(trainer_type: str = "simple", **kwargs):
    """
    Factory function to create trainers.

    Args:
        trainer_type: Type of trainer ("simple" or "advanced")
        **kwargs: Additional arguments passed to trainer constructor

    Returns:
        Trainer instance
    """
    trainers = {
        "simple": SimpleTrainer,
        "advanced": AdvancedTrainer,
    }

    if trainer_type not in trainers:
        raise ValueError(
            f"Unknown trainer type: {trainer_type}. Choose from {list(trainers.keys())}"
        )

    return trainers[trainer_type](**kwargs)
