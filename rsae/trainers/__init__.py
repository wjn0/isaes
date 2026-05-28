"""Trainer classes for SAE training.

This module provides a unified trainer architecture with:
- BaseTrainer: Abstract base class with common training infrastructure
- SyntheticTrainer: For synthetic data with ground truth metrics
- TransformerTrainer: For transformer activations with buffer integration
"""

from .base_trainer import BaseTrainer
from .synthetic_trainer import SyntheticTrainer
from .transformer_trainer import TransformerTrainer

__all__ = [
    'BaseTrainer',
    'SyntheticTrainer',
    'TransformerTrainer',
]
