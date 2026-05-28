"""Regularized Sparse Autoencoders for improved stability."""

from .rip_topk import RIPTopK
from .synthetic_data import generate_synthetic_data, SyntheticDataConfig

__version__ = "0.1.0"
__all__ = ["RIPTopK", "generate_synthetic_data", "SyntheticDataConfig"]
