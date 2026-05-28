"""
Oracle baseline for sparse coding using Orthogonal Matching Pursuit (OMP).

This module provides an optimal baseline for evaluating sparse autoencoders
by using OMP to solve the sparse coding problem when the true dictionary is known.
"""

import torch
import numpy as np
from typing import Tuple


def omp_batch(D: np.ndarray, Y: np.ndarray, k: int) -> np.ndarray:
    """
    Batched Orthogonal Matching Pursuit with fixed number of iterations.

    Parameters
    ----------
    D : ndarray, shape (m, n)
        Dictionary (columns are atoms, assumed roughly normalized)
    Y : ndarray, shape (batch, m)
        Batch of signals
    k : int
        Number of OMP iterations (sparsity level)

    Returns
    -------
    X : ndarray, shape (batch, n)
        Sparse coefficient vectors
    """
    batch_size, m = Y.shape
    n = D.shape[1]

    residuals = Y.copy()
    supports = np.zeros((batch_size, k), dtype=np.int64)

    for iteration in range(k):
        # Batch correlation: (batch, m) @ (m, n) -> (batch, n)
        correlations = residuals @ D

        # Select atom with max absolute correlation per sample
        new_indices = np.argmax(np.abs(correlations), axis=1)
        supports[:, iteration] = new_indices

        # Build batched dictionary submatrices for current supports
        current_k = iteration + 1
        current_supports = supports[:, :current_k]

        # Gather columns: D[:, current_supports] -> (m, batch, current_k)
        # Transpose to (batch, m, current_k)
        Ds = D[:, current_supports].transpose(1, 0, 2)

        # Solve least squares via normal equations (batched)
        # Gram matrix: (batch, current_k, current_k)
        Gram = np.einsum('bmi,bmj->bij', Ds, Ds)
        # Right-hand side: (batch, current_k)
        rhs = np.einsum('bmi,bm->bi', Ds, Y)

        # Solve: Gram @ coeffs = rhs
        coeffs = np.linalg.solve(Gram, rhs)

        # Update residuals
        residuals = Y - np.einsum('bmi,bi->bm', Ds, coeffs)

    # Build sparse coefficient matrix
    X = np.zeros((batch_size, n))
    batch_idx = np.arange(batch_size)[:, None]
    X[batch_idx, supports] = coeffs

    return X


class OMPOracle:
    """
    Oracle sparse coder using Orthogonal Matching Pursuit.

    Given the true dictionary, this uses OMP to find the optimal sparse codes
    that reconstruct the observations. This serves as an upper bound baseline
    for what a learned encoder can achieve.

    Args:
        dictionary: The true dictionary matrix of shape (observed_dim, concept_dim)
        k: Target sparsity level (number of non-zero coefficients)
    """

    def __init__(self, dictionary: torch.Tensor, k: int):
        self.dictionary = dictionary
        self.k = k
        self.device = dictionary.device

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode observations into sparse codes using OMP.

        Args:
            x: Observations of shape (batch_size, observed_dim)

        Returns:
            Sparse codes of shape (batch_size, concept_dim)
        """
        # Move to CPU for numpy
        x_cpu = x.detach().cpu().numpy()
        D_cpu = self.dictionary.detach().cpu().numpy()

        # Batched OMP
        codes = omp_batch(D_cpu, x_cpu, self.k)

        # Convert back to torch
        return torch.from_numpy(codes).to(self.device)

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode and reconstruct observations.

        Args:
            x: Observations of shape (batch_size, observed_dim)

        Returns:
            codes: Sparse codes of shape (batch_size, concept_dim)
            reconstruction: Reconstructed observations of shape (batch_size, observed_dim)
        """
        codes = self.encode(x).float()
        reconstruction = codes @ self.dictionary.T
        return codes, reconstruction


def get_oracle_codes(x: torch.Tensor, dictionary: torch.Tensor, k: int) -> torch.Tensor:
    """
    Legacy function for backward compatibility.

    Get oracle sparse codes using Orthogonal Matching Pursuit.

    Args:
        x: Observations of shape (batch_size, observed_dim)
        dictionary: Dictionary of shape (observed_dim, concept_dim)
        k: Sparsity level (number of non-zero coefficients)

    Returns:
        Sparse codes of shape (batch_size, concept_dim)
    """
    oracle = OMPOracle(dictionary, k)
    return oracle.encode(x)
