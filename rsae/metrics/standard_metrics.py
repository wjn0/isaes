"""Standard metrics for evaluating sparse autoencoders.

This module contains metrics that operate on a single SAE model without requiring
ground truth or paired model comparisons. For identifiability metrics that require
ground truth or model pairs, see identifiability_metrics.py.
"""

import torch
from typing import Dict, Union
from .metrics import compute_concept_connectedness
from ..rip_loss import compute_rip_loss
from ..utils.model_utils import extract_decoder_weights


# ============================================================================
# Helper Functions for Model Interaction
# ============================================================================

def _extract_decoder_weights(model) -> torch.Tensor:
    """
    Extract decoder weight matrix from any SAE model.

    Args:
        model: SAE model with decoder weights

    Returns:
        Decoder weights D of shape [nb_concepts, input_dim]
    """
    return extract_decoder_weights(model)


def _extract_activations_from_output(output, model=None):
    """
    Extract activations from model forward pass output.

    Args:
        output: Output from model forward pass
        model: Optional model reference for fallback encoding

    Returns:
        Activations tensor or None if extraction fails
    """
    if isinstance(output, tuple) and len(output) == 3:
        _, activations, _ = output
        return activations
    elif isinstance(output, tuple) and len(output) == 2:
        activations, _ = output
        return activations
    else:
        # For non-standard outputs, return None
        return None


def _collect_activations_from_model(
    model,
    observations: torch.Tensor,
    batch_size: int = 512,
    binarize: bool = False,
) -> torch.Tensor:
    """
    Collect activations from a model for a dataset.

    Args:
        model: Trained SAE model
        observations: Input observations [n_samples, input_dim]
        batch_size: Batch size for forward passes
        binarize: If True, return binary activations (>0)

    Returns:
        Activations tensor [n_samples, n_concepts]
    """
    model.eval()
    device = next(model.parameters()).device

    activations_list = []
    n_samples = observations.shape[0]

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = observations[i:i + batch_size].to(device)

            output = model(batch)
            activations = _extract_activations_from_output(output, model)

            # Fallback: try encode method
            if activations is None and hasattr(model, 'encode'):
                activations = model.encode(batch)

            if activations is not None:
                if binarize:
                    activations = (activations != 0).float()
                activations_list.append(activations.cpu())

    if len(activations_list) == 0:
        raise RuntimeError("Failed to extract activations from model")

    return torch.cat(activations_list, dim=0)


# ============================================================================
# Reconstruction Metrics
# ============================================================================

def compute_reconstruction_metrics_batch(
    sae,
    observations: torch.Tensor,
    batch_size: int = 256,
) -> Dict[str, float]:
    """
    Compute reconstruction MSE, explained variance, and sparsity in a single data pass.

    This is more efficient than calling compute_reconstruction_mse,
    compute_explained_variance, and compute_sparsity separately, as it
    processes the data only once.

    Args:
        sae: Trained sparse autoencoder model
        observations: Test observations [n_samples, observed_dim]
        batch_size: Batch size for evaluation

    Returns:
        Dictionary containing:
            - reconstruction_mse: Mean squared reconstruction error
            - explained_variance: Fraction of variance explained (0 to 1)
            - sparsity: Average L0 sparsity (number of active features per sample)
    """
    sae.eval()
    device = next(sae.parameters()).device

    n_samples = observations.shape[0]
    total_mse = 0.0
    total_sparsity = 0.0
    n_batches = 0

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = observations[i:i + batch_size].to(device)

            # Single forward pass - get all outputs
            output = sae(batch)

            # Extract activations and reconstruction
            if isinstance(output, tuple) and len(output) == 3:
                _, activations, reconstruction = output
            elif isinstance(output, tuple) and len(output) == 2:
                activations, reconstruction = output
                # Fallback for activations
                if activations is None and hasattr(sae, 'encode'):
                    activations = sae.encode(batch)
            else:
                reconstruction = output
                activations = None
                # Try to get activations
                if hasattr(sae, 'encode'):
                    activations = sae.encode(batch)

            # Compute MSE for this batch
            batch_mse = torch.mean((reconstruction - batch) ** 2)
            total_mse += batch_mse.item()

            # Compute sparsity for this batch
            if activations is not None:
                batch_sparsity = torch.mean((activations != 0).float().sum(dim=1))
                total_sparsity += batch_sparsity.item()

            n_batches += 1

    # Compute average MSE
    avg_mse = total_mse / n_batches

    # Compute explained variance from MSE
    total_variance = torch.var(observations).item()
    explained_variance = 1 - (avg_mse / total_variance)

    # Compute average sparsity
    avg_sparsity = total_sparsity / n_batches if n_batches > 0 else 0.0

    return {
        'reconstruction_mse': avg_mse,
        'explained_variance': explained_variance,
        'sparsity': avg_sparsity,
    }


# ============================================================================
# Connectedness Metrics
# ============================================================================

def compute_concept_connectedness_from_model(
    sae,
    observations: torch.Tensor,
    batch_size: int = 256,
) -> Dict[str, float]:
    """
    Compute concept connectedness for a trained SAE model.

    Args:
        sae: Trained sparse autoencoder
        observations: Observations to compute activations on [n_samples, observed_dim]
        batch_size: Batch size for computing activations

    Returns:
        Dictionary with connectedness metrics (see compute_concept_connectedness)
    """
    all_activations = _collect_activations_from_model(sae, observations, batch_size, binarize=False)
    return compute_concept_connectedness(all_activations)


# ============================================================================
# Dictionary Property Metrics
# ============================================================================

def compute_coherence(model) -> float:
    """
    Compute coherence metric (maximum off-diagonal correlation in Gram matrix).

    Coherence measures the maximum absolute cosine similarity between any two
    dictionary vectors. Lower coherence indicates better separation between concepts.

    Args:
        model: SAE model with decoder weights

    Returns:
        Maximum absolute off-diagonal element of normalized Gram matrix
    """
    D = _extract_decoder_weights(model)

    gram = D @ D.T
    norms = torch.sqrt(torch.diagonal(gram))
    gram_normalized = gram / (norms.unsqueeze(1) @ norms.unsqueeze(0) + 1e-8)
    gram_off_diag = gram_normalized - torch.eye(gram_normalized.shape[0], device=gram.device)
    coherence = torch.max(torch.abs(gram_off_diag)).item()

    return coherence


def compute_max_eigenvalue(model) -> float:
    """
    Compute the largest eigenvalue of D^T D where D is [input_dim, nb_concepts].

    Since _extract_decoder_weights returns [nb_concepts, input_dim],
    we compute D @ D.T to get the [nb_concepts, nb_concepts] Gram matrix.

    Args:
        model: SAE model with decoder weights

    Returns:
        Largest eigenvalue of the Gram matrix
    """
    D = _extract_decoder_weights(model)  # [nb_concepts, input_dim]
    gram_matrix = D @ D.T  # [nb_concepts, nb_concepts]
    eigenvalues = torch.linalg.eigvalsh(gram_matrix)  # ascending order
    return eigenvalues[-1].item()


def compute_operator_norm(model_or_matrix: Union[torch.nn.Module, torch.Tensor]) -> float:
    """
    Compute operator (spectral) norm from model or matrix.

    The operator norm is the largest singular value, which measures the maximum
    scaling factor the linear map can apply to any unit vector.

    Unified implementation that accepts either:
    - Model with decoder weights (extracts weights)
    - Raw matrix tensor

    Args:
        model_or_matrix: Model with decoder or raw tensor [nb_concepts, input_dim]

    Returns:
        Operator norm (largest singular value)
    """
    if isinstance(model_or_matrix, torch.Tensor):
        D = model_or_matrix
    else:
        D = _extract_decoder_weights(model_or_matrix)

    with torch.no_grad():
        operator_norm = torch.linalg.matrix_norm(D, ord=2).item()

    return operator_norm


def compute_avg_concept_norm(model_or_matrix: Union[torch.nn.Module, torch.Tensor]) -> float:
    """
    Compute average L2 norm of concept vectors from model or matrix.

    This measures the typical scale of individual concept vectors, complementing
    the operator norm which measures the maximum scaling.

    Unified implementation that accepts either:
    - Model with decoder weights (extracts weights)
    - Raw matrix tensor

    Args:
        model_or_matrix: Model with decoder or raw tensor [nb_concepts, input_dim]

    Returns:
        Average L2 norm of concept vectors (columns when transposed)
    """
    if isinstance(model_or_matrix, torch.Tensor):
        D = model_or_matrix
    else:
        D = _extract_decoder_weights(model_or_matrix)

    # D is [nb_concepts, input_dim], transpose to get concept vectors as columns
    D_T = D.T  # [input_dim, nb_concepts]

    with torch.no_grad():
        concept_norms = torch.norm(D_T, dim=0)  # [nb_concepts]
        avg_norm = concept_norms.mean().item()

    return avg_norm


def compute_rip_loss_for_model(model, pre_activations: torch.Tensor, x: torch.Tensor) -> float:
    """
    Compute RIP loss for any model.

    Args:
        model: SAE model with decoder/dictionary and rip_loss_* attributes
        pre_activations: Pre-sparsified concept activations [batch_size, nb_concepts]
        x: Input observations (unused, kept for API compatibility)

    Returns:
        RIP loss value (scalar float)
    """
    return model.compute_rip_loss(pre_activations, x).item()


# ============================================================================
# Diagnostic Metrics
# ============================================================================

def compute_activation_statistics(model, observations: torch.Tensor, batch_size: int = 512) -> Dict[str, float]:
    """
    Compute statistics about model activations to diagnose issues.

    Args:
        model: Trained model
        observations: Full dataset [n_samples, input_dim]
        batch_size: Batch size for computing activations

    Returns:
        Dictionary with activation statistics:
            Per-sample statistics:
            - mean_norm: Mean L2 norm per sample
            - min_norm: Minimum L2 norm
            - max_norm: Maximum L2 norm
            - num_zero_norm: Number of samples with zero norm
            - mean_sparsity: Mean number of active features
            - min_sparsity: Minimum sparsity
            - max_sparsity: Maximum sparsity
            - num_zero_active: Number of samples with no active features

            Per-concept statistics:
            - dead_concept_count: Number of concepts never activated
            - dead_concept_proportion: Proportion of concepts never activated
            - mean_concept_activation_freq: Average activation frequency across concepts
            - concept_activation_freq_variance: Variance of activation frequencies across concepts
    """
    activations = _collect_activations_from_model(model, observations, batch_size, binarize=False)

    # Compute per-sample L2 norms
    sample_norms = torch.sqrt(torch.sum(activations ** 2, dim=1))  # [n_samples]

    # Compute sparsity per sample (number of active features)
    sparsity = (activations != 0).sum(dim=1).float()  # [n_samples]

    # Compute per-concept statistics
    concept_active = (activations != 0)  # [n_samples, n_concepts]
    concept_activation_freq = concept_active.float().mean(dim=0)  # [n_concepts]

    stats = {
        # Per-sample statistics
        'mean_norm': sample_norms.mean().item(),
        'min_norm': sample_norms.min().item(),
        'max_norm': sample_norms.max().item(),
        'num_zero_norm': (sample_norms < 1e-6).sum().item(),
        'mean_sparsity': sparsity.mean().item(),
        'min_sparsity': sparsity.min().item(),
        'max_sparsity': sparsity.max().item(),
        'num_zero_active': (sparsity == 0).sum().item(),
        # Per-concept statistics
        'dead_concept_count': (concept_activation_freq == 0).sum().item(),
        'dead_concept_proportion': (concept_activation_freq == 0).float().mean().item(),
        'mean_concept_activation_freq': concept_activation_freq.mean().item(),
        'concept_activation_freq_variance': concept_activation_freq.var().item(),
    }

    return stats
