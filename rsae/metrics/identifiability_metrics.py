"""Identifiability metrics for evaluating sparse autoencoders.

This module contains metrics that require either:
1. Ground truth mixing matrix/latents (for synthetic data evaluation)
2. Two independently trained models (for pairwise comparison)

For standard metrics that only require a single model, see standard_metrics.py.
"""

import torch
from typing import Dict, Tuple
from .metrics import (
    compute_dictionary_mcc,
    compute_dictionary_alignment,
    align_activations,
)
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


def _apply_dictionary_alignment(Z2: torch.Tensor, model1, model2) -> torch.Tensor:
    """
    Align activations from model2 to model1 using dictionary-based matching.

    Extracts decoder weights from both models, computes optimal alignment using
    cosine similarity of dictionary vectors, and reorders Z2 columns accordingly.

    Args:
        Z2: Activations from model2 [batch_size, nb_concepts]
        model1: First SAE model
        model2: Second SAE model

    Returns:
        Aligned Z2 with columns reordered to match model1
    """
    # Extract dictionaries
    dict1 = _extract_decoder_weights(model1)
    dict2 = _extract_decoder_weights(model2)

    # Get alignment indices
    _, col_ind, signs = compute_dictionary_alignment(dict1, dict2)

    # Reorder Z2 columns to align with Z1
    # Only use valid indices (in case of padding from mismatched sizes)
    n_concepts_2 = Z2.shape[1]
    valid_indices = col_ind[col_ind < n_concepts_2]
    Z2_aligned = Z2[:, valid_indices]

    # Apply sign corrections
    Z2_aligned = Z2_aligned * signs.to(Z2.device).unsqueeze(0)

    return Z2_aligned


# ============================================================================
# Batch Alignment and Metric Computation (Optimized)
# ============================================================================

def compute_all_alignments(
    Z1: torch.Tensor,
    Z2: torch.Tensor,
    dict1: torch.Tensor = None,
    dict2: torch.Tensor = None
) -> Dict[str, torch.Tensor]:
    """
    Compute all alignment types once and return as a dictionary.

    This eliminates redundant Hungarian matching calls when computing multiple metrics.

    Args:
        Z1: Activations from first model/ground truth [n_samples, n_concepts]
        Z2: Activations from second model [n_samples, n_concepts]
        dict1: Optional dictionary from first model/ground truth [n_concepts, input_dim]
        dict2: Optional dictionary from second model [n_concepts, input_dim]

    Returns:
        Dict with keys:
            - 'activation_aligned': Z2 aligned via activation cosine similarity
            - 'activation_aligned_indices': column permutation from activation alignment
            - 'activation_aligned_signs': sign corrections from activation alignment
            - 'dict_aligned': Z2 aligned via dictionary cosine similarity (if dicts provided)
            - 'dict_aligned_indices': column permutation from dictionary alignment (if dicts provided)
            - 'dict_aligned_signs': sign corrections from dictionary alignment (if dicts provided)
    """
    alignments = {}

    # 1. Activation-based alignment (Hungarian on Z1, Z2)
    Z2_activation_aligned, alignment = align_activations(Z1, Z2)
    alignments['activation_aligned'] = Z2_activation_aligned
    alignments['activation_aligned_indices'] = alignment

    # Extract signs from the alignment by comparing aligned vs original
    # Z2_activation_aligned = Z2[:, alignment] * signs
    # So signs = sign(Z2_activation_aligned / (Z2[:, alignment] + epsilon))
    # But simpler: check if the first non-zero element has the same sign
    Z2_reordered = Z2[:, alignment]
    # Compute signs by comparing dot products (positive = same sign, negative = flipped)
    activation_signs = torch.sign(
        (Z2_activation_aligned * Z2_reordered).sum(dim=0)
    )
    # Handle zero columns (keep positive)
    activation_signs = torch.where(
        activation_signs == 0,
        torch.ones_like(activation_signs),
        activation_signs
    )
    alignments['activation_aligned_signs'] = activation_signs

    # 2. Dictionary-based alignment (only if dictionaries provided)
    if dict1 is not None and dict2 is not None:
        _, dict_col_ind, dict_signs = compute_dictionary_alignment(dict1, dict2)

        n_concepts_2 = Z2.shape[1]
        valid_indices = dict_col_ind[dict_col_ind < n_concepts_2]
        Z2_dict_aligned = Z2[:, valid_indices]

        # Apply sign corrections
        Z2_dict_aligned = Z2_dict_aligned * dict_signs.to(Z2.device).unsqueeze(0)

        alignments['dict_aligned'] = Z2_dict_aligned
        alignments['dict_aligned_indices'] = dict_col_ind
        alignments['dict_aligned_signs'] = dict_signs

    return alignments


def _collect_paired_activations(
    model1,
    model2,
    observations: torch.Tensor,
    batch_size: int = 512,
    binarize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect activations from two models for the same dataset.

    Args:
        model1: First trained model
        model2: Second trained model
        observations: Input observations [n_samples, input_dim]
        batch_size: Batch size for forward passes
        binarize: If True, return binary activations (>0)

    Returns:
        Tuple of (activations1, activations2) both [n_samples, n_concepts]
    """
    model1.eval()
    model2.eval()
    device = next(model1.parameters()).device

    Z1_list = []
    Z2_list = []
    n_samples = observations.shape[0]

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = observations[i:i + batch_size].to(device)

            # Get activations from both models
            _, acts1, _ = model1(batch)
            _, acts2, _ = model2(batch)

            if binarize:
                acts1 = (acts1 != 0).float()
                acts2 = (acts2 != 0).float()

            Z1_list.append(acts1)
            Z2_list.append(acts2)

    Z1 = torch.cat(Z1_list, dim=0)
    Z2 = torch.cat(Z2_list, dim=0)

    return Z1, Z2


def _collect_model_and_groundtruth_activations(
    model,
    observations: torch.Tensor,
    latents: torch.Tensor,
    batch_size: int = 512,
    binarize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect activations from model and prepare ground truth latents.

    Args:
        model: Trained model
        observations: Input observations [n_samples, input_dim]
        latents: Ground truth latent activations [n_samples, concept_dim]
        batch_size: Batch size for forward passes
        binarize: If True, return binary activations (>0)

    Returns:
        Tuple of (model_activations, ground_truth_activations)
    """
    model.eval()
    device = next(model.parameters()).device

    # Collect model activations
    Z_model_list = []
    n_samples = observations.shape[0]

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = observations[i:i + batch_size].to(device)
            _, acts, _ = model(batch)

            if binarize:
                acts = (acts != 0).float()

            Z_model_list.append(acts)

    Z_model = torch.cat(Z_model_list, dim=0)
    Z_true = latents.to(device)

    if binarize:
        Z_true = (Z_true != 0).float()

    return Z_model, Z_true


# ============================================================================
# Shared Computation Helpers (Deduplicated Logic)
# ============================================================================

_DEFAULT_MAX_CHUNK_BYTES = 256 * 1024 * 1024  # 256 MB


def _should_chunk(Z: torch.Tensor, max_chunk_bytes: int = _DEFAULT_MAX_CHUNK_BYTES) -> bool:
    return Z.numel() * Z.element_size() > max_chunk_bytes


def _infer_chunk_size(Z: torch.Tensor, max_chunk_bytes: int = _DEFAULT_MAX_CHUNK_BYTES) -> int:
    if Z.numel() == 0:
        return 1
    per_row_bytes = Z.shape[1] * Z.element_size()
    if per_row_bytes <= 0:
        return Z.shape[0]
    chunk_size = max(1, int(max_chunk_bytes // per_row_bytes))
    return min(chunk_size, Z.shape[0])


def _compute_iou_metric(
    Z1: torch.Tensor,
    Z2_aligned: torch.Tensor,
) -> float:
    """
    Compute mean Intersection over Union (IoU) for binary activations.

    Helper function to deduplicate IoU computation logic.

    Args:
        Z1: Activations from first model [n_samples, n_concepts]
        Z2_aligned: Activations from second model (already aligned) [n_samples, n_concepts]

    Returns:
        Mean IoU across all samples
    """
    n_samples = Z1.shape[0]
    if not _should_chunk(Z1):
        Z1_binary = Z1 != 0
        Z2_binary = Z2_aligned != 0
        intersection = (Z1_binary & Z2_binary).sum(dim=1)  # [n_samples]
        union = (Z1_binary | Z2_binary).sum(dim=1)  # [n_samples]
        iou = torch.where(union > 0, intersection / union, torch.ones_like(union, dtype=torch.float))
        return iou.mean().item()

    chunk_size = _infer_chunk_size(Z1)
    iou_sum = torch.zeros((), device=Z1.device, dtype=torch.float32)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        Z1_chunk = Z1[start:end] != 0
        Z2_chunk = Z2_aligned[start:end] != 0
        intersection = (Z1_chunk & Z2_chunk).sum(dim=1)
        union = (Z1_chunk | Z2_chunk).sum(dim=1)
        iou_chunk = torch.where(
            union > 0,
            intersection / union,
            torch.ones_like(union, dtype=torch.float)
        )
        iou_sum += torch.sum(iou_chunk, dtype=torch.float32)

    return (iou_sum / n_samples).item()


def _compute_l2_distance_metric(
    Z1: torch.Tensor,
    Z2_aligned: torch.Tensor,
) -> float:
    """
    Compute mean L2 distance between activation vectors.

    Helper function to deduplicate L2 distance computation logic.

    Args:
        Z1: Activations from first model [n_samples, n_concepts]
        Z2_aligned: Activations from second model (already aligned) [n_samples, n_concepts]

    Returns:
        Mean L2 distance per sample
    """
    n_samples = Z1.shape[0]
    if not _should_chunk(Z1):
        l2_distances = torch.sqrt(torch.sum((Z1 - Z2_aligned) ** 2, dim=1))  # [n_samples]
        return l2_distances.mean().item()

    chunk_size = _infer_chunk_size(Z1)
    l2_sum = torch.zeros((), device=Z1.device, dtype=torch.float32)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        diff = Z1[start:end] - Z2_aligned[start:end]
        l2 = torch.sqrt(torch.sum(diff * diff, dim=1))
        l2_sum += torch.sum(l2, dtype=torch.float32)

    return (l2_sum / n_samples).item()


def _compute_mse_metric(
    Z1: torch.Tensor,
    Z2_aligned: torch.Tensor,
) -> float:
    """
    Compute mean squared error between activations.

    Helper function to deduplicate MSE computation logic.

    Args:
        Z1: Activations from first model [n_samples, n_concepts]
        Z2_aligned: Activations from second model (already aligned) [n_samples, n_concepts]

    Returns:
        Mean squared error across all activations
    """
    if not _should_chunk(Z1):
        mse = torch.mean((Z1 - Z2_aligned) ** 2)
        return mse.item()

    n_samples, n_concepts = Z1.shape
    chunk_size = _infer_chunk_size(Z1)
    mse_sum = torch.zeros((), device=Z1.device, dtype=torch.float32)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        diff = Z1[start:end] - Z2_aligned[start:end]
        mse_sum += torch.sum(diff * diff, dtype=torch.float32)

    mse = mse_sum / (n_samples * n_concepts)
    return mse.item()


def _compute_normalized_l2_metric(
    Z1: torch.Tensor,
    Z2_aligned: torch.Tensor,
) -> float:
    """
    Compute normalized L2 distance between activations.

    Normalizes by the median distance between random pairs of activation vectors.

    Helper function to deduplicate normalized L2 computation logic.

    Args:
        Z1: Activations from first model [n_samples, n_concepts]
        Z2_aligned: Activations from second model (already aligned) [n_samples, n_concepts]

    Returns:
        Normalized L2 distance (mean L2 distance / mean norm)
    """
    n_samples = Z1.shape[0]
    if not _should_chunk(Z1):
        l2_distances = torch.sqrt(torch.sum((Z1 - Z2_aligned) ** 2, dim=1))  # [n_samples]
        Z2_aligned_reordered = Z2_aligned[torch.randperm(Z2_aligned.size(0))]
        random_distances = torch.sqrt(torch.sum((Z1 - Z2_aligned_reordered) ** 2, dim=1))
        return (l2_distances.mean() / random_distances.mean()).item()

    chunk_size = _infer_chunk_size(Z1)
    perm = torch.randperm(n_samples, device=Z1.device)
    l2_sum = torch.zeros((), device=Z1.device, dtype=torch.float32)
    rand_sum = torch.zeros((), device=Z1.device, dtype=torch.float32)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        Z1_chunk = Z1[start:end]
        Z2_chunk = Z2_aligned[start:end]
        diff = Z1_chunk - Z2_chunk
        l2 = torch.sqrt(torch.sum(diff * diff, dim=1))
        l2_sum += torch.sum(l2, dtype=torch.float32)

        Z2_rand = Z2_aligned.index_select(0, perm[start:end])
        diff_rand = Z1_chunk - Z2_rand
        rand = torch.sqrt(torch.sum(diff_rand * diff_rand, dim=1))
        rand_sum += torch.sum(rand, dtype=torch.float32)

    normalized_l2 = (l2_sum / n_samples) / (rand_sum / n_samples)
    return normalized_l2.item()


def _compute_correlation_metric(
    Z1: torch.Tensor,
    Z2_aligned: torch.Tensor,
) -> float:
    """
    Compute mean absolute Pearson correlation coefficient between activation vectors.

    Helper function to deduplicate correlation computation logic.

    Args:
        Z1: Activations from first model [n_samples, n_concepts]
        Z2_aligned: Activations from second model (already aligned) [n_samples, n_concepts]

    Returns:
        Mean absolute Pearson correlation coefficient across all concepts
    """
    if not _should_chunk(Z1):
        Z1_centered = Z1 - Z1.mean(dim=0, keepdim=True)
        Z2_centered = Z2_aligned - Z2_aligned.mean(dim=0, keepdim=True)
        covariance = (Z1_centered * Z2_centered).mean(dim=0)
        std1 = Z1_centered.std(dim=0)
        std2 = Z2_centered.std(dim=0)
        correlation = covariance / (std1 * std2 + 1e-8)  # Avoid division by zero
        return correlation.abs().mean().item()

    n_samples = Z1.shape[0]
    chunk_size = _infer_chunk_size(Z1)
    mean1 = Z1.mean(dim=0)
    mean2 = Z2_aligned.mean(dim=0)
    cov_sum = torch.zeros_like(mean1, dtype=torch.float32)
    var1_sum = torch.zeros_like(mean1, dtype=torch.float32)
    var2_sum = torch.zeros_like(mean2, dtype=torch.float32)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        Z1_chunk = Z1[start:end] - mean1
        Z2_chunk = Z2_aligned[start:end] - mean2
        cov_sum += torch.sum(Z1_chunk * Z2_chunk, dim=0, dtype=torch.float32)
        var1_sum += torch.sum(Z1_chunk * Z1_chunk, dim=0, dtype=torch.float32)
        var2_sum += torch.sum(Z2_chunk * Z2_chunk, dim=0, dtype=torch.float32)

    covariance = cov_sum / n_samples
    std1 = torch.sqrt(var1_sum / n_samples)
    std2 = torch.sqrt(var2_sum / n_samples)
    correlation = covariance / (std1 * std2 + 1e-8)
    return correlation.abs().mean().item()


def _compute_explained_variance_metric(
    observations: torch.Tensor,
    reconstruction: torch.Tensor,
) -> float:
    """
    Compute explained variance (R² score) for reconstruction.

    Formula: 1 - (SS_res / SS_tot)
    where SS_tot = sum of variance per dimension
    and SS_res = sum of mean squared residuals per dimension

    Helper function to deduplicate explained variance computation logic.

    Args:
        observations: Original observations [n_samples, observed_dim]
        reconstruction: Reconstructed observations [n_samples, observed_dim]

    Returns:
        Explained variance as float (can be negative if worse than mean)
    """
    ss_tot = torch.var(observations, dim=0).sum()
    ss_res = torch.mean((observations - reconstruction) ** 2, dim=0).sum()
    explained_var = 1 - (ss_res / (ss_tot + 1e-8))
    return explained_var.item()


# ============================================================================
# Dictionary-based Metrics (Model-to-Model or Model-to-Ground-Truth)
# ============================================================================

def _compute_dictionary_mcc_from_tensors(dict1: torch.Tensor, dict2: torch.Tensor, ind=None, signs=None) -> float:
    """
    Compute MCC between two dictionaries (tensors).

    Args:
        dict1: First dictionary [n_concepts, input_dim]
        dict2: Second dictionary [n_concepts, input_dim]
        ind: Optional alignment indices to use in dict2
        signs: Optional sign corrections to apply to dict2 after reordering

    Returns:
        Average absolute cosine similarity of matched concept pairs (0 to 1, higher is better)
    """
    dict1 = dict1.detach()
    dict2 = dict2.detach()

    if ind is None:
        return compute_dictionary_mcc(dict1, dict2)

    dict1 /= torch.norm(dict1, dim=1, keepdim=True) + 1e-8
    dict2 /= torch.norm(dict2, dim=1, keepdim=True) + 1e-8

    # Reorder dict2 according to alignment
    dict2_aligned = dict2[ind]

    # Apply sign corrections if provided
    if signs is not None:
        dict2_aligned = dict2_aligned * signs.to(dict2.device).unsqueeze(1)

    # Compute similarity between matched pairs
    similarity = (dict1 * dict2_aligned).sum(dim=1)

    return similarity.abs().mean().item()


def compute_pw_dictionary_mcc(sae1, sae2, ind=None) -> float:
    """
    Compute pairwise MCC between two trained SAE models.

    Args:
        sae1: First trained sparse autoencoder
        sae2: Second trained sparse autoencoder
        ind: Optional alignment indices to use in sae2

    Returns:
        Average absolute cosine similarity of matched concept pairs (0 to 1, higher is better)
    """
    dict1 = _extract_decoder_weights(sae1).detach()
    dict2 = _extract_decoder_weights(sae2).detach()
    return _compute_dictionary_mcc_from_tensors(dict1, dict2, ind)


def compute_gt_dictionary_mcc(sae, true_mixing_matrix: torch.Tensor) -> float:
    """
    Compute MCC between a trained SAE and the ground truth mixing matrix.

    Args:
        sae: Trained sparse autoencoder
        true_mixing_matrix: True mixing matrix [concept_dim, observed_dim]

    Returns:
        Average absolute cosine similarity of matched concept pairs (0 to 1, higher is better)
    """
    learned_dict = _extract_decoder_weights(sae)
    return compute_dictionary_mcc(true_mixing_matrix, learned_dict)


# ============================================================================
# Identifiability Metrics (Pairwise Model Comparison)
# ============================================================================


def _compute_bias_difference(model1, model2):
    return torch.norm(model1.observation_bias - model2.observation_bias)

def compute_pairwise_identifiability_metrics(
    Z1: torch.Tensor,
    Z2: torch.Tensor,
    dict1: torch.Tensor = None,
    dict2: torch.Tensor = None
) -> Dict[str, float]:
    """
    Core function to compute identifiability metrics between two sets of activations.

    This is the shared implementation used by both model-to-model and ground-truth comparisons.

    Args:
        Z1: First set of activations [n_samples, n_concepts]
        Z2: Second set of activations [n_samples, n_concepts]
        dict1: Optional first dictionary [n_concepts, input_dim] (needed for dictionary-based metrics)
        dict2: Optional second dictionary [n_concepts, input_dim] (needed for dictionary-based metrics)

    Returns:
        Dict with all metric names as keys (without 'identifiability/' prefix):
            - 'pw_mcc': Pairwise dictionary MCC (if dicts provided)
            - 'pw_mcc_z': Dictionary MCC using activation-based alignment (if dicts provided)
            - 'iou': Mean IoU (activation-aligned)
            - 'raw_l2': Raw L2 distance (activation-aligned)
            - 'mse': MSE (activation-aligned)
            - 'mean_code_norm': Mean L2 norm of reference model codes
            - 'normalized_l2': Normalized L2 (activation-aligned)
            - 'correlation': Mean absolute Pearson correlation (activation-aligned)
            - 'iou_dict': Mean IoU (dictionary-aligned, if dicts provided)
            - 'raw_l2_dict': Raw L2 distance (dictionary-aligned, if dicts provided)
            - 'mse_dict': MSE (dictionary-aligned, if dicts provided)
            - 'normalized_l2_dict': Normalized L2 (dictionary-aligned, if dicts provided)
            - 'correlation_dict': Mean absolute Pearson correlation (dictionary-aligned, if dicts provided)
    """
    metrics = {}

    # Validate inputs
    assert Z1.shape == Z2.shape, f"Activation shapes must match: Z1 {Z1.shape} vs Z2 {Z2.shape}"
    assert Z1.shape[0] > 0 and Z1.shape[1] > 0, f"Invalid activation shape: {Z1.shape}"

    # ===== STEP 1: Compute all alignments once =====
    alignments = compute_all_alignments(Z1, Z2, dict1, dict2)

    # ===== STEP 2: Dictionary-based metrics (only if dicts provided) =====
    if dict1 is not None and dict2 is not None:
        metrics['pw_mcc'] = _compute_dictionary_mcc_from_tensors(dict1, dict2)
        metrics['pw_mcc_z'] = _compute_dictionary_mcc_from_tensors(
            dict1, dict2,
            ind=alignments['activation_aligned_indices'],
            signs=alignments['activation_aligned_signs']
        )

    # ===== STEP 4: Activation-aligned metrics =====
    Z2_aligned = alignments['activation_aligned']

    # IoU (binary activations)
    metrics['iou'] = _compute_iou_metric(Z1, Z2_aligned)

    # Raw L2
    metrics['raw_l2'] = _compute_l2_distance_metric(Z1, Z2_aligned)

    # MSE
    metrics['mse'] = _compute_mse_metric(Z1, Z2_aligned)

    # Mean code norm (for reference)
    metrics['mean_code_norm'] = torch.sqrt(torch.sum(Z1 ** 2, dim=1)).mean().item()

    # Normalized L2
    metrics['normalized_l2'] = _compute_normalized_l2_metric(Z1, Z2_aligned)

    # Correlation
    metrics['correlation'] = _compute_correlation_metric(Z1, Z2_aligned)

    # ===== STEP 5: Dictionary-aligned metrics (only if dicts provided) =====
    if 'dict_aligned' in alignments:
        Z2_dict_aligned = alignments['dict_aligned']

        # IoU (binary activations)
        metrics['iou_dict'] = _compute_iou_metric(Z1, Z2_dict_aligned)

        # Raw L2
        metrics['raw_l2_dict'] = _compute_l2_distance_metric(Z1, Z2_dict_aligned)

        # MSE
        metrics['mse_dict'] = _compute_mse_metric(Z1, Z2_dict_aligned)

        # Normalized L2
        metrics['normalized_l2_dict'] = _compute_normalized_l2_metric(Z1, Z2_dict_aligned)

        # Correlation
        metrics['correlation_dict'] = _compute_correlation_metric(Z1, Z2_dict_aligned)

    return metrics


def compute_model_identifiability_metrics(
    model1,
    model2,
    observations: torch.Tensor,
    batch_size: int = 512
) -> Dict[str, float]:
    """
    Compute identifiability metrics between two trained models.

    Thin wrapper that collects activations from both models, extracts dictionaries,
    and calls the core pairwise metrics function.

    Args:
        model1: First trained model
        model2: Second trained model
        observations: Full dataset [n_samples, input_dim]
        batch_size: Batch size for computing activations

    Returns:
        Dict with all metric names as keys (without 'identifiability/' prefix)
    """
    # Collect activations from both models
    Z1, Z2 = _collect_paired_activations(model1, model2, observations, batch_size, binarize=False)

    # Extract dictionaries
    dict1 = _extract_decoder_weights(model1)
    dict2 = _extract_decoder_weights(model2)

    # Compute all pairwise metrics
    metrics = compute_pairwise_identifiability_metrics(Z1, Z2, dict1, dict2)

    # Add model-specific metrics
    metrics['bias_difference'] = _compute_bias_difference(model1, model2)

    return metrics
