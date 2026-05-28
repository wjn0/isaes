"""Pure tensor-based metrics for evaluating sparse autoencoders.

This module contains metrics that operate on tensors directly.
For model-specific metrics, see standard_metrics.py and identifiability_metrics.py.
"""

import torch
import numpy as np
import time
import warnings
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from typing import Dict, Tuple, Optional


# ============================================================================
# Dictionary Comparison Metrics
# ============================================================================

def compute_dictionary_mcc(
    dictionary1: torch.Tensor,
    dictionary2: torch.Tensor,
) -> float:
    """
    Compute MCC (Maximum Cosine Correlation) between two dictionary matrices.

    Uses Hungarian algorithm to find optimal 1-to-1 matching between dictionary concepts,
    then computes average cosine similarity of matched pairs.

    Args:
        dictionary1: First dictionary [n_concepts_1, observed_dim]
        dictionary2: Second dictionary [n_concepts_2, observed_dim]

    Returns:
        Average cosine similarity of matched pairs (0 to 1, higher is better)
    """
    # Shape assertions
    assert dictionary1.ndim == 2, f"dictionary1 must be 2D, got shape {dictionary1.shape}"
    assert dictionary2.ndim == 2, f"dictionary2 must be 2D, got shape {dictionary2.shape}"
    assert dictionary1.shape[1] == dictionary2.shape[1], \
        f"Dictionaries must have same observed_dim: {dictionary1.shape[1]} vs {dictionary2.shape[1]}"

    n_concepts_1 = dictionary1.shape[0]
    n_concepts_2 = dictionary2.shape[0]
    observed_dim = dictionary1.shape[1]

    # Warn if not overcomplete
    if n_concepts_1 <= observed_dim or n_concepts_2 <= observed_dim:
        warnings.warn(
            f"Dictionary is not overcomplete: {n_concepts_1} and {n_concepts_2} concepts with "
            f"{observed_dim} dimensions. This may indicate undercomplete setup.",
            RuntimeWarning
        )

    # Normalize concept vectors (rows)
    dict1_norm = dictionary1 / (
        torch.norm(dictionary1, dim=1, keepdim=True) + 1e-8
    )
    dict2_norm = dictionary2 / (
        torch.norm(dictionary2, dim=1, keepdim=True) + 1e-8
    )

    # Compute cosine similarity matrix
    # [n_concepts_1, n_concepts_2]
    similarity = (dict1_norm @ dict2_norm.T).abs()

    # Use Hungarian algorithm to find optimal 1-to-1 matching
    # We need to maximize similarity, so use negative of similarity as cost
    cost_matrix = -similarity.detach().cpu().numpy()

    # Handle mismatched sizes: pad the smaller dimension if needed
    if n_concepts_1 > n_concepts_2:
        # More concepts in dict1: pad dict2
        cost_matrix = np.pad(cost_matrix, ((0, 0), (0, n_concepts_1 - n_concepts_2)), constant_values=0)
    elif n_concepts_2 > n_concepts_1:
        # More concepts in dict2: pad dict1
        cost_matrix = np.pad(cost_matrix, ((0, n_concepts_2 - n_concepts_1), (0, 0)), constant_values=0)

    # Run Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Compute average similarity of matched pairs (only count valid matches)
    valid_matches = row_ind < n_concepts_1
    if n_concepts_2 < n_concepts_1:
        valid_matches = valid_matches & (col_ind < n_concepts_2)

    matched_similarities = similarity[row_ind[valid_matches], col_ind[valid_matches]]

    # Print quantiles of matched similarities for debugging
    quantiles = np.quantile(
        matched_similarities.detach().cpu().numpy(),
        [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    )
    print("Matched similarities quantiles (min, 10%, 25%, 50%, 75%, 90%, max):", quantiles.tolist())

    return matched_similarities.mean().item()


def compute_dictionary_alignment(
    dictionary1: torch.Tensor,
    dictionary2: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """
    Compute optimal alignment between two dictionary matrices using Hungarian algorithm.

    Similar to compute_dictionary_mcc but returns alignment indices instead of score.
    These indices can be used to reorder activations from model2 to align with model1.

    Args:
        dictionary1: First dictionary [n_concepts_1, observed_dim]
        dictionary2: Second dictionary [n_concepts_2, observed_dim]

    Returns:
        Tuple of (row_ind, col_ind, signs) from Hungarian algorithm
        - row_ind: Indices into dictionary1 (and activations from model1)
        - col_ind: Indices into dictionary2 (can reorder model2 activations)
        - signs: Sign corrections (+1 or -1) for matched pairs to handle anti-correlated features
    """
    # Shape assertions
    assert dictionary1.ndim == 2, f"dictionary1 must be 2D, got shape {dictionary1.shape}"
    assert dictionary2.ndim == 2, f"dictionary2 must be 2D, got shape {dictionary2.shape}"
    assert dictionary1.shape[1] == dictionary2.shape[1], \
        f"Dictionaries must have same observed_dim: {dictionary1.shape[1]} vs {dictionary2.shape[1]}"

    n_concepts_1 = dictionary1.shape[0]
    n_concepts_2 = dictionary2.shape[0]
    observed_dim = dictionary1.shape[1]

    # Warn if not overcomplete
    if n_concepts_1 <= observed_dim or n_concepts_2 <= observed_dim:
        warnings.warn(
            f"Dictionary is not overcomplete: {n_concepts_1} and {n_concepts_2} concepts with "
            f"{observed_dim} dimensions. This may indicate undercomplete setup.",
            RuntimeWarning
        )

    # Normalize concept vectors (rows)
    dict1_norm = dictionary1 / (
        torch.norm(dictionary1, dim=1, keepdim=True) + 1e-8
    )
    dict2_norm = dictionary2 / (
        torch.norm(dictionary2, dim=1, keepdim=True) + 1e-8
    )

    # Compute cosine similarity matrix
    # [n_concepts_1, n_concepts_2]
    similarity_signed = dict1_norm @ dict2_norm.T
    similarity = similarity_signed.abs()  # Use abs for matching

    # Use Hungarian algorithm to find optimal 1-to-1 matching
    # We need to maximize similarity, so use negative of similarity as cost
    cost_matrix = -similarity.detach().cpu().numpy()

    # Handle mismatched sizes: pad the smaller dimension if needed
    if n_concepts_1 > n_concepts_2:
        # More concepts in dict1: pad dict2
        cost_matrix = np.pad(cost_matrix, ((0, 0), (0, n_concepts_1 - n_concepts_2)), constant_values=0)
    elif n_concepts_2 > n_concepts_1:
        # More concepts in dict2: pad dict1
        cost_matrix = np.pad(cost_matrix, ((0, n_concepts_2 - n_concepts_1), (0, 0)), constant_values=0)

    # Run Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Extract signs for matched pairs
    valid_matches = (row_ind < n_concepts_1) & (col_ind < n_concepts_2)
    signs = torch.sign(similarity_signed[row_ind[valid_matches], col_ind[valid_matches]])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)

    return row_ind, col_ind, signs


# ============================================================================
# Activation Alignment
# ============================================================================

def align_activations(
    Z1: torch.Tensor,
    Z2: torch.Tensor,
    epsilon: float = 1e-8,
    warn_slow: bool = True,
    timings: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Align columns of Z2 to Z1 using correlation coefficient and Hungarian algorithm.

    Args:
        Z1: First activation matrix [n_samples, n_features]
        Z2: Second activation matrix [n_samples, n_features]
        epsilon: Small constant for numerical stability (default: 1e-8)
        warn_slow: If True, warn if Hungarian alignment takes > 1s
        timings: Optional dict to store detailed timing breakdowns

    Returns:
        Z2_aligned: Z2 with columns reordered to align with Z1
        alignment: Alignment indices tensor
    """
    device = Z1.device

    if timings is None:
        timings = {}

    # Time centering and normalization
    t0 = time.time()
    # Center each column (subtract mean across samples)
    Z1_centered = Z1 - Z1.mean(dim=0, keepdim=True)
    Z2_centered = Z2 - Z2.mean(dim=0, keepdim=True)
    # Normalize by standard deviation (equivalent to L2 norm of centered data)
    Z1_norm = Z1_centered / (torch.norm(Z1_centered, dim=0, keepdim=True) + epsilon)
    Z2_norm = Z2_centered / (torch.norm(Z2_centered, dim=0, keepdim=True) + epsilon)
    timings['normalization'] = time.time() - t0

    # Time correlation computation
    t0 = time.time()
    correlation_signed = torch.einsum('si,sj->ij', Z1_norm, Z2_norm)
    correlation = correlation_signed.abs()  # Use abs for matching
    timings['similarity_computation'] = time.time() - t0

    # Time Hungarian algorithm
    t0 = time.time()
    cost_matrix = -correlation.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    hungarian_time = time.time() - t0
    timings['hungarian_algorithm'] = hungarian_time

    # Extract signs for matched pairs
    signs = torch.sign(correlation_signed[row_ind, col_ind]).to(device)
    # Handle zero correlations (keep positive)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)

    # Warn if slow
    if warn_slow and hungarian_time > 1.0:
        warnings.warn(
            f"Hungarian alignment took {hungarian_time:.2f}s, which is unusually slow. "
            f"Consider reducing the number of concepts or using a faster alignment method.",
            RuntimeWarning
        )

    # Time reordering
    t0 = time.time()
    n_features = Z1.shape[1]
    alignment = torch.zeros(n_features, dtype=torch.long, device=device)
    alignment[row_ind] = torch.tensor(col_ind, dtype=torch.long, device=device)
    Z2_aligned = Z2[:, alignment]

    # Apply sign correction for anti-correlated features
    Z2_aligned = Z2_aligned * signs.unsqueeze(0)  # broadcast over samples
    timings['reordering'] = time.time() - t0

    return Z2_aligned, alignment


# ============================================================================
# Concept Connectedness
# ============================================================================

def compute_concept_connectedness(
    concepts: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute connectedness metrics for concept co-occurrence graph.

    Builds a graph where nodes are concepts and edges exist if two concepts
    co-occur in at least one sample. Then computes:
    - Number of connected components
    - Average shortest path length within the largest component

    Args:
        concepts: Concept activations [n_samples, n_concepts]

    Returns:
        Dictionary with:
            - n_connected_components: Number of connected components
            - avg_shortest_path: Average shortest path in largest component
            - largest_component_size: Size of largest connected component
    """
    n_samples, n_concepts = concepts.shape
    assert n_samples > 0 and n_concepts > 0, f"Invalid activation shape: {concepts.shape}"

    # Binarize concepts (active = 1, inactive = 0)
    binary_concepts = (concepts != 0).float()

    # Build co-occurrence matrix: C[i,j] = 1 if concepts i and j co-occur
    # Co-occurrence means they're both active in at least one sample
    # C = binary_concepts.T @ binary_concepts gives counts, we want binary
    cooccurrence = (binary_concepts.T @ binary_concepts) > 0

    # Set diagonal to False (no self-loops)
    cooccurrence.fill_diagonal_(False)

    # Convert to scipy sparse matrix for graph algorithms
    adj_matrix = csr_matrix(cooccurrence.cpu().numpy())

    # Find connected components
    n_components, labels = connected_components(
        csgraph=adj_matrix,
        directed=False,
        return_labels=True
    )

    # Find largest component
    component_sizes = np.bincount(labels)
    largest_component_idx = np.argmax(component_sizes)
    largest_component_size = component_sizes[largest_component_idx]

    # Compute average shortest path in largest component
    if largest_component_size > 1:
        # Get nodes in largest component
        nodes_in_largest = np.where(labels == largest_component_idx)[0]

        # Extract subgraph for largest component
        subgraph = adj_matrix[nodes_in_largest, :][:, nodes_in_largest]

        # Compute all-pairs shortest paths
        dist_matrix = shortest_path(
            csgraph=subgraph,
            directed=False,
            unweighted=True
        )

        # Average over all pairs (excluding diagonal and infinities)
        finite_distances = dist_matrix[np.isfinite(dist_matrix)]
        finite_distances = finite_distances[finite_distances > 0]  # Exclude diagonal

        if len(finite_distances) > 0:
            avg_shortest_path = float(np.mean(finite_distances))
        else:
            avg_shortest_path = 0.0
    else:
        avg_shortest_path = 0.0

    return {
        'n_connected_components': int(n_components),
        'avg_shortest_path': avg_shortest_path,
        'largest_component_size': int(largest_component_size),
    }
