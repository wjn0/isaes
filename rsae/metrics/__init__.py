"""Metrics for evaluating sparse autoencoders.

This package provides metrics organized into two categories:
1. Standard metrics (single model, no ground truth needed)
2. Identifiability metrics (require ground truth or model pairs)

For backward compatibility, all metrics are re-exported from the root level.
"""

# Pure tensor-based metrics from metrics.py
from .metrics import (
    compute_dictionary_mcc,
    compute_dictionary_alignment,
    align_activations,
    compute_concept_connectedness,
)

# Standard metrics (single model, no ground truth)
from .standard_metrics import (
    # Reconstruction metrics
    compute_reconstruction_metrics_batch,

    # Connectedness metrics
    compute_concept_connectedness_from_model,

    # Dictionary property metrics
    compute_coherence,
    compute_max_eigenvalue,
    compute_operator_norm,
    compute_avg_concept_norm,
    compute_rip_loss_for_model,

    # Diagnostic metrics
    compute_activation_statistics,
)

# Identifiability metrics (require ground truth or model pairs)
from .identifiability_metrics import (
    # Dictionary-based metrics
    compute_pw_dictionary_mcc,
    compute_gt_dictionary_mcc,

    # Batch metric computation (optimized)
    compute_all_alignments,
    compute_pairwise_identifiability_metrics,
    compute_model_identifiability_metrics,
)

# Backward compatibility
compute_all_identifiability_metrics = compute_model_identifiability_metrics

__all__ = [
    # Pure tensor-based metrics
    'compute_dictionary_mcc',
    'compute_dictionary_alignment',
    'align_activations',
    'compute_concept_connectedness',

    # Dictionary-based metrics
    'compute_pw_dictionary_mcc',
    'compute_gt_dictionary_mcc',

    # Reconstruction metrics
    'compute_reconstruction_metrics_batch',

    # Connectedness metrics
    'compute_concept_connectedness_from_model',

    # Dictionary property metrics
    'compute_coherence',
    'compute_max_eigenvalue',
    'compute_operator_norm',
    'compute_avg_concept_norm',
    'compute_rip_loss_for_model',

    # Batch metric computation (optimized)
    'compute_all_alignments',
    'compute_pairwise_identifiability_metrics',
    'compute_model_identifiability_metrics',
    'compute_all_identifiability_metrics',  # Backward compatibility

    # Diagnostic metrics
    'compute_activation_statistics',
]
