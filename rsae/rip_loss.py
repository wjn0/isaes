"""Standalone RIP (Restricted Isometry Property) loss computation."""

import torch
from typing import Optional

from .utils.topk import apply_topk_and_scatter


def compute_rip_loss(
    x: torch.Tensor,
    preactivations: torch.Tensor,
    encoder: torch.Tensor,
    dictionary: torch.Tensor,
    k: int,
    weighted: bool = False,
    multiplier: int = 2,
    use_abstopk: bool = False,
    mixup: bool = False,
) -> torch.Tensor:
    """
    Compute the Restricted Isometry Property (RIP) regularization loss for the decoder.

    The RIP property encourages the decoder to act as an approximate isometry,
    such that D @ D.T ≈ I. This is measured by comparing the norm of the
    round-trip projection (p → p @ D → p @ D @ D.T) to the direct projection.

    The final interpretation is the average squared deviation from 1 of the eigenvalues
    of the k * multiplier sized Gram submatrices corresponding to inputs.

    Args:
        x: Original input [batch_size, observed_dim] (unused, kept for compatibility)
        preactivations: Pre-sparsified concept activations [batch_size, nb_concepts]
        encoder: Encoder weights [observed_dim, nb_concepts] (unused, kept for compatibility)
        dictionary: Dictionary/decoder weights [nb_concepts, observed_dim]
        k: Sparsity level used in concept activations
        weighted: Whether to use original code values (True) or random Gaussian
                 N(0,1) substitutes with same sparsity pattern (False). Default: False.
        multiplier: Multiplier for RIP dimension (k'). Default: 2.
        use_abstopk: Whether to use magnitude-based TopK (True) or positive-only TopK (False).
        mixup: Whether to apply mixup between samples before computing RIP loss.

    Returns:
        RIP loss value (scalar tensor) - decoder loss only
    """
    rip_dim = k * multiplier

    # Sparsify to Top(RIP dim) activations
    code_interp = apply_topk_and_scatter(
        preactivations,
        rip_dim,
        use_abstopk
    )

    # Mixup
    if mixup:
        code_interp = (code_interp + torch.roll(code_interp, shifts=1, dims=0)) / 2.

    # Optionally replace with random Gaussian values
    if not weighted:
        code_interp = torch.randn_like(code_interp) * (code_interp != 0.).float()

    # Decoder RIP loss
    projections = code_interp @ dictionary
    projections2 = (projections @ dictionary.T) * (code_interp != 0.)
    htr = torch.sum(projections**2, dim=1).mean()
    htr2 = torch.sum(projections2**2, dim=1).mean()

    decoder_loss = htr2 / htr**2 * rip_dim - 1.

    return decoder_loss


def compute_auxk_loss(
    activations: torch.Tensor,
    x: torch.Tensor,
    decoder_weights: torch.Tensor,
    auxiliary_k: int,
    activation_counts: torch.Tensor,
    use_abstopk: bool = True,
) -> torch.Tensor:
    """
    Compute AuxK reconstruction loss using top-K dead concept features.

    This targets dead concepts specifically - concepts that have not activated
    in the sliding window. Among dead concepts only, select the top-K by
    magnitude and compute reconstruction loss. This encourages the model to
    maintain useful representations for inactive concepts.

    IMPORTANT: Pass pre-sparsified activations (before TopK), not sparse activations.
    Dead concepts need non-zero values to be selected by the top-K operation.

    Args:
        activations: Pre-sparsified concept activations [batch_size, nb_concepts]
        x: Original input [batch_size, input_dim]
        decoder_weights: Dictionary/decoder weights [nb_concepts, input_dim]
        auxiliary_k: Number of top features to use for auxiliary reconstruction
        activation_counts: Per-concept activation counts in sliding window [nb_concepts]
        use_abstopk: Whether to use magnitude-based TopK (True) or positive-only TopK (False).

    Returns:
        AuxK loss value (scalar tensor) - MSE between aux reconstruction and input,
        normalized by the variance of ``x`` across the batch.
        Returns 0.0 if no dead concepts exist.
    """
    batch_size = activations.shape[0]
    nb_concepts = activations.shape[1]

    # Identify dead concepts (count == 0)
    dead_mask = activation_counts == 0  # [nb_concepts]
    num_dead = dead_mask.sum().item()

    # If no dead concepts, return zero loss
    if num_dead == 0:
        return torch.tensor(0.0, device=activations.device, dtype=activations.dtype)

    # Get indices of dead concepts
    dead_indices = torch.nonzero(dead_mask, as_tuple=True)[0]  # [num_dead]

    # Extract activations for dead concepts only
    dead_activations = activations[:, dead_indices]  # [batch_size, num_dead]

    # Determine effective k (can't select more than num_dead)
    k = min(auxiliary_k, num_dead)

    # Apply topk and scatter on dead concept activations only
    sparse_dead_activations = apply_topk_and_scatter(dead_activations, k, use_abstopk)  # [batch_size, num_dead]

    # Map back to full activation tensor
    aux_activations = torch.zeros(batch_size, nb_concepts, device=activations.device, dtype=activations.dtype)
    aux_activations[:, dead_indices] = sparse_dead_activations

    # Reconstruct using only the selected dead concepts
    aux_reconstruction = aux_activations @ decoder_weights

    # Compute MSE loss
    auxk_loss = torch.mean((x - aux_reconstruction) ** 2)

    # Normalization factor
    norm_factor = torch.mean((x - x.mean(dim=0, keepdim=True)) ** 2).item()

    return auxk_loss / norm_factor
