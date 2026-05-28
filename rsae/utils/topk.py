import torch


def apply_topk_and_scatter(pre_codes: torch.Tensor, top_k: int, use_abstopk: bool) -> torch.Tensor:
    """
    Apply TopK sparsification to pre-codes and scatter values.

    Args:
        pre_codes: Pre-codes tensor of shape [batch_size, nb_concepts]
        top_k: Number of top activations to keep
        use_abstopk: If True, use magnitude-based TopK (preserves signs).
               If False, apply ReLU then TopK (positive-only, original behavior).
    """
    if use_abstopk:
        # Magnitude-based TopK: select by absolute value, preserve signs
        topk = torch.topk(pre_codes.abs(), top_k, dim=-1)
        z_topk = torch.zeros_like(pre_codes).scatter(-1, topk.indices, pre_codes.gather(-1, topk.indices))
    else:
        # Positive-only TopK: apply ReLU to ensure positive values (original behavior)
        pre_codes_positive = torch.relu(pre_codes)
        topk = torch.topk(pre_codes_positive, top_k, dim=-1)
        z_topk = torch.zeros_like(pre_codes_positive).scatter(-1, topk.indices, topk.values)

    return z_topk
