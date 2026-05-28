"""Model introspection utilities for SAE models."""

import torch
from typing import Optional, Union
from omegaconf import DictConfig


def extract_decoder_weights(model: torch.nn.Module, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Extract decoder weight matrix from any SAE model.

    This function supports multiple SAE architectures by checking for various
    attribute patterns commonly used for decoder/dictionary weights.

    Args:
        model: SAE model with decoder weights
        device: Optional device to move weights to. If None, keeps weights on original device.

    Returns:
        Decoder weights D of shape [nb_concepts, input_dim]

    Raises:
        AttributeError: If decoder weights cannot be found in the model

    Examples:
        >>> weights = extract_decoder_weights(model)
        >>> weights_cpu = extract_decoder_weights(model, device=torch.device('cpu'))
    """
    decoder_weights = None

    # Try different attribute patterns for decoder weights
    if hasattr(model, 'dictionary') and hasattr(model.dictionary, '_weights'):
        decoder_weights = model.dictionary._weights
    elif hasattr(model, 'decoder') and hasattr(model.decoder, 'weight'):
        decoder_weights = model.decoder.weight.T
    else:
        # Search through modules for common patterns
        for module in model.modules():
            if hasattr(module, '_weights'):
                decoder_weights = module._weights
                break
            elif isinstance(module, torch.nn.Linear):
                # Use first Linear layer as fallback
                decoder_weights = module.weight
                break

    if decoder_weights is None:
        raise AttributeError(
            "Could not find decoder weights in model. "
            "Expected one of: model.dictionary._weights, model.decoder.weight, "
            "or a module with _weights attribute, or a Linear layer."
        )

    # Move to specified device if requested
    if device is not None:
        decoder_weights = decoder_weights.to(device)

    return decoder_weights


def create_rip_topk_model_from_config(
    cfg: DictConfig,
    input_dim: int,
    nb_concepts: int,
    device: Union[str, torch.device] = "cpu",
) -> "RIPTopK":
    """
    Create a RIPTopK model from a Hydra configuration.

    This is the canonical way to instantiate a RIPTopK model to ensure
    consistency across training scripts.

    Args:
        cfg: Hydra configuration with model parameters in cfg.model and
            optional top_k in cfg.data.k
        input_dim: Dimension of input data
        nb_concepts: Number of concepts/features in sparse representation
        device: Device to place the model on

    Returns:
        Configured RIPTopK model (not yet moved to device - caller should call .to(device))

    Example:
        >>> model = create_rip_topk_model_from_config(cfg, input_dim=512, nb_concepts=1024, device="cuda")
        >>> model = model.to(device)
    """
    from rsae import RIPTopK

    # Get top_k from data config (required)
    top_k = cfg.data.k

    # Get model config with defaults
    model_cfg = cfg.model

    model = RIPTopK(
        input_shape=input_dim,
        nb_concepts=nb_concepts,
        top_k=top_k,
        # TopK variants
        batch_topk=model_cfg.get("batch_topk", False),
        matryoshka=model_cfg.get("matryoshka", False),
        matryoshka_group_sizes=model_cfg.get("matryoshka_group_sizes", None),
        # RIP loss configuration
        rip_weight=model_cfg.rip_weight,
        rip_loss_weighted=model_cfg.get("rip_loss_weighted", False),
        rip_loss_multiplier=model_cfg.get("rip_loss_multiplier", 2),
        # Encoder configuration
        use_abstopk=model_cfg.get("use_abstopk", False),
        mlp_encoder=model_cfg.get("mlp_encoder", False),
        num_encode_steps=model_cfg.get("num_encode_steps", 1),
        initial_step_size=model_cfg.get("initial_step_size", 0.9),
        learned_step_size=model_cfg.get("learned_step_size", True),
        step_size_rank=model_cfg.get("step_size_rank", 0),
        # Auxiliary loss configuration
        auxiliary_k=model_cfg.get("auxiliary_k", None),
        auxk_weight=model_cfg.get("auxk_weight", 1.0 / 32),
        l1_weight=model_cfg.get("l1_weight", 0.0),
        # Normalization and tracking
        normalization=model_cfg.normalization,
        activation_window_batches=model_cfg.get("activation_window_batches", 64),
        whiten=model_cfg.get("whiten", False),
        # Device
        device=str(device),
    )

    return model
