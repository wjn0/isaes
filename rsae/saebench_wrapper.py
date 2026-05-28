"""SAEBench wrapper for RIPTopK SAE.

This module provides a wrapper class that adapts the RIPTopK SAE
to be compatible with SAEBench's BaseSAE interface.
"""

import torch
import torch.nn as nn
from typing import Optional
import sys
from pathlib import Path

# Add SAEBench to path
saebench_path = Path(__file__).parent.parent / "SAEBench"
sys.path.insert(0, str(saebench_path))

from sae_bench.custom_saes.base_sae import BaseSAE
import sae_bench.custom_saes.custom_sae_config as sae_config
from rsae.modules import MLPEncoder


class RIPTopKSAEBench(BaseSAE):
    """
    SAEBench-compatible wrapper for RIPTopK SAE.

    This class wraps a RIPTopK model to provide the interface expected by SAEBench:
    - encode(x) → sparse activations
    - decode(z) → reconstruction
    - forward(x) → reconstruction
    - W_enc, W_dec, b_enc, b_dec weight attributes

    Args:
        riptopk_model: Trained RIPTopK instance
        d_in: Input dimension (e.g., 512 for Pythia-70m)
        d_sae: Number of SAE features/concepts
        model_name: Name of the language model (e.g., "pythia-70m-deduped")
        hook_layer: Layer number where SAE operates
        device: Device for computation
        dtype: Data type for tensors
        hook_name: Optional custom hook name (default: f"blocks.{hook_layer}.hook_resid_post")
    """

    def __init__(
        self,
        riptopk_model,
        d_in: int,
        d_sae: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: Optional[str] = None,
    ):
        # Initialize BaseSAE parent class
        super().__init__(
            d_in=d_in,
            d_sae=d_sae,
            model_name=model_name,
            hook_layer=hook_layer,
            device=device,
            dtype=dtype,
            hook_name=hook_name,
        )

        # Store reference to the wrapped RIPTopK model
        self.riptopk = riptopk_model.to(device=device, dtype=dtype)

        # Per-sample normalization stats cached by encode() for use in decode().
        # See _normalize_input / _denormalize_output.
        self._cached_mean: Optional[torch.Tensor] = None
        self._cached_norm: Optional[torch.Tensor] = None

        # Extract and map weights from RIPTopK to SAEBench format
        self._map_weights_from_riptopk()

    def _map_weights_from_riptopk(self):
        """
        Extract weights from RIPTopK model and map them to SAEBench format.

        SAEBench expects:
        - W_enc: [input_dim, nb_concepts]
        - W_dec: [nb_concepts, input_dim]
        - b_enc: [nb_concepts] (optional)
        - b_dec: [input_dim] (optional)
        """
        with torch.no_grad():
            # Encoder weights: [input_dim, nb_concepts]
            if hasattr(self.riptopk, "_get_encoder_weights"):
                encoder_weight = self.riptopk._get_encoder_weights()
            elif isinstance(self.riptopk.encoder, nn.Linear):
                encoder_weight = self.riptopk.encoder.weight.T
            elif isinstance(self.riptopk.encoder, MLPEncoder):
                encoder_weight = self.riptopk.encoder.fc2.weight.T
            else:
                raise AttributeError("Could not infer RIPTopK encoder weights for SAEBench")

            self.W_enc.data = encoder_weight.to(device=self.W_enc.device, dtype=self.W_enc.dtype)

            # Encoder bias (optional)
            encoder_bias = None
            if isinstance(self.riptopk.encoder, nn.Linear):
                encoder_bias = self.riptopk.encoder.bias
            elif isinstance(self.riptopk.encoder, MLPEncoder):
                encoder_bias = self.riptopk.encoder.fc2.bias
            elif isinstance(self.riptopk.encoder, nn.Sequential):
                for module in reversed(self.riptopk.encoder):
                    if isinstance(module, nn.Linear) and module.bias is not None:
                        encoder_bias = module.bias
                        break

            if encoder_bias is not None:
                self.b_enc.data = encoder_bias.to(device=self.b_enc.device, dtype=self.b_enc.dtype)
            else:
                self.b_enc.data.zero_()

            # Decoder weights: [nb_concepts, input_dim]
            if hasattr(self.riptopk, "_get_decoder_weights"):
                decoder_weight = self.riptopk._get_decoder_weights()
            elif hasattr(self.riptopk, "decoder") and hasattr(self.riptopk.decoder, "weight"):
                decoder_weight = self.riptopk.decoder.weight.T
            else:
                raise AttributeError("Could not infer RIPTopK decoder weights for SAEBench")

            self.W_dec.data = decoder_weight.to(device=self.W_dec.device, dtype=self.W_dec.dtype)

            # Decoder bias: use observation bias if present
            if hasattr(self.riptopk, "observation_bias"):
                self.b_dec.data = self.riptopk.observation_bias.data.to(
                    device=self.b_dec.device, dtype=self.b_dec.dtype
                )
            else:
                self.b_dec.data.zero_()

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        # Match BaseTrainer._normalize_data: per-sample mean-center + L2-norm
        # along the feature dim. SAEs are trained on unit-norm vectors and the
        # decoder produces normalized reconstructions, so we must apply the
        # same transform here and stash the stats for decode() to invert.
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        norm = torch.linalg.vector_norm(centered, dim=-1, keepdim=True).clamp(min=1e-8)
        self._cached_mean = mean
        self._cached_norm = norm
        return centered / norm

    def _denormalize_output(self, y: torch.Tensor) -> torch.Tensor:
        if self._cached_norm is None or self._cached_mean is None:
            raise RuntimeError(
                "decode() called without a matching encode() — per-sample "
                "normalization stats are not cached. The wrapper's decode is "
                "stateful: it must run on the output of the most recent encode "
                "call on the same batch."
            )
        return y * self._cached_norm + self._cached_mean

    def _center_input(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.riptopk, "whiten") and self.riptopk.whiten:
            x = self.riptopk.whitener(x)
        if hasattr(self.riptopk, "observation_bias"):
            x = x - self.riptopk.observation_bias
        return x

    def _decenter_output(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.riptopk, "observation_bias"):
            x = x + self.riptopk.observation_bias
        if hasattr(self.riptopk, "whiten") and self.riptopk.whiten:
            x = self.riptopk.whitener.unwhiten(x)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to sparse activations.

        Args:
            x: Input tensor [..., d_in]

        Returns:
            Sparse activations [..., d_sae]
        """
        x_normalized = self._normalize_input(x)
        x_centered = self._center_input(x_normalized)
        _, z_topk = self.riptopk.encode(x_centered)
        return z_topk

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse activations to reconstruction.

        Note: stateful — uses the per-sample mean/norm cached by the most
        recent call to encode(). Reconstructions are returned in the same
        scale as the original input passed to encode().

        Args:
            z: Sparse activations [..., d_sae]

        Returns:
            Reconstruction [..., d_in]
        """
        recon_normalized = self.riptopk.decode(z)
        recon_normalized = self._decenter_output(recon_normalized)
        return self._denormalize_output(recon_normalized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode then decode.

        Args:
            x: Input tensor [batch_size, d_in]

        Returns:
            Reconstruction [batch_size, d_in]
        """
        z = self.encode(x)
        return self.decode(z)


def load_riptopk_sae(
    checkpoint_path: str,
    model_name: str = "pythia-70m-deduped",
    hook_layer: int = 3,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> RIPTopKSAEBench:
    """
    Load a trained RIPTopK SAE from checkpoint and wrap it for SAEBench.

    Args:
        checkpoint_path: Path to saved RIPTopK checkpoint (.pt file)
        model_name: Language model name
        hook_layer: Layer number
        device: Device for computation
        dtype: Data type

    Returns:
        RIPTopKSAEBench instance ready for SAEBench evaluation
    """
    from rsae import RIPTopK

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model config from checkpoint
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    d_in = checkpoint.get('d_in')
    nb_concepts = checkpoint.get('nb_concepts')
    if d_in is None:
        if 'observation_bias' in state_dict:
            d_in = state_dict['observation_bias'].shape[0]
        elif 'decoder.weight' in state_dict:
            d_in = state_dict['decoder.weight'].shape[0]
    if nb_concepts is None:
        if 'decoder.weight' in state_dict:
            nb_concepts = state_dict['decoder.weight'].shape[1]
        elif 'activation_counts' in state_dict:
            nb_concepts = state_dict['activation_counts'].shape[0]
        elif 'activation_frequencies' in state_dict:
            nb_concepts = state_dict['activation_frequencies'].shape[0]
    top_k = checkpoint.get('top_k', 32)  # Default to 32 if not in checkpoint

    # Create RIPTopK instance
    riptopk = RIPTopK(
        input_shape=d_in,
        nb_concepts=nb_concepts,
        top_k=top_k,
        device=device,
    )

    # Load weights
    if 'model_state_dict' in checkpoint:
        riptopk.load_state_dict(checkpoint['model_state_dict'])
    else:
        riptopk.load_state_dict(checkpoint)

    riptopk.eval()

    # Wrap in SAEBench interface
    sae_bench = RIPTopKSAEBench(
        riptopk_model=riptopk,
        d_in=d_in,
        d_sae=nb_concepts,
        model_name=model_name,
        hook_layer=hook_layer,
        device=torch.device(device),
        dtype=dtype,
    )

    return sae_bench


def load_riptopk_from_dictionary_learning_format(
    repo_id: str,
    filename: str,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer: Optional[int] = None,
    local_dir: str = "downloaded_saes",
) -> RIPTopKSAEBench:
    """
    Load a RIPTopK SAE saved in dictionary_learning format.

    This is useful if you trained with dictionary_learning's data pipeline
    and saved in their standard format (ae.pt + config.json).

    Args:
        repo_id: HuggingFace repo ID
        filename: Path to ae.pt within repo
        model_name: Language model name
        device: Device for computation
        dtype: Data type
        layer: Layer number (will be read from config if None)
        local_dir: Local directory for downloads

    Returns:
        RIPTopKSAEBench instance
    """
    import json
    from huggingface_hub import hf_hub_download
    from rsae import RIPTopK

    # Download weights
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
        local_dir=local_dir,
    )

    pt_params = torch.load(path_to_params, map_location=torch.device("cpu"))

    # Download config
    config_filename = filename.replace("ae.pt", "config.json")
    path_to_config = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
        force_download=False,
        local_dir=local_dir,
    )

    with open(path_to_config) as f:
        config = json.load(f)

    if layer is None:
        layer = config["trainer"]["layer"]

    # Extract dimensions from saved weights
    # dictionary_learning format: {encoder.weight, decoder.weight, encoder.bias, bias}
    d_in = pt_params['bias'].shape[0]
    nb_concepts = pt_params['encoder.bias'].shape[0]
    top_k = config["trainer"].get("k", 32)

    # Create RIPTopK model
    riptopk = RIPTopK(
        input_shape=d_in,
        nb_concepts=nb_concepts,
        top_k=top_k,
        device=str(device),
    )

    # Map dictionary_learning weights to RIPTopK
    with torch.no_grad():
        enc_weight = pt_params['encoder.weight']
        if enc_weight.shape == (d_in, nb_concepts):
            enc_weight = enc_weight.T
        if isinstance(riptopk.encoder, nn.Linear):
            riptopk.encoder.weight.data = enc_weight
            if riptopk.encoder.bias is not None:
                riptopk.encoder.bias.data = pt_params['encoder.bias']
        elif isinstance(riptopk.encoder, MLPEncoder):
            riptopk.encoder.fc2.weight.data = enc_weight
            riptopk.encoder.fc2.bias.data = pt_params['encoder.bias']
        else:
            raise AttributeError("Unsupported RIPTopK encoder type for dictionary_learning weights")

        dec_weight = pt_params['decoder.weight']
        if dec_weight.shape == (nb_concepts, d_in):
            dec_weight = dec_weight.T
        riptopk.decoder.weight.data = dec_weight

        # Map decoder bias into observation_bias for centered reconstruction
        if hasattr(riptopk, "observation_bias"):
            riptopk.observation_bias.data = pt_params['bias']

    riptopk.eval()

    # Wrap for SAEBench
    sae_bench = RIPTopKSAEBench(
        riptopk_model=riptopk,
        d_in=d_in,
        d_sae=nb_concepts,
        model_name=model_name,
        hook_layer=layer,
        device=device,
        dtype=dtype,
    )

    return sae_bench
