"""RIPTopK: TopK SAE with Restricted Isometry Property regularization."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .rip_loss import compute_rip_loss as compute_rip_loss_fn
from .rip_loss import compute_auxk_loss as compute_auxk_loss_fn
from .utils.model_utils import extract_decoder_weights
from .utils.topk import apply_topk_and_scatter
from .modules import MLPEncoder
from .whitener import ZCAWhitener


class TiedEncoder(nn.Module):
    """Encoder module that uses decoder weights (tied weights)."""

    def __init__(self, decoder: nn.Linear):
        super().__init__()
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using transposed decoder weights."""
        return torch.nn.functional.linear(x, self.decoder.weight.t())


class RIPTopK(nn.Module):
    """
    TopK Sparse Autoencoder with Restricted Isometry Property (RIP) regularization.

    This extends the standard TopK SAE by adding a regularization term that enforces
    norm preservation on k'-sparse codes (where k' = k * rip_loss_multiplier) in the
    decoder subspace, plus AuxK regularization to encourage good reconstruction with
    fewer features.

    Args:
        input_shape: Dimension of input data (int or tuple)
        nb_concepts: Number of concepts/features in sparse representation
        top_k: Number of top activations to keep per sample
        rip_weight: Weight for RIP regularization term (default: 0.1)
        rip_loss_weighted: Use original code values (True) or Gaussian substitutes (False) (default: False)
        rip_loss_multiplier: RIP dimension multiplier (k' = k * multiplier) (default: 2)
        use_abstopk: Whether to use magnitude-based TopK (True) or positive-only TopK (False) (default: True)
        auxiliary_k: Number of top features for auxiliary reconstruction (default: None, computes as input_dim // 2)
        auxk_weight: Weight for AuxK regularization term (default: 1/32)
        device: Device to use for computation (default: 'cpu')
        normalization: Normalization mode for dictionary ('identity' or 'l2', default: 'l2')
        activation_window_batches: Number of batches to track in sliding window (default: 64)
    """

    def __init__(
        self,
        input_shape: int,
        nb_concepts: int,
        top_k: int = None,
        batch_topk: bool = False,
        matryoshka: bool = False,
        matryoshka_group_sizes: Optional[Tuple[int]] = None,
        rip_weight: float = 0.1,
        rip_loss_weighted: bool = False,
        rip_loss_multiplier: int = 2,
        use_abstopk: bool = True,
        mlp_encoder: bool = False,
        num_encode_steps: int = 5,
        initial_step_size: float = 0.9,
        learned_step_size: bool = True,
        step_size_rank: int = 0,
        auxiliary_k: Optional[int] = None,
        auxk_weight: float = 1.0 / 32,
        l1_weight: float = 0.,
        device: str = 'cpu',
        normalization: Optional[str] = 'l2',
        activation_window_batches: int = 64,
        whiten: bool = False,
    ):
        super().__init__()

        # Extract input_dim and store number of concepts for downstream tooling
        self.input_dim = input_shape if isinstance(input_shape, int) else input_shape[0]
        self.nb_concepts = nb_concepts

        # AuxK parameters
        self.auxiliary_k = auxiliary_k if auxiliary_k is not None else self.input_dim // 2
        self.auxk_weight = auxk_weight

        self.top_k = top_k
        self.rip_weight = rip_weight
        self.rip_loss_weighted = rip_loss_weighted
        self.rip_loss_multiplier = rip_loss_multiplier
        self.l1_weight = l1_weight
        self.use_abstopk = use_abstopk
        self.normalization = normalization
        self.activation_window_batches = activation_window_batches

        # Whitening
        self.whiten = whiten
        if self.whiten:
            self.whitener = ZCAWhitener(dim=self.input_dim, device=device)

        # If true, top_k is applied per batch instead of per sample (BatchTopK)
        self.batch_topk = batch_topk
        self.register_buffer(
            'threshold',
            torch.tensor(0.0, device=device)
        )  # Inference threshold
        self.batch_topk_threshold_ema = 0.99  # EMA factor for threshold updating

        # If true, use Matryoshka reconstruction loss (MatryoshkaTopK/MatryoshkaBatchTopK)
        self.matryoshka = matryoshka
        if self.matryoshka:
            if matryoshka_group_sizes is None:
                matryoshka_group_sizes = (nb_concepts // 4, nb_concepts // 4, nb_concepts // 4, nb_concepts // 4)
            self.group_boundaries = list(np.cumsum(matryoshka_group_sizes))
            assert self.group_boundaries[-1] == nb_concepts, "Matryoshka group sizes must sum to nb_concepts"

        # Observation bias: learned bias applied before encoding and after decoding
        self.observation_bias = nn.Parameter(torch.zeros(self.input_dim, device=device))

        # Encoder: linear or MLP
        if mlp_encoder:
            self.encoder = MLPEncoder(self.input_dim, nb_concepts)
        else:
            self.encoder = nn.Linear(self.input_dim, nb_concepts, bias=False)

        # Encoder: multi-step
        self.num_encode_steps = num_encode_steps
        self.learned_step_size = learned_step_size
        self.step_size_rank = step_size_rank
        # Parameterize as factor F where step_size = I - F
        # If initial_step_size = 0.9, then F = I - 0.9*I = 0.1*I
        if step_size_rank > 0:
            # Low-rank parameterization: F = U @ V.T
            # Initialize U and V such that U @ V.T = (1 - initial_step_size) * I (truncated)
            # The trainer will reinitialize these with proper SVD
            init_scale = np.sqrt(1.0 - initial_step_size)
            self._step_size_U = nn.Parameter(
                torch.eye(nb_concepts, step_size_rank) * init_scale
            )
            self._step_size_V = nn.Parameter(
                torch.eye(nb_concepts, step_size_rank) * init_scale
            )
        else:
            # Full-rank parameterization
            self._step_size_factor = nn.Parameter(
                torch.eye(nb_concepts) * (1.0 - initial_step_size)
            )

        # Decoder: linear layer without bias
        self.decoder = nn.Linear(nb_concepts, self.input_dim, bias=False)

        # Register buffers for tracking activation counts (batch-based sliding window)
        # Running total of activations in current window
        self.register_buffer(
            'activation_counts',
            torch.zeros(nb_concepts, device=device, dtype=torch.long)
        )

        # Circular buffer: stores activation counts per batch
        # Shape: [activation_window_batches, nb_concepts]
        self.register_buffer(
            'activation_history',
            torch.zeros(activation_window_batches, nb_concepts, device=device, dtype=torch.long)
        )

        # Circular buffer pointer (which batch to replace next)
        self.register_buffer(
            'batch_ptr',
            torch.tensor(0, device=device, dtype=torch.long)
        )

        # Total inputs seen (for warmup tracking)
        self.register_buffer(
            'total_inputs_seen',
            torch.tensor(0, device=device, dtype=torch.long)
        )

        # Register buffers for tracking MSE and AuxK normalization factors
        self.register_buffer(
            'mse_normalization_factor',
            torch.tensor(1.0, device=device)
        )
    
    def encode(self, x):
        if self.num_encode_steps <= 1:
            return self._encode_single_step(x)
        
        step_size = self._get_step_size()
        pre_codes, z_topk = self._encode_single_step(x)
        for _ in range(self.num_encode_steps - 1):
            pre_codes_step = z_topk @ step_size + pre_codes
            z_topk = apply_topk_and_scatter(
                pre_codes_step,
                self.top_k,
                use_abstopk=self.use_abstopk,
            )

        return pre_codes_step, z_topk

    def _encode_single_step(self, x):
        """
        Encode input data to latent representation with TopK sparsification in a single
        step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns
        -------
        pre_codes : torch.Tensor
            Pre-codes tensor of shape (batch_size, nb_components) before top-k operation.
        z : torch.Tensor
            Codes, latent representation tensor (z) of shape (batch_size, nb_components).
        """
        # Get pre-activations from encoder
        pre_codes = self.encoder(x)

        # Apply TopK or threshold-based sparsification
        if self.batch_topk:
            if self.training:
                # Training: Use exact BatchTopK
                batch_size = pre_codes.shape[0]
                flat_codes = pre_codes.view(-1)
                z_topk = apply_topk_and_scatter(flat_codes, batch_size * self.top_k, use_abstopk=self.use_abstopk)
                z_topk = z_topk.view(batch_size, -1)

                # Update threshold for inference
                self.update_threshold(z_topk)
            else:
                # Inference: Use learned threshold
                if self.use_abstopk:
                    z_topk = pre_codes * (pre_codes.abs() > self.threshold)
                else:
                    # Apply ReLU to match training behavior (positive-only)
                    pre_codes_positive = torch.relu(pre_codes)
                    z_topk = pre_codes_positive * (pre_codes_positive > self.threshold)
        else:
            # Standard per-sample TopK
            z_topk = apply_topk_and_scatter(pre_codes, self.top_k, use_abstopk=self.use_abstopk)

        return pre_codes, z_topk

    @torch.no_grad()
    def update_threshold(self, z_topk: torch.Tensor):
        """
        At inference time, we need a fixed threshold for consistent inference.

        Args:
            z_topk: A batch of BatchTopK'ed activations of shape [batch_size, nb_concepts]
        """
        # When use_abstopk=True, activations can be negative, so use absolute values
        if self.use_abstopk:
            batch_threshold = z_topk.abs().min()
        else:
            batch_threshold = z_topk.min()
        self.threshold = (1 - self.batch_topk_threshold_ema) * self.threshold + self.batch_topk_threshold_ema * batch_threshold

    def decode(self,
               z: torch.Tensor) -> torch.Tensor:
        """
        Decode via dictionary.

        If training, normalize the dictionary before using it (if normalization != 'identity').

        Args:
            z: Sparse activations of shape [batch_size, nb_concepts]

        Returns:
            x_reconstruction: Reconstructed inputs of shape [batch_size, input_dim]
        """
        def _normalize(d):
            return d / (torch.norm(d, p=2, dim=0, keepdim=True) + 1e-8)

        if self.training and self.normalization == 'l2':
            self.decoder.weight.data = _normalize(self.decoder.weight)

        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with RIP regularization and dead concept tracking.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            pre_activations: Pre-activation values [batch_size, nb_concepts]
            activations: Sparse activations after TopK [batch_size, nb_concepts]
            reconstruction: Reconstructed input [batch_size, input_dim]
        """
        # Whiten input if enabled, then apply observation bias
        if self.whiten:
            x_whitened = self.whitener(x)
            x_centered = x_whitened - self.observation_bias
        else:
            # Apply pre-encoder bias
            x_centered = x - self.observation_bias

        # Encode
        pre_activations, activations = self.encode(x_centered)

        # Decode
        reconstruction = self.decode(activations)

        # Apply observation bias, then unwhiten (or just apply bias if not whitening)
        if self.whiten:
            reconstruction = reconstruction + self.observation_bias
            reconstruction = self.whitener.unwhiten(reconstruction)
        else:
            reconstruction = reconstruction + self.observation_bias

        # Update activation counts (during training only)
        if self.training:
            with torch.no_grad():
                batch_size = activations.shape[0]

                # Count activations: which concepts are active (survived TopK) in this batch
                batch_counts = (activations != 0.).sum(dim=0)  # [nb_concepts]

                # Batch-based circular buffer update
                # 1. Subtract oldest batch from running count
                self.activation_counts -= self.activation_history[self.batch_ptr]

                # 2. Store new batch counts in circular buffer
                self.activation_history[self.batch_ptr] = batch_counts

                # 3. Add new batch to running count
                self.activation_counts += batch_counts

                # 4. Advance circular pointer
                self.batch_ptr = (self.batch_ptr + 1) % self.activation_window_batches

                # 5. Update total inputs seen
                self.total_inputs_seen += batch_size

        return pre_activations, activations, reconstruction
    
    @torch.no_grad()
    def tie_weights(self):
        """Tie encoder and decoder weights."""
        self.encoder_untied = self.encoder
        self.encoder = TiedEncoder(self.decoder)

        # Save original num_encode_steps and set to 1 while tied
        self.num_encode_steps_untied = self.num_encode_steps
        self.num_encode_steps = 1

    @torch.no_grad()
    def untie_weights(self):
        """Untie encoder and decoder weights."""
        if not hasattr(self, 'encoder_untied'):
            raise RuntimeError(
                "Cannot untie weights: encoder_untied not found. "
                "Make sure tie_weights() was called before untie_weights()."
            )

        # Restore original encoder module
        self.encoder = self.encoder_untied

        # Copy current decoder weights (transposed) to encoder
        # This preserves the learned decoder weights in the encoder
        self.encoder.weight.data = self.decoder.weight.data.t().clone()

        # Restore original num_encode_steps
        if hasattr(self, 'num_encode_steps_untied'):
            self.num_encode_steps = self.num_encode_steps_untied
            del self.num_encode_steps_untied

        # Clean up
        del self.encoder_untied

    def _get_step_size(self):
        """Get step size matrix.

        The step size is parameterized as I - F where F is either:
        - Full-rank: F = _step_size_factor
        - Low-rank: F = _step_size_U @ _step_size_V.T
        This ensures the learned factor F is directly optimized.
        """
        if self.learned_step_size:
            if self.step_size_rank > 0:
                # Low-rank parameterization: F = U @ V.T
                nb_concepts = self._step_size_U.shape[0]
                identity = torch.eye(nb_concepts, device=self._step_size_U.device)
                F = self._step_size_U @ self._step_size_V.T
                return identity - F
            else:
                # Full-rank parameterization
                nb_concepts = self._step_size_factor.shape[0]
                identity = torch.eye(nb_concepts, device=self._step_size_factor.device)
                return identity - self._step_size_factor
        else:
            D = self.decoder.weight.detach()
            nb_concepts = D.shape[1]
            return torch.eye(nb_concepts, device=D.device) - 5.e-2 * D.t() @ D

    @torch.no_grad()
    def _initialize_normalization_factor(self, batch: torch.tensor):
        """
        Initialize MSE normalization factor based on a data batch.

        Parameters
        ----------
        batch : torch.Tensor
            Input batch tensor of shape (batch_size, input_dim).
        """
        mu = batch.mean(dim=0, keepdim=True)
        norm_const = torch.mean((batch - mu)**2)
        self.mse_normalization_factor.fill_(norm_const)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Handle backward compatibility when loading old checkpoints."""
        # Check if loading old checkpoint (has activation_frequencies but not activation_counts)
        old_freq_key = prefix + 'activation_frequencies'
        new_count_key = prefix + 'activation_counts'

        if old_freq_key in state_dict and new_count_key not in state_dict:
            print("Loading old checkpoint with EMA-based activation tracking. Initializing count-based buffers to zero.")
            # Don't copy old values, just let new buffers initialize to zero
            # Remove old key to avoid unexpected_keys warning
            state_dict.pop(old_freq_key, None)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def compute_matryoshka_reconstruction_loss(self, x):
        """
        Compute Matryoshka reconstruction loss with normalization.

        Matryoshka training means each nested subset of features should reconstruct well.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            mse_loss: Normalized MSE reconstruction loss (scalar tensor)
        """
        # Store original x for computing reconstruction error
        x_original = x

        # Whiten input if enabled, then apply observation bias
        if self.whiten:
            x_whitened = self.whitener(x)
            x_centered = x_whitened - self.observation_bias
        else:
            x_centered = x - self.observation_bias

        _, z_topk = self.encode(x_centered)

        # Compute reconstruction loss for each nested group
        x_recons = torch.zeros(len(self.group_boundaries), device=x.device, dtype=x.dtype)

        for i, group_end in enumerate(self.group_boundaries):
            # Decode using cumulative features [0:group_end]
            decoder_subset = self.decoder.weight[:, :group_end]
            x_recon = torch.nn.functional.linear(z_topk[:, :group_end], decoder_subset)

            # Apply observation bias, then unwhiten (or just apply bias if not whitening)
            if self.whiten:
                x_recon = x_recon + self.observation_bias
                x_recon = self.whitener.unwhiten(x_recon)
            else:
                x_recon = x_recon + self.observation_bias

            # Compute reconstruction error for this nested level (in original space)
            x_recons[i] = (x_recon - x_original).pow(2).mean()

        # Average across all nested levels and normalize
        mse_loss = x_recons.mean() / self.mse_normalization_factor

        return mse_loss

    def _get_encoder_weights(self) -> torch.Tensor:
        """
        Extract encoder weight matrix from the model.

        Returns:
            Encoder weights E of shape [observed_dim, nb_concepts]
        """
        if isinstance(self.encoder, nn.Linear):
            return self.encoder.weight.T
        elif isinstance(self.encoder, nn.Sequential):
            linear = self.encoder[-1]
            return linear.weight.T
        elif isinstance(self.encoder, MLPEncoder):
            return self.encoder.fc2.weight.T
        else:  # TiedEncoder
            return self.decoder.weight

    def _get_decoder_weights(self) -> torch.Tensor:
        """
        Extract decoder weight matrix from the model.

        Returns:
            Decoder weights D of shape [nb_concepts, input_dim]
        """
        return extract_decoder_weights(self)

    def compute_rip_loss(self, pre_activations: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Restricted Isometry Property (RIP) regularization loss.

        Args:
            pre_activations: Pre-sparsified concept activations [batch_size, nb_concepts]
            x: Input observations [batch_size, observed_dim] (unused, kept for API compatibility)

        Returns:
            RIP loss value (scalar tensor)
        """
        E = self._get_encoder_weights()
        D = self._get_decoder_weights()

        batch_size, nb_concepts = pre_activations.shape
        assert D.shape[0] == nb_concepts, \
            f"Dictionary shape mismatch: pre_activations have {nb_concepts} concepts but dictionary has {D.shape[0]}"

        return compute_rip_loss_fn(
            x,
            pre_activations,
            E,
            D,
            k=self.top_k,
            weighted=self.rip_loss_weighted,
            multiplier=self.rip_loss_multiplier,
            use_abstopk=self.use_abstopk,
        )

    def compute_auxk_loss(self, pre_activations: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute AuxK reconstruction loss using top-K dead concept features.

        Targets dead concepts specifically - selects the top auxiliary_k features
        (by magnitude) from dead concepts only. This encourages the model to maintain
        useful representations for inactive concepts.

        Args:
            pre_activations: Pre-sparsified concept activations [batch_size, nb_concepts]
            x: Target input [batch_size, input_dim]

        Returns:
            AuxK loss value (scalar tensor). Returns 0.0 if no dead concepts exist.
        """
        D = self._get_decoder_weights()

        return compute_auxk_loss_fn(
            pre_activations,
            x,
            D,
            self.auxiliary_k,
            self.activation_counts,
            self.use_abstopk,
        )

    def compute_l1_loss(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 sparsity penalty on activations.

        Computes mean L1 norm per active feature: sum(|activations|) / (batch_size * k).
        This normalization makes the loss consistent across different dictionary sizes.

        Args:
            activations: Sparse activations after TopK [batch_size, nb_concepts]

        Returns:
            L1 loss value (scalar tensor)
        """
        batch_size = activations.shape[0]
        return torch.sum(torch.abs(activations)) / (batch_size * self.top_k)

    def compute_loss(
        self,
        x: torch.Tensor,
        pre_activations: torch.Tensor,
        activations: torch.Tensor,
        reconstruction: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute total loss including reconstruction, RIP, and AuxK regularization.

        Args:
            x: Original input [batch_size, input_dim]
            pre_activations: Non-sparsified activations [batch_size, nb_concepts]
            activations: Sparse activations [batch_size, nb_concepts]
            reconstruction: Reconstructed input [batch_size, input_dim]

        Returns:
            Total loss (scalar)
        """
        # Reconstruction loss (MSE)
        if self.matryoshka:
            recon_loss = self.compute_matryoshka_reconstruction_loss(x)
        else:
            recon_loss = torch.mean((x - reconstruction) ** 2) / self.mse_normalization_factor

        # Add RIP regularization
        if self.rip_weight > 0:
            rip_loss = self.compute_rip_loss(pre_activations, x)
        else:
            rip_loss = 0.0

        # Add AuxK regularization
        if self.auxk_weight > 0:
            residual = x - reconstruction
            # Whiten residual since decoder operates in whitened space
            if self.whiten:
                residual = self.whitener(residual)
            # WARN: `.detach()` here ensures no backprop through whitener for AuxK
            # branch.
            auxk_loss = self.compute_auxk_loss(pre_activations, residual.detach())
        else:
            auxk_loss = 0.0

        # Add L1 penalty
        if self.l1_weight > 0:
            l1_loss = self.compute_l1_loss(activations)
        else:
            l1_loss = 0.0

        total_loss = (recon_loss
                     + self.rip_weight * rip_loss
                     + self.auxk_weight * auxk_loss
                     + self.l1_weight * l1_loss)

        return total_loss

    def get_rip_metrics(self, pre_activations: torch.Tensor = None, x: torch.Tensor = None) -> dict:
        """
        Compute metrics related to the Restricted Isometry Property.

        Args:
            pre_activations: Optional pre-sparsified activations for computing RIP loss [batch_size, nb_concepts]
            x: Optional input samples for computing RIP loss [batch_size, input_dim]

        Returns:
            Dictionary with RIP-related metrics:
                - rip_loss: Current RIP regularization loss (if pre_activations and x provided)
                - coherence: Maximum absolute off-diagonal element of normalized Gram matrix
                - condition_number: Condition number of Gram matrix
        """
        D = self._get_decoder_weights()

        # Compute Gram matrix on decoder: D @ D.T
        gram = D @ D.T

        # Normalize Gram matrix
        norms = torch.sqrt(torch.diagonal(gram))
        gram_normalized = gram / (norms.unsqueeze(1) @ norms.unsqueeze(0) + 1e-8)

        # Coherence: max off-diagonal element
        gram_off_diag = gram_normalized - torch.eye(gram_normalized.shape[0], device=gram.device)
        coherence = torch.max(torch.abs(gram_off_diag)).item()

        # Condition number
        eigenvalues = torch.linalg.eigvalsh(gram)
        condition_number = (torch.max(eigenvalues) / (torch.min(eigenvalues) + 1e-8)).item()

        metrics = {
            'coherence': coherence,
            'condition_number': condition_number,
        }

        # Compute RIP loss if pre_activations and inputs are provided
        if pre_activations is not None and x is not None:
            metrics['rip_loss'] = self.compute_rip_loss(pre_activations, x).item()

        return metrics
