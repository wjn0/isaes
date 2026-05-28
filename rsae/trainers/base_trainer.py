"""Base Trainer class for Sparse Autoencoder training.

This module provides the abstract BaseTrainer class with common training
infrastructure shared by all concrete trainer implementations.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.optim as optim
import mlflow
from omegaconf import DictConfig

from rsae.metrics import (
    compute_reconstruction_metrics_batch,
    compute_coherence,
    compute_max_eigenvalue,
    compute_operator_norm,
    compute_avg_concept_norm,
    compute_rip_loss_for_model,
    compute_activation_statistics,
)
from rsae.baselines.oracle import OMPOracle


def compute_geometric_median(data: torch.Tensor, max_iter: int = 100, tol: float = 1e-5) -> torch.Tensor:
    """
    Compute the geometric median of a set of points using the Weiszfeld algorithm.

    The geometric median is the point that minimizes the sum of Euclidean distances
    to all points in the dataset.

    Parameters
    ----------
    data : torch.Tensor
        Input data of shape (n_samples, n_features)
    max_iter : int
        Maximum number of iterations for the algorithm
    tol : float
        Convergence tolerance

    Returns
    -------
    torch.Tensor
        Geometric median of shape (n_features,)
    """
    # Initialize with the mean
    median = data.mean(dim=0)

    for _ in range(max_iter):
        # Compute distances to all points
        distances = torch.norm(data - median.unsqueeze(0), dim=1, keepdim=True)

        # Avoid division by zero
        distances = torch.clamp(distances, min=1e-8)

        # Compute weights (inverse distances)
        weights = 1.0 / distances

        # Compute new median
        new_median = (data * weights).sum(dim=0) / weights.sum()

        # Check convergence
        if torch.norm(new_median - median) < tol:
            break

        median = new_median
    
    return median


class BaseTrainer(ABC):
    """
    Abstract base trainer for Sparse Autoencoder models.

    This class encapsulates common training infrastructure including:
    - Data preprocessing (scaling, geometric median computation)
    - Model weight initialization
    - Learning rate scheduling (3-phase: warmup → constant → decay)
    - RIP weight scheduling
    - Training loop with logging
    - Common evaluation metrics

    Subclasses must implement:
    - _get_next_batch(): How to iterate through training data
    - evaluate(): Full evaluation including subclass-specific metrics
    - _should_checkpoint(): Determine if checkpoint should be saved
    - _save_checkpoint(): Save model and training state

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object
    model : torch.nn.Module
        SAE model to train
    device : torch.device
        Device to run training on
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        device: torch.device,
    ):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.model_type = cfg.model.type
        self.is_rip_model = self.model_type in ["riptopk", "ripbatchtopk"]

        # Validate tied weights configuration
        self._validate_tied_weights_config()

        # Track whether weights are currently tied
        self.weights_are_tied = False

    # ========================================================================
    # Data Preprocessing
    # ========================================================================

    def _normalize_data(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Normalize observations using per-sample centering and scaling.

        For each sample x: x_normalized = (x - mean(x)) / ||x - mean(x)||

        Parameters
        ----------
        observations : torch.Tensor
            Input data of shape (n_samples, input_dim) or (input_dim,)

        Returns
        -------
        torch.Tensor
            Normalized observations with same shape as input
        """
        # Handle both single samples and batches
        if observations.ndim == 1:
            # Single sample: (input_dim,)
            centered = observations - observations.mean()
            norm = torch.norm(centered)

            # Handle edge case: zero norm after centering
            if norm < 1e-8:
                return torch.zeros_like(observations)

            return centered / norm

        elif observations.ndim == 2:
            # Batch: (n_samples, input_dim)
            # Center each sample (subtract mean across features)
            centered = observations - observations.mean(dim=1, keepdim=True)

            # Compute L2 norm per sample
            norms = torch.norm(centered, dim=1, keepdim=True)

            # Handle edge cases: zero norm samples
            # Replace zero norms with 1 to avoid division by zero
            safe_norms = torch.where(norms < 1e-8, torch.ones_like(norms), norms)

            return centered / safe_norms

        else:
            raise ValueError(f"Expected 1D or 2D tensor, got shape {observations.shape}")

    def _compute_geometric_median(
        self, observations: torch.Tensor, n_samples: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute geometric median from a subsample of observations.

        Parameters
        ----------
        observations : torch.Tensor
            Input data of shape (n_samples, input_dim)
        n_samples : int, optional
            Number of samples to use. If None, uses cfg.training.geometric_median.n_samples

        Returns
        -------
        torch.Tensor
            Geometric median of shape (input_dim,)
        """
        # Get n_samples from config if not provided
        if n_samples is None:
            n_samples = self.cfg.training.get('geometric_median', {}).get('n_samples', 50000)

        # Subsample observations
        n_total = observations.shape[0]
        if n_total > n_samples:
            indices = torch.randperm(n_total)[:n_samples]
            subsample = observations[indices]
        else:
            subsample = observations

        # Compute geometric median on CPU for numerical stability
        subsample_cpu = subsample.cpu()

        # Get tolerance from config
        tol = self.cfg.training.get('geometric_median', {}).get('tolerance', 1e-5)
        geometric_median = compute_geometric_median(subsample_cpu, tol=tol)

        print(f"Computed geometric median from {min(n_total, n_samples)} samples")
        print(f"  Geometric median norm: {torch.norm(geometric_median):.4f}")

        return geometric_median.to(self.device)

    def _validate_tied_weights_config(self):
        """
        Validate that tied weights configuration is compatible with model.

        Raises
        ------
        ValueError
            If tied weights are enabled but model has MLP encoder
            If both tied weights and hot restart are enabled
        """
        # Get configurations
        tied_cfg = self.cfg.model.get('tied_weights', {})
        tied_enabled = tied_cfg.get('enabled', False)
        hot_restart_cfg = self.cfg.model.get('hot_restart', {})
        hot_restart_enabled = hot_restart_cfg.get('enabled', False)

        # Check hot restart compatibility with tied weights
        if tied_enabled and hot_restart_enabled:
            raise ValueError(
                "Hot restart (model.hot_restart.enabled=true) is not compatible "
                "with tied weights (model.tied_weights.enabled=true). "
                "Please disable one of these features."
            )

        if not tied_enabled:
            return  # No further validation needed if tied weights disabled

        # Check if model has mlp_encoder
        mlp_encoder = self.cfg.model.get('mlp_encoder', False)

        if mlp_encoder:
            raise ValueError(
                "Tied weights (model.tied_weights.enabled=true) are only supported "
                "with linear encoder (model.mlp_encoder=false). "
                f"Current configuration has model.mlp_encoder={mlp_encoder}. "
                "Please set model.mlp_encoder=false or disable tied weights."
            )

        # Validate untie_fraction is in valid range
        untie_fraction = tied_cfg.get('untie_fraction', 1.0)
        if not (0.0 <= untie_fraction <= 1.0):
            raise ValueError(
                f"model.tied_weights.untie_fraction must be in [0.0, 1.0], "
                f"got {untie_fraction}"
            )

        print(f"Tied weights validation passed:")
        print(f"  enabled: {tied_enabled}")
        print(f"  untie_fraction: {untie_fraction}")

    def _initialize_model_weights(self, observations: torch.Tensor, preserve_decoder: bool = False):
        """
        Initialize model weights with custom strategy:
        1. Initialize decoder to random unit-norm columns (skipped if preserve_decoder=True)
        2. Tie encoder weights to decoder.T
        3. Initialize step_size factor F = (1/L) * D.T @ D where L ≈ ||D.T @ D||_op
           (truncated SVD form when step_size_rank > 0)
        4. Initialize whitener from observations (if model.whiten=True)
        5. Set observation bias = geometric median of whitened data (or normalized data
           if no whitening)
        6. Tie encoder/decoder weights if model.tied_weights.enabled

        Parameters
        ----------
        observations : torch.Tensor
            Data to use for initialization
        preserve_decoder : bool
            If True, skip decoder initialization and preserve existing weights (for hot restart)
        """
        print("\nInitializing model weights with custom strategy...")

        with torch.no_grad():
            # Access linear layers directly
            encoder_linear = self.model.encoder  # Custom encoder module is Linear
            decoder_linear = self.model.decoder  # DictionaryLayer acts like a linear layer

            # Store original shapes for assertion
            original_encoder_shape = encoder_linear.weight.shape
            original_decoder_shape = decoder_linear.weight.shape

            print(f"  Original encoder weight shape: {original_encoder_shape}")
            print(f"  Original decoder weight shape: {original_decoder_shape}")

            # Initialize decoder weights (skip if preserving during hot restart)
            if not preserve_decoder:
                D_init = torch.randn_like(decoder_linear.weight.data)
                assert D_init.shape[1] > D_init.shape[0], "Dictionary not overcomplete!"
                decoder_linear.weight.data = D_init / torch.norm(D_init, p=2, dim=0, keepdim=True)

                print(f"  Initialized decoder weights to random directions")
                print(f"    Mean l2 norm of decoder weights: {torch.norm(decoder_linear.weight.data, dim=0).mean():.4f}")
            else:
                print(f"  Preserving existing decoder weights")
                print(f"    Mean l2 norm of decoder weights: {torch.norm(decoder_linear.weight.data, dim=0).mean():.4f}")

            # Set encoder weights = decoder weights transposed
            encoder_linear.weight.data = decoder_linear.weight.data.clone().T
            print(f"  Set encoder weights = decoder weights^T")

            # Initialize step_size_factor F where step_size = I - F
            # Set F = (1/L) * (D.T @ D) where L is the largest eigenvalue of D.T @ D
            D = decoder_linear.weight.data  # [input_dim, nb_concepts]
            gram_matrix = D.T @ D  # [nb_concepts, nb_concepts]

            # Compute largest eigenvalue
            eigenvalues = torch.linalg.eigvalsh(gram_matrix)  # Returns eigenvalues in ascending order
            L = eigenvalues.max().item() * 1.1  # Largest eigenvalue

            # Compute initial F = (1/L) * gram_matrix
            F_init = (1.0 / L) * gram_matrix

            # Check if using low-rank parameterization
            step_size_rank = getattr(self.model, 'step_size_rank', 0)
            if step_size_rank > 0:
                # Low-rank parameterization: F = U @ V.T
                # Compute truncated SVD of F_init
                U_full, S, Vh_full = torch.linalg.svd(F_init, full_matrices=False)

                # Truncate to step_size_rank
                U_truncated = U_full[:, :step_size_rank]  # [nb_concepts, rank]
                S_truncated = S[:step_size_rank]  # [rank]
                V_truncated = Vh_full[:step_size_rank, :].T  # [nb_concepts, rank]

                # Scale U and V by sqrt of singular values so they're similar scale
                sqrt_S = torch.sqrt(S_truncated)  # [rank]
                self.model._step_size_U.data = U_truncated * sqrt_S.unsqueeze(0)  # [nb_concepts, rank]
                self.model._step_size_V.data = V_truncated * sqrt_S.unsqueeze(0)  # [nb_concepts, rank]

                print(f"  Initialized step_size low-rank factors (rank={step_size_rank})")
                print(f"    Largest eigenvalue of D.T @ D: {L:.4f}")
                print(f"    Top {step_size_rank} singular values: {S_truncated.tolist()}")
                print(f"    Approximation error: {torch.norm(F_init - self.model._step_size_U.data @ self.model._step_size_V.data.T).item():.6f}")
            else:
                # Full-rank parameterization
                self.model._step_size_factor.data = F_init

                print(f"  Initialized step_size_factor matrix")
                print(f"    Largest eigenvalue of D.T @ D: {L:.4f}")
                print(f"    step_size_factor = (1/{L:.4f}) * D.T @ D")

            # Initialize whitening transform FIRST (before geometric median computation)
            if self.model.whiten:
                n_whitening_samples = self.cfg.training.get('whitening', {}).get('n_samples', 50000)
                # Use raw observations for whitening statistics
                whitening_data = observations[:n_whitening_samples].to(self.device)
                self.model.whitener.initialize_from_data(whitening_data)
                print(f"  Initialized whitening transform from {whitening_data.shape[0]} samples")
                print(f"    Mean norm: {torch.norm(self.model.whitener.running_mean):.4f}")
                print(f"    Cov trace: {self.model.whitener.running_cov.trace():.4f}")

                # Compute geometric median on WHITENED observations
                n_gm_samples = self.cfg.training.get('geometric_median', {}).get('n_samples', 50000)
                gm_data = observations[:n_gm_samples].to(self.device)
                whitened_obs = self.model.whitener.whiten(gm_data)
                geometric_median = self._compute_geometric_median(whitened_obs, n_samples=n_gm_samples)
                print(f"  Computed geometric median from {n_gm_samples} WHITENED samples")
            else:
                # No whitening: compute geometric median on normalized observations
                normalized_obs = self._normalize_data(observations)
                geometric_median = self._compute_geometric_median(normalized_obs, n_samples=50000)

            # Set observation bias = geometric_median (of whitened or normalized data)
            self.model.observation_bias.copy_(
                self.model.observation_bias.new_tensor(geometric_median)
            )
            print(f"  Set observation bias = geometric_median")
            print(f"    Observation bias norm: {torch.norm(self.model.observation_bias):.4f}")

            # Tie weights if configured
            tied_cfg = self.cfg.model.get('tied_weights', {})
            if tied_cfg.get('enabled', False):
                print(f"  Tying encoder and decoder weights...")
                original_num_steps = self.model.num_encode_steps
                self.model.tie_weights()
                self.weights_are_tied = True
                print(f"    Weights tied successfully")
                print(f"    Set num_encode_steps: {original_num_steps} -> 1 (will restore on untie)")

                untie_fraction = tied_cfg.get('untie_fraction', 1.0)
                if untie_fraction < 1.0:
                    max_steps = self.cfg.training.max_steps
                    untie_step = int(untie_fraction * max_steps)
                    print(f"    Weights will untie at step {untie_step} (fraction={untie_fraction})")
                else:
                    print(f"    Weights will remain tied throughout training")

    # ========================================================================
    # Learning Rate Scheduling
    # ========================================================================

    def _create_lr_scheduler(self, optimizer: optim.Optimizer) -> optim.lr_scheduler.LambdaLR:
        """
        Create 3-phase learning rate scheduler:
        - Phase 1 (warmup): Linear ramp from 0 to full LR
        - Phase 2 (constant): Constant LR
        - Phase 3 (decay): Linear decay to 0

        Parameters
        ----------
        optimizer : optim.Optimizer
            Optimizer to schedule

        Returns
        -------
        optim.lr_scheduler.LambdaLR
            Learning rate scheduler
        """
        warmup_steps = self.cfg.training.get('lr_schedule', {}).get('warmup_steps', 1000)
        decay_start_fraction = self.cfg.training.get('lr_schedule', {}).get('decay_start_fraction', 0.8)
        max_steps = self.cfg.training.max_steps
        decay_start_step = int(decay_start_fraction * max_steps)

        def lr_lambda(current_step):
            """
            Learning rate schedule:
            - Warmup: 0 to warmup_steps - linear warmup from 0 to 1
            - Constant: warmup_steps to decay_start_step - constant at 1
            - Decay: decay_start_step to max_steps - linear decay from 1 to 0
            """
            if current_step < warmup_steps:
                # Warmup phase
                return (current_step + 1) / warmup_steps
            elif current_step < decay_start_step or decay_start_step == max_steps:
                # Constant phase
                return 1.0
            else:
                # Decay phase
                return (max_steps - current_step) / (max_steps - decay_start_step)

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _store_model_config(self):
        """
        Store model configuration parameters for hot restart.

        Extracts configuration directly from the model object to ensure
        consistency between trainers.
        """
        model = self.model
        cfg = self.cfg

        # Get nb_concepts from encoder output dimension
        if hasattr(model.encoder, 'out_features'):
            nb_concepts = model.encoder.out_features
        else:
            # MLPEncoder case
            nb_concepts = model.encoder.fc2.out_features

        # Get initial_step_size from diagonal of step_size_factor
        # step_size = I - F, so initial_step_size = 1 - F_diag_mean
        step_size_rank = getattr(model, 'step_size_rank', 0)
        if step_size_rank > 0:
            # Low-rank: F = U @ V.T, compute F and take diagonal mean
            F = model._step_size_U.data @ model._step_size_V.data.T
            initial_step_size = 1.0 - F.diag().mean().item()
        else:
            initial_step_size = 1.0 - model._step_size_factor.data.diag().mean().item()

        # Get matryoshka_group_sizes if applicable
        matryoshka_group_sizes = None
        if model.matryoshka:
            boundaries = model.group_boundaries
            matryoshka_group_sizes = tuple(
                boundaries[i] - (boundaries[i-1] if i > 0 else 0)
                for i in range(len(boundaries))
            )

        self._model_config = {
            'input_shape': model.input_dim,
            'nb_concepts': nb_concepts,
            'top_k': model.top_k,
            'batch_topk': model.batch_topk,
            'matryoshka': model.matryoshka,
            'matryoshka_group_sizes': matryoshka_group_sizes,
            'rip_weight': model.rip_weight,
            'rip_loss_weighted': model.rip_loss_weighted,
            'rip_loss_multiplier': model.rip_loss_multiplier,
            'use_abstopk': model.use_abstopk,
            'mlp_encoder': cfg.model.get('mlp_encoder', False),
            'num_encode_steps': model.num_encode_steps,
            'initial_step_size': initial_step_size,
            'learned_step_size': model.learned_step_size,
            'step_size_rank': getattr(model, 'step_size_rank', 0),
            'auxiliary_k': model.auxiliary_k,
            'auxk_weight': model.auxk_weight,
            'l1_weight': model.l1_weight,
            'device': str(self.device),
            'normalization': model.normalization,
            'activation_window_batches': model.activation_window_batches,
            'whiten': model.whiten,
        }

    # ========================================================================
    # Hot Restart
    # ========================================================================

    def _perform_hot_restart(self, optimizer: optim.Optimizer,
                             scheduler: optim.lr_scheduler.LambdaLR,
                             step: int) -> tuple:
        """
        Perform hot restart by recreating the model object with preserved decoder.

        This creates a brand new RIPTopK model and reinitializes all parameters
        except the decoder (dictionary), which is preserved from the current model.

        Requires subclasses to set:
        - self._model_config: dict of model constructor kwargs
        - self._stored_init_sample: tensor of initialization samples

        Returns
        -------
        tuple
            (new_optimizer, scheduler)
        """
        print(f"\n{'='*80}")
        print(f"HOT RESTART at step {step}")
        print(f"Recreating model object with preserved decoder")
        print(f"{'='*80}")

        # Import here to avoid circular dependency
        from rsae.rip_topk import RIPTopK

        with torch.no_grad():
            # 1. Preserve decoder weights from current model
            preserved_decoder = self.model.decoder.weight.data.clone()
            print(f"  Preserved decoder: shape={preserved_decoder.shape}, "
                  f"mean_norm={torch.norm(preserved_decoder, dim=0).mean():.4f}")

            # 1b. Preserve whitener state if applicable
            preserved_whitener_state = None
            if self.model.whiten:
                preserved_whitener_state = {
                    'running_mean': self.model.whitener.running_mean.clone(),
                    'running_cov': self.model.whitener.running_cov.clone(),
                    'W': self.model.whitener.W.clone(),
                }
                print(f"  Preserved whitener state")

            # 2. Create brand new model with same configuration
            new_model = RIPTopK(**self._model_config).to(self.device)
            print(f"  Created new RIPTopK model")

            # 3. Load preserved decoder into new model
            new_model.decoder.weight.data.copy_(preserved_decoder)
            print(f"  Loaded preserved decoder into new model")

            # 4. Reinitialize all other parameters using stored init sample
            # Normalize the stored initialization sample
            init_sample = self._stored_init_sample.to(self.device)
            normalized_init_sample = self._normalize_data(init_sample)

            # Call initialization with preserve_decoder=True
            old_model = self.model
            self.model = new_model
            self._initialize_model_weights(normalized_init_sample, preserve_decoder=True)
            del old_model  # Free memory
            print(f"  Reinitialized encoder, observation_bias, and step_size")
            print(f"  Replaced model object")

            # 4b. Restore whitener state if applicable
            if preserved_whitener_state is not None:
                self.model.whitener.running_mean.copy_(preserved_whitener_state['running_mean'])
                self.model.whitener.running_cov.copy_(preserved_whitener_state['running_cov'])
                self.model.whitener.W.copy_(preserved_whitener_state['W'])
                print(f"  Restored whitener state")

        # 5. Reinitialize MSE normalization factor on next forward pass
        # (it will be recomputed automatically since it's a new model)
        print(f"  MSE normalization will be recomputed on next batch")

        # 6. Create new optimizer for new model
        new_optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.cfg.training.optimizer.lr,
            betas=(0.9, 0.999)
        )
        print(f"  Created fresh optimizer for new model")
        print(f"  Preserving LR scheduler (current_lr={scheduler.get_last_lr()[0]:.6f})")
        print(f"{'='*80}\n")

        return new_optimizer, scheduler

    # ========================================================================
    # Training Loop
    # ========================================================================

    def train(self) -> torch.nn.Module:
        """
        Train the model with the configured hyperparameters.

        Returns
        -------
        torch.nn.Module
            Trained model
        """
        # Setup optimizer with specified Adam betas
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.cfg.training.optimizer.lr,
            betas=(0.9, 0.999)  # Adam defaults
        )

        # Setup learning rate scheduler
        scheduler = self._create_lr_scheduler(optimizer)

        # Training loop parameters
        max_steps = self.cfg.training.max_steps
        log_interval = self.cfg.training.log_interval
        eval_interval = self.cfg.training.eval_interval

        # Get hot restart configuration
        hot_restart_cfg = self.cfg.model.get('hot_restart', {})
        hot_restart_enabled = hot_restart_cfg.get('enabled', False)
        hot_restart_interval = hot_restart_cfg.get('interval', 10000)

        print(f"\nTraining for {max_steps:,} steps...")
        print(f"  Log interval: {log_interval}")
        print(f"  Eval interval: {eval_interval}")

        # Get non-iterated warmup configuration
        noniterated_warmup_steps = self.cfg.model.get('noniterated_warmup_steps', -1)
        original_num_encode_steps = None
        if noniterated_warmup_steps >= 0:
            if not hasattr(self.model, 'num_encode_steps'):
                raise ValueError("noniterated_warmup_steps requires a model with num_encode_steps")
            if self.model.num_encode_steps <= 1:
                raise ValueError(
                    f"noniterated_warmup_steps >= 0 requires num_encode_steps > 1, "
                    f"but got num_encode_steps={self.model.num_encode_steps}"
                )
            original_num_encode_steps = self.model.num_encode_steps
            self.model.num_encode_steps = 1
            print(f"  Non-iterated warmup: using num_encode_steps=1 for first {noniterated_warmup_steps} steps")

        # Track if we need to reinitialize normalization factor
        need_normalization_init = False

        for step in range(max_steps):
            self.model.train()

            # Get batch from subclass
            batch = self._get_next_batch()

            if step == 0 or need_normalization_init:
                self.model._initialize_normalization_factor(batch)
                print("  Initialized model normalization factor: {:.4f}".format(self.model.mse_normalization_factor.item()))
                need_normalization_init = False

            # Forward pass
            # Extract device type for autocast (needs "cuda" or "cpu", not "cuda:0" or torch.device)
            if isinstance(self.device, torch.device):
                device_type = self.device.type
            elif isinstance(self.device, str):
                device_type = self.device.split(':')[0]  # Extract "cuda" from "cuda:0"
            else:
                device_type = str(self.device)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                optimizer.zero_grad()
                pre_activations, activations, reconstruction = self.model(batch)

                # Compute loss
                if self.is_rip_model:
                    loss = self.model.compute_loss(batch, pre_activations, activations, reconstruction)
                else:
                    loss = torch.mean((batch - reconstruction) ** 2)

                # Backward pass
                loss.backward()

                # Compute gradient norms before clipping (on logging steps)
                gradient_norms = {}
                if step % log_interval == 0:
                    gradient_norms = self._compute_gradient_norms()

                # Clip gradient norms
                max_grad_norm = self.cfg.training.get('max_grad_norm', 1.0)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                optimizer.step()

            # Check if we should untie weights at this step
            if self.weights_are_tied:
                tied_cfg = self.cfg.model.get('tied_weights', {})
                untie_fraction = tied_cfg.get('untie_fraction', 1.0)
                untie_step = int(untie_fraction * max_steps)

                # Untie at the START of the untie_step (before processing that step's batch)
                # This way the model is already untied when we process the untie_step
                if step == untie_step:
                    print(f"\n{'='*80}")
                    print(f"UNTYING WEIGHTS at step {step} (fraction={untie_fraction:.3f})")
                    print(f"{'='*80}")
                    self.model.untie_weights()
                    self.weights_are_tied = False
                    print(f"Weights untied successfully. Encoder and decoder are now independent.")
                    print(f"Restored num_encode_steps to {self.model.num_encode_steps}")
                    print(f"{'='*80}\n")

            # Check if we should end non-iterated warmup
            if noniterated_warmup_steps >= 0 and step == noniterated_warmup_steps:
                self.model.num_encode_steps = original_num_encode_steps
                print(f"\n{'='*80}")
                print(f"NON-ITERATED WARMUP COMPLETE at step {step}")
                print(f"Restored num_encode_steps to {self.model.num_encode_steps}")
                print(f"{'='*80}\n")
                noniterated_warmup_steps = -1  # Disable further checks

            scheduler.step()

            # Logging
            if step % log_interval == 0:
                self._log_training_metrics(step, batch, pre_activations.float(), activations.float(), reconstruction.float(), loss.float(), scheduler, gradient_norms)

            # Periodic full evaluation
            if step % eval_interval == 0 and step > 0:
                eval_metrics = self.evaluate()
                if self.cfg.mlflow.enabled:
                    mlflow.log_metrics(eval_metrics, step=step)

                print(f"Step {step}/{max_steps} - Loss: {loss.item():.4f}")

                # Reset to training mode after evaluation
                self.model.train()

            # Checkpointing
            if self._should_checkpoint(step):
                self._save_checkpoint(step, optimizer, scheduler)

            # Check if we should perform hot restart
            if hot_restart_enabled and step > 0 and step % hot_restart_interval == 0:
                optimizer, scheduler = self._perform_hot_restart(optimizer, scheduler, step)
                need_normalization_init = True  # Reinit normalization on next batch

        return self.model

    def _log_training_metrics(
        self,
        step: int,
        batch: torch.Tensor,
        pre_activations: torch.Tensor,
        activations: torch.Tensor,
        reconstruction: torch.Tensor,
        loss: torch.Tensor,
        scheduler: optim.lr_scheduler.LambdaLR,
        gradient_norms: Dict[str, float] = None,
    ):
        """Log training metrics to MLflow."""
        with torch.no_grad():
            metrics = {
                "train/step": step,
                "train/loss": loss.item(),
                "train/lr": scheduler.get_last_lr()[0],  # Get current LR from scheduler
            }

            if self.is_rip_model:
                recon_loss = torch.mean((batch - reconstruction.detach()) ** 2) / self.model.mse_normalization_factor.item()
                rip_loss_value = self.model.compute_rip_loss(pre_activations.detach(), batch)
                # Compute residual and whiten if needed (decoder operates in whitened space)
                residual = batch - reconstruction.detach()
                if self.model.whiten:
                    residual = self.model.whitener(residual)
                auxk_loss_value = self.model.compute_auxk_loss(pre_activations.detach(), residual)
                l1_loss_value = self.model.compute_l1_loss(activations.detach())

                metrics["train/recon_loss"] = recon_loss.item()
                metrics["train/rip_loss"] = rip_loss_value.item()
                metrics["train/auxk_loss"] = auxk_loss_value.item()
                metrics["train/l1_loss"] = l1_loss_value.item()
                metrics["train/auxk_weight"] = self.model.auxk_weight

                # Log dead concept statistics if tracking is enabled
                if hasattr(self.model, 'activation_counts'):
                    act_counts = self.model.activation_counts
                    total_inputs = max(self.model.total_inputs_seen.item(), 1)  # Avoid division by zero

                    # Dead concept statistics
                    dead_mask = act_counts == 0
                    metrics["train/dead_concept_count"] = dead_mask.sum().item()
                    metrics["train/dead_concept_proportion"] = dead_mask.float().mean().item()

                    # Activation count statistics
                    metrics["train/min_activation_count"] = act_counts.min().item()
                    metrics["train/max_activation_count"] = act_counts.max().item()
                    metrics["train/mean_activation_count"] = act_counts.float().mean().item()

                    # Activation rate statistics (counts / window_size for interpretability)
                    window_size = min(total_inputs, self.model.activation_window_batches * 16384)  # Approximate window size
                    act_rates = act_counts.float() / max(window_size, 1)
                    metrics["train/min_activation_rate"] = act_rates.min().item()
                    metrics["train/max_activation_rate"] = act_rates.max().item()
                    metrics["train/mean_activation_rate"] = act_rates.mean().item()

                    # Tracking metadata
                    metrics["train/total_inputs_seen"] = total_inputs
                    # Estimate batches seen (assuming constant batch size, approximate)
                    batches_seen = self.model.batch_ptr.item()
                    if batches_seen == 0 and total_inputs > 0:
                        # Haven't wrapped around yet
                        batches_seen = min(total_inputs // 1024, self.model.activation_window_batches)  # Rough estimate
                    metrics["train/window_fill_fraction"] = min(batches_seen / max(self.model.activation_window_batches, 1), 1.0)

            # Add gradient norms if provided
            if gradient_norms:
                metrics.update(gradient_norms)

            if self.cfg.mlflow.enabled:
                mlflow.log_metrics(metrics, step=step)

    def _compute_gradient_norms(self) -> Dict[str, float]:
        """Compute L2 norms of gradients for key RIPTopK parameters."""
        if not self.is_rip_model:
            return {}

        norms = {}

        # Decoder gradient norm
        if self.model.decoder.weight.grad is not None:
            norms["train/grad_norm/decoder"] = self.model.decoder.weight.grad.norm().item()

        # Encoder gradient norm
        if hasattr(self.model.encoder, 'weight') and self.model.encoder.weight.grad is not None:
            norms["train/grad_norm/encoder"] = self.model.encoder.weight.grad.norm().item()

        # Observation bias gradient norm
        if self.model.observation_bias.grad is not None:
            norms["train/grad_norm/observation_bias"] = self.model.observation_bias.grad.norm().item()

        # Step size factor gradient norm (only if learned)
        if self.model.learned_step_size:
            step_size_rank = getattr(self.model, 'step_size_rank', 0)
            if step_size_rank > 0:
                # Low-rank parameterization
                if self.model._step_size_U.grad is not None:
                    norms["train/grad_norm/step_size_U"] = self.model._step_size_U.grad.norm().item()
                if self.model._step_size_V.grad is not None:
                    norms["train/grad_norm/step_size_V"] = self.model._step_size_V.grad.norm().item()
            else:
                # Full-rank parameterization
                if self.model._step_size_factor.grad is not None:
                    norms["train/grad_norm/step_size_factor"] = self.model._step_size_factor.grad.norm().item()

        return norms

    # ========================================================================
    # Common Evaluation Metrics
    # ========================================================================

    def _compute_common_metrics(self, observations: torch.Tensor) -> Dict[str, float]:
        """
        Compute evaluation metrics common to all trainers.

        Parameters
        ----------
        observations : torch.Tensor
            Data to evaluate on (must be already normalized by subclass)

        Returns
        -------
        Dict[str, float]
            Dictionary of common evaluation metrics
        """
        batch_size = self.cfg.training.batch_size
        metrics = {}

        with torch.no_grad():
            # Reconstruction metrics - computed in single pass for efficiency
            print("  Computing reconstruction metrics (MSE + variance + sparsity)...")
            t0 = time.time()
            recon_metrics = compute_reconstruction_metrics_batch(
                self.model, observations, batch_size
            )
            recon_time = time.time() - t0

            # Unpack metrics
            metrics["eval/recon_mse"] = recon_metrics['reconstruction_mse'] / self.model.mse_normalization_factor.item()
            metrics["eval/explained_variance"] = recon_metrics['explained_variance']
            metrics["eval/sparsity"] = recon_metrics['sparsity']
            print(f"    -> {recon_time:.3f}s (combined MSE + variance + sparsity)")

            # Dictionary properties
            print("  Computing coherence...")
            t0 = time.time()
            metrics["eval/coherence"] = compute_coherence(self.model)
            print(f"    -> {time.time() - t0:.3f}s")

            print("  Computing max_eigenvalue...")
            t0 = time.time()
            metrics["eval/max_eigenvalue"] = compute_max_eigenvalue(self.model)
            print(f"    -> {time.time() - t0:.3f}s")

            print("  Computing operator_norm...")
            t0 = time.time()
            metrics["eval/operator_norm"] = compute_operator_norm(self.model)
            print(f"    -> {time.time() - t0:.3f}s")

            print("  Computing avg_concept_norm...")
            t0 = time.time()
            metrics["eval/avg_concept_norm"] = compute_avg_concept_norm(self.model)
            print(f"    -> {time.time() - t0:.3f}s")

            # RIP loss
            print("  Computing rip_loss...")
            t0 = time.time()
            eval_batch = observations[:batch_size].to(self.device)
            eval_pre_acts, eval_acts, _ = self.model(eval_batch)
            metrics["eval/rip_loss"] = compute_rip_loss_for_model(self.model, eval_pre_acts, eval_batch)
            print(f"    -> {time.time() - t0:.3f}s")

            # Activation statistics
            print("  Computing activation_statistics...")
            t0 = time.time()
            activation_stats = compute_activation_statistics(self.model, observations, batch_size)
            for stat_name, stat_value in activation_stats.items():
                metrics[f"eval/{stat_name}"] = stat_value
            print(f"    -> {time.time() - t0:.3f}s")

            # Oracle baseline metrics
            print("  Computing oracle baseline metrics...")
            t0 = time.time()
            oracle_metrics = self._compute_oracle_metrics(observations)
            metrics.update(oracle_metrics)
            print(f"    -> {time.time() - t0:.3f}s")

        return metrics

    def _estimate_oracle(self, observations: torch.Tensor) -> tuple:
        """
        Fit an OMP oracle on the learned dictionary and compute baseline metrics.

        This compares the learned encoder to an oracle that uses Orthogonal
        Matching Pursuit (OMP) to find the optimal sparse codes for the current
        learned dictionary.

        Parameters
        ----------
        observations : torch.Tensor
            Data to evaluate on (must be already normalized by subclass)

        Returns
        -------
        tuple
            (oracle, metrics) where oracle is the fitted OMPOracle and metrics
            is a dictionary containing oracle comparison metrics
        """
        metrics = {}
        batch_size = self.cfg.training.batch_size
        eval_batch = observations[:batch_size].to(self.device)

        # Get learned dictionary (decoder weights)
        dictionary = self.model.decoder.weight.data  # (input_dim, nb_concepts)
        k = self.model.top_k

        # Create oracle with learned dictionary
        oracle = OMPOracle(dictionary, k)

        # Apply same preprocessing as model: whiten (if enabled) then center
        observation_bias = self.model.observation_bias
        if self.model.whiten:
            whitened_batch = self.model.whitener(eval_batch)
            centered_batch = whitened_batch - observation_bias
        else:
            centered_batch = eval_batch - observation_bias

        # Oracle encoding and reconstruction (on centered, potentially whitened data)
        oracle_codes, oracle_recon_centered = oracle(centered_batch)

        # Reverse the preprocessing: add bias, then unwhiten (if enabled)
        if self.model.whiten:
            oracle_recon_whitened = oracle_recon_centered + observation_bias
            oracle_recon = self.model.whitener.unwhiten(oracle_recon_whitened)
        else:
            oracle_recon = oracle_recon_centered + observation_bias

        # Model encoding and reconstruction
        _, model_codes, model_recon = self.model(eval_batch)

        # Oracle MSE (normalized same as model)
        oracle_mse = torch.mean((eval_batch - oracle_recon) ** 2).item()
        oracle_mse_normalized = oracle_mse / self.model.mse_normalization_factor.item()
        metrics["eval/oracle_mse"] = oracle_mse_normalized

        # Model MSE for this batch
        model_mse = torch.mean((eval_batch - model_recon) ** 2).item()

        # Relative MSE (model/oracle) - lower is better, 1.0 = optimal
        if model_mse > 0:
            metrics["eval/oracle_mse_relative"] = oracle_mse / model_mse
        else:
            metrics["eval/oracle_mse_relative"] = 1.0

        # Support overlap: fraction of matching active concepts (IoU)
        oracle_support = (oracle_codes != 0)
        model_support = (model_codes != 0)
        intersection = (oracle_support & model_support).sum(dim=1).float()
        union = (oracle_support | model_support).sum(dim=1).float()
        iou = (intersection / union.clamp(min=1)).mean().item()
        metrics["eval/oracle_support_overlap"] = iou

        return oracle, metrics

    def _compute_oracle_metrics(self, observations: torch.Tensor) -> Dict[str, float]:
        """
        Compute oracle baseline metrics using OMP on the learned dictionary.

        Parameters
        ----------
        observations : torch.Tensor
            Data to evaluate on (must be already normalized by subclass)

        Returns
        -------
        Dict[str, float]
            Dictionary containing oracle comparison metrics
        """
        _, metrics = self._estimate_oracle(observations)
        return metrics

    # ========================================================================
    # Abstract Methods (must be implemented by subclasses)
    # ========================================================================

    @abstractmethod
    def _get_next_batch(self) -> torch.Tensor:
        """
        Get next training batch.

        Returns
        -------
        torch.Tensor
            Batch tensor on correct device and correct dtype, shape (batch_size, input_dim)
        """
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """
        Compute full evaluation metrics including subclass-specific metrics.

        Returns
        -------
        Dict[str, float]
            Dictionary of evaluation metrics
        """
        pass

    @abstractmethod
    def _should_checkpoint(self, step: int) -> bool:
        """
        Determine if checkpoint should be saved at current step.

        Parameters
        ----------
        step : int
            Current training step

        Returns
        -------
        bool
            True if checkpoint should be saved
        """
        pass

    @abstractmethod
    def _save_checkpoint(self, step: int, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LambdaLR):
        """
        Save training checkpoint.

        Parameters
        ----------
        step : int
            Current training step
        optimizer : optim.Optimizer
            Optimizer state
        scheduler : optim.lr_scheduler.LambdaLR
            LR scheduler state
        """
        pass
