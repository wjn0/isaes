"""Synthetic Trainer for SAE training with ground truth comparisons.

This module provides the SyntheticTrainer class for training on synthetic data
with comprehensive ground truth identifiability metrics.
"""

from typing import Dict, Optional
import torch
import torch.optim as optim
from omegaconf import DictConfig

from .base_trainer import BaseTrainer
from rsae.metrics import (
    compute_gt_dictionary_mcc,
)


class SyntheticTrainer(BaseTrainer):
    """
    Trainer for synthetic data with ground truth comparisons.

    Supports two modes:
    1. Traditional mode: Pre-loaded tensors (observations, concepts, mixing_matrix)
    2. Online mode: Generate batches on-the-fly using SyntheticDataGenerator

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object
    model : torch.nn.Module
        SAE model to train
    observations : torch.Tensor or None
        Training data observations (None for online mode)
    concepts : torch.Tensor or None
        Ground truth concept activations (for eval, can be None in online mode)
    mixing_matrix : torch.Tensor
        Ground truth dictionary matrix (always required)
    device : torch.device
        Device to run training on
    generator : SyntheticDataGenerator, optional
        Generator for online mode (if observations is None)
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        observations: Optional[torch.Tensor],
        concepts: Optional[torch.Tensor],
        mixing_matrix: torch.Tensor,
        device: torch.device,
        generator: Optional['SyntheticDataGenerator'] = None,
    ):
        # Store mixing matrix and generator
        self.mixing_matrix = mixing_matrix
        self.generator = generator

        # Initialize weights and store init sample for hot restarts
        # Collect multiple batches for proper whitening/geometric median initialization
        if generator is not None:
            # Online mode: collect multiple batches for initialization
            init_sample = self._collect_initialization_sample_static(generator, None, cfg)
            self._stored_init_sample = init_sample.cpu()  # Store for hot restarts
            self._initialize_model_weights_impl(model, init_sample, device, cfg)
        else:
            # Traditional mode: collect from observations and normalize
            init_sample = self._collect_initialization_sample_static(None, observations, cfg)
            normalized_init_sample = self._normalize_data_static(init_sample)
            self._stored_init_sample = init_sample.cpu()  # Store unnormalized for hot restarts
            self._initialize_model_weights_impl(model, normalized_init_sample, device, cfg)

        # Call parent constructor
        super().__init__(cfg=cfg, model=model, device=device)

        # Store model config for hot restarts
        self._store_model_config()

        # Store data
        if generator is not None:
            # Online mode: no pre-loaded observations
            self.observations = None
            self.concepts = concepts  # Can be None
        else:
            # Traditional mode: store normalized data (normalize full dataset for training)
            device_obj = torch.device(device) if isinstance(device, str) else device
            normalized_observations = self._normalize_data(observations)
            self.observations = normalized_observations.to(device_obj)
            self.concepts = concepts

        # Current iteration state (for epoch-level shuffling in traditional mode)
        self.current_epoch_indices = None
        self.current_batch_idx = 0

    def _initialize_model_weights_impl(
        self,
        model: torch.nn.Module,
        observations: torch.Tensor,
        device: torch.device,
        cfg: DictConfig,
        preserve_decoder: bool = False
    ):
        """
        Helper to initialize weights before super().__init__().

        This is needed because we want to initialize weights before calling
        the parent constructor, but we can't call self._initialize_model_weights()
        before super().__init__() sets up self.cfg, self.model, etc.

        Parameters
        ----------
        preserve_decoder : bool
            If True, skip decoder initialization (for hot restart)
        """
        # Temporarily set up what we need for initialization
        self.cfg = cfg
        self.model = model
        self.device = device
        self._initialize_model_weights(observations, preserve_decoder=preserve_decoder)

    def _normalize_data_static(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Static wrapper for _normalize_data callable before super().__init__().

        This is needed because we want to normalize data before calling the parent
        constructor, but we can't call instance methods before super().__init__().
        """
        return BaseTrainer._normalize_data(self, observations)

    def _collect_initialization_sample_static(
        self,
        generator,
        observations: torch.Tensor,
        cfg: DictConfig
    ) -> torch.Tensor:
        """
        Collect sample for initialization (whitening + geometric median).

        Works in both online mode (generator) and traditional mode (observations).
        Uses multiple batches to collect enough samples, similar to TransformerTrainer.

        Parameters
        ----------
        generator : SyntheticDataGenerator or None
            Generator for online mode
        observations : torch.Tensor or None
            Pre-loaded observations for traditional mode
        cfg : DictConfig
            Configuration object

        Returns
        -------
        torch.Tensor
            Collected samples for initialization
        """
        gm_samples = cfg.training.get('geometric_median', {}).get('n_samples', 50000)
        whitening_samples = cfg.training.get('whitening', {}).get('n_samples', 50000)
        n_samples = max(gm_samples, whitening_samples)
        batch_size = cfg.training.batch_size

        if generator is not None:
            # Online mode: generate multiple batches
            samples = []
            collected = 0
            print(f"Collecting {n_samples} samples for initialization (GM: {gm_samples}, whitening: {whitening_samples})...")
            while collected < n_samples:
                batch, _ = generator.generate_batch(batch_size)
                samples.append(batch.cpu())
                collected += batch.shape[0]
            result = torch.cat(samples, dim=0)[:n_samples]
        else:
            # Traditional mode: sample from observations
            if observations.shape[0] >= n_samples:
                result = observations[:n_samples]
            else:
                print(f"Warning: Only {observations.shape[0]} samples available (requested {n_samples})")
                result = observations

        print(f"Collected {result.shape[0]} samples for initialization")
        return result

    # ========================================================================
    # Data Iteration
    # ========================================================================

    def _get_next_batch(self) -> torch.Tensor:
        """
        Get next batch.

        Returns
        -------
        torch.Tensor
            Batch tensor on correct device
        """
        batch_size = self.cfg.training.batch_size

        if self.generator is not None:
            # Online mode: generate fresh batch (already on device)
            observations, _ = self.generator.generate_batch(batch_size)
            return observations.float()
        else:
            # Traditional mode: epoch-level shuffling
            data_size = self.observations.shape[0]

            # Check if we need to start a new epoch
            if self.current_epoch_indices is None or self.current_batch_idx >= data_size:
                # New epoch: reshuffle
                self.current_epoch_indices = torch.randperm(data_size)
                self.current_batch_idx = 0

            # Get batch
            start = self.current_batch_idx
            end = min(start + batch_size, data_size)
            indices = self.current_epoch_indices[start:end]
            self.current_batch_idx = end

            return self.observations[indices].to(self.device).float()

    # ========================================================================
    # Evaluation
    # ========================================================================

    def evaluate(self) -> Dict[str, float]:
        """
        Compute full evaluation including ground truth metrics.

        Returns
        -------
        Dict[str, float]
            Dictionary of evaluation metrics
        """
        self.model.eval()
        metrics = {}

        with torch.no_grad():
            # Get evaluation batch
            if self.generator is not None:
                # Online mode: generate eval batch (already on device)
                eval_batch, eval_concepts = self.generator.generate_batch(
                    batch_size=min(self.cfg.training.batch_size * 10, 10000)
                )
                # Already on correct device
                eval_concepts = eval_concepts
            else:
                # Traditional mode: use pre-loaded data
                eval_batch = self.observations
                eval_concepts = self.concepts

            # Common metrics
            metrics.update(self._compute_common_metrics(eval_batch))

            # GT-specific metrics
            if self.mixing_matrix is not None:
                print("  Computing gt_mcc...")
                metrics["eval/gt_mcc"] = compute_gt_dictionary_mcc(
                    self.model, self.mixing_matrix.to(self.device)
                )

                if eval_concepts is not None:
                    # Collect activations (alignment happens in metrics function)
                    Z_model, Z_true = self._collect_activations(eval_batch, eval_concepts)

                    # Compute GT activation metrics
                    gt_metrics = self._compute_gt_activation_metrics(Z_model, Z_true)
                    metrics.update(gt_metrics)

        return metrics

    def _collect_activations(self, eval_batch: torch.Tensor, eval_concepts: torch.Tensor):
        """
        Collect model activations (without pre-alignment).

        Alignment is handled by the metrics function to ensure consistency
        between activation-based and dictionary-based alignments.

        Parameters
        ----------
        eval_batch : torch.Tensor
            Observations to evaluate on
        eval_concepts : torch.Tensor
            Ground truth concepts

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (Z_model, Z_true) - raw model and true activations
        """
        batch_size = self.cfg.training.batch_size

        # Subsample for efficiency
        max_samples = self.cfg.training.get('evaluation', {}).get('max_subsample_size', 100000)
        n_samples = eval_batch.shape[0]

        if n_samples > max_samples:
            indices = torch.randperm(n_samples)[:max_samples]
            obs_subset = eval_batch[indices]
            concepts_subset = eval_concepts[indices]
            print(f"    Subsampled {max_samples} / {n_samples} activations for evaluation")
        else:
            obs_subset = eval_batch
            concepts_subset = eval_concepts

        # Collect model activations (batched)
        Z_model_list = []
        n_subset = obs_subset.shape[0]

        for i in range(0, n_subset, batch_size):
            batch = obs_subset[i:i + batch_size].to(self.device)
            _, acts, _ = self.model(batch)
            Z_model_list.append(acts.cpu())

        Z_model = torch.cat(Z_model_list, dim=0)
        Z_true = concepts_subset.cpu()

        # No pre-alignment - let the metrics function handle it
        return Z_model, Z_true

    def _compute_oracle_metrics(self, observations: torch.Tensor) -> Dict[str, float]:
        """
        Compute oracle baseline metrics including ground-truth identifiability.

        Extends base class to also compute identifiability metrics comparing
        oracle codes against ground truth activations.

        Parameters
        ----------
        observations : torch.Tensor
            Data to evaluate on (must be already normalized)

        Returns
        -------
        Dict[str, float]
            Dictionary containing oracle comparison metrics and GT identifiability
        """
        # Get oracle and base metrics from parent
        oracle, metrics = self._estimate_oracle(observations)

        # Compute GT identifiability metrics for oracle if we have ground truth
        if self.generator is not None:
            # Online mode: generate eval batch with ground truth
            batch_size = self.cfg.training.batch_size
            eval_batch, eval_concepts = self.generator.generate_batch(batch_size)
        elif self.concepts is not None:
            # Traditional mode: use stored data
            batch_size = self.cfg.training.batch_size
            eval_batch = observations[:batch_size].to(self.device)
            eval_concepts = self.concepts[:batch_size]
        else:
            # No ground truth available
            return metrics

        # Center observations using learned observation bias (same as model does)
        observation_bias = self.model.observation_bias
        centered_batch = eval_batch - observation_bias

        # Get oracle codes
        oracle_codes, _ = oracle(centered_batch)

        # Compute GT identifiability metrics for oracle
        gt_metrics = self._compute_gt_activation_metrics(
            oracle_codes.cpu(), eval_concepts.cpu()
        )

        # Add oracle prefix to GT metrics
        for k, v in gt_metrics.items():
            # Replace 'eval/gt_' with 'eval/oracle_gt_'
            oracle_key = k.replace('eval/gt_', 'eval/oracle_gt_')
            metrics[oracle_key] = v

        return metrics

    def _compute_gt_activation_metrics(
        self,
        Z_model: torch.Tensor,
        Z_true: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute ground truth activation comparison metrics.

        Thin wrapper that calls the core pairwise metrics function with true
        mixing matrix and learned dictionary for dictionary-aligned metrics.
        The metrics function handles alignment internally.

        Parameters
        ----------
        Z_model : torch.Tensor
            Raw model activations (not pre-aligned)
        Z_true : torch.Tensor
            True activations

        Returns
        -------
        Dict[str, float]
            GT activation metrics
        """
        from rsae.metrics import compute_pairwise_identifiability_metrics
        from rsae.utils.model_utils import extract_decoder_weights

        # Extract dictionaries for dictionary-aligned metrics
        true_dict = self.mixing_matrix.to(self.device)  # [true_concepts, input_dim]
        learned_dict = extract_decoder_weights(self.model)  # [nb_concepts, input_dim]

        # Compute all pairwise metrics (including dictionary-aligned)
        # The function handles both activation-based and dictionary-based alignment internally
        print("  Computing GT metrics...")
        metrics_dict = compute_pairwise_identifiability_metrics(
            Z_true, Z_model, dict1=true_dict, dict2=learned_dict
        )

        # Add 'eval/gt_' prefix and return
        return {f'eval/gt_{k}': v for k, v in metrics_dict.items()}

    # ========================================================================
    # Checkpointing (disabled for synthetic trainer)
    # ========================================================================

    def _should_checkpoint(self, step: int) -> bool:
        """
        No checkpointing for synthetic trainer (fast training).

        Returns
        -------
        bool
            Always False
        """
        return False

    def _save_checkpoint(self, step: int, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LambdaLR):
        """
        No-op for synthetic trainer.

        Parameters
        ----------
        step : int
            Current training step (unused)
        optimizer : optim.Optimizer
            Optimizer state (unused)
        scheduler : optim.lr_scheduler.LambdaLR
            LR scheduler state (unused)
        """
        pass
