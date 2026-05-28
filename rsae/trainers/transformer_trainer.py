"""Transformer Trainer for SAE training on transformer activations.

This module provides the TransformerTrainer class for training on dynamically
generated transformer activations with buffer integration and checkpointing support.
"""

from pathlib import Path
from typing import Dict, Union, Optional
import torch
import torch.optim as optim
from omegaconf import DictConfig

from .base_trainer import BaseTrainer


class TransformerTrainer(BaseTrainer):
    """
    Trainer for transformer activations with buffer integration and checkpointing.

    This trainer handles ActivationBuffer data sources with support for:
    - Train/val splits
    - Checkpoint management (periodic + evaluation)
    - No ground truth metrics (real data)

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object
    model : torch.nn.Module
        SAE model to train
    train_buffer : ActivationBuffer
        Training data buffer
    val_buffer : ActivationBuffer, optional
        Validation data buffer (for evaluation)
    device : torch.device
        Device to run training on
    output_dir : Path
        Directory for saving checkpoints
    experiment_name : str
        Name of experiment (for checkpoint organization)
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        train_buffer,  # ActivationBuffer
        device: torch.device,
        output_dir: Path,
        experiment_name: str,
        val_buffer=None,  # Optional[ActivationBuffer]
    ):
        # Collect sample for initialization
        init_sample = self._collect_initialization_sample(train_buffer, cfg, device)

        # Store for hot restarts (on CPU to save GPU memory)
        self._stored_init_sample = init_sample.cpu()

        # Normalize and initialize (CHANGED from _scale_data)
        normalized_init_sample = self._normalize_data_static(init_sample)
        self._initialize_model_weights_impl(model, normalized_init_sample, device, cfg)

        # Call parent constructor
        super().__init__(cfg=cfg, model=model, device=device)

        # Store buffer and output info
        self.train_buffer = train_buffer
        self.val_buffer = val_buffer
        self.output_dir = output_dir
        self.experiment_name = experiment_name

        # Store model config for hot restarts
        self._store_model_config()

        # Create buffer iterator
        self.buffer_iter = iter(train_buffer)

        # Validation
        if cfg.training.get('eval_interval') is not None and val_buffer is None:
            print("Warning: eval_interval set but no validation buffer provided")

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
            If True, skip decoder initialization and preserve existing weights (for hot restart)
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

    def _collect_initialization_sample(self, buffer, cfg: DictConfig, device: torch.device) -> torch.Tensor:
        """
        Collect sample for geometric median initialization.

        Parameters
        ----------
        buffer : ActivationBuffer
            Data buffer to sample from
        cfg : DictConfig
            Configuration object
        device : torch.device
            Device to use

        Returns
        -------
        torch.Tensor
            Sample data for initialization
        """
        n_samples = cfg.training.get('geometric_median', {}).get('n_samples', 50000)
        samples = []
        buffer_iter = iter(buffer)
        collected = 0

        print(f"Collecting {n_samples} samples for geometric median initialization...")
        while collected < n_samples:
            try:
                batch = next(buffer_iter)
                # Move to CPU to save GPU memory
                samples.append(batch.cpu())
                collected += batch.shape[0]
                if collected >= n_samples:
                    break
            except StopIteration:
                print(f"Warning: Buffer exhausted after {collected} samples (requested {n_samples})")
                break

        if len(samples) == 0:
            raise ValueError("No samples collected from buffer!")

        result = torch.cat(samples, dim=0)[:n_samples]
        print(f"Collected {result.shape[0]} samples from buffer")
        return result

    # ========================================================================
    # Data Iteration
    # ========================================================================

    def _get_next_batch(self) -> torch.Tensor:
        """
        Get next batch from buffer and normalize it.

        Returns
        -------
        torch.Tensor
            Normalized batch tensor on correct device
        """
        batch = next(self.buffer_iter)

        # Handle variable batch sizes from buffer (shouldn't happen but be defensive)
        max_batch_size = self.cfg.training.batch_size
        if batch.shape[0] > max_batch_size:
            indices = torch.randperm(batch.shape[0])[:max_batch_size]
            batch = batch[indices]

        # CRITICAL: Normalize batch before returning (fixes initialization/training mismatch bug!)
        batch = batch.float()
        normalized_batch = self._normalize_data(batch)

        return normalized_batch.to(self.device)

    # ========================================================================
    # Evaluation
    # ========================================================================

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on validation set (no GT metrics for real data).

        Returns
        -------
        Dict[str, float]
            Dictionary of evaluation metrics
        """
        self.model.eval()

        # Use validation buffer if available, otherwise training
        eval_buffer = self.val_buffer if self.val_buffer is not None else self.train_buffer

        with torch.no_grad():
            # Collect evaluation sample
            eval_data = self._collect_eval_sample(eval_buffer)

            # Compute common metrics (no GT)
            metrics = self._compute_common_metrics(eval_data)

            # Add loss breakdown for RIP models
            if self.is_rip_model:
                loss_metrics = self._compute_loss_breakdown(eval_data)
                metrics.update(loss_metrics)

        return metrics

    def _collect_eval_sample(self, buffer) -> torch.Tensor:
        """
        Collect and normalize evaluation sample.

        Parameters
        ----------
        buffer : ActivationBuffer
            Buffer to sample from

        Returns
        -------
        torch.Tensor
            Normalized evaluation data on CPU
        """
        n_samples = self.cfg.training.get('eval_samples', 10000)
        samples = []
        buffer_iter = iter(buffer)
        collected = 0

        while collected < n_samples:
            try:
                batch = next(buffer_iter)
                samples.append(batch.cpu())
                collected += batch.shape[0]
                if collected >= n_samples:
                    break
            except StopIteration:
                break

        if len(samples) == 0:
            raise ValueError("No samples collected for evaluation!")

        eval_data = torch.cat(samples, dim=0)[:n_samples].float()

        # Normalize evaluation data
        normalized_eval_data = self._normalize_data(eval_data)

        return normalized_eval_data

    def _compute_loss_breakdown(self, data: torch.Tensor) -> Dict[str, float]:
        """
        Compute detailed loss breakdown for RIP models.

        Parameters
        ----------
        data : torch.Tensor
            Data to evaluate on (must be already normalized)

        Returns
        -------
        Dict[str, float]
            Loss breakdown metrics
        """
        metrics = {}

        if not self.is_rip_model:
            return metrics

        # Compute in batches (data is already normalized by _collect_eval_sample)
        batch_size = self.cfg.training.batch_size
        total_rip = 0.0
        total_auxk = 0.0
        total_k_est = 0.0
        n_batches = 0

        for i in range(0, data.shape[0], batch_size):
            batch = data[i:i + batch_size].to(self.device)
            pre_acts, acts, recon = self.model(batch)

            total_rip += self.model.compute_rip_loss(pre_acts, batch).item()
            total_k_est += (acts != 0).float().sum(dim=1).mean().item()
            # Whiten residual if needed (decoder operates in whitened space)
            residual = batch - recon
            if self.model.whiten:
                residual = self.model.whitener(residual)
            total_auxk += self.model.compute_auxk_loss(pre_acts, residual).item()
            n_batches += 1

        # Note: eval/rip_loss is already computed in _compute_common_metrics
        # We add the breakdown here
        metrics["eval/auxk_loss"] = total_auxk / n_batches
        metrics["eval/k_est"] = total_k_est / n_batches

        return metrics

    # ========================================================================
    # Checkpointing
    # ========================================================================

    def _should_checkpoint(self, step: int) -> bool:
        """
        Checkpoint at evaluation and periodic intervals.

        Parameters
        ----------
        step : int
            Current training step

        Returns
        -------
        bool
            True if checkpoint should be saved
        """
        save_interval = self.cfg.training.get('save_interval', 50000)
        eval_interval = self.cfg.training.get('eval_interval', None)

        # Checkpoint at evaluation
        if eval_interval is not None and step % eval_interval == 0 and step > 0:
            return True

        # Periodic checkpoint
        if step % save_interval == 0 and step > 0:
            return True

        return False

    def _save_checkpoint(self, step: int, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LambdaLR):
        """
        Save checkpoint to disk.

        Parameters
        ----------
        step : int
            Current training step
        optimizer : optim.Optimizer
            Optimizer state
        scheduler : optim.lr_scheduler.LambdaLR
            LR scheduler state
        """
        checkpoint_path = (
            self.output_dir / self.experiment_name / f"checkpoint_step_{step}.pt"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'cfg': self.cfg,
            'weights_are_tied': getattr(self, 'weights_are_tied', False),
        }, checkpoint_path)

        print(f"  Saved checkpoint: {checkpoint_path}")
