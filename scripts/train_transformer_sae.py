"""Train RIPTopK SAE on transformer activations using dictionary_learning infrastructure.

This script:
1. Collects activations from a specified transformer layer using dictionary_learning's ActivationBuffer
2. Trains a RIPTopK SAE with RIP + AuxK + reconstruction loss
3. Logs the trained model + Hydra config to MLflow (used downstream by the
   SAEBench wrapper to evaluate the checkpoint)
"""

import json
from pathlib import Path
import sys
from typing import Dict, Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from tqdm import tqdm
import mlflow
import mlflow.pytorch

# Add dictionary_learning to path
dict_learning_path = Path(__file__).parent.parent / "dictionary_learning"
sys.path.insert(0, str(dict_learning_path))

from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from nnsight import LanguageModel

from rsae import RIPTopK
from rsae.metrics import compute_coherence
from rsae.utils import flatten_dict, create_rip_topk_model_from_config


def infer_d_submodule(submodule, model_name: str) -> int:
    """
    Infer the dimension of a submodule by inspecting its components.

    Tries multiple strategies:
    1. Check if submodule has out_features directly
    2. Check common output projections (MLP, attention)
    3. Check layernorm dimensions

    Args:
        submodule: The model submodule (e.g., a transformer layer)
        model_name: Model name for better error messages

    Returns:
        Inferred dimension (d_submodule)
    """
    # Strategy 1: Check if submodule has out_features directly
    if hasattr(submodule, 'out_features'):
        return submodule.out_features

    # Strategy 2: Check common output projection patterns
    # For transformer layers, the output dimension is typically the residual stream dimension
    # Look for the final linear projection in common sub-components

    # Pythia/GPT-NeoX: mlp.dense_4h_to_h or attention.dense
    if hasattr(submodule, 'mlp') and hasattr(submodule.mlp, 'dense_4h_to_h'):
        if hasattr(submodule.mlp.dense_4h_to_h, 'out_features'):
            return submodule.mlp.dense_4h_to_h.out_features

    if hasattr(submodule, 'attention') and hasattr(submodule.attention, 'dense'):
        if hasattr(submodule.attention.dense, 'out_features'):
            return submodule.attention.dense.out_features

    # Gemma/Llama: mlp.down_proj or self_attn.o_proj
    if hasattr(submodule, 'mlp') and hasattr(submodule.mlp, 'down_proj'):
        if hasattr(submodule.mlp.down_proj, 'out_features'):
            return submodule.mlp.down_proj.out_features

    if hasattr(submodule, 'self_attn') and hasattr(submodule.self_attn, 'o_proj'):
        if hasattr(submodule.self_attn.o_proj, 'out_features'):
            return submodule.self_attn.o_proj.out_features

    # Strategy 3: Check layernorm normalized_shape
    # LayerNorm stores the dimension in normalized_shape
    if hasattr(submodule, 'input_layernorm'):
        if hasattr(submodule.input_layernorm, 'normalized_shape'):
            # normalized_shape is a tuple like (512,)
            return submodule.input_layernorm.normalized_shape[0]

    if hasattr(submodule, 'ln_1'):  # GPT-2 style
        if hasattr(submodule.ln_1, 'normalized_shape'):
            return submodule.ln_1.normalized_shape[0]

    # If all strategies fail, raise an error
    raise ValueError(
        f"Could not infer d_submodule from {type(submodule).__name__} in model {model_name}. "
        f"Please specify --d-model explicitly."
    )


def collect_activations(cfg: DictConfig, device: str) -> ActivationBuffer:
    """
    Collect activations from a transformer model using ActivationBuffer.

    Args:
        cfg: Hydra configuration
        device: Device for computation

    Returns:
        ActivationBuffer filled with activations
    """
    model_name = cfg.data.model_name
    hook_layer = cfg.data.hook_layer
    hook_name = cfg.data.hook_name if cfg.data.hook_name else f"blocks.{hook_layer}.hook_resid_post"
    n_contexts = cfg.data.n_contexts
    dataset_name = cfg.data.dataset_name
    d_model = cfg.data.d_model

    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Hook: {hook_name}")
    print(f"  Contexts: {n_contexts:,}")

    print("\nLoading language model...")
    model = LanguageModel(model_name, device_map=device)
    print("Model loaded successfully!")

    print("\nCreating data generator...")
    data = hf_dataset_to_generator(dataset_name)

    # Get submodule from model
    # For Pythia models: model.gpt_neox.layers[layer].output
    # Parse hook_name to get the submodule
    print(f"\nGetting submodule: {hook_name}")
    if "blocks" in hook_name and "hook_resid_post" in hook_name:
        # TransformerLens style: "blocks.3.hook_resid_post"
        # Convert to model-specific path
        if "pythia" in model_name.lower():
            submodule = model.gpt_neox.layers[hook_layer]
        elif "gemma" in model_name.lower():
            submodule = model.model.layers[hook_layer]
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
    else:
        raise ValueError(f"Unsupported hook_name format: {hook_name}")

    # Infer d_model if not provided
    if d_model is None:
        print("\nAuto-detecting model dimension from submodule...")
        d_model = infer_d_submodule(submodule, model_name)
        print(f"  Detected d_model: {d_model}")
    else:
        print(f"\nUsing provided d_model: {d_model}")

    print(f"\nCollecting activations...")
    # Store activations on CPU to avoid GPU OOM with large buffers
    # (activations will be moved to GPU in training loop batches)
    activation_buffer = ActivationBuffer(
        data=data,
        model=model,
        submodule=submodule,
        d_submodule=d_model,  # Explicitly pass dimension
        n_ctxs=n_contexts,
        # device='cpu',  # Force CPU storage to avoid OOM with large batch sizes
        device=device,
    )

    return activation_buffer


def evaluate_model(
    model: RIPTopK,
    val_buffer: ActivationBuffer,
    batch_size: int,
    n_samples: int = 10000,
    device: str = "cuda",
) -> dict:
    """
    Evaluate model on validation data.

    Args:
        model: Trained RIPTopK model
        val_buffer: Validation buffer (separate from training)
        batch_size: Batch size for evaluation
        n_samples: Number of samples to evaluate on
        device: Device for computation

    Returns:
        Dictionary of evaluation metrics with 'eval/' prefix
    """
    model.eval()
    metrics = {}

    with torch.no_grad():
        # Validate we have enough samples
        val_size = len(val_buffer)
        if n_samples > val_size:
            print(f"Warning: Requested {n_samples} eval samples, but val has {val_size}. Using {val_size}.")
            n_samples = val_size

        # Collect samples for evaluation
        eval_samples = []
        buffer_iter = iter(val_buffer)
        collected = 0

        while collected < n_samples:
            batch = next(buffer_iter)
            eval_samples.append(batch)
            collected += batch.shape[0]

        eval_data = torch.cat(eval_samples, dim=0)[:n_samples].to(device)

        # Compute metrics in batches
        all_reconstructions = []
        all_activations = []
        total_recon_loss = 0.0
        total_rip_loss = 0.0
        total_auxk_loss = 0.0
        total_k_est = 0.0
        n_batches = 0

        for i in range(0, eval_data.shape[0], batch_size):
            batch = eval_data[i:i + batch_size]
            pre_acts, acts, recon = model(batch)

            all_reconstructions.append(recon.cpu())
            all_activations.append(acts.cpu())

            # Compute losses
            total_recon_loss += torch.mean((batch - recon) ** 2).item()
            total_rip_loss += model.compute_rip_loss(pre_acts, batch).item()
            total_k_est += (acts != 0).float().sum(dim=1).mean().item()  # actual sparsity
            # Whiten residual if needed (decoder operates in whitened space)
            residual = batch - recon
            if model.whiten:
                residual = model.whitener(residual)
            total_auxk_loss += model.compute_auxk_loss(pre_acts, residual).item()
            n_batches += 1

        # Average losses
        metrics['eval/recon_mse'] = total_recon_loss / n_batches
        metrics['eval/rip_loss'] = total_rip_loss / n_batches
        metrics['eval/auxk_loss'] = total_auxk_loss / n_batches
        metrics['eval/k_est'] = total_k_est / n_batches

        # Concatenate all activations and reconstructions
        all_activations = torch.cat(all_activations, dim=0)
        all_reconstructions = torch.cat(all_reconstructions, dim=0)
        eval_data_cpu = eval_data.cpu()

        # Sparsity (L0)
        metrics['eval/sparsity'] = (all_activations > 0).float().sum(dim=-1).mean().item()

        # Explained variance
        residuals = eval_data_cpu - all_reconstructions
        total_variance = torch.var(eval_data_cpu)
        residual_variance = torch.var(residuals)
        metrics['eval/explained_variance'] = (1 - residual_variance / total_variance).item()

        # Coherence (max absolute dot product between dictionary vectors)
        metrics['eval/coherence'] = compute_coherence(model)

        # Dead concept tracking
        if hasattr(model, 'activation_counts'):
            act_counts = model.activation_counts
            total_inputs = max(model.total_inputs_seen.item(), 1)

            # Dead concept statistics
            dead_mask = act_counts == 0
            metrics['eval/dead_concept_count'] = dead_mask.sum().item()
            metrics['eval/dead_concept_proportion'] = dead_mask.float().mean().item()

            # Activation count statistics
            metrics['eval/min_activation_count'] = act_counts.min().item()
            metrics['eval/max_activation_count'] = act_counts.max().item()
            metrics['eval/mean_activation_count'] = act_counts.float().mean().item()

            # Activation rate statistics (counts / window_size for interpretability)
            window_size = min(total_inputs, model.activation_window_batches * 16384)  # Approximate window size
            act_rates = act_counts.float() / max(window_size, 1)
            metrics['eval/min_activation_rate'] = act_rates.min().item()
            metrics['eval/max_activation_rate'] = act_rates.max().item()
            metrics['eval/mean_activation_rate'] = act_rates.mean().item()

    return metrics


def save_checkpoint(
    model: RIPTopK,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LambdaLR,
    step: int,
    output_dir: Path,
    experiment_name: str,
    d_in: int,
    nb_concepts: int,
    top_k: int,
    rip_weight: float,
    auxk_weight: float,
) -> Path:
    """Save a training checkpoint.

    Args:
        model: RIPTopK model to save
        optimizer: Optimizer state to save
        scheduler: Learning rate scheduler state to save
        step: Current training step
        output_dir: Base directory for outputs
        experiment_name: Name of the experiment
        d_in: Input dimension
        nb_concepts: Number of concepts
        top_k: Sparsity level
        rip_weight: RIP loss weight
        auxk_weight: AuxK loss weight

    Returns:
        Path to the saved checkpoint
    """
    checkpoint_path = output_dir / experiment_name / f"checkpoint_step_{step}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'd_in': d_in,
        'nb_concepts': nb_concepts,
        'top_k': top_k,
        'rip_weight': rip_weight,
        'auxk_weight': auxk_weight,
    }, checkpoint_path)

    print(f"  Saved checkpoint: {checkpoint_path}")
    return checkpoint_path


def train_riptopk(
    cfg: DictConfig,
    train_buffer: ActivationBuffer,
    d_in: int,
    output_dir: Path,
    experiment_name: str,
    device: str,
    val_buffer: Optional[ActivationBuffer] = None,
) -> RIPTopK:
    """
    Train a RIPTopK SAE using the unified TransformerTrainer.

    Args:
        cfg: Hydra configuration
        train_buffer: Training buffer
        d_in: Input dimension
        output_dir: Directory to save checkpoints
        experiment_name: Name for this experiment
        device: Device for training
        val_buffer: Validation buffer (optional, for evaluation)

    Returns:
        Trained RIPTopK model
    """
    from rsae.trainers import TransformerTrainer

    # Extract parameters from config for logging
    nb_concepts = cfg.data.nb_concepts
    top_k = cfg.data.k
    rip_weight = cfg.model.rip_weight
    auxk_weight = cfg.model.get('auxk_weight', 1.0 / 32)

    # Create RIPTopK model using shared utility
    print(f"\nInitializing RIPTopK SAE:")
    print(f"  d_in: {d_in}")
    print(f"  nb_concepts: {nb_concepts}")
    print(f"  top_k: {top_k}")
    print(f"  rip_weight: {rip_weight}")
    print(f"  auxk_weight: {auxk_weight}")

    model = create_rip_topk_model_from_config(
        cfg=cfg,
        input_dim=d_in,
        nb_concepts=nb_concepts,
        device=device,
    ).to(device)

    # Create TransformerTrainer
    trainer = TransformerTrainer(
        cfg=cfg,
        model=model,
        train_buffer=train_buffer,
        device=device,
        output_dir=output_dir,
        experiment_name=experiment_name,
        val_buffer=val_buffer,
    )

    # Train the model
    model = trainer.train()

    return model


@hydra.main(version_base=None, config_path="../configs", config_name=None)
def main(cfg: DictConfig):
    """Main training function."""
    print("=" * 80)
    print("Transformer SAE Training with Hydra + MLflow")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Setup device
    if cfg.device.use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg.device.gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Initialize MLflow
    if cfg.mlflow.enabled:
        from rsae.utils import get_or_create_experiment

        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        experiment_id = get_or_create_experiment(cfg.mlflow.experiment_name)
        mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"{cfg.model.name}_{cfg.experiment.name}",
            tags={
                "job_type": cfg.experiment.job_type,
                "group": cfg.experiment.get("group", ""),
            }
        )
        mlflow.log_params({
            f"{k}": v for k, v in flatten_dict(OmegaConf.to_container(cfg, resolve=True)).items()
        })

    # Collect activations
    print("\nCollecting activations...")
    train_buffer = collect_activations(cfg, str(device))
    val_buffer = None  # Validation not supported with dynamic collection
    print("Activation buffer created successfully!")

    # Get input dimension
    d_in = train_buffer.d_submodule
    print(f"  Input dimension (d_model): {d_in}")

    # Train model
    print("\nTraining RIPTopK SAE...")
    model = train_riptopk(
        cfg=cfg,
        train_buffer=train_buffer,
        d_in=d_in,
        output_dir=Path("outputs"),
        experiment_name=cfg.experiment.name,
        device=str(device),
        val_buffer=val_buffer,
    )

    # Save final model
    if cfg.mlflow.enabled and cfg.mlflow.log_artifacts:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_dict(config_dict, "model/extra_files/config.yaml")
        mlflow.pytorch.log_model(model, name="model")
        print("Model saved to MLflow!")

    # End MLflow run
    if cfg.mlflow.enabled:
        mlflow.end_run()

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
