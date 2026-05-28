"""Train a Sparse Autoencoder with Hydra configuration and MLflow logging.

This script provides a structured way to train SAE models with comprehensive
experiment tracking and reproducibility via Hydra and MLflow.
"""

import math
import os
from pathlib import Path
from typing import Optional, Dict, Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch


def assign_gpu_from_job_num(cfg: DictConfig) -> None:
    """Assign GPU based on Hydra job number for local parallel runs.

    Must be called before any CUDA operations.
    """
    num_gpus = cfg.device.get('num_gpus', None)
    if num_gpus is None:
        return

    # Access job number from HydraConfig (not the job config)
    hydra_cfg = HydraConfig.get()
    job_num = hydra_cfg.job.get('num', 0)
    gpu_id = job_num % num_gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"Job {job_num}: Assigned to GPU {gpu_id} (CUDA_VISIBLE_DEVICES={gpu_id})")
import mlflow
import mlflow.pytorch

from rsae import generate_synthetic_data, SyntheticDataConfig
from rsae.trainers import SyntheticTrainer
from rsae.metrics import (
    compute_operator_norm,
    compute_avg_concept_norm,
    compute_concept_connectedness,
)
from rsae.rip_loss import compute_rip_loss
from rsae.utils import flatten_dict, get_or_create_experiment, create_rip_topk_model_from_config


def resolve_learned_concepts(cfg: DictConfig) -> int:
    """Resolve learned_concepts from regime specification or nb_concepts."""
    # For OpenPhenom data, use nb_concepts directly if regime is None
    if cfg.data.regime is None:
        return cfg.data.nb_concepts

    true_concepts = cfg.data.true_concepts
    regime = cfg.data.regime

    if regime == "matched":
        return true_concepts
    elif regime in ["under", "underparameterized"]:
        return int(math.ceil(0.75 * true_concepts))
    elif regime in ["over", "overparameterized"]:
        return int(math.ceil(2.0 * true_concepts))
    else:
        # Assume it's an integer
        return int(regime)


def setup_device(cfg: DictConfig) -> torch.device:
    """Setup compute device."""
    if cfg.device.use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg.device.gpu_id}")
    else:
        device = torch.device("cpu")
    return device


def create_model(cfg: DictConfig, learned_concepts: int, device: torch.device):
    """Create model based on configuration."""
    model_type = cfg.model.type
    input_dim = cfg.data.input_dim

    if model_type == "riptopk":
        model = create_rip_topk_model_from_config(
            cfg=cfg,
            input_dim=input_dim,
            nb_concepts=learned_concepts,
            device=device,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Only 'riptopk' is supported.")

    return model


def generate_data(cfg: DictConfig, device: torch.device):
    """
    Generate synthetic data or setup online generator.

    Returns:
        Tuple of (observations, concepts, mixing_matrix, generator):
        - Traditional mode: (observations, concepts, mixing_matrix, None)
        - Online mode: (None, None, mixing_matrix, generator)
    """
    from rsae.synthetic_data import generate_mixing_matrix, SyntheticDataGenerator

    true_concepts = cfg.data.true_concepts

    config = SyntheticDataConfig(
        n_samples=cfg.data.get('n_samples', None),  # Allow None
        concept_dim=true_concepts,
        observed_dim=cfg.data.input_dim,
        k=cfg.data.k,
        noise_std=cfg.data.noise_std,
        seed=cfg.seed.data,
        distribution=cfg.data.get('distribution', 'gaussian'),
        mixture_scale=cfg.data.get('mixture_scale', 0.5),
        num_mixtures=cfg.data.get('num_mixtures', None),
    )

    # Generate mixing matrix (always needed for ground truth)
    if config.seed is not None:
        torch.manual_seed(config.seed)
    mixing_matrix = generate_mixing_matrix(
        config.observed_dim,
        config.concept_dim,
        distribution=config.distribution,
        mixture_scale=config.mixture_scale,
        num_mixtures=config.num_mixtures,
    )

    # Check if online mode
    if config.is_online_mode:
        print("  Online generation mode enabled (n_samples=None)")
        print("  Data will be generated on-the-fly during training")
        print(f"  Generated mixing matrix: {mixing_matrix.shape}")
        # Create generator (generate on device for speed)
        generator = SyntheticDataGenerator(config, mixing_matrix, device=device)
        return None, None, mixing_matrix, generator

    # Traditional mode: pre-generate all data
    print(f"  Traditional mode: Pre-generating {config.n_samples:,} samples")

    # Get caching configuration
    use_cache = cfg.get('data_cache', {}).get('use_cache', True)
    cache_dir = cfg.get('data_cache', {}).get('cache_dir', None)
    if cache_dir is not None:
        cache_dir = Path(cache_dir)

    # Generate synthetic data (reuse mixing_matrix for consistency)
    observations, concepts, _ = generate_synthetic_data(
        config,
        verbose=False,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )

    return observations.to(device), concepts, mixing_matrix, None


def compute_and_log_data_metrics(
    concepts: torch.Tensor,
    mixing_matrix: torch.Tensor,
    device: torch.device,
    cfg: DictConfig
):
    """Compute and log metrics about the synthetic data."""
    metrics = {}

    # Compute ground truth dictionary properties
    D_synth = mixing_matrix.to(device)
    sample_concepts = concepts[:cfg.training.batch_size].to(device)

    metrics["data/synthetic_rip_loss"] = compute_rip_loss(
        sample_concepts @ D_synth,
        sample_concepts,
        D_synth.T,
        D_synth,
        k=cfg.data.k,
        weighted=cfg.model.get('rip_loss_weighted', False),
        use_abstopk=cfg.model.get('use_abstopk', False),
        multiplier=1,  # no multiplier, because concepts are $k$-sparse already
    ).item()
    metrics["data/synthetic_operator_norm"] = compute_operator_norm(D_synth)
    metrics["data/synthetic_avg_concept_norm"] = compute_avg_concept_norm(D_synth)

    # Compute ground truth connectedness
    connectedness = compute_concept_connectedness(concepts)
    metrics["data/synthetic_n_components"] = connectedness['n_connected_components']
    metrics["data/synthetic_largest_component"] = connectedness['largest_component_size']
    metrics["data/synthetic_avg_shortest_path"] = connectedness['avg_shortest_path']

    return metrics




@hydra.main(version_base=None, config_path="../configs", config_name=None)
def main(cfg: DictConfig):
    """Main training function."""
    # Assign GPU before any CUDA operations (for local parallel runs)
    assign_gpu_from_job_num(cfg)

    print("=" * 80)
    print("SAE Training with Hydra + MLflow")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Setup device
    device = setup_device(cfg)
    print(f"Device: {device}")

    # Resolve learned concepts
    learned_concepts = resolve_learned_concepts(cfg)
    cfg.data.learned_concepts = learned_concepts
    print(f"True concepts: {cfg.data.true_concepts}")
    print(f"Learned concepts: {learned_concepts}")

    # Initialize MLflow
    if cfg.mlflow.enabled:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        # Use race-condition-safe experiment creation
        experiment_id = get_or_create_experiment(cfg.mlflow.experiment_name)
        mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"{cfg.model.name}_{cfg.experiment.name}",
            tags={
                "job_type": cfg.experiment.job_type,
                "group": cfg.experiment.get("group", ""),
            }
        )
        # Log all parameters
        mlflow.log_params({
            f"{k}": v for k, v in flatten_dict(OmegaConf.to_container(cfg, resolve=True)).items()
        })

    # Generate or load data based on config
    data_source = cfg.data.get('data_source', 'synthetic')

    if data_source == 'synthetic':
        print("\nGenerating synthetic data...")
        observations, concepts, mixing_matrix, generator = generate_data(cfg, device)

        if generator is not None:
            print("  Online mode: Using batch generator")
        else:
            print(f"  Data shape: observations={observations.shape}, concepts={concepts.shape}")
        print("Data generated successfully!")
    else:
        raise ValueError(f"Unknown data_source: {data_source}")

    # Log data metrics (only for traditional mode with pre-generated data)
    if data_source == 'synthetic' and generator is None:
        data_metrics = compute_and_log_data_metrics(concepts, mixing_matrix, device, cfg)
        if cfg.mlflow.enabled:
            mlflow.log_metrics(data_metrics)
        print("\nData metrics:")
        for key, value in data_metrics.items():
            print(f"  {key}: {value:.4f}")
    else:
        print("\nSkipping pre-generation data metrics (online mode)")

    # Set random seed for training (after data generation)
    torch.manual_seed(cfg.seed.train)

    # Create model
    print("\nCreating model...")
    print(f"  Model: {cfg.model.name}")
    print(f"  Input dim: {cfg.data.input_dim}")
    print(f"  Learned concepts: {learned_concepts}")
    model = create_model(cfg, learned_concepts, device)
    model = model.to(device)  # Ensure all parameters are on correct device
    print("Model created successfully!")

    # Create trainer and train model
    print("\nInitializing trainer...")
    trainer = SyntheticTrainer(
        cfg=cfg,
        model=model,
        observations=observations,
        concepts=concepts,
        mixing_matrix=mixing_matrix,
        device=str(device),
        generator=generator,
    )

    print(f"\nTraining for {cfg.training.max_steps} steps...")
    model = trainer.train()
    print("Training finished!")

    # Final evaluation
    print("\nComputing final evaluation metrics...")
    final_metrics = trainer.evaluate()

    # Log final metrics
    if cfg.mlflow.enabled:
        mlflow.log_metrics(final_metrics, step=cfg.training.max_steps)

    print("Evaluation complete!")
    print("\nFinal evaluation metrics:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save model
    if cfg.mlflow.enabled and cfg.mlflow.log_artifacts:
        # Save config as a separate artifact
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_dict(config_dict, "model/extra_files/config.yaml")

        # Save model using MLflow's PyTorch integration
        mlflow.pytorch.log_model(
            model,
            name="model",
        )
        print("Model saved to MLflow!")

    # End MLflow run
    if cfg.mlflow.enabled:
        mlflow.end_run()

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
