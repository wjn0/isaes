"""Compare two trained models for identifiability assessment.

This script loads two models (trained with different seeds) and computes
identifiability metrics between them.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import mlflow
import mlflow.pytorch
import yaml

# Add dictionary_learning to path
dict_learning_path = Path(__file__).parent.parent / "dictionary_learning"
sys.path.insert(0, str(dict_learning_path))

from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from nnsight import LanguageModel

from rsae import RIPTopK, generate_synthetic_data, SyntheticDataConfig
from rsae.buffers import VisionActivationBuffer
from rsae.baselines.oracle import OMPOracle
from rsae.metrics import (
    # Batch computation (optimized)
    compute_model_identifiability_metrics,
)
from rsae.metrics.identifiability_metrics import compute_pairwise_identifiability_metrics
from rsae.utils import flatten_dict, get_or_create_experiment
from rsae.utils.model_utils import extract_decoder_weights



def load_model_from_mlflow(run_id: str, tracking_uri: str, device: torch.device):
    """Load a model from an MLflow run."""
    mlflow.set_tracking_uri(tracking_uri)

    # Load model using MLflow
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    model.eval()

    # Load config from artifacts
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    artifact_path = client.download_artifacts(run_id, "model/extra_files/config.yaml")

    with open(artifact_path) as f:
        config = yaml.safe_load(f)

    return model, config


def create_model_from_config(config: Dict[str, Any], device: torch.device):
    """Create a model from a config dictionary."""
    model_type = config['model']['type']
    input_dim = config['data']['input_dim']
    learned_concepts = config['data']['learned_concepts']
    k = config['data']['k']

    if model_type == "riptopk":
        model = RIPTopK(
            input_shape=input_dim,
            nb_concepts=learned_concepts,
            top_k=k,
            rip_weight=config['model']['rip_weight'],
            connectedness_weight=config['model']['connectedness_weight'],
            device=str(device),
            normalization=config['model']['normalization'],
            activation_window_batches=config['model'].get('activation_window_batches', 64),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Only 'riptopk' is supported.")

    return model


def generate_or_load_data_from_config(config: Dict[str, Any], device: torch.device, args=None):
    """Generate or load data based on data source in config."""
    data_source = config['data'].get('data_source', 'synthetic')

    if data_source == 'synthetic':
        print(f"Generating synthetic data from config...")

        # For online generation (n_samples = null), use max_samples from args
        n_samples = config['data']['n_samples']
        if n_samples is None:
            n_samples = args.max_samples if args and hasattr(args, 'max_samples') else 100_000
            print(f"  n_samples is null, generating {n_samples} samples for evaluation")

        data_config = SyntheticDataConfig(
            n_samples=n_samples,
            concept_dim=config['data']['true_concepts'],
            observed_dim=config['data']['input_dim'],
            k=config['data']['k'],
            noise_std=config['data']['noise_std'],
            seed=config['seed']['data'],
            distribution=config['data'].get('distribution', 'gaussian'),
            mixture_scale=config['data'].get('mixture_scale', 0.5),
            num_mixtures=config['data'].get('num_mixtures', None),
        )

        observations, concepts, mixing_matrix = generate_synthetic_data(
            data_config,
        )

        return observations.to(device), concepts, mixing_matrix

    elif data_source == 'transformer':
        print(f"Collecting transformer activations for identifiability comparison...")

        # Extract config
        model_name = config['data']['model_name']
        hook_layer = config['data']['hook_layer']
        dataset_name = config['data']['dataset_name']
        n_contexts = config['data'].get('n_contexts', 100000)
        d_model = config['data'].get('d_model', None)

        print(f"  Model: {model_name}")
        print(f"  Layer: {hook_layer}")
        print(f"  Dataset: {dataset_name}")
        print(f"  Contexts: {n_contexts}")

        # Load model
        model = LanguageModel(model_name, device_map=str(device))

        # Get submodule
        if "pythia" in model_name.lower():
            submodule = model.gpt_neox.layers[hook_layer]
        elif "gemma" in model_name.lower():
            submodule = model.model.layers[hook_layer]
        else:
            raise ValueError(f"Unsupported model architecture: {model_name}")

        # Infer d_model if not provided (following train_transformer_sae.py logic)
        if d_model is None:
            # Simple inference: check common output projection patterns
            if hasattr(submodule, 'mlp') and hasattr(submodule.mlp, 'dense_4h_to_h'):
                d_model = submodule.mlp.dense_4h_to_h.out_features
            elif hasattr(submodule, 'input_layernorm') and hasattr(submodule.input_layernorm, 'normalized_shape'):
                d_model = submodule.input_layernorm.normalized_shape[0]
            else:
                raise ValueError(f"Could not infer d_model. Please specify in config.")
            print(f"  Auto-detected d_model: {d_model}")

        # Create data generator
        data_generator = hf_dataset_to_generator(dataset_name)

        # Collect activations using ActivationBuffer
        activation_buffer = ActivationBuffer(
            data=data_generator,
            model=model,
            submodule=submodule,
            d_submodule=d_model,
            n_ctxs=n_contexts,
            device=str(device),
        )

        # Extract all activations from buffer by iterating
        print(f"  Extracting activations from buffer...")
        all_activations = []
        buffer_iter = iter(activation_buffer)
        collected = 0

        while collected < n_contexts:
            try:
                batch = next(buffer_iter)
                all_activations.append(batch)
                collected += batch.shape[0]
            except StopIteration:
                break

        observations = torch.cat(all_activations, dim=0)[:n_contexts]

        print(f"  Collected {observations.shape[0]} activations, dim={observations.shape[1]}")

        # No ground truth for real data
        return observations, None, None

    elif data_source == 'vision':
        print(f"Collecting vision model activations for identifiability comparison...")

        # Extract config
        model_name = config['data']['model_name']
        hook_layer = config['data']['hook_layer']
        dataset_name = config['data']['dataset_name']
        d_model = config['data'].get('d_model', None)
        patches_per_image = config['data'].get('patches_per_image', 196)
        refresh_batch_size = config['data'].get('refresh_batch_size', 64)

        # Limit n_images based on max_samples to avoid OOM
        # (buffer pre-allocates n_images * patches_per_image activations)
        max_samples = args.max_samples if args and hasattr(args, 'max_samples') else 100_000
        config_n_images = config['data'].get('n_images', 10000)
        # Calculate minimum images needed to get max_samples patches
        n_images_for_max_samples = (max_samples + patches_per_image - 1) // patches_per_image
        n_images = min(config_n_images, n_images_for_max_samples)
        print(f"  Limiting n_images from {config_n_images} to {n_images} based on max_samples={max_samples}")

        print(f"  Model: {model_name}")
        print(f"  Layer: {hook_layer}")
        print(f"  Dataset: {dataset_name}")
        print(f"  n_images: {n_images}")
        print(f"  d_model: {d_model}")
        print(f"  patches_per_image: {patches_per_image}")

        # Create image generator (same as train_vision_sae.py)
        from datasets import load_dataset

        def imagenet_to_generator(dataset_name: str = "imagenet-1k", split: str = "train"):
            dataset = load_dataset(
                dataset_name, split=split, streaming=True, trust_remote_code=True
            )
            for x in iter(dataset):
                img = x["image"]
                if img.mode != "RGB":
                    img = img.convert("RGB")
                yield img

        data = imagenet_to_generator(dataset_name)

        # Create VisionActivationBuffer
        activation_buffer = VisionActivationBuffer(
            data=data,
            model_name=model_name,
            hook_layer=hook_layer,
            d_submodule=d_model,
            n_images=n_images,
            patches_per_image=patches_per_image,
            refresh_batch_size=refresh_batch_size,
            out_batch_size=512,  # Use reasonable default batch size
            device=str(device),
            include_cls=False,
        )

        # Extract activations from buffer
        print(f"  Extracting activations from buffer...")
        all_activations = []
        buffer_iter = iter(activation_buffer)
        target_samples = n_images * patches_per_image
        collected = 0

        while collected < target_samples:
            try:
                batch = next(buffer_iter)
                all_activations.append(batch)
                collected += batch.shape[0]
            except StopIteration:
                break

        observations = torch.cat(all_activations, dim=0)[:target_samples]

        print(f"  Collected {observations.shape[0]} activations, dim={observations.shape[1]}")

        # No ground truth for real vision data
        return observations, None, None

    else:
        raise ValueError(f"Unknown data_source: {data_source}")


def compute_identifiability_metrics(
    model1,
    model2,
    observations: torch.Tensor,
    batch_size: int = 512
) -> Dict[str, float]:
    """
    Compute identifiability metrics between two models.

    Uses optimized batch computation to eliminate redundant forward passes
    and alignment calculations. This is ~10-20x faster than calling individual
    metric functions.
    """
    # Compute all metrics at once (optimized)
    metrics_dict = compute_model_identifiability_metrics(model1, model2, observations, batch_size)

    # Add 'identifiability/' prefix to all keys
    return {f'identifiability/{k}': v for k, v in metrics_dict.items()}


def compute_gt_metrics(
    model1,
    model2,
    observations: torch.Tensor,
    concepts: torch.Tensor,
    mixing_matrix: torch.Tensor,
    batch_size: int = 512
) -> Dict[str, float]:
    """
    Recompute ground truth metrics for both models against ground truth.

    This ensures gt_* metrics use the exact same data and alignment procedures
    as the identifiability/* metrics, eliminating inconsistencies that could arise
    from using different data samples or alignment methods.

    Args:
        model1: First trained model
        model2: Second trained model
        observations: Input observations [n_samples, input_dim]
        concepts: Ground truth concept activations [n_samples, concept_dim]
        mixing_matrix: Ground truth mixing matrix [concept_dim, observed_dim]
        batch_size: Batch size for computing activations

    Returns:
        Dict with eval/gt_* metrics, averaged across both models
    """
    from rsae.metrics.identifiability_metrics import (
        _collect_model_and_groundtruth_activations,
        compute_pairwise_identifiability_metrics
    )
    from rsae.utils.model_utils import extract_decoder_weights

    # Collect activations for model1 vs ground truth
    Z_model1, Z_true1 = _collect_model_and_groundtruth_activations(
        model1, observations, concepts, batch_size, binarize=False
    )

    # Collect activations for model2 vs ground truth
    Z_model2, Z_true2 = _collect_model_and_groundtruth_activations(
        model2, observations, concepts, batch_size, binarize=False
    )

    # Extract dictionaries
    dict1 = extract_decoder_weights(model1)
    dict2 = extract_decoder_weights(model2)
    true_dict = mixing_matrix.to(Z_true1.device)

    # Compute GT metrics for model1 (comparing ground truth to model1)
    gt_metrics1 = compute_pairwise_identifiability_metrics(
        Z_true1, Z_model1, dict1=true_dict, dict2=dict1
    )

    # Compute GT metrics for model2 (comparing ground truth to model2)
    gt_metrics2 = compute_pairwise_identifiability_metrics(
        Z_true2, Z_model2, dict1=true_dict, dict2=dict2
    )

    # Average metrics across both models (matching find_identifiability_pairs.py behavior)
    averaged_metrics = {}
    for key in gt_metrics1.keys():
        averaged_metrics[key] = (gt_metrics1[key] + gt_metrics2[key]) / 2

    # Add eval/gt_ prefix to match training convention
    return {f'eval/gt_{k}': v for k, v in averaged_metrics.items()}


def compute_oracle_identifiability_metrics(
    model1,
    model2,
    observations: torch.Tensor,
    k: int,
    batch_size: int = 512
) -> Dict[str, float]:
    """
    Compute identifiability metrics between oracle codes from both models' dictionaries.

    For each model, extracts its learned dictionary and uses OMP to find optimal
    sparse codes. Then compares these oracle codes using standard identifiability metrics.
    This isolates dictionary quality from encoder quality.

    Note: Only uses a single batch for efficiency since OMP runs on CPU.

    Args:
        model1: First trained model
        model2: Second trained model
        observations: Input observations [n_samples, input_dim]
        k: Sparsity level for OMP
        batch_size: Batch size for computing oracle codes

    Returns:
        Dict with oracle/* metrics
    """
    device = next(model1.parameters()).device

    # Extract dictionaries from both models
    dict1 = extract_decoder_weights(model1)  # [nb_concepts, input_dim]
    dict2 = extract_decoder_weights(model2)

    # Create oracles using each model's dictionary
    # Note: OMPOracle expects dictionary of shape (input_dim, nb_concepts)
    oracle1 = OMPOracle(dict1.T, k)
    oracle2 = OMPOracle(dict2.T, k)

    # Use single batch for efficiency (OMP is expensive on CPU)
    batch = observations[:batch_size].to(device)
    print(f"  Computing oracle codes for {batch.shape[0]} samples...")

    with torch.no_grad():
        oracle_Z1, _ = oracle1(batch)
        oracle_Z2, _ = oracle2(batch)

    # Compute identifiability metrics between oracle codes
    oracle_metrics = compute_pairwise_identifiability_metrics(
        oracle_Z1, oracle_Z2, dict1, dict2
    )

    # Add oracle/ prefix to all keys
    return {f'oracle/{k}': v for k, v in oracle_metrics.items()}


def compute_debugging_metrics(
    model1,
    model2,
    observations: torch.Tensor,
    concepts: torch.Tensor,
    mixing_matrix: torch.Tensor,
    batch_size: int = 512
) -> Dict[str, Any]:
    """
    Compute debugging diagnostics to understand identifiability metrics.

    Features:
    1. Data consistency checks (verify same data is used)
    2. Per-concept correlation statistics (identify problematic concepts)
    3. Alignment comparison (activation-based vs dictionary-based)

    Args:
        model1: First trained model
        model2: Second trained model
        observations: Input observations [n_samples, input_dim]
        concepts: Ground truth concept activations [n_samples, concept_dim]
        mixing_matrix: Ground truth mixing matrix [concept_dim, observed_dim]
        batch_size: Batch size for computing activations

    Returns:
        Dict with debugging information
    """
    from rsae.metrics.identifiability_metrics import (
        _collect_paired_activations,
        _collect_model_and_groundtruth_activations,
        compute_all_alignments
    )
    from rsae.utils.model_utils import extract_decoder_weights

    debug_info = {}

    # 1. Data consistency checks
    debug_info['data_shape'] = list(observations.shape)
    debug_info['data_mean'] = float(observations.mean().item())
    debug_info['data_std'] = float(observations.std().item())
    debug_info['data_min'] = float(observations.min().item())
    debug_info['data_max'] = float(observations.max().item())

    if concepts is not None:
        debug_info['concepts_shape'] = list(concepts.shape)
        debug_info['concepts_sparsity'] = float((concepts == 0).float().mean().item())
        debug_info['concepts_active_per_sample'] = float((concepts != 0).sum(dim=1).float().mean().item())

    # 2. Collect all activations
    Z1, Z2 = _collect_paired_activations(model1, model2, observations, batch_size)

    Z_model1, Z_true = _collect_model_and_groundtruth_activations(
        model1, observations, concepts, batch_size, binarize=False
    )

    # 3. Activation statistics
    debug_info['model1_sparsity'] = float((Z1 == 0).float().mean().item())
    debug_info['model2_sparsity'] = float((Z2 == 0).float().mean().item())
    debug_info['model1_active_per_sample'] = float((Z1 != 0).sum(dim=1).float().mean().item())
    debug_info['model2_active_per_sample'] = float((Z2 != 0).sum(dim=1).float().mean().item())

    # 4. Alignment comparison
    dict1 = extract_decoder_weights(model1)
    dict2 = extract_decoder_weights(model2)

    alignments = compute_all_alignments(Z1, Z2, dict1, dict2)

    # Compare activation-based vs dictionary-based alignment
    if 'dict_aligned_indices' in alignments:
        # Convert to numpy arrays (handle both torch tensors and numpy arrays)
        act_indices_raw = alignments['activation_aligned_indices']
        dict_indices_raw = alignments['dict_aligned_indices']

        if isinstance(act_indices_raw, torch.Tensor):
            act_indices = act_indices_raw.cpu().numpy()
        else:
            act_indices = np.asarray(act_indices_raw)

        if isinstance(dict_indices_raw, torch.Tensor):
            dict_indices = dict_indices_raw.cpu().numpy()
        else:
            dict_indices = np.asarray(dict_indices_raw)

        # Compute alignment agreement
        agreement = (act_indices == dict_indices).mean()
        debug_info['alignment_agreement'] = float(agreement)

        # Find misaligned concepts
        disagreements = np.where(act_indices != dict_indices)[0]
        debug_info['num_misaligned'] = int(len(disagreements))
        debug_info['misaligned_indices'] = disagreements[:10].tolist()  # Top 10

    # 5. Per-concept GT correlations (model1 vs ground truth)
    Z1_centered = Z_model1 - Z_model1.mean(dim=0, keepdim=True)
    Z_true_centered = Z_true - Z_true.mean(dim=0, keepdim=True)

    covariance = (Z1_centered * Z_true_centered).mean(dim=0)
    std1 = Z1_centered.std(dim=0)
    std_true = Z_true_centered.std(dim=0)

    per_concept_corr_gt = (covariance / (std1 * std_true + 1e-8)).abs()

    debug_info['gt_correlation_per_concept'] = {
        'mean': float(per_concept_corr_gt.mean().item()),
        'std': float(per_concept_corr_gt.std().item()),
        'min': float(per_concept_corr_gt.min().item()),
        'max': float(per_concept_corr_gt.max().item()),
        'median': float(per_concept_corr_gt.median().item()),
        'q25': float(per_concept_corr_gt.quantile(0.25).item()),
        'q75': float(per_concept_corr_gt.quantile(0.75).item()),
    }

    # Find best/worst aligned concepts
    sorted_indices = torch.argsort(per_concept_corr_gt, descending=True)
    debug_info['best_aligned_concepts'] = sorted_indices[:10].tolist()
    debug_info['worst_aligned_concepts'] = sorted_indices[-10:].tolist()

    # 6. Per-concept pairwise correlations (model1 vs model2)
    Z2_aligned = alignments['activation_aligned']
    Z1_centered_pair = Z1 - Z1.mean(dim=0, keepdim=True)
    Z2_centered = Z2_aligned - Z2_aligned.mean(dim=0, keepdim=True)

    covariance_pair = (Z1_centered_pair * Z2_centered).mean(dim=0)
    std1_pair = Z1_centered_pair.std(dim=0)
    std2 = Z2_centered.std(dim=0)

    per_concept_corr_pair = (covariance_pair / (std1_pair * std2 + 1e-8)).abs()

    debug_info['pairwise_correlation_per_concept'] = {
        'mean': float(per_concept_corr_pair.mean().item()),
        'std': float(per_concept_corr_pair.std().item()),
        'min': float(per_concept_corr_pair.min().item()),
        'max': float(per_concept_corr_pair.max().item()),
        'median': float(per_concept_corr_pair.median().item()),
        'q25': float(per_concept_corr_pair.quantile(0.25).item()),
        'q75': float(per_concept_corr_pair.quantile(0.75).item()),
    }

    return debug_info


def main():
    parser = argparse.ArgumentParser(
        description="Compare two models for identifiability assessment"
    )
    parser.add_argument('run1', type=str, help='First MLflow run ID')
    parser.add_argument('run2', type=str, help='Second MLflow run ID')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--tracking-uri', type=str, default='./mlruns',
                        help='MLflow tracking URI (default: ./mlruns)')
    parser.add_argument('--log-to-mlflow', action='store_true', help='Log results to MLflow')
    parser.add_argument('--experiment-name', type=str, default='rsae-experiments',
                        help='MLflow experiment for logging results')
    parser.add_argument('--original-metrics', type=str, default=None,
                        help='JSON string of original training metrics to log')
    parser.add_argument('--max-samples', type=int, default=100000,
                        help='Maximum number of samples to use for identifiability metrics (default: 100000)')
    parser.add_argument('--debug', action='store_true',
                        help='Compute and log detailed debugging metrics (synthetic data only)')

    args = parser.parse_args()

    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("=" * 80)
    print("Identifiability Assessment")
    print("=" * 80)
    print(f"Model 1: {args.run1}")
    print(f"Model 2: {args.run2}")
    print(f"Device: {device}")
    print("=" * 80)

    # Load models
    print("\nLoading models from MLflow...")
    model1, config1 = load_model_from_mlflow(args.run1, args.tracking_uri, device)
    model2, config2 = load_model_from_mlflow(args.run2, args.tracking_uri, device)
    print("Models loaded successfully!")

    # Verify configs match (except for seed)
    assert config1['model']['type'] == config2['model']['type'], \
        "Models must be of the same type"
    assert config1['data']['k'] == config2['data']['k'], \
        "Sparsity levels must match"
    assert config1['seed'].get('data') == config2['seed'].get('data'), \
        "Data seeds must match"

    # For OpenPhenom, input_dim is determined at runtime, so it may not be in config
    # For synthetic, it must match
    if config1['data'].get('data_source', 'synthetic') == 'synthetic':
        assert config1['data']['input_dim'] == config2['data']['input_dim'], \
            "Data dimensions must match"

    # Generate or load data using config from first model
    print("\nLoading/generating evaluation data...")
    observations, concepts, mixing_matrix = generate_or_load_data_from_config(config1, device, args)
    print(f"Data shape: {observations.shape}")

    # Subsample for memory efficiency
    if observations.shape[0] > args.max_samples:
        print(f"Subsampling from {observations.shape[0]} to {args.max_samples} samples for identifiability metrics...")
        indices = torch.randperm(observations.shape[0])[:args.max_samples]
        observations = observations[indices]
        if concepts is not None:
            concepts = concepts[indices]
        print(f"Subsampled data shape: {observations.shape}")

    # Compute identifiability metrics
    print("\nComputing identifiability metrics...")
    metrics = compute_identifiability_metrics(
        model1, model2, observations, args.batch_size
    )

    # Recompute GT metrics if we have ground truth (synthetic mode)
    if concepts is not None and mixing_matrix is not None:
        print("\nRecomputing ground truth metrics...")
        gt_metrics = compute_gt_metrics(
            model1, model2, observations, concepts, mixing_matrix, args.batch_size
        )
        metrics.update(gt_metrics)  # Adds eval/gt_* metrics

    # Compute oracle identifiability metrics
    print("\nComputing oracle identifiability metrics...")
    k = config1['data']['k']
    oracle_metrics = compute_oracle_identifiability_metrics(
        model1, model2, observations, k, args.batch_size
    )
    metrics.update(oracle_metrics)  # Adds oracle/* metrics

    # Compute debugging diagnostics if requested
    debug_info = None
    if args.debug and concepts is not None and mixing_matrix is not None:
        print("\nComputing debugging diagnostics...")
        debug_info = compute_debugging_metrics(
            model1, model2, observations, concepts, mixing_matrix, args.batch_size
        )

    # Display results
    print("\n" + "=" * 80)
    print("IDENTIFIABILITY METRICS")
    print("=" * 80)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print("=" * 80)

    # Display debug info if computed
    if debug_info is not None:
        print("\n" + "=" * 80)
        print("DEBUGGING DIAGNOSTICS")
        print("=" * 80)
        print(json.dumps(debug_info, indent=2, default=str))
        print("=" * 80)

    print("\nInterpretation:")
    print("- PW-MCC: Higher = better identifiability (1.0 = perfect)")
    print("- IoU: Higher = better identifiability (1.0 = perfect)")
    print("- Raw L2/MSE/Normalized L2: Lower = better identifiability (0.0 = perfect)")
    print("- Correlation: Higher = better identifiability (1.0 = perfect)")
    print("\nOracle metrics (oracle/*):")
    print("- Uses OMP with each model's dictionary to find optimal sparse codes")
    print("- Isolates dictionary quality from encoder quality")
    print("- Higher oracle identifiability suggests similar dictionaries")
    print("\nFor synthetic data:")
    print("- eval/gt_* metrics: Recomputed from same data as identifiability/*")
    print("- original/gt_* metrics: From training (may differ if data/alignment differs)")
    print("- Compare eval/gt_correlation vs original/gt_correlation to check consistency")

    # Log to MLflow if requested
    if args.log_to_mlflow:
        print("\nLogging results to MLflow...")
        mlflow.set_tracking_uri(args.tracking_uri)
        # Use race-condition-safe experiment creation
        experiment_id = get_or_create_experiment(args.experiment_name)

        # Flatten config for logging
        flattened_config = flatten_dict(config1)

        # Extract key fields for tags (for easy filtering)
        model_type = config1['model']['type']
        data_seed = config1.get('data', {}).get('seed') or config1.get('seed', {}).get('data')
        rip_weight = config1['model'].get('rip_weight', None)
        connectedness_weight = config1['model'].get('connectedness_weight', None)

        mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"compare_{args.run1[:8]}_vs_{args.run2[:8]}",
            tags={
                "job_type": "identifiability",
                "run1": args.run1,
                "run2": args.run2,
                "model_type": model_type,
                "data_seed": str(data_seed),
                "rip_weight": str(rip_weight) if rip_weight is not None else "None",
                "connectedness_weight": str(connectedness_weight) if connectedness_weight is not None else "None",
            }
        )

        # Log ALL config parameters (flattened)
        comparison_params = {
            'run1': args.run1,
            'run2': args.run2,
        }
        all_params = {**comparison_params, **flattened_config}
        mlflow.log_params(all_params)
        print(f"Logged {len(flattened_config)} config parameters")

        mlflow.log_metrics(metrics)

        # Log original metrics if provided
        if args.original_metrics:
            try:
                original_metrics_dict = json.loads(args.original_metrics)
                # Log with "original/" prefix
                original_metrics_prefixed = {
                    f"original/{key}": value
                    for key, value in original_metrics_dict.items()
                }
                mlflow.log_metrics(original_metrics_prefixed)
                print(f"Logged {len(original_metrics_dict)} original metrics")
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse original metrics JSON: {e}", file=sys.stderr)

        # Log debug metrics if computed
        if debug_info is not None:
            flat_debug = flatten_dict({'debug': debug_info})
            mlflow.log_params(flat_debug)
            print(f"Logged {len(flat_debug)} debug parameters")

        mlflow.end_run()
        print("Results logged to MLflow!")

    print("\nDone!")


if __name__ == "__main__":
    main()
