#!/usr/bin/env python3
"""Run SAEBench evaluations on a model from MLflow.

This script:
1. Downloads a trained model from MLflow
2. Runs SAEBench evaluations
3. Saves results to eval_results/

Usage:
    python scripts/run_saebench_from_mlflow.py \\
        --run-id YOUR_RUN_ID \\
        --eval-types core sparse_probing \\
        --device cuda
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set

import mlflow
import mlflow.pytorch

# Add SAEBench to path
saebench_path = Path(__file__).parent.parent / "SAEBench"
sys.path.insert(0, str(saebench_path))

from sae_bench.custom_saes import run_all_evals_custom_saes
import sae_bench.sae_bench_utils.general_utils as general_utils

from rsae.saebench_wrapper import RIPTopKSAEBench
from rsae.utils import flatten_dict, get_or_create_experiment
from rsae.utils.model_utils import extract_decoder_weights


def infer_sae_dimensions(model) -> tuple[int, int]:
    """Infer (d_in, d_sae) from a RIPTopK-like model."""
    d_in = getattr(model, "input_dim", None)
    d_sae = getattr(model, "nb_concepts", None)

    if d_in is None or d_sae is None:
        try:
            decoder_weights = extract_decoder_weights(model)
            d_sae = d_sae or decoder_weights.shape[0]
            d_in = d_in or decoder_weights.shape[1]
        except Exception:
            pass

    if d_in is None and hasattr(model, "decoder") and hasattr(model.decoder, "out_features"):
        d_in = model.decoder.out_features
    if d_sae is None and hasattr(model, "decoder") and hasattr(model.decoder, "in_features"):
        d_sae = model.decoder.in_features

    if d_in is None or d_sae is None:
        raise AttributeError("Could not infer d_in or d_sae from model")

    return d_in, d_sae


def extract_sae_hyperparameters(model) -> Dict[str, Any]:
    """Extract all SAE hyperparameters from model for MLflow logging.

    Args:
        model: Trained RIPTopK model

    Returns:
        Dictionary of hyperparameters with 'sae/' prefix
    """
    d_in, d_sae = infer_sae_dimensions(model)
    hyperparams: Dict[str, Any] = {
        # Core architecture
        "sae/d_in": d_in,
        "sae/nb_concepts": d_sae,
    }

    # Optional attributes (log if present)
    for key, attr in [
        ("sae/top_k", "top_k"),
        ("sae/rip_weight", "rip_weight"),
        ("sae/rip_loss_weighted", "rip_loss_weighted"),
        ("sae/auxk_weight", "auxk_weight"),
        ("sae/auxiliary_k", "auxiliary_k"),
        ("sae/activation_window_batches", "activation_window_batches"),
        ("sae/normalization", "normalization"),
    ]:
        value = getattr(model, attr, None)
        if value is not None:
            hyperparams[key] = value

    if hasattr(model, "rip_loss_mixup"):
        hyperparams["sae/rip_loss_mixup"] = str(model.rip_loss_mixup)

    return hyperparams


def log_eval_results_to_mlflow(json_path: Path, eval_type: str) -> None:
    """Parse and log a single evaluation result to current MLflow run.

    Args:
        json_path: Path to evaluation result JSON (relative to eval_results/)
        eval_type: Type of evaluation (core, sparse_probing, etc.)
    """
    full_path = Path("eval_results") / json_path

    try:
        with open(full_path) as f:
            result = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  ✗ Error parsing {json_path}: {e}")
        return
    except FileNotFoundError:
        print(f"  ✗ File not found: {full_path}")
        return

    # Validate required fields
    required_keys = ['eval_type_id', 'eval_result_metrics', 'eval_config']
    missing = [k for k in required_keys if k not in result]
    if missing:
        print(f"  ✗ Missing required keys in {json_path}: {missing}")
        return

    # Verify eval type matches
    if result['eval_type_id'] != eval_type:
        print(f"  ⚠ Eval type mismatch: expected {eval_type}, got {result['eval_type_id']}")

    print(f"  Logging {eval_type} results...")

    # Log metrics with namespace
    metrics = flatten_dict(result['eval_result_metrics'], parent_key=f"saebench/{eval_type}", sep="/")
    # Filter out non-numeric values and sentinel values
    metrics_to_log = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and v != -1:  # -1 is common sentinel for "not computed"
            metrics_to_log[k] = float(v)

    if metrics_to_log:
        mlflow.log_metrics(metrics_to_log)
        print(f"    ✓ Logged {len(metrics_to_log)} metrics")

    # Log eval config as parameters
    config = flatten_dict(result['eval_config'], parent_key=f"eval_config/{eval_type}", sep="/")
    if config:
        mlflow.log_params(config)
        print(f"    ✓ Logged {len(config)} config parameters")

    # Log full result as artifact
    artifact_dir = f"saebench/{eval_type}"
    mlflow.log_dict(result, f"{artifact_dir}/results.json")

    # Log details separately if they exist
    if 'eval_result_details' in result and result['eval_result_details']:
        mlflow.log_dict({'details': result['eval_result_details']}, f"{artifact_dir}/details.json")
        print(f"    ✓ Logged full results and details as artifacts")
    else:
        print(f"    ✓ Logged full results as artifact")


def log_all_eval_results(
    new_result_files: Set[Path],
    sae_name: str,
    training_run_id: str,
    training_experiment_name: str,
    model_name: str,
    hook_layer: int,
    eval_types: List[str],
    tracking_uri: str,
    sae_hyperparams: Dict[str, Any],
    model_variant: str = None,
    saebench_experiment_name: str = None,
) -> None:
    """Log all evaluation results to MLflow in new saebench experiment.

    Creates a new run in the SAEbench experiment. By default this is
    "{training_experiment_name}-saebench", but it can be overridden via
    saebench_experiment_name.

    Args:
        new_result_files: Set of new result file paths (relative to eval_results/)
        sae_name: Name of the SAE
        training_run_id: Run ID from training
        training_experiment_name: Name of training experiment
        model_name: Model name (e.g., pythia-70m-deduped)
        hook_layer: Layer number
        eval_types: List of eval types run
        tracking_uri: MLflow tracking URI
        sae_hyperparams: SAE hyperparameters to log
        model_variant: SAE variant name (model.name) for grouping results
        saebench_experiment_name: Explicit SAEbench experiment name. Falls back
            to "{training_experiment_name}-saebench" when None.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    # Get/create saebench experiment
    if saebench_experiment_name is None:
        saebench_experiment_name = f"{training_experiment_name}-saebench"
    # Use race-condition-safe experiment creation
    experiment_id = get_or_create_experiment(saebench_experiment_name)

    # Start MLflow run
    run_name = f"{sae_name}_saebench"
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        print(f"  MLflow run: {run.info.run_id}")

        # Set tags for traceability
        tags = {
            "training_run_id": training_run_id,
            "training_experiment": training_experiment_name,
            "sae_name": sae_name,
            "model_name": model_name,
            "hook_layer": str(hook_layer),
            "eval_types": ",".join(eval_types),
            "job_type": "saebench_eval",
        }
        if model_variant:
            tags["model.name"] = model_variant
        mlflow.set_tags(tags)

        # Log SAE hyperparameters
        params_to_log = dict(sae_hyperparams)
        if model_variant:
            params_to_log["model.name"] = model_variant
        mlflow.log_params(params_to_log)
        print(f"  ✓ Logged {len(params_to_log)} SAE hyperparameters")

        # Log each result file
        success_count = 0
        for result_path in sorted(new_result_files):
            # Extract eval type from path (e.g., "core/file.json" -> "core")
            eval_type = result_path.parts[0] if len(result_path.parts) > 1 else "unknown"

            try:
                log_eval_results_to_mlflow(result_path, eval_type)
                success_count += 1
            except Exception as e:
                print(f"  ✗ Error logging {result_path}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n  ✓ Successfully logged {success_count}/{len(new_result_files)} result files")
        print(f"  View in MLflow UI: {tracking_uri}")


def cleanup_result_files(result_files: Set[Path], eval_results_dir: Path) -> None:
    """Clean up result files after successful MLflow logging.

    Args:
        result_files: Set of result file paths (relative to eval_results/)
        eval_results_dir: Base directory for eval results
    """
    deleted_count = 0
    failed_deletions = []

    for result_path in result_files:
        full_path = eval_results_dir / result_path
        try:
            if full_path.exists():
                full_path.unlink()
                deleted_count += 1
        except Exception as e:
            failed_deletions.append((result_path, str(e)))

    print(f"\n  Cleaned up {deleted_count}/{len(result_files)} result files")
    if failed_deletions:
        print(f"  ✗ Failed to delete {len(failed_deletions)} files:")
        for path, error in failed_deletions:
            print(f"    - {path}: {error}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAEBench evaluations on a model from MLflow"
    )

    # MLflow arguments
    parser.add_argument("--run-id", type=str, required=True,
                        help="MLflow run ID")
    parser.add_argument("--tracking-uri", type=str, default="./mlruns",
                        help="MLflow tracking URI (default: ./mlruns)")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="MLflow experiment name (optional, for display)")
    parser.add_argument("--saebench-experiment-name", type=str, default=None,
                        help="MLflow experiment name to log SAEbench results to "
                             "(default: '{training_experiment_name}-saebench')")

    # Evaluation arguments
    parser.add_argument("--eval-types", type=str, nargs="+",
                        default=["core", "sparse_probing"],
                        choices=["absorption", "autointerp", "core", "ravel",
                                 "scr", "tpp", "sparse_probing", "unlearning"],
                        help="Evaluation types to run")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Force rerun even if results exist")
    parser.add_argument("--save-activations", action="store_true",
                        help="Save activations for reuse (requires ~100GB disk space)")

    # Output arguments
    parser.add_argument("--sae-name", type=str, default=None,
                        help="Name for this SAE in results (default: run_id)")
    parser.add_argument("--model-variant", type=str, default=None,
                        help="SAE variant name from training (model.name), used "
                             "to group saebench results by variant")

    # Device arguments
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for evaluation (cuda/cpu)")

    # Model evaluation config
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for LLM evaluation (optional)")
    parser.add_argument("--dtype", type=str, default=None,
                        choices=["float32", "float16", "bfloat16"],
                        help="Data type for evaluation (optional)")

    # API key for autointerp
    parser.add_argument("--api-key-file", type=str, default="openai_api_key.txt",
                        help="Path to file containing OpenAI API key (for autointerp)")

    return parser.parse_args()


# Model configs (defaults for common models from SAEBench)
MODEL_CONFIGS = {
    "pythia-70m": {
        "batch_size": 512,
        "dtype": "float32",
    },
    "pythia-70m-deduped": {
        "batch_size": 512,
        "dtype": "float32",
    },
    "pythia-160m": {
        "batch_size": 256,
        "dtype": "float32",
    },
    "pythia-160m-deduped": {
        "batch_size": 256,
        "dtype": "float32",
    },
    "gemma-2-2b": {
        "batch_size": 32,
        "dtype": "bfloat16",
    },
}

DEFAULT_MODEL_CONFIG = {
    "batch_size": 256,
    "dtype": "float32",
}


def main():
    args = parse_args()

    # Setup MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=args.tracking_uri)

    print("=" * 80)
    print("SAEBench Evaluation from MLflow")
    print("=" * 80)
    print(f"Run ID: {args.run_id}")
    print(f"Tracking URI: {args.tracking_uri}")
    print()

    # Step 1: Get run info
    print("Step 1: Loading run metadata from MLflow...")
    run = client.get_run(args.run_id)
    params = run.data.params

    # Get original experiment for later use
    original_experiment = client.get_experiment(run.info.experiment_id)

    # Extract model info from params
    model_name = params.get('data.model_name', params.get('data/model_name', 'pythia-70m-deduped'))
    hook_layer = int(params.get('data.hook_layer', params.get('data/hook_layer', 3)))

    # SAE variant name (cfg.model.name) — propagated for grouping in saebench
    # results. Prefer the CLI value, fall back to the training run's params.
    model_variant = args.model_variant or params.get('model.name') or params.get('model/name')

    # Strip organization prefix for SAEBench / TransformerLens compatibility
    # (e.g. "EleutherAI/pythia-70m-deduped" -> "pythia-70m-deduped"). We do NOT
    # rewrite the model identity (e.g. swap "pythia-160m" for the deduped
    # variant) — those are different models with different residual streams,
    # so evaluating an SAE on the wrong base model silently produces garbage.
    if '/' in model_name:
        saebench_model_name = model_name.split('/')[-1]
    else:
        saebench_model_name = model_name

    print(f"  Model: {model_name}")
    print(f"  Layer: {hook_layer}")
    print(f"  Status: {run.info.status}")
    print()

    # Get model config (use saebench_model_name for lookup)
    if saebench_model_name in MODEL_CONFIGS:
        model_config = MODEL_CONFIGS[saebench_model_name].copy()
        print(f"Using predefined config for {saebench_model_name}")
    else:
        model_config = DEFAULT_MODEL_CONFIG.copy()
        print(f"Using default config for {model_name}")

    # Override with command-line arguments if provided
    if args.batch_size is not None:
        model_config["batch_size"] = args.batch_size
    if args.dtype is not None:
        model_config["dtype"] = args.dtype

    print(f"  batch_size: {model_config['batch_size']}")
    print(f"  dtype: {model_config['dtype']}")
    print()

    # Step 2: Load model from MLflow
    print("Step 2: Loading model from MLflow...")
    model_uri = f"runs:/{args.run_id}/model"
    device = general_utils.setup_environment()
    dtype = general_utils.str_to_dtype(model_config["dtype"])

    try:
        # Load the raw model
        native_model = mlflow.pytorch.load_model(model_uri, map_location=device)
        native_model.eval()
        print(f"✓ Loaded model from MLflow")
        d_in, d_sae = infer_sae_dimensions(native_model)
        print(f"  d_in: {d_in}")
        print(f"  d_sae: {d_sae}")
        if hasattr(native_model, "top_k"):
            print(f"  k: {native_model.top_k}")
        print()

        # Wrap in SAEBench interface
        print("Step 3: Wrapping model in SAEBench interface...")
        sae = RIPTopKSAEBench(
            riptopk_model=native_model,
            d_in=d_in,
            d_sae=d_sae,
            model_name=saebench_model_name,  # Use short name for SAEBench
            hook_layer=hook_layer,
            device=device,
            dtype=dtype,
        )
        print("✓ SAE wrapped successfully")
        print()

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 4: Test SAE
    print("Step 4: Testing SAE...")
    try:
        sae.test_sae(saebench_model_name)  # Use short name for SAEBench
        print("✓ SAE validation passed!")
    except Exception as e:
        print(f"✗ SAE validation failed: {e}")
        print("Continuing anyway, but results may be incorrect...")

    # Check decoder normalization
    if sae.check_decoder_norms():
        print("✓ Decoder weights are normalized")
    else:
        print("✗ Warning: Decoder weights are not normalized")
    print()

    # Step 5: Run evaluations
    print("Step 5: Running SAEBench Evaluations")
    print("=" * 80)

    # Generate SAE name
    if args.sae_name is None:
        args.sae_name = f"riptopk_{model_name.replace('/', '-')}_l{hook_layer}_{args.run_id[:8]}"

    selected_saes = [(args.sae_name, sae)]

    # Load API key if needed
    api_key = None
    if "autointerp" in args.eval_types:
        try:
            with open(args.api_key_file) as f:
                api_key = f.read().strip()
            print(f"✓ Loaded OpenAI API key from {args.api_key_file}")
        except FileNotFoundError:
            print(f"✗ Warning: Could not load API key from {args.api_key_file}")
            print("  Autointerp evaluation will be skipped")

    print(f"\nRunning evaluations: {', '.join(args.eval_types)}")
    print(f"Results will be saved to: eval_results/")
    print()

    # Track result files before running evals (to identify new files created by this run)
    eval_results_dir = Path("eval_results")
    existing_files = set()
    if eval_results_dir.exists():
        existing_files = {
            f.relative_to(eval_results_dir)
            for f in eval_results_dir.rglob("*.json")
        }

    # Run evaluations
    run_all_evals_custom_saes.run_evals(
        model_name=saebench_model_name,  # Use short name for SAEBench
        selected_saes=selected_saes,
        llm_batch_size=model_config["batch_size"],
        llm_dtype=model_config["dtype"],
        device=device,
        eval_types=args.eval_types,
        api_key=api_key,
        force_rerun=args.force_rerun,
        save_activations=args.save_activations,
    )

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)
    print("\nResults locations:")
    for eval_type in args.eval_types:
        print(f"  {eval_type}: eval_results/{eval_type}/")

    # Find new result files (filter to only this SAE's results)
    new_files = set()
    if eval_results_dir.exists():
        all_new_files = {
            f.relative_to(eval_results_dir)
            for f in eval_results_dir.rglob("*.json")
        } - existing_files

        # Filter to only files belonging to our SAE (to avoid logging other parallel jobs' results)
        new_files = {
            f for f in all_new_files
            if args.sae_name in str(f)
        }
        print(f"\n✓ Generated {len(new_files)} new result files for {args.sae_name}")
        if len(all_new_files) > len(new_files):
            print(f"  (Ignored {len(all_new_files) - len(new_files)} files from other SAEs)")

    # Step 6 - Log results to MLflow
    if new_files:
        print("\n" + "=" * 80)
        print("Step 6: Logging Results to MLflow")
        print("=" * 80)

        # Extract SAE hyperparameters from model
        sae_hyperparams = extract_sae_hyperparameters(native_model)

        try:
            log_all_eval_results(
                new_result_files=new_files,
                sae_name=args.sae_name,
                training_run_id=args.run_id,
                training_experiment_name=original_experiment.name,
                model_name=model_name,
                hook_layer=hook_layer,
                eval_types=args.eval_types,
                tracking_uri=args.tracking_uri,
                sae_hyperparams=sae_hyperparams,
                model_variant=model_variant,
                saebench_experiment_name=args.saebench_experiment_name,
            )
            print("\n✓ Successfully logged results to MLflow!")

            # Clean up result files after successful MLflow logging
            print("\n" + "=" * 80)
            print("Step 7: Cleaning Up Result Files")
            print("=" * 80)
            cleanup_result_files(new_files, eval_results_dir)
            print("✓ Result files cleaned up - MLflow is now source of truth")

        except Exception as e:
            print(f"\n✗ Error logging to MLflow: {e}")
            print("  Results are still saved locally in eval_results/")
            print("  Skipping cleanup to preserve results for manual inspection")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠ No new result files found - skipping MLflow logging")
        print("  This may happen if results already exist (use --force-rerun to regenerate)")

    print("\nTo visualize results, see SAEBench/sae_bench_demo.ipynb")


if __name__ == "__main__":
    sys.exit(main() or 0)
