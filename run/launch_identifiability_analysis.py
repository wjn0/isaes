#!/usr/bin/env python3
"""Run identifiability analysis on all pairs from an experiment.

This script combines find_identifiability_pairs.py and compare_identifiability.py
to run a complete identifiability analysis pipeline.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

from rsae.utils import get_or_create_experiment


def find_pairs(experiment_name: str, grouping_keys: str, tracking_uri: str, min_pairs: int) -> List[Dict[str, Any]]:
    """Find identifiability pairs using find_identifiability_pairs.py."""
    cmd = [
        "python", "scripts/find_identifiability_pairs.py",
        experiment_name,
        "--grouping-keys", grouping_keys,
        "--tracking-uri", tracking_uri,
        "--min-pairs", str(min_pairs),
        "--json"
    ]

    print("Step 1: Finding identifiability pairs...")
    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error finding pairs: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Parse JSON output
    pairs = json.loads(result.stdout)
    return pairs


def run_comparison(pair: Dict[str, Any], compare_options: List[str], current: int, total: int):
    """Run comparison for a single pair (sequential execution)."""
    run1 = pair['run1']
    run2 = pair['run2']

    # Extract metadata for display
    group = pair['group']
    model_seed1 = pair['model_seed1']
    model_seed2 = pair['model_seed2']

    print()
    print(f"Pair {current}/{total}")
    print(f"  Config:")
    # Display all grouping keys dynamically
    for key, value in group.items():
        # Format nested dict keys
        if isinstance(value, dict):
            for subkey, subval in value.items():
                print(f"    - {key}.{subkey}: {subval}")
        else:
            print(f"    - {key}: {value}")
    print(f"  Run 1: {run1[:8]} (model_seed={model_seed1})")
    print(f"  Run 2: {run2[:8]} (model_seed={model_seed2})")
    print()

    # Build command
    cmd = [
        "python", "scripts/compare_identifiability.py",
        run1, run2
    ] + compare_options

    # Add original metrics if available
    if 'original_metrics' in pair and pair['original_metrics']:
        cmd.extend(["--original-metrics", json.dumps(pair['original_metrics'])])

    # Run comparison (shell=False to avoid argument splitting)
    result = subprocess.run(cmd, shell=False)

    if result.returncode != 0:
        print(f"Warning: Comparison failed for pair {run1[:8]} vs {run2[:8]}", file=sys.stderr)

    print("-" * 80)


def run_comparison_slurm(original_experiment_name: str, pair: Dict[str, Any], compare_options: List[str],
                         slurm_args: Dict[str, Any], current: int, total: int) -> str:
    """Submit comparison job to Slurm (parallel execution)."""
    run1 = pair['run1']
    run2 = pair['run2']

    # Extract metadata
    group = pair['group']
    # Get first two values from group for display (flexible with different grouping keys)
    group_display = ', '.join(f"{k}={v}" for k, v in list(group.items())[:2])

    # Build command
    cmd = [
        "python", "scripts/compare_identifiability.py",
        run1, run2
    ] + compare_options

    # Add original metrics if available
    if 'original_metrics' in pair and pair['original_metrics']:
        cmd.extend(["--original-metrics", json.dumps(pair['original_metrics'])])

    # Build full command string with proper shell quoting
    cmd_str = " ".join(shlex.quote(arg) for arg in cmd)

    # Create unique job name
    job_name = f"compare_{original_experiment_name}_{run1[:8]}_{run2[:8]}"

    # Submit to slurm
    sbatch_cmd = [
        "sbatch",
        f"--partition={slurm_args['partition']}",
        f"--time={slurm_args['job_time']}",
        f"--gres=gpu:{slurm_args['job_gpus']}",
        f"--mem={slurm_args['job_mem']}",
        f"--cpus-per-task={slurm_args['job_cpus']}",
        f"--job-name={job_name}",
        f"--output=logs/{job_name}_%j.out",
        f"--error=logs/{job_name}_%j.err",
        f"--wrap={cmd_str}"
    ]

    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Warning: Failed to submit job for pair {run1[:8]} vs {run2[:8]}", file=sys.stderr)
        print(f"Error: {result.stderr}", file=sys.stderr)
        return None

    # Extract job ID from sbatch output
    job_id = result.stdout.strip().split()[-1]

    print(f"  Pair {current}/{total}: {run1[:8]} vs {run2[:8]} -> Job {job_id} ({group_display})")

    return job_id


def query_comparison_runs(experiment_name: str, tracking_uri: str) -> List[Dict[str, Any]]:
    """Query all finished comparison runs from MLflow experiment.

    Returns:
        List of dicts containing run data with keys: run_id, params, metrics
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Get comparison experiment
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Warning: Experiment '{experiment_name}' not found", file=sys.stderr)
            return []
    except Exception as e:
        print(f"Error accessing MLflow: {e}", file=sys.stderr)
        return []

    # Get all finished runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'"
    )

    run_data = []
    for run in runs:
        data = {
            'run_id': run.info.run_id,
            'params': dict(run.data.params),
            'metrics': dict(run.data.metrics),
        }
        run_data.append(data)

    return run_data


def get_param_value(params: Dict[str, str], key: str, default=None) -> Optional[float]:
    """Extract parameter value, handling nested keys and type conversion."""
    # Try both dotted and slash notation
    for separator in ['.', '/']:
        param_key = key.replace('.', separator)
        if param_key in params:
            try:
                val = params[param_key]
                if val.lower() == 'none':
                    return default
                return float(val)
            except (ValueError, AttributeError):
                return default
    return default


def aggregate_data(data: List[Dict[str, Any]], x_key: str, y_key: str,
                   group_keys: List[str], filter_conditions: Optional[Dict[str, float]] = None) -> Dict:
    """Aggregate data for plotting.
    Args:
        data: List of run data dicts
        x_key: Parameter key for x-axis (e.g., 'data.mixture_scale')
        y_key: Metric key for y-axis (e.g., 'original/rip_loss')
        group_keys: Parameter keys for grouping (e.g., ['model.rip_weight'])
        filter_conditions: Optional dict of param filters (e.g., {'model.auxk_weight': 0})
    Returns:
        Dict mapping group_value -> {x_values: [...], y_values: [...]}
    """
    # Group data
    grouped = defaultdict(lambda: ([], []))  # Stores (x_values, y_values)

    for run in data:
        # Check filter conditions
        if filter_conditions:
            skip = False
            for param_key, target_value in filter_conditions.items():
                value = get_param_value(run['params'], param_key)
                if value is None or abs(value - target_value) > 1e-6:
                    skip = True
                    break
            if skip:
                continue

        # Extract x and y values
        x_val = get_param_value(run['params'], x_key)
        y_val = run['metrics'].get(y_key)

        if x_val is None or y_val is None:
            continue

        # Create group tuple
        group_vals = tuple(get_param_value(run['params'], gk) for gk in group_keys)

        grouped[group_vals][0].append(x_val)
        grouped[group_vals][1].append(y_val)

    # Convert to the desired output format
    result = {}
    for group_vals, (x_values, y_values) in grouped.items():
        result[group_vals] = {
            'x_values': x_values,
            'y_values': y_values,
        }

    return result


def plot_mixture_scale_vs_rip_loss(data: List[Dict[str, Any]], output_path: str):
    """Plot 1: mixture_scale vs rip_loss, grouped by rip_weight."""
    aggregated = aggregate_data(
        data,
        x_key='data.mixture_scale',
        y_key='original/train/rip_loss',
        group_keys=['model.rip_weight']
    )

    if not aggregated:
        print("Warning: No data for mixture_scale vs rip_loss plot", file=sys.stderr)
        return False

    plt.figure(figsize=(10, 6))

    for (rip_weight,), values in sorted(aggregated.items()):
        if rip_weight is None:
            continue
        plt.scatter(values['x_values'], values['y_values'],
                    label=f'rip_weight={rip_weight:.2f}', alpha=0.6)

    plt.xlabel('Mixture Scale')
    plt.ylabel('RIP Loss')
    plt.title('Mixture Scale vs RIP Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return True


def plot_mixture_scale_vs_iou(data: List[Dict[str, Any]], output_path: str):
    """Plot 2: mixture_scale vs identifiability/iou, grouped by rip_weight."""
    aggregated = aggregate_data(
        data,
        x_key='data.mixture_scale',
        y_key='identifiability/iou',
        group_keys=['model.rip_weight']
    )

    if not aggregated:
        print("Warning: No data for mixture_scale vs iou plot", file=sys.stderr)
        return False

    plt.figure(figsize=(10, 6))

    for (rip_weight,), values in sorted(aggregated.items()):
        if rip_weight is None:
            continue
        plt.scatter(values['x_values'], values['y_values'],
                    label=f'rip_weight={rip_weight:.2f}', alpha=0.6)

    plt.xlabel('Mixture Scale')
    plt.ylabel('Identifiability (IoU)')
    plt.title(f'Mixture Scale vs Identifiability/IoU')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return True


def plot_mixture_scale_vs_pw_mcc(data: List[Dict[str, Any]], output_path: str):
    """Plot 3: mixture_scale vs pw_mcc, grouped by rip_weight."""
    aggregated = aggregate_data(
        data,
        x_key='data.mixture_scale',
        y_key='identifiability/pw_mcc',
        group_keys=['model.rip_weight']
    )

    if not aggregated:
        print("Warning: No data for mixture_scale vs pw_mcc plot", file=sys.stderr)
        return False

    plt.figure(figsize=(10, 6))

    for (rip_weight,), values in sorted(aggregated.items()):
        if rip_weight is None:
            continue
        plt.scatter(values['x_values'], values['y_values'],
                    label=f'rip_weight={rip_weight:.2f}', alpha=0.6)

    plt.xlabel('Mixture Scale')
    plt.ylabel('PW-MCC')
    plt.title(f'Mixture Scale vs PW-MCC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return True


def plot_auxk_weight_vs_dead_proportion(data: List[Dict[str, Any]], output_path: str):
    """Plot 4: auxk_weight vs dead_concept_proportion, grouped by mixture_scale and rip_weight."""
    aggregated = aggregate_data(
        data,
        x_key='model.auxk_weight',
        y_key='original/eval/dead_concept_proportion',
        group_keys=['data.mixture_scale', 'model.rip_weight']
    )

    if not aggregated:
        print("Warning: No data for auxk_weight vs dead_concept_proportion plot", file=sys.stderr)
        return False

    plt.figure(figsize=(12, 6))

    # Get unique mixture_scales and rip_weights for color and marker mapping
    mixture_scales = sorted(set(group[0] for group in aggregated.keys() if group[0] is not None))
    rip_weights = sorted(set(group[1] for group in aggregated.keys() if group[1] is not None))

    # Color map for mixture_scale
    colors = plt.cm.viridis(np.linspace(0, 1, len(mixture_scales)))
    color_map = {ms: colors[i] for i, ms in enumerate(mixture_scales)}

    # Marker map for rip_weight
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    marker_map = {rw: markers[i % len(markers)] for i, rw in enumerate(rip_weights)}

    for (mixture_scale, rip_weight), values in sorted(aggregated.items()):
        if mixture_scale is None or rip_weight is None:
            continue

        color = color_map.get(mixture_scale, 'black')
        marker = marker_map.get(rip_weight, 'o')

        plt.scatter(values['x_values'], values['y_values'],
                    color=color, marker=marker,
                    label=f'mix={mixture_scale:.2f}, rip={rip_weight:.2f}', alpha=0.7)

    # Create custom legend to avoid duplicate labels
    from matplotlib.lines import Line2D
    legend_elements = []
    for ms in mixture_scales:
        legend_elements.append(Line2D([0], [0], color=color_map[ms], lw=4, label=f'mix_scale={ms:.2f}'))
    for rw in rip_weights:
        legend_elements.append(Line2D([0], [0], marker=marker_map[rw], color='grey', label=f'rip_weight={rw:.2f}', linestyle='None'))

    plt.xlabel('AuxK Weight')
    plt.ylabel('Dead Concept Proportion')
    plt.title('AuxK Weight vs Dead Concept Proportion')
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return True


def generate_summary_plots(comparison_experiment: str, tracking_uri: str, output_dir: str = 'plots'):
    """Generate summary plots from comparison experiment results.

    Args:
        comparison_experiment: Name of the MLflow comparison experiment
        tracking_uri: MLflow tracking URI
        output_dir: Directory to save plots (default: 'plots')

    Returns:
        List of generated plot paths
    """
    print("\n" + "=" * 80)
    print("Generating Summary Plots")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Query comparison runs
    print(f"Querying comparison runs from experiment: {comparison_experiment}")
    data = query_comparison_runs(comparison_experiment, tracking_uri)

    if not data:
        print("No comparison runs found. Skipping plot generation.")
        return []

    print(f"Found {len(data)} comparison runs")

    # Generate plots
    plot_paths = []

    print("\nGenerating plots...")

    # Plot 1: mixture_scale vs rip_loss
    plot_path = os.path.join(output_dir, 'mixture_scale_vs_rip_loss.png')
    print(f"  1. mixture_scale vs train/rip_loss -> {plot_path}")
    if plot_mixture_scale_vs_rip_loss(data, plot_path):
        plot_paths.append(plot_path)

    # Plot 2: mixture_scale vs iou
    plot_path = os.path.join(output_dir, 'mixture_scale_vs_iou.png')
    print(f"  2. mixture_scale vs identifiability/iou -> {plot_path}")
    if plot_mixture_scale_vs_iou(data, plot_path):
        plot_paths.append(plot_path)

    # Plot 3: mixture_scale vs pw_mcc
    plot_path = os.path.join(output_dir, 'mixture_scale_vs_pw_mcc.png')
    print(f"  3. mixture_scale vs pw_mcc -> {plot_path}")
    if plot_mixture_scale_vs_pw_mcc(data, plot_path):
        plot_paths.append(plot_path)

    # Plot 4: auxk_weight vs dead_concept_proportion
    plot_path = os.path.join(output_dir, 'auxk_weight_vs_dead_proportion.png')
    print(f"  4. auxk_weight vs eval/dead_concept_proportion -> {plot_path}")
    if plot_auxk_weight_vs_dead_proportion(data, plot_path):
        plot_paths.append(plot_path)

    print(f"\nGenerated {len(plot_paths)} plots")
    print("=" * 80)

    return plot_paths


def main():
    parser = argparse.ArgumentParser(
        description="Run identifiability analysis on all pairs from an experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/run_identifiability_analysis.py rsae-gaussian-large-4 \\
    --grouping-keys seed.data,model.rip_weight,data.mixture_scale,model.auxk_weight \\
    --device cuda --log-to-mlflow
        """
    )

    # Required arguments
    parser.add_argument('experiment_name', type=str,
                       help='MLflow experiment name')

    # Options for find_identifiability_pairs
    parser.add_argument('--grouping-keys', type=str, required=True,
                       help='Comma-separated list of config keys to group by (e.g., seed.data,model.rip_weight,data.mixture_scale)')
    parser.add_argument('--tracking-uri', type=str, default='./mlruns',
                       help='MLflow tracking URI (default: ./mlruns)')
    parser.add_argument('--min-pairs', type=int, default=2,
                       help='Minimum number of runs per group (default: 2)')

    # Options for compare_identifiability
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for comparison (cuda/cpu, default: cuda)')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size for evaluation (default: 512)')
    parser.add_argument('--log-to-mlflow', action='store_true',
                       help='Log comparison results to MLflow')
    parser.add_argument('--comparison-experiment', type=str, default=None,
                       help='MLflow experiment name for logging comparison results (default: {experiment_name}-id-comparison)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with logging of original metrics')

    # Slurm options for parallel execution
    parser.add_argument('--use-slurm', action='store_true',
                       help='Submit comparison jobs to Slurm for parallel execution')
    parser.add_argument('--partition', type=str, default='gpu100',
                       help='Slurm partition to use (default: gpu100)')
    parser.add_argument('--job-time', type=str, default='02:00:00',
                       help='Time limit for each job (default: 02:00:00)')
    parser.add_argument('--job-gpus', type=int, default=1,
                       help='Number of GPUs per job (default: 1)')
    parser.add_argument('--job-mem', type=str, default='128G',
                       help='Memory per job (default: 128G)')
    parser.add_argument('--job-cpus', type=int, default=4,
                       help='CPUs per job (default: 4)')

    # Override options for activation collection
    parser.add_argument('--max-samples', type=int, default=100000,
                       help='Maximum number of samples to use for identifiability metrics (default: 100000)')

    args = parser.parse_args()

    # Set default comparison experiment name
    if args.comparison_experiment is None:
        args.comparison_experiment = f"{args.experiment_name}-id-comparison"

    print("=" * 80)
    print("Identifiability Analysis Pipeline")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Comparison experiment: {args.comparison_experiment}")
    if args.use_slurm:
        print(f"Execution mode: Parallel (Slurm)")
        print(f"  Partition: {args.partition}")
        print(f"  Job time: {args.job_time}")
        print(f"  GPUs per job: {args.job_gpus}")
        print(f"  Memory per job: {args.job_mem}")
        print(f"  CPUs per job: {args.job_cpus}")
    else:
        print(f"Execution mode: Sequential")
    print()

    # Step 1: Find pairs
    pairs = find_pairs(args.experiment_name, args.grouping_keys, args.tracking_uri, args.min_pairs)

    total_pairs = len(pairs)
    print(f"Found {total_pairs} pairs to compare")
    print()

    if total_pairs == 0:
        print("No pairs found. Exiting.")
        return

    # Step 1.5: Pre-create MLflow comparison experiment to avoid race conditions
    if args.log_to_mlflow:
        print("Ensuring MLflow comparison experiment exists...")
        import mlflow
        mlflow.set_tracking_uri(args.tracking_uri)
        get_or_create_experiment(args.comparison_experiment)
        print(f"✓ MLflow experiment ready: {args.comparison_experiment}")
        print()

    # Step 2: Run comparisons
    print("Step 2: Running comparisons...")
    print("=" * 80)

    # Build options for compare_identifiability
    compare_options = [
        "--tracking-uri", args.tracking_uri,
        "--device", args.device,
        "--batch-size", str(args.batch_size),
    ]

    if args.log_to_mlflow:
        compare_options.extend([
            "--log-to-mlflow",
            "--experiment-name", args.comparison_experiment
        ])

    if args.debug:
        compare_options.append("--debug")

    # Add max samples option
    if args.max_samples != 100000:  # Only add if not default
        compare_options.extend(["--max-samples", str(args.max_samples)])

    # Run comparisons (sequential or parallel)
    if args.use_slurm:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Prepare slurm arguments
        slurm_args = {
            'partition': args.partition,
            'job_time': args.job_time,
            'job_gpus': args.job_gpus,
            'job_mem': args.job_mem,
            'job_cpus': args.job_cpus,
        }

        print("Submitting comparison jobs to Slurm...")
        print()

        job_ids = []
        for i, pair in enumerate(pairs, 1):
            job_id = run_comparison_slurm(args.experiment_name, pair, compare_options, slurm_args, i, total_pairs)
            if job_id:
                job_ids.append(job_id)

        print()
        print("=" * 80)
        print(f"✓ Submitted {len(job_ids)} comparison jobs to Slurm")
        print("=" * 80)
        print(f"Experiment: {args.comparison_experiment}")
        print(f"Jobs submitted: {len(job_ids)}")
        print()
        print("Check job status:")
        print("  squeue -u $USER")
        print()
        print("View job logs:")
        print("  tail -f logs/compare_*.out")
        print()
        print("View MLflow results:")
        print("  mlflow ui --backend-store-uri ./mlruns")
        print()
        print("NOTE: To generate summary plots after jobs complete, run:")
        print(f"  python scripts/generate_summary_plots.py {args.comparison_experiment} --tracking-uri {args.tracking_uri}")
        print("=" * 80)
    else:
        # Sequential execution
        for i, pair in enumerate(pairs, 1):
            run_comparison(pair, compare_options, i, total_pairs)

        print()
        print("=" * 80)
        print("Analysis complete!")
        print("=" * 80)

        # Step 3: Generate summary plots (only for sequential execution)
        if args.log_to_mlflow:
            plot_paths = generate_summary_plots(
                args.comparison_experiment,
                args.tracking_uri,
                output_dir='plots'
            )

            # Log plots to MLflow summary run
            if plot_paths:
                print("\nLogging plots to MLflow...")
                mlflow.set_tracking_uri(args.tracking_uri)
                # Use race-condition-safe experiment creation
                experiment_id = get_or_create_experiment(args.comparison_experiment)

                with mlflow.start_run(
                    experiment_id=experiment_id,
                    run_name=f"summary_{args.experiment_name}",
                    tags={"job_type": "summary", "source_experiment": args.experiment_name}
                ):
                    # Log each plot as an artifact
                    for plot_path in plot_paths:
                        mlflow.log_artifact(plot_path)

                    # Log summary statistics
                    mlflow.log_metric("total_comparisons", total_pairs)

                print(f"Logged {len(plot_paths)} plots to MLflow summary run")
                print()
                print("View summary plots:")
                print("  mlflow ui --backend-store-uri ./mlruns")
                print("=" * 80)


if __name__ == "__main__":
    main()
