#!/usr/bin/env python3
"""Run SAEbench evaluations on all models in an MLflow experiment.

This script:
1. Queries all finished runs from a specified MLflow experiment
2. Submits SAEbench evaluation for each run as a Slurm job
3. Provides progress tracking and job management

Usage:
    python run/run_saebench.py \\
        --experiment-name pythia160m \\
        --eval-types core sparse_probing \\
        --partition gpu \\
        --device cuda
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

import mlflow
from mlflow.tracking import MlflowClient

from rsae.utils import get_or_create_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAEbench evaluations on all models in an experiment"
    )

    # Experiment selection
    parser.add_argument("--experiment-name", type=str, required=True,
                        help="MLflow experiment name to process")
    parser.add_argument("--saebench-experiment-name", type=str, default=None,
                        help="MLflow experiment name to log SAEbench results to "
                             "(default: '{experiment-name}-saebench')")
    parser.add_argument("--tracking-uri", type=str, default="./mlruns",
                        help="MLflow tracking URI (default: ./mlruns)")

    # Evaluation arguments (passed through to run_saebench_from_mlflow.py)
    parser.add_argument("--eval-types", type=str, nargs="+",
                        default=["core", "sparse_probing"],
                        choices=["absorption", "autointerp", "core", "ravel",
                                 "scr", "tpp", "sparse_probing", "unlearning"],
                        help="Evaluation types to run")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for evaluation (cuda/cpu)")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Force rerun even if results exist")
    parser.add_argument("--save-activations", action="store_true",
                        help="Save activations for reuse")

    # Slurm configuration
    parser.add_argument("--partition", type=str, required=True,
                        help="Slurm partition to use")
    parser.add_argument("--job-time", type=str, default="04:00:00",
                        help="Time limit per job (default: 04:00:00)")
    parser.add_argument("--job-gpus", type=int, default=1,
                        help="GPUs per job (default: 1)")
    parser.add_argument("--job-mem", type=str, default="64G",
                        help="Memory per job (default: 64G)")
    parser.add_argument("--job-cpus", type=int, default=8,
                        help="CPUs per job (default: 8)")

    # Filtering
    parser.add_argument("--filter", type=str, default=None,
                        help="MLflow filter string (e.g., 'params.model.rip_weight = \"0.01\"')")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="Maximum number of runs to process (for testing)")

    return parser.parse_args()


def get_experiment_runs(
    experiment_name: str,
    tracking_uri: str,
    filter_string: str = None,
    max_runs: int = None
) -> List[Any]:
    """Query all finished runs from experiment.

    Args:
        experiment_name: Name of MLflow experiment
        tracking_uri: MLflow tracking URI
        filter_string: Optional filter string for runs
        max_runs: Optional limit on number of runs

    Returns:
        List of MLflow Run objects
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    print(f"Found experiment: {experiment_name}")
    print(f"  ID: {experiment.experiment_id}")

    # Build filter string
    base_filter = "attributes.status = 'FINISHED'"
    if filter_string:
        full_filter = f"{base_filter} and {filter_string}"
    else:
        full_filter = base_filter

    # Query runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=full_filter,
        order_by=["start_time DESC"],
        max_results=max_runs if max_runs else 1000,
    )

    print(f"  Found {len(runs)} finished runs")

    if max_runs and len(runs) > max_runs:
        print(f"  Limiting to first {max_runs} runs")
        runs = runs[:max_runs]

    return runs


def submit_saebench_job(
    run_id: str,
    run_number: int,
    total_runs: int,
    eval_types: List[str],
    slurm_args: Dict[str, Any],
    script_args: Dict[str, Any],
    model_variant: str = None,
) -> str:
    """Submit SAEbench evaluation as Slurm job.

    Args:
        run_id: MLflow run ID to evaluate
        run_number: Current run number (for display)
        total_runs: Total number of runs
        eval_types: List of eval types to run
        slurm_args: Slurm configuration (partition, time, etc.)
        script_args: Arguments to pass to run_saebench_from_mlflow.py
        model_variant: Value of training run's `model.name` param (SAE variant
            name, e.g. topk_sae/isae/isae_me), forwarded so saebench results
            can be grouped by SAE variant.

    Returns:
        Slurm job ID
    """
    # Build command for run_saebench_from_mlflow.py
    cmd = [
        "python", "scripts/run_saebench_from_mlflow.py",
        "--run-id", run_id,
        "--tracking-uri", script_args["tracking_uri"],
        "--eval-types", *eval_types,
        "--device", script_args["device"],
    ]

    if script_args.get("saebench_experiment_name"):
        cmd.extend(["--saebench-experiment-name",
                    script_args["saebench_experiment_name"]])
    if model_variant:
        cmd.extend(["--model-variant", model_variant])
    if script_args.get("force_rerun"):
        cmd.append("--force-rerun")
    if script_args.get("save_activations"):
        cmd.append("--save-activations")

    # Shell-quote for safety
    cmd_str = " ".join(shlex.quote(arg) for arg in cmd)

    # Job name
    job_name = f"saebench_{run_id[:8]}"

    # Build sbatch command
    sbatch_cmd = [
        "sbatch",
        f"--partition={slurm_args['partition']}",
        f"--time={slurm_args['job_time']}",
        f"--gres=gpu:{slurm_args['job_gpus']}",
        f"--mem={slurm_args['job_mem']}",
        f"--cpus-per-task={slurm_args['job_cpus']}",
        f"--job-name={job_name}",
        f"--output=logs/saebench_{run_id[:8]}_%j.out",
        f"--error=logs/saebench_{run_id[:8]}_%j.err",
        f"--wrap={cmd_str}",
    ]

    # Submit job
    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ✗ Failed to submit job for run {run_id[:8]}")
        print(f"    Error: {result.stderr}")
        return None

    # Extract job ID
    job_id = result.stdout.strip().split()[-1]

    print(f"  Run {run_number}/{total_runs}: {run_id[:8]} -> Job {job_id}")

    return job_id


def main():
    args = parse_args()

    print("=" * 80)
    print("SAEbench Batch Evaluation")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Eval types: {', '.join(args.eval_types)}")
    print(f"Partition: {args.partition}")
    print(f"Device: {args.device}")
    print("=" * 80)
    print()

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Get all runs from experiment
    print("Step 1: Querying experiment runs...")
    runs = get_experiment_runs(
        args.experiment_name,
        args.tracking_uri,
        args.filter,
        args.max_runs,
    )

    if not runs:
        print("No runs found. Exiting.")
        return 0

    print()

    # Step 1.5: Pre-create SAEbench experiment to avoid race conditions
    saebench_experiment_name = (
        args.saebench_experiment_name or f"{args.experiment_name}-saebench"
    )
    print("Ensuring SAEbench MLflow experiment exists...")
    mlflow.set_tracking_uri(args.tracking_uri)
    get_or_create_experiment(saebench_experiment_name)
    print(f"✓ MLflow experiment ready: {saebench_experiment_name}")
    print()

    # Submit jobs
    print("Step 2: Submitting Slurm jobs...")
    print()

    slurm_args = {
        "partition": args.partition,
        "job_time": args.job_time,
        "job_gpus": args.job_gpus,
        "job_mem": args.job_mem,
        "job_cpus": args.job_cpus,
    }

    script_args = {
        "tracking_uri": args.tracking_uri,
        "device": args.device,
        "force_rerun": args.force_rerun,
        "save_activations": args.save_activations,
        "saebench_experiment_name": saebench_experiment_name,
    }

    job_ids = []
    for i, run in enumerate(runs, 1):
        # Pull model.name (SAE variant) from training run params for grouping
        model_variant = run.data.params.get("model.name") or run.data.params.get("model/name")
        job_id = submit_saebench_job(
            run_id=run.info.run_id,
            run_number=i,
            total_runs=len(runs),
            eval_types=args.eval_types,
            slurm_args=slurm_args,
            script_args=script_args,
            model_variant=model_variant,
        )
        if job_id:
            job_ids.append(job_id)

    print()
    print("=" * 80)
    print(f"✓ Submitted {len(job_ids)}/{len(runs)} jobs")
    print("=" * 80)
    print()
    print("Check job status:")
    print("  squeue -u $USER")
    print()
    print("View job logs:")
    print("  tail -f logs/saebench_*.out")
    print()
    print("Monitor results in MLflow:")
    print(f"  mlflow ui --backend-store-uri {args.tracking_uri}")
    print(f"  Experiment: {saebench_experiment_name}")
    print()
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
