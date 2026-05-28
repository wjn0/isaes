#!/usr/bin/env python3
"""Aggregate identifiability metrics from MLflow comparison experiments.

This script queries a comparison experiment (e.g., {experiment}-id-comparison),
groups runs by specified parameters, and computes statistics (mean, SEM) for
each metric within each group.

Example usage:
    python scripts/aggregate_identifiability_metrics.py \
        pythia160m-noinit-2-id-comparison \
        --grouping-keys model.rip_weight,data.mixture_scale
"""

import argparse
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
from tabulate import tabulate


def extract_param_value(params: Dict[str, str], key: str) -> Optional[Any]:
    """Extract parameter value from flattened params with type conversion.

    Args:
        params: Dictionary of flattened parameters (e.g., {'model.rip_weight': '0.001'})
        key: Dot-notation key to extract (e.g., 'model.rip_weight')

    Returns:
        Converted value (bool, None, int, float, or string), or None if key doesn't exist
    """
    value_str = params.get(key)
    if value_str is None:
        return None

    try:
        # Handle special cases
        if value_str.lower() == 'none':
            return None
        if value_str.lower() in ('true', 'false'):
            return value_str.lower() == 'true'

        # Try numeric conversion
        try:
            return int(value_str)
        except ValueError:
            try:
                return float(value_str)
            except ValueError:
                return value_str  # Keep as string
    except Exception as e:
        print(f"Warning: Failed to parse parameter {key}={value_str}: {e}", file=sys.stderr)
        return value_str


def fetch_comparison_runs(experiment_name: str, tracking_uri: str) -> List[Dict[str, Any]]:
    """Query all finished comparison runs from MLflow experiment.

    Args:
        experiment_name: Name of comparison experiment
        tracking_uri: MLflow tracking URI

    Returns:
        List of dicts containing run data with keys: run_id, params, metrics, tags
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Get comparison experiment
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Error: Experiment '{experiment_name}' not found", file=sys.stderr)
            print("\nAvailable comparison experiments:", file=sys.stderr)
            for exp in client.search_experiments():
                if 'comparison' in exp.name or 'id-comparison' in exp.name:
                    print(f"  - {exp.name}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error accessing MLflow: {e}", file=sys.stderr)
        sys.exit(1)

    # Get all finished comparison runs
    try:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED' and tags.job_type = 'identifiability'",
            max_results=5000
        )
    except Exception as e:
        print(f"Error searching runs: {e}", file=sys.stderr)
        sys.exit(1)

    if not runs:
        print(f"Warning: No finished comparison runs found in experiment '{experiment_name}'", file=sys.stderr)
        print("Check if comparison jobs have completed successfully.", file=sys.stderr)
        sys.exit(0)

    run_data = []
    for run in runs:
        data = {
            'run_id': run.info.run_id,
            'params': dict(run.data.params),
            'metrics': dict(run.data.metrics),
            'tags': dict(run.data.tags)
        }
        run_data.append(data)

    return run_data


def group_runs_by_params(
    runs: List[Dict],
    grouping_keys: List[str]
) -> Tuple[Dict[Tuple, List[Dict]], int, Dict[str, int]]:
    """Group runs by specified parameter keys.

    Args:
        runs: List of run dicts
        grouping_keys: List of parameter keys to group by

    Returns:
        Tuple of (grouped_runs, skipped_count, missing_keys_count) where:
            - grouped_runs: Dict mapping group_tuple -> list of runs
            - skipped_count: Number of runs skipped due to missing keys
            - missing_keys_count: Dict of key -> count of runs missing that key
    """
    grouped = defaultdict(list)
    missing_keys_count = defaultdict(int)
    skipped_runs = 0

    for run in runs:
        group_values = []
        has_missing = False

        for key in grouping_keys:
            value = extract_param_value(run['params'], key)
            if value is None:
                missing_keys_count[key] += 1
                has_missing = True
                break
            group_values.append(value)

        if has_missing:
            skipped_runs += 1
            continue

        # Create tuple key for grouping
        group_key = tuple(group_values)
        grouped[group_key].append(run)

    return dict(grouped), skipped_runs, dict(missing_keys_count)


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute mean, SEM, and other statistics for a list of values.

    Args:
        values: List of numeric values

    Returns:
        Dict with 'mean', 'sem', 'std', 'n'
    """
    if not values:
        return {'mean': np.nan, 'sem': np.nan, 'std': np.nan, 'n': 0}

    values_array = np.array(values, dtype=float)
    n = len(values_array)
    mean = np.mean(values_array)

    if n > 1:
        std = np.std(values_array, ddof=1)  # Sample std
        sem = std / np.sqrt(n)
    else:
        std = 0.0
        sem = 0.0

    return {
        'mean': float(mean),
        'sem': float(sem),
        'std': float(std),
        'n': int(n)
    }


def aggregate_metrics(
    grouped_runs: Dict[Tuple, List[Dict]],
    grouping_keys: List[str]
) -> Dict[Tuple, Dict[str, Any]]:
    """Aggregate metrics across groups.

    Args:
        grouped_runs: Dict mapping group_tuple -> list of runs
        grouping_keys: List of parameter keys used for grouping

    Returns:
        Nested dict: group_tuple -> {'group_params': {...}, 'metrics': {...}}
    """
    aggregated = {}

    # Collect all unique metric names across all runs
    all_metric_keys = set()
    for runs in grouped_runs.values():
        for run in runs:
            all_metric_keys.update(run['metrics'].keys())

    for group_tuple, runs in grouped_runs.items():
        # Create human-readable group params dict
        group_params = {key: value for key, value in zip(grouping_keys, group_tuple)}

        # Aggregate metrics for this group
        metrics_data = defaultdict(list)

        for run in runs:
            for metric_key in all_metric_keys:
                value = run['metrics'].get(metric_key)
                if value is not None:
                    metrics_data[metric_key].append(value)

        # Compute statistics for each metric
        metrics_stats = {}
        for metric_key, values in metrics_data.items():
            if values:  # Only compute stats for metrics with at least one value
                metrics_stats[metric_key] = compute_statistics(values)

        aggregated[group_tuple] = {
            'group_params': group_params,
            'metrics': metrics_stats
        }

    return aggregated


def format_results_table(
    aggregated_data: Dict[Tuple, Dict],
    grouping_keys: List[str],
    metric_filter: Optional[str] = None
) -> str:
    """Format aggregated results as a console table.

    Args:
        aggregated_data: Aggregated metrics data
        grouping_keys: List of grouping parameter keys
        metric_filter: Optional regex pattern to filter metrics

    Returns:
        Formatted table string
    """
    # Compile metric filter if provided
    if metric_filter:
        try:
            pattern = re.compile(metric_filter)
        except re.error as e:
            print(f"Warning: Invalid regex pattern '{metric_filter}': {e}", file=sys.stderr)
            pattern = None
    else:
        pattern = None

    # Build table rows
    rows = []
    for group_tuple, data in sorted(aggregated_data.items()):
        group_params = data['group_params']

        for metric_name, stats in sorted(data['metrics'].items()):
            # Apply metric filter
            if pattern and not pattern.search(metric_name):
                continue

            row = []
            # Add grouping parameter values
            for key in grouping_keys:
                row.append(group_params.get(key, 'N/A'))

            # Add metric info
            row.extend([
                metric_name,
                f"{stats['mean']:.6f}",
                f"{stats['sem']:.6f}",
                stats['n']
            ])
            rows.append(row)

    if not rows:
        return "No metrics to display (check metric filter)"

    # Define headers
    headers = grouping_keys + ['Metric', 'Mean', 'SEM', 'N']

    # Format table
    table = tabulate(
        rows,
        headers=headers,
        tablefmt='grid',
        numalign='right',
        stralign='left'
    )

    return table


def main():
    """Main function to orchestrate the aggregation workflow."""
    parser = argparse.ArgumentParser(
        description="Aggregate identifiability metrics from MLflow comparison experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/aggregate_identifiability_metrics.py \\
      pythia160m-noinit-2-id-comparison \\
      --grouping-keys model.rip_weight

  # Multiple grouping keys
  python scripts/aggregate_identifiability_metrics.py \\
      pythia160m-noinit-2-id-comparison \\
      --grouping-keys model.rip_weight,data.mixture_scale

  # Filter to specific metrics
  python scripts/aggregate_identifiability_metrics.py \\
      pythia160m-noinit-2-id-comparison \\
      --grouping-keys model.rip_weight \\
      --metric-filter "identifiability/test/.*"
        """
    )

    parser.add_argument(
        'experiment_name',
        type=str,
        help='MLflow comparison experiment name (e.g., pythia160m-noinit-2-id-comparison)'
    )
    parser.add_argument(
        '--grouping-keys',
        type=str,
        required=True,
        help='Comma-separated parameter keys to group by (e.g., model.rip_weight,data.mixture_scale)'
    )
    parser.add_argument(
        '--tracking-uri',
        type=str,
        default='./mlruns',
        help='MLflow tracking URI (default: ./mlruns)'
    )
    parser.add_argument(
        '--metric-filter',
        type=str,
        default=None,
        help='Regex pattern to filter metrics (e.g., "identifiability/test/.*")'
    )

    args = parser.parse_args()

    # Parse grouping keys
    grouping_keys = [key.strip() for key in args.grouping_keys.split(',')]

    # Print header
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Grouping by: {', '.join(grouping_keys)}")
    if args.metric_filter:
        print(f"Metric filter: {args.metric_filter}")
    print("=" * 80)
    print()

    # Fetch runs from MLflow
    print("Fetching runs from MLflow...", file=sys.stderr)
    runs = fetch_comparison_runs(args.experiment_name, args.tracking_uri)
    print(f"Found {len(runs)} finished comparison runs", file=sys.stderr)
    print()

    # Group runs by parameters
    print("Grouping runs by parameters...", file=sys.stderr)
    grouped_runs, skipped_count, missing_keys = group_runs_by_params(runs, grouping_keys)
    print(f"Created {len(grouped_runs)} groups", file=sys.stderr)

    if skipped_count > 0:
        print(f"\nWarning: Skipped {skipped_count} runs due to missing parameters:", file=sys.stderr)
        for key, count in sorted(missing_keys.items()):
            print(f"  - {key}: missing in {count} run(s)", file=sys.stderr)
    print()

    # Aggregate metrics
    print("Aggregating metrics...", file=sys.stderr)
    aggregated = aggregate_metrics(grouped_runs, grouping_keys)

    # Compute summary statistics
    total_metrics = sum(len(data['metrics']) for data in aggregated.values())
    runs_per_group = [len(runs) for runs in grouped_runs.values()]
    avg_runs = np.mean(runs_per_group) if runs_per_group else 0
    min_runs = min(runs_per_group) if runs_per_group else 0
    max_runs = max(runs_per_group) if runs_per_group else 0

    print(f"Aggregated {total_metrics} metric-group combinations", file=sys.stderr)
    print(f"Runs per group: avg={avg_runs:.1f}, min={min_runs}, max={max_runs}", file=sys.stderr)
    print()

    # Format and print results
    table = format_results_table(aggregated, grouping_keys, args.metric_filter)
    print(table)

    # Print summary footer
    print()
    print("=" * 80)
    print(f"Summary: {len(grouped_runs)} groups, {len(runs)} total runs, {skipped_count} skipped")
    print("=" * 80)


if __name__ == "__main__":
    main()
