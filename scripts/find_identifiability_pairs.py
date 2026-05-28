"""Find pairs of runs suitable for identifiability assessment.

This script groups runs from an MLflow experiment by their configuration
(data seed, model architecture, hyperparameters) and outputs pairs of runs
that differ only in their model training seed.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

import mlflow
from mlflow.tracking import MlflowClient


def get_run_config(client: MlflowClient, run, tracking_uri: str) -> Dict:
    """Get config from run parameters.

    First tries to reconstruct from logged parameters (fast, already in memory),
    then falls back to loading from artifacts (slow, requires download).
    """
    run_id = run.info.run_id

    # Try reconstructing from params first (fast - data already loaded)
    try:
        params = run.data.params

        # Reconstruct nested config from flattened params
        config = {}
        for key, value in params.items():
            parts = key.split('.')
            current = config

            # Navigate/create nested structure
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the value, converting types
            try:
                # Try to convert to appropriate type
                if value.lower() in ('true', 'false'):
                    current[parts[-1]] = value.lower() == 'true'
                elif value.lower() == 'none':
                    current[parts[-1]] = None
                else:
                    try:
                        # Try int first
                        current[parts[-1]] = int(value)
                    except ValueError:
                        try:
                            # Try float
                            current[parts[-1]] = float(value)
                        except ValueError:
                            # Keep as string
                            current[parts[-1]] = value
            except AttributeError:
                # value might already be the right type
                current[parts[-1]] = value

        return config
    except Exception:
        # Fall back to loading from artifacts (slow - requires download)
        try:
            artifact_path = client.download_artifacts(run_id, "model/extra_files/config.yaml")
            with open(artifact_path) as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Warning: Could not load config for run {run_id}: {e}")
            return None


def get_grouping_key(config: Dict, grouping_keys: List[str]) -> Tuple[str, List[str]]:
    """Extract grouping key from config based on specified keys.

    Args:
        config: Full config dictionary
        grouping_keys: List of dot-notation keys to extract (e.g., ['seed.data', 'model.rip_weight'])

    Returns:
        Tuple of (JSON string of extracted key-value pairs (hashable), list of missing keys)
    """
    try:
        extracted = {}
        missing_keys = []

        for key_path in grouping_keys:
            parts = key_path.split('.')
            current = config

            # Navigate nested structure
            try:
                for part in parts:
                    current = current[part]

                # Store the value in a nested structure matching the key path
                target = extracted
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                target[parts[-1]] = current

            except (KeyError, TypeError):
                # Key doesn't exist in this config
                missing_keys.append(key_path)

        # If any keys are missing, return None for the grouping key
        if missing_keys:
            return None, missing_keys

        # Convert to JSON string for hashable key
        # Sort keys for deterministic ordering
        return json.dumps(extracted, sort_keys=True), []

    except Exception as e:
        print(f"Warning: Error creating grouping key: {e}")
        return None, []


def main():
    parser = argparse.ArgumentParser(
        description="Find pairs of runs for identifiability assessment"
    )
    parser.add_argument('experiment_name', type=str,
                       help='MLflow experiment name (e.g., rsae-gaussian-large-4)')
    parser.add_argument('--grouping-keys', type=str, required=True,
                       help='Comma-separated list of config keys to group by (e.g., seed.data,model.rip_weight,data.mixture_scale)')
    parser.add_argument('--tracking-uri', type=str, default='./mlruns',
                       help='MLflow tracking URI (default: ./mlruns)')
    parser.add_argument('--min-pairs', type=int, default=2,
                       help='Minimum number of runs per group to output (default: 2)')
    parser.add_argument('--output-commands', action='store_true',
                       help='Output ready-to-run commands instead of just pairs')
    parser.add_argument('--json', action='store_true',
                       help='Output in JSON format for programmatic use')

    args = parser.parse_args()

    # Parse grouping keys
    grouping_keys = [key.strip() for key in args.grouping_keys.split(',')]

    # Setup MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)

    # Get experiment
    try:
        experiment = client.get_experiment_by_name(args.experiment_name)
        if experiment is None:
            print(f"Error: Experiment '{args.experiment_name}' not found")
            print("\nAvailable experiments:")
            for exp in client.search_experiments():
                print(f"  - {exp.name}")
            return
    except Exception as e:
        print(f"Error accessing MLflow: {e}")
        return

    if not args.json:
        print(f"Searching experiment: {args.experiment_name}")
        print(f"Experiment ID: {experiment.experiment_id}")
        print("=" * 80)

    # Get all runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'"
    )

    if not args.json:
        print(f"Found {len(runs)} finished runs")
        print("\nGrouping runs by configuration...")

    # Group runs by configuration (without learning rate)
    groups = defaultdict(list)
    run_configs = {}
    run_objects = {}  # Store run objects to access metrics
    group_configs = {}  # Store sample config for each group (for display)

    # Diagnostics
    runs_with_valid_config = 0
    runs_with_grouping_key = 0
    missing_keys_count = defaultdict(int)  # Track which keys are missing

    for run in runs:
        run_id = run.info.run_id
        config = get_run_config(client, run, args.tracking_uri)

        if config is None:
            continue

        runs_with_valid_config += 1

        grouping_key, missing_keys = get_grouping_key(config, grouping_keys)
        if grouping_key is None:
            # Track which keys are missing
            for key in missing_keys:
                missing_keys_count[key] += 1
            continue

        runs_with_grouping_key += 1

        groups[grouping_key].append(run_id)
        run_configs[run_id] = config
        run_objects[run_id] = run

        # Store first config we see for this group (for display)
        if grouping_key not in group_configs:
            group_configs[grouping_key] = config

    if not args.json:
        print(f"Diagnostics:")
        print(f"  - Total finished runs: {len(runs)}")
        print(f"  - Runs with valid config: {runs_with_valid_config}")
        print(f"  - Runs with grouping key: {runs_with_grouping_key}")
        print(f"  - Unique configurations: {len(groups)}")
        print(f"  - Grouping keys: {', '.join(grouping_keys)}")

        # Show missing keys
        if missing_keys_count:
            print(f"\n  Warning: Some runs are missing required grouping keys:")
            for key, count in sorted(missing_keys_count.items()):
                print(f"    - {key}: missing in {count} run(s)")

        # Show distribution of runs per group
        runs_per_group = [len(run_ids) for run_ids in groups.values()]
        if runs_per_group:
            print(f"\n  - Runs per group (min/avg/max): {min(runs_per_group)}/{sum(runs_per_group)/len(runs_per_group):.1f}/{max(runs_per_group)}")
        print()

    # Output pairs
    total_pairs = 0
    groups_with_pairs = 0
    json_output = []  # Collect JSON data if needed

    for group_key, run_ids in sorted(groups.items()):
        if len(run_ids) < args.min_pairs:
            continue

        groups_with_pairs += 1

        # Get config for this group
        config = group_configs[group_key]

        # Extract values for the grouping keys
        def get_value_by_path(cfg: Dict, key_path: str):
            """Extract value from nested dict using dot notation."""
            parts = key_path.split('.')
            current = cfg
            try:
                for part in parts:
                    current = current[part]
                return current
            except (KeyError, TypeError):
                return 'N/A'

        grouping_values = {key: get_value_by_path(config, key) for key in grouping_keys}

        if not args.json:
            print(f"Group {groups_with_pairs}:")
            print(f"  Config:")
            for key, value in grouping_values.items():
                print(f"    - {key}: {value}")
            print(f"  Runs: {len(run_ids)}")

            # Show varying parameters for each run (those not in grouping keys)
            # Common case: show model training seeds
            print(f"  Run details:")
            for run_id in run_ids:
                details = [f"id={run_id[:8]}"]
                # Show seed.model or seed.train if not in grouping keys
                if 'seed.model' not in grouping_keys and 'seed.train' not in grouping_keys:
                    model_seed = run_configs[run_id].get('seed', {}).get('model') or \
                                 run_configs[run_id].get('seed', {}).get('train')
                    if model_seed is not None:
                        details.append(f"model_seed={model_seed}")
                print(f"    {', '.join(details)}")

            print(f"  Pairs:")

        # Generate all pairs
        for i in range(len(run_ids)):
            for j in range(i + 1, len(run_ids)):
                run1, run2 = run_ids[i], run_ids[j]
                total_pairs += 1

                if args.json:
                    # Collect data for JSON output
                    model_seed1 = run_configs[run1].get('seed', {}).get('model') or \
                                  run_configs[run1].get('seed', {}).get('train')
                    model_seed2 = run_configs[run2].get('seed', {}).get('model') or \
                                  run_configs[run2].get('seed', {}).get('train')

                    # Extract and average eval/* and train/* metrics from both runs
                    original_metrics = {}
                    metrics1 = run_objects[run1].data.metrics
                    metrics2 = run_objects[run2].data.metrics

                    # Get all metric keys from both runs
                    all_metric_keys = set(metrics1.keys()) | set(metrics2.keys())

                    for key in all_metric_keys:
                        if key.startswith('eval/') or key.startswith('train/'):
                            val1 = metrics1.get(key)
                            val2 = metrics2.get(key)

                            # Average if both present, otherwise use the one available
                            if val1 is not None and val2 is not None:
                                original_metrics[key] = (val1 + val2) / 2
                            elif val1 is not None:
                                original_metrics[key] = val1
                            elif val2 is not None:
                                original_metrics[key] = val2

                    json_output.append({
                        "run1": run1,
                        "run2": run2,
                        "group": grouping_values,
                        "model_seed1": model_seed1,
                        "model_seed2": model_seed2,
                        "original_metrics": original_metrics
                    })
                elif args.output_commands:
                    print(f"    python scripts/compare_identifiability.py {run1} {run2}")
                else:
                    print(f"    {run1} {run2}")

        if not args.json:
            print()

    if args.json:
        # Output JSON to stdout
        print(json.dumps(json_output, indent=2))
    else:
        print("=" * 80)
        print(f"Summary:")
        print(f"  Total groups with pairs: {groups_with_pairs}")
        print(f"  Total pairs: {total_pairs}")

        if args.output_commands:
            print(f"\nTo run all comparisons, copy the commands above or pipe to bash:")
            print(f"  python scripts/find_identifiability_pairs.py {args.experiment_name} --output-commands | grep 'python scripts' | bash")


if __name__ == "__main__":
    main()
