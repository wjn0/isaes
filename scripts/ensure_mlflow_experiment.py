#!/usr/bin/env python
"""Ensure an MLflow experiment exists before launching jobs.

This script creates an MLflow experiment if it doesn't exist, preventing
race conditions when multiple jobs try to create the same experiment simultaneously.

Usage:
    python scripts/ensure_mlflow_experiment.py EXPERIMENT_NAME [--tracking-uri URI]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rsae.utils import get_or_create_experiment
import mlflow


def main():
    parser = argparse.ArgumentParser(
        description="Ensure an MLflow experiment exists"
    )
    parser.add_argument(
        "experiment_name",
        help="Name of the MLflow experiment to create"
    )
    parser.add_argument(
        "--tracking-uri",
        default="./mlruns",
        help="MLflow tracking URI (default: ./mlruns)"
    )

    args = parser.parse_args()

    # Set tracking URI
    mlflow.set_tracking_uri(args.tracking_uri)

    # Create or get experiment
    try:
        experiment_id = get_or_create_experiment(args.experiment_name)
        print(f"✓ MLflow experiment ready: {args.experiment_name} (ID: {experiment_id})")
        return 0
    except Exception as e:
        print(f"✗ Failed to create experiment: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
