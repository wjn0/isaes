"""Utilities for MLflow logging."""

import mlflow
from typing import Dict, Any


def get_or_create_experiment(experiment_name: str) -> str:
    """
    Safely get or create an MLflow experiment, handling race conditions.

    When multiple processes try to create the same experiment simultaneously,
    MLflow can create duplicates. This function handles that race condition
    by:
    1. First trying to get the experiment by name
    2. If it doesn't exist, trying to create it
    3. If creation fails (another process created it), getting it again

    Args:
        experiment_name: Name of the experiment

    Returns:
        experiment_id: The experiment ID (as a string)

    Example:
        >>> experiment_id = get_or_create_experiment("my-experiment")
        >>> mlflow.start_run(experiment_id=experiment_id)
    """
    client = mlflow.tracking.MlflowClient()

    # Try to get existing experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id

    # Experiment doesn't exist, try to create it
    try:
        experiment_id = client.create_experiment(experiment_name)
        return experiment_id
    except Exception as e:
        # Another process may have created it between our check and create
        # Try to get it again
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is not None:
            return experiment.experiment_id
        else:
            # Still doesn't exist, something else is wrong
            raise e


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary for MLflow parameter logging.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion
        sep: Separator between nested keys (default: '.')

    Returns:
        Flattened dictionary with keys like 'parent.child.grandchild'

    Example:
        >>> flatten_dict({'a': {'b': 1, 'c': 2}, 'd': 3})
        {'a.b': 1, 'a.c': 2, 'd': 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            # MLflow params must be strings, numbers, or booleans
            if v is None or isinstance(v, (str, int, float, bool)):
                items.append((new_key, v))
            else:
                items.append((new_key, str(v)))
    return dict(items)
