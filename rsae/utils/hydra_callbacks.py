"""Hydra callbacks for MLflow integration."""

from hydra.experimental.callback import Callback
from omegaconf import DictConfig
import mlflow
from rsae.utils.mlflow_utils import get_or_create_experiment


class MLflowExperimentCallback(Callback):
    """
    Hydra callback to pre-create MLflow experiments before multirun jobs launch.

    This prevents race conditions when multiple jobs try to create the same
    experiment simultaneously. The callback runs on_multirun_start, which executes
    on the local machine BEFORE the submitit launcher spawns jobs.
    """

    def on_multirun_start(self, config: DictConfig, **kwargs) -> None:
        """
        Called once before multirun launches any jobs.

        This runs on the LOCAL machine before the launcher (submitit) spawns
        any jobs, making it the perfect place to pre-create MLflow experiments.

        Args:
            config: The Hydra configuration (multirun overrides resolved)
            **kwargs: Additional keyword arguments from Hydra
        """
        # Check if MLflow is enabled
        if not config.get('mlflow', {}).get('enabled', False):
            return

        # Get experiment name and tracking URI
        experiment_name = config.mlflow.get('experiment_name')
        tracking_uri = config.mlflow.get('tracking_uri', './mlruns')

        if not experiment_name:
            # No experiment name configured, skip
            return

        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Pre-create experiment
        print(f"\n{'='*80}")
        print(f"MLflow Experiment Pre-Creation (Hydra Callback)")
        print(f"{'='*80}")
        print(f"Experiment: {experiment_name}")
        print(f"Tracking URI: {tracking_uri}")

        try:
            experiment_id = get_or_create_experiment(experiment_name)
            print(f"✓ Experiment ready (ID: {experiment_id})")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"✗ Failed to create experiment: {e}")
            print(f"{'='*80}\n")
            # Don't fail the run, let individual jobs handle it
            # (they have the same retry logic in get_or_create_experiment)
