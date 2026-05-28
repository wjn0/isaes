# README

## MLflow

MLflow is used to view experimental results in a dashboard. It can be launched with:

    $ bash run/launch_mlflow_ui.sh

## Synthetic experiments

### Running Parameter Sweeps

This project uses Hydra's native multirun with the Slurm submitit launcher for parameter sweeps.

**Quick Start:**

Launch the synthetic experiment sweep with all models:
```bash
PYTHONPATH=. uv run python scripts/train_sae.py --config-name sweep/synthetic --multirun model=topk_sae,abstopk_sae,topk_isae,topk_isae_me hydra.launcher.partition=<partition> mlflow.experiment_name=synthetic
```

And the LLM experiments follow the form:
```bash
PYTHONPATH=. uv run python scripts/train_transformer_sae.py --config-name sweep/pythia160m --multirun model=topk_sae,abstopk_sae,topk_isae,topk_isae_me hydra.launcher.partition=<partition> mlflow.experiment_name=pythia160m
```

And the vision experiments follow the form:
```bash
PYTHONPATH=. uv run python scripts/train_vision_sae.py --config-name sweep/dinov2 --multirun model=topk_sae,abstopk_sae,topk_isae,topk_isae_me hydra.launcher.partition=<partition> mlflow.experiment_name=dinov2
```

Available models: `topk_sae`, `abstopk_sae`, `topk_isae`, `topk_isae_me`.

Once the experiments are complete, run identifiability analysis for the synthetic models:
```bash
PYTHONPATH=. srun -c 6 --mem=64G --pty --gres=gpu:1 --partition=<partition> \
    uv run run/launch_identifiability_analysis.py synthetic \
    --grouping-keys seed.data,model.name \
    --device cuda \
    --log-to-mlflow \
    --use-slurm
```

and for LLMs/vision:
```bash
PYTHONPATH=. srun -c 6 --mem=64G --pty --gres=gpu:1 --partition=<partition> \
    uv run run/launch_identifiability_analysis.py pythia160m \
    --grouping-keys model.name \
    --device cuda \
    --log-to-mlflow \
    --use-slurm
```

Results will be logged to MLflow.

Get a copy-pasteable summary output with standard errors:

```bash
PYTHONPATH=. uv run python scripts/aggregate_identifiability_metrics.py \
    pythia160m-id-comparison
```

### LLM benchmarking

The `SAEBench` submodule pinned by this repo is a fork (`wjn0/SAEBench`) with a small patch to `MODEL_CONFIGS` in `run_all_evals_custom_saes.py` so that Pythia-160m can be evaluated. Make sure submodules are initialized (`git submodule update --init --recursive`), then assess SAEBench performance:
```bash
PYTHONPATH=. srun -c 6 --mem=100G --pty --gres=gpu:1 --partition=<partition> uv run python run/run_saebench.py --experiment-name pythia160m --eval-types core sparse_probing --partition gpu --device cuda
```
