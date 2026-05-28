#!/usr/bin/env python3
"""Run SAEBench evals on ALL SAEs from an MLflow experiment in a single process.

The per-checkpoint orchestrator (run/run_saebench.py -> run_saebench_from_mlflow.py)
submits one Slurm job per SAE. Those jobs race on SAEBench's shared, hardcoded
caches (e.g. ``SAEBench/artifacts/absorption/probes`` and ``artifacts/tpp/...``),
which corrupts probe CSVs (absorption) and triggers ``shutil.rmtree`` TOCTOU
failures during cleanup (scr/tpp).

SAEBench is designed to evaluate *many* SAEs in one process: ``selected_saes`` is a
list, each shared cache is built once, every SAE is evaluated, and cleanup happens
once at the end. This script does exactly that -- load every checkpoint's SAE, run
the requested evals once over all of them, then log each SAE's results to its own
MLflow run in the SAEbench experiment.

Example:
    PYTHONPATH=. uv run python scripts/run_saebench_multi_from_mlflow.py \
        --experiment-name pythia160m \
        --saebench-experiment-name pythia160m-saebench \
        --eval-types absorption --device cuda --force-rerun
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# Importing the single-run module also inserts SAEBench/ onto sys.path and gives us
# its battle-tested helpers, so we reuse rather than duplicate them.
sys.path.insert(0, str(Path(__file__).parent))
from run_saebench_from_mlflow import (  # noqa: E402
    infer_sae_dimensions,
    extract_sae_hyperparameters,
    log_all_eval_results,
    cleanup_result_files,
    MODEL_CONFIGS,
    DEFAULT_MODEL_CONFIG,
)

from sae_bench.custom_saes import run_all_evals_custom_saes  # noqa: E402
import sae_bench.sae_bench_utils.general_utils as general_utils  # noqa: E402

from rsae.saebench_wrapper import RIPTopKSAEBench  # noqa: E402


# Eval families that show up as a metric namespace prefix, used for --skip-existing.
EVAL_FAMILIES = [
    "absorption", "autointerp", "core", "ravel",
    "scr", "tpp", "sparse_probing", "sparse_probing_sae_probes", "unlearning",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment-name", type=str, default=None,
                   help="Source MLflow training experiment (its finished runs are the SAEs).")
    p.add_argument("--run-ids", type=str, nargs="+", default=None,
                   help="Explicit training run IDs to evaluate (overrides --experiment-name).")
    p.add_argument("--saebench-experiment-name", type=str, default=None,
                   help="Target SAEbench experiment (default: '{experiment-name}-saebench').")
    p.add_argument("--tracking-uri", type=str, default="./mlruns")
    p.add_argument("--eval-types", type=str, nargs="+",
                   default=["core", "sparse_probing"],
                   choices=["absorption", "autointerp", "core", "ravel", "scr", "tpp",
                            "sparse_probing", "sparse_probing_sae_probes", "unlearning"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--save-activations", action="store_true")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip checkpoints that already have all requested eval-types "
                        "logged in the target SAEbench experiment.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--dtype", type=str, default=None,
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--max-saes", type=int, default=None, help="Limit SAEs (for testing).")
    return p.parse_args()


def resolve_run_ids(client, args):
    if args.run_ids:
        return list(args.run_ids)
    if not args.experiment_name:
        sys.exit("ERROR: provide --run-ids or --experiment-name.")
    exp = client.get_experiment_by_name(args.experiment_name)
    if exp is None:
        sys.exit(f"ERROR: experiment '{args.experiment_name}' not found.")
    runs = client.search_runs([exp.experiment_id],
                              filter_string="attributes.status = 'FINISHED'",
                              max_results=5000)
    return [r.info.run_id for r in runs]


def already_done(client, saebench_experiment_name, eval_types):
    """Map training_run_id -> set of eval families already logged in the target exp."""
    exp = client.get_experiment_by_name(saebench_experiment_name)
    done = defaultdict(set)
    if exp is None:
        return done
    for r in client.search_runs([exp.experiment_id], max_results=5000):
        tr = r.data.tags.get("training_run_id")
        if not tr:
            continue
        for k in r.data.metrics:
            if k.startswith("saebench/"):
                fam = k.split("/")[1]
                if fam in EVAL_FAMILIES:
                    done[tr].add(fam)
    return done


def main():
    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)

    saebench_experiment_name = (
        args.saebench_experiment_name
        or (f"{args.experiment_name}-saebench" if args.experiment_name else None)
    )
    if saebench_experiment_name is None:
        sys.exit("ERROR: provide --saebench-experiment-name (or --experiment-name).")

    run_ids = resolve_run_ids(client, args)

    if args.skip_existing:
        done = already_done(client, saebench_experiment_name, args.eval_types)
        wanted = set(args.eval_types)
        kept = [rid for rid in run_ids if not wanted.issubset(done.get(rid, set()))]
        skipped = [rid for rid in run_ids if rid not in kept]
        if skipped:
            print(f"Skipping {len(skipped)} checkpoints already complete for "
                  f"{args.eval_types}: {[r[:8] for r in skipped]}")
        run_ids = kept

    if args.max_saes:
        run_ids = run_ids[:args.max_saes]
    if not run_ids:
        print("Nothing to evaluate. Exiting.")
        return 0

    print("=" * 80)
    print(f"Multi-SAE SAEBench eval | {len(run_ids)} SAEs | eval_types={args.eval_types}")
    print(f"Target experiment: {saebench_experiment_name}")
    print("=" * 80)

    device = general_utils.setup_environment()

    # Load + wrap every SAE; collect (sae_name, wrapped_sae) and per-SAE metadata.
    selected_saes = []
    meta = []  # list of dicts, one per SAE
    saebench_model_name = None
    hook_layer = None
    for i, run_id in enumerate(run_ids, 1):
        run = client.get_run(run_id)
        params = run.data.params
        full_model_name = params.get("data.model_name", params.get("data/model_name",
                                                                    "pythia-70m-deduped"))
        this_layer = int(params.get("data.hook_layer", params.get("data/hook_layer", 3)))
        model_variant = params.get("model.name") or params.get("model/name")
        short_name = full_model_name.split("/")[-1] if "/" in full_model_name else full_model_name

        # All SAEs in one run_evals() call must share the same base model + layer.
        if saebench_model_name is None:
            saebench_model_name, hook_layer = short_name, this_layer
        elif (short_name, this_layer) != (saebench_model_name, hook_layer):
            sys.exit(f"ERROR: mixed base models/layers in one batch: "
                     f"{(short_name, this_layer)} != {(saebench_model_name, hook_layer)}. "
                     f"Run them as separate jobs.")

        # Pick dtype now so we can load consistently.
        model_config = MODEL_CONFIGS.get(saebench_model_name, DEFAULT_MODEL_CONFIG).copy()
        if args.batch_size is not None:
            model_config["batch_size"] = args.batch_size
        if args.dtype is not None:
            model_config["dtype"] = args.dtype
        dtype = general_utils.str_to_dtype(model_config["dtype"])

        print(f"[{i}/{len(run_ids)}] loading SAE from run {run_id[:8]} ({model_variant})")
        native_model = mlflow.pytorch.load_model(f"runs:/{run_id}/model", map_location=device)
        native_model.eval()
        d_in, d_sae = infer_sae_dimensions(native_model)
        sae = RIPTopKSAEBench(
            riptopk_model=native_model, d_in=d_in, d_sae=d_sae,
            model_name=saebench_model_name, hook_layer=hook_layer,
            device=device, dtype=dtype,
        )
        sae_name = f"riptopk_{full_model_name.replace('/', '-')}_l{hook_layer}_{run_id[:8]}"
        selected_saes.append((sae_name, sae))
        meta.append({
            "run_id": run_id, "sae_name": sae_name, "native_model": native_model,
            "full_model_name": full_model_name, "hook_layer": hook_layer,
            "model_variant": model_variant,
            "training_experiment_name": client.get_experiment(run.info.experiment_id).name,
        })

    model_config = MODEL_CONFIGS.get(saebench_model_name, DEFAULT_MODEL_CONFIG).copy()
    if args.batch_size is not None:
        model_config["batch_size"] = args.batch_size
    if args.dtype is not None:
        model_config["dtype"] = args.dtype

    # Snapshot result files so we can attribute new ones to each SAE afterwards.
    eval_results_dir = Path("eval_results")
    existing = set()
    if eval_results_dir.exists():
        existing = {f.relative_to(eval_results_dir) for f in eval_results_dir.rglob("*.json")}

    print(f"\nRunning {args.eval_types} over {len(selected_saes)} SAEs in one process...\n")
    run_all_evals_custom_saes.run_evals(
        model_name=saebench_model_name,
        selected_saes=selected_saes,
        llm_batch_size=model_config["batch_size"],
        llm_dtype=model_config["dtype"],
        device=device,
        eval_types=args.eval_types,
        api_key=None,
        force_rerun=args.force_rerun,
        save_activations=args.save_activations,
    )

    # Attribute new result files to each SAE and log to its own MLflow run.
    all_new = set()
    if eval_results_dir.exists():
        all_new = {f.relative_to(eval_results_dir)
                   for f in eval_results_dir.rglob("*.json")} - existing
    print(f"\n{len(all_new)} new result files; logging per SAE...")

    logged = 0
    for m in meta:
        sae_files = {f for f in all_new if m["sae_name"] in str(f)}
        if not sae_files:
            print(f"  ! no result files for {m['sae_name']} ({m['run_id'][:8]}) -- skipping log")
            continue
        log_all_eval_results(
            new_result_files=sae_files,
            sae_name=m["sae_name"],
            training_run_id=m["run_id"],
            training_experiment_name=m["training_experiment_name"],
            model_name=m["full_model_name"],
            hook_layer=m["hook_layer"],
            eval_types=args.eval_types,
            tracking_uri=args.tracking_uri,
            sae_hyperparams=extract_sae_hyperparameters(m["native_model"]),
            model_variant=m["model_variant"],
            saebench_experiment_name=saebench_experiment_name,
        )
        cleanup_result_files(sae_files, eval_results_dir)
        logged += 1

    print(f"\n✓ Logged {logged}/{len(meta)} SAEs to '{saebench_experiment_name}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
