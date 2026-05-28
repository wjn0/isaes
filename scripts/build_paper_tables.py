#!/usr/bin/env python3
"""Build LaTeX result tables from MLflow comparison experiments.

Given one or more MLflow experiment names, this groups runs by ``model.name``
(and optionally splits into separate tables by a secondary param such as
``data.mixture_scale``), averages the requested metrics across the runs in each
group, and emits a LaTeX ``tabular`` per table.

Two table layouts are produced, selected automatically from each run's
``job_type`` tag:

* identifiability (``job_type=identifiability``): the paired identifiability
  table -- MSE, R_S, SAE IoU/l2, Oracle IoU/l2, DCS.
* saebench (``job_type=saebench_eval``): a compact table -- CE loss, sparse
  probing accuracy.

Examples:
    # A single experiment:
    python scripts/build_paper_tables.py pythia160m-id-comparison

    # Stack two experiments row-wise into one tabular:
    python scripts/build_paper_tables.py synthetic-id-comparison+pythia160m-id-comparison

    # Show mean +/- SEM and four decimals:
    python scripts/build_paper_tables.py exp-name --show-sem --precision 4
"""

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

# model.name -> display label, in canonical row order.
MODEL_ORDER = ["topk_sae", "abstopk_sae", "isae", "isae_me"]
MODEL_DISPLAY = {
    "topk_sae": "TopK",
    "abstopk_sae": "AbsTopK",
    "isae": "iSAE",
    "isae_me": "iSAE-ME",
}
# Models whose row label is bolded (the proposed methods).
HIGHLIGHT_MODELS = {"isae", "isae_me"}

# Friendly block labels used when combining tables (see --combine / "a+b" groups).
# Keys are MLflow experiment names; add entries as needed for your own runs.
EXPERIMENT_LABELS = {}
SPLIT_LABELS = {
    "data.mixture_scale": "Mixture scale",
}


@dataclass
class Column:
    """One metric column in a table."""

    metric: str           # MLflow metric key
    header: str           # short label (used only for the saebench layout / logs)
    lower_better: bool    # direction for "best" highlighting
    bold_best: bool       # whether the best value in this column is bolded


# Identifiability table columns, in display order. The fancy multi-column header
# is emitted verbatim (see ID_HEADER); these specs drive the data rows + bolding.
ID_COLUMNS = [
    Column("original/eval/recon_mse",            "MSE",     lower_better=True,  bold_best=True),
    Column("original/eval/rip_loss",             "R_S",     lower_better=True,  bold_best=False),
    Column("identifiability/iou_dict",           "SAE IoU", lower_better=False, bold_best=True),
    Column("identifiability/normalized_l2_dict", "SAE l2",  lower_better=True,  bold_best=True),
    Column("oracle/iou_dict",                    "Or IoU",  lower_better=False, bold_best=False),
    Column("oracle/normalized_l2_dict",          "Or l2",   lower_better=True,  bold_best=False),
    Column("identifiability/pw_mcc",             "DCS",     lower_better=False, bold_best=True),
]

# Verbatim header for the identifiability table (mirrors the paper format).
ID_HEADER = r"""\begin{tabular}{lccccccc}
\toprule
Model &
MSE &
$\mathcal{R}_S$ &
\multicolumn{5}{c}{Pairwise Identifiability} \\
\cmidrule(lr){4-8}
& & &
\multicolumn{2}{c}{SAE} &
\multicolumn{2}{c}{Oracle} &
DCS \\
\cmidrule(lr){4-5} \cmidrule(lr){6-7}
& & &
IoU & $\ell_2$ &
IoU & $\ell_2$ &
{} \\
\midrule"""

SAEBENCH_COLUMNS = [
    Column("saebench/core/model_performance_preservation/ce_loss_with_sae",
           "CE Loss", lower_better=True, bold_best=True),
    Column("saebench/sparse_probing/sae/sae_test_accuracy",
           "Sparse Probing", lower_better=False, bold_best=True),
    # SCR / TPP are reported at ablation threshold k=20 (a SAEBench default headline);
    # both higher-is-better. Edit the _threshold_20 suffix to report another k.
    Column("saebench/scr/scr_metrics/scr_metric_threshold_20",
           "SCR", lower_better=False, bold_best=True),
    Column("saebench/tpp/tpp_metrics/tpp_threshold_20_total_metric",
           "TPP", lower_better=False, bold_best=True),
    # Absorption and RAVEL columns are omitted by default; both require ~2B+ param
    # base models to produce informative scores. Add Columns here to include them.
]

# job_type tag -> (columns, layout name)
LAYOUTS = {
    "identifiability": ("id", ID_COLUMNS),
    "saebench_eval": ("saebench", SAEBENCH_COLUMNS),
}


@dataclass
class Group:
    """Aggregated metrics for one (model, split) cell of a table."""

    model: str
    n: int
    mean: dict = field(default_factory=dict)   # metric -> mean
    sem: dict = field(default_factory=dict)    # metric -> sem
    count: dict = field(default_factory=dict)  # metric -> n runs contributing


@dataclass
class Block:
    """One labeled block of model rows within a (possibly combined) table."""

    exp_name: str
    split_desc: Optional[str]   # e.g. "Mixture scale = 0.0", or None when unsplit
    groups: dict = field(default_factory=dict)  # model.name -> Group


def fetch_runs(client, experiment_name):
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"  ERROR: experiment '{experiment_name}' not found, skipping.", file=sys.stderr)
        return []
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=5000,
    )
    return runs


def detect_layout(runs):
    """Pick the table layout from the most common job_type tag."""
    counts = defaultdict(int)
    for r in runs:
        counts[r.data.tags.get("job_type")] += 1
    job_type = max(counts, key=counts.get) if counts else None
    if job_type in LAYOUTS:
        return LAYOUTS[job_type]
    # Fallback: sniff metric keys.
    metric_keys = {k for r in runs for k in r.data.metrics}
    if any(k.startswith("saebench/") for k in metric_keys):
        return LAYOUTS["saebench_eval"]
    return LAYOUTS["identifiability"]


def active_split_keys(runs, split_by):
    """Return split keys that take more than one distinct (non-None) value."""
    active = []
    for key in split_by:
        values = {r.data.params.get(key) for r in runs}
        values.discard(None)
        if len(values) > 1:
            active.append(key)
    return active


def aggregate(runs, columns):
    """Group runs by model.name and average each column's metric.

    Returns {model.name: Group}.
    """
    by_model = defaultdict(list)
    for r in runs:
        model = r.data.params.get("model.name")
        if model is None:
            continue
        by_model[model].append(r)

    groups = {}
    for model, model_runs in by_model.items():
        g = Group(model=model, n=len(model_runs))
        for col in columns:
            vals = [r.data.metrics[col.metric] for r in model_runs if col.metric in r.data.metrics]
            if vals:
                arr = np.asarray(vals, dtype=float)
                g.mean[col.metric] = float(arr.mean())
                g.sem[col.metric] = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
                g.count[col.metric] = len(arr)
        groups[model] = g
    return groups


def ordered_models(groups):
    known = [m for m in MODEL_ORDER if m in groups]
    extra = sorted(m for m in groups if m not in MODEL_ORDER)
    return known + extra


def best_values(groups, columns):
    """For each bold_best column, the best (min/max) mean across model rows."""
    best = {}
    for col in columns:
        if not col.bold_best:
            continue
        vals = [g.mean[col.metric] for g in groups.values() if col.metric in g.mean]
        if vals:
            best[col.metric] = min(vals) if col.lower_better else max(vals)
    return best


def fmt_cell(value, sem, is_best, precision, show_sem):
    if value is None:
        return "--"
    s = f"{value:.{precision}f}"
    if show_sem and sem is not None:
        s = f"${s} \\pm {sem:.{precision}f}$"
    if is_best:
        s = rf"\textbf{{{s}}}"
    return s


def blocks_for_experiment(client, exp_name, split_by):
    """Build one Block per (active-)split value for a single experiment.

    Returns (blocks, layout, columns); ([], None, None) when the experiment has
    no runs.
    """
    runs = fetch_runs(client, exp_name)
    if not runs:
        return [], None, None

    layout, columns = detect_layout(runs)
    splits = active_split_keys(runs, split_by)

    buckets = defaultdict(list)
    for r in runs:
        buckets[tuple(r.data.params.get(s) for s in splits)].append(r)

    blocks = []
    for split_vals in sorted(buckets, key=lambda k: tuple(str(v) for v in k)):
        groups = aggregate(buckets[split_vals], columns)
        if not groups:
            continue
        desc = None
        if splits:
            desc = ", ".join(f"{SPLIT_LABELS.get(k, k)} = {v}"
                             for k, v in zip(splits, split_vals))
        blocks.append(Block(exp_name, desc, groups))
    return blocks, layout, columns


def build_label(block, multi_exp):
    """Block subheader text: experiment name (when >1 experiment) and/or split."""
    parts = []
    if multi_exp:
        parts.append(EXPERIMENT_LABELS.get(block.exp_name, block.exp_name))
    if block.split_desc:
        parts.append(block.split_desc)
    return " -- ".join(parts) if parts else None


def render_blocks(blocks, columns, layout, multi_exp, precision, show_sem, bold_models):
    """Render one tabular. Multiple blocks are stacked row-wise, each preceded by
    a labeled subheader row and separated by a \\midrule. A single block renders
    plainly (no subheader), preserving the standalone-table layout."""
    n_table_cols = 1 + len(columns)
    emit_labels = len(blocks) > 1

    lines = []
    if layout == "id":
        lines.append(ID_HEADER)
    else:
        spec = "l" + "c" * len(columns)
        lines.append(rf"\begin{{tabular}}{{{spec}}}")
        lines.append(r"\toprule")
        lines.append("Model & " + " & ".join(c.header for c in columns) + r" \\")
        lines.append(r"\midrule")

    for i, block in enumerate(blocks):
        if i > 0:
            lines.append(r"\midrule")
        if emit_labels:
            label = build_label(block, multi_exp)
            if label:
                lines.append(rf"\multicolumn{{{n_table_cols}}}{{l}}{{\emph{{{label}}}}} \\")

        best = best_values(block.groups, columns)
        for model in ordered_models(block.groups):
            g = block.groups[model]
            disp = MODEL_DISPLAY.get(model, model)
            if bold_models and model in HIGHLIGHT_MODELS:
                disp = rf"\textbf{{{disp}}}"
            cells = [disp]
            for col in columns:
                value = g.mean.get(col.metric)
                sem = g.sem.get(col.metric)
                is_best = (
                    col.bold_best
                    and value is not None
                    and col.metric in best
                    and np.isclose(value, best[col.metric])
                )
                cells.append(fmt_cell(value, sem, is_best, precision, show_sem))
            lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Build LaTeX result tables from MLflow comparison experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("experiments", nargs="+",
                        help="MLflow experiment names. Join names with '+' (e.g. 'a+b') to "
                             "stack them row-wise into one table with labeled blocks.")
    parser.add_argument("--tracking-uri", default="./mlruns",
                        help="MLflow tracking URI (default: ./mlruns).")
    parser.add_argument("--split-by", default="data.mixture_scale",
                        help="Comma-separated params that split an experiment into separate "
                             "tables when they take >1 value (default: data.mixture_scale).")
    parser.add_argument("--precision", type=int, default=3,
                        help="Decimal places for metric values (default: 3).")
    parser.add_argument("--show-sem", action="store_true",
                        help="Append +/- SEM to each value.")
    parser.add_argument("--no-bold", action="store_true",
                        help="Disable bolding of best values and proposed-method row labels.")
    parser.add_argument("--combine", action="store_true",
                        help="Stack an experiment's split tables row-wise into a single "
                             "tabular with labeled blocks (e.g. the two synthetic scales). "
                             "'+'-joined experiments are always combined regardless.")
    args = parser.parse_args()

    experiments = args.experiments
    split_by = [s.strip() for s in args.split_by.split(",") if s.strip()]
    bold = not args.no_bold

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)

    def counts_str(block):
        return ", ".join(f"{MODEL_DISPLAY.get(m, m)}={block.groups[m].n}"
                         for m in ordered_models(block.groups))

    def coverage_str(block, columns):
        # Per-column total runs contributing — surfaces uneven metric coverage
        # (e.g. when a metric is logged on fewer runs than others).
        return ", ".join(
            f"{c.header}={sum(block.groups[m].count.get(c.metric, 0) for m in block.groups)}"
            for c in columns
        )

    # Each positional arg is a '+'-joined group of experiments to stack together.
    for raw in experiments:
        group = [e.strip() for e in raw.split("+") if e.strip()]

        all_blocks, layout, columns = [], None, None
        for exp in group:
            blocks, lay, cols = blocks_for_experiment(client, exp, split_by)
            if lay is not None:
                layout, columns = lay, cols
            all_blocks.extend(blocks)
        if not all_blocks:
            continue

        multi_exp = len({b.exp_name for b in all_blocks}) > 1
        combine = args.combine or len(group) > 1
        group_label = " + ".join(EXPERIMENT_LABELS.get(e, e) for e in group)
        print(f"% ==> {group_label}", file=sys.stderr)

        if combine:
            # One stacked tabular for the whole group.
            for b in all_blocks:
                tag = build_label(b, multi_exp) or b.exp_name
                print(f"%    [{tag}]  (n: {counts_str(b)})", file=sys.stderr)
                print(f"%      coverage: {coverage_str(b, columns)}", file=sys.stderr)
            print(f"% {group_label} (combined)")
            print(render_blocks(all_blocks, columns, layout, multi_exp,
                                args.precision, args.show_sem, bold))
            print()
        else:
            # Standalone table per block (backward-compatible layout).
            for b in all_blocks:
                caption = b.exp_name + (f" | {b.split_desc}" if b.split_desc else "")
                print(f"%    {caption}  (n: {counts_str(b)})", file=sys.stderr)
                print(f"%      coverage: {coverage_str(b, columns)}", file=sys.stderr)
                print(f"% {caption}")
                print(render_blocks([b], columns, layout, False,
                                    args.precision, args.show_sem, bold))
                print()


if __name__ == "__main__":
    main()
