"""Offline command-line demonstration and CSV evaluation."""
from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np
import scipy

from . import __version__
from .data import load_draws_bytes, write_synthetic
from .evaluation import MODEL_NAMES, summarize, walk_forward
from .probability import UNIFORM_LOGP


# Exogenous generation boundaries, never estimated from outcomes.
MACHINE_RESET_DATES = ("2010-11-09", "2026-05-05")
# Earlier Mark Six draws used 45 or 47 numbers, so the 6-of-49 null does not apply.
FIRST_SIX_OF_49_DRAW = "2002-07-04"


def _dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def run(input_path, output_path, *, warmup=60, reset_date=None, seed=20260914,
        dataset_kind="user_supplied_unverified"):
    """Evaluate inputs without changing them; refuse to replace any output.

    ``reset_date`` is None, one YYYY-MM-DD string, or a list of them.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    output = Path(output_path)
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    input_bytes = Path(input_path).read_bytes()
    data = load_draws_bytes(input_bytes)
    if data.dates[0] < FIRST_SIX_OF_49_DRAW:
        raise ValueError(f"Input starts {data.dates[0]}, but draws before {FIRST_SIX_OF_49_DRAW} "
                         "used 45 or 47 numbers and cannot be scored against the 6-of-49 null")
    # Synthetic dates are artificial indices, not days on a physical machine.
    reset_dates = set() if dataset_kind == "synthetic_demo" else set(MACHINE_RESET_DATES)
    for requested in [reset_date] if isinstance(reset_date, str) else reset_date or ():
        try:
            boundary = date.fromisoformat(requested).isoformat()
        except ValueError:
            boundary = None
        if boundary != requested:
            raise ValueError(f"reset-date must use YYYY-MM-DD, got {requested!r}")
        reset_dates.add(boundary)
    reset_dates = sorted(reset_dates)
    reset_indices = sorted({next((i for i, d in enumerate(data.dates) if d >= boundary),
                                len(data.dates)) for boundary in reset_dates})
    result = walk_forward(data.outcomes, warmup=warmup, reset_indices=reset_indices)
    summary = {"schema_version": 1, "dataset_kind": dataset_kind,
               "input_draws": len(data.dates), "warmup_draws": warmup,
               "evaluation_draws": len(result["indices"]),
               "machine_policy": "independent_generation_fit_no_transfer",
               "betting_eligible": False,
               "betting_status": "research_scores_only_no_per_line_EV_or_verified_draw_terms",
               "evaluation_start": data.dates[warmup], "evaluation_end": data.dates[-1],
               "models": summarize(result, data.outcomes, seed=seed),
               "interpretation": "Log-score gains measure forecast quality, not cash return. Synthetic data cannot establish a physical effect."}
    provenance = {"schema_version": 1, "dataset_kind": dataset_kind,
                  "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
                  "package_version": __version__, "python": platform.python_version(),
                  "numpy": np.__version__, "scipy": scipy.__version__,
                  "seed": seed, "reset_date": reset_date, "reset_dates": reset_dates,
                  "reset_indices": reset_indices, "warmup_draws": warmup,
                  "machine_policy": "independent_generation_fit_no_transfer",
                  "physical_measurements": "not_available; fitted statistical effects only",
                  "number_selection": "draw_probabilities_only_no_popularity_features",
                  "operational_EV_rule": "expected_gross_HKD_per_full_line > 10; this evaluator does not certify EV",
                  "models": list(MODEL_NAMES), "prior_strength": 20,
                  "single_ball_alternative_mass": 0.5, "spike_bias_probability": 1/49,
                  "mixture_prior": [0.5, 0.25, 0.25], "mixture_learning_rate": 0.25,
                  "mixture_prior_mix": 0.01,
                  "bootstrap_replicates": 2000, "bootstrap_block_length": 8,
                  "selection": "Fixed model settings; no evaluation-window tuning",
                  "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(Path(__file__).parent.glob("*.py"))}}
    # Build a complete run next to its destination, then move only after success.
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".marksix-run-", dir=output.parent))
    try:
        _dump(staging / "summary.json", summary)
        _dump(staging / "provenance.json", provenance)
        if dataset_kind == "synthetic_demo":
            (staging / "synthetic_draws.csv").write_bytes(input_bytes)
        with (staging / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["date", "draw_id", "model", "log_gain", "w_uniform", "w_sparse_single_ball", "w_spike_slab"])
            for row, index in enumerate(result["indices"]):
                for j, name in enumerate(MODEL_NAMES):
                    writer.writerow([data.dates[index], data.draw_ids[index], name,
                                     float(result["log_probabilities"][row, j] - UNIFORM_LOGP),
                                     *result["mixture_weights"][row]])
        # Reserving an absent directory prevents a concurrent run being replaced.
        output.mkdir(exist_ok=False)
        try:
            for child in staging.iterdir():
                child.rename(output / child.name)
        except BaseException:
            # This run created the directory, so a partial publish is removed.
            shutil.rmtree(output, ignore_errors=True)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(prog="marksix", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--output", required=True, help="New output directory; existing paths are rejected")
        sub.add_argument("--warmup", type=int, default=60, help="Initial draws used only for training (default: 60)")
        sub.add_argument("--seed", type=int, default=20260914, help="Non-negative seed for generation and bootstrap")
        sub.add_argument("--reset-date", action="append",
                         help="Additional pre-specified YYYY-MM-DD boundary (repeatable); known machine changes always reset")
        if name == "evaluate":
            sub.add_argument("--input", required=True, help="CSV with date, draw_id, and n1 through n6")
        else:
            sub.add_argument("--draws", type=int, default=240, help="Synthetic fair draws to generate (default: 240)")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            with tempfile.TemporaryDirectory(prefix="marksix-synthetic-") as tmp:
                source = Path(tmp) / "synthetic_draws.csv"
                write_synthetic(source, draws=args.draws, seed=args.seed)
                summary = run(source, args.output, warmup=args.warmup,
                              reset_date=args.reset_date, seed=args.seed, dataset_kind="synthetic_demo")
        else:
            summary = run(args.input, args.output, warmup=args.warmup,
                          reset_date=args.reset_date, seed=args.seed)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    count = summary["evaluation_draws"]
    print(f"Evaluated {count} draw{'s' * (count != 1)} ({summary['dataset_kind']}).")
    for row in summary["models"]:
        print(f"{row['model']:22} {row['total_log_gain']:+.4f} total log-score nats vs uniform")
    print(f"Saved scores, summary and provenance to {args.output}")
    return 0
