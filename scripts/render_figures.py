#!/usr/bin/env python3
"""Render figures from saved historical and synthetic evaluation results."""
from __future__ import annotations

import argparse
import csv
from itertools import accumulate
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parents[1]


def set_style():
    """Keep SVG text selectable, with font fallbacks, and omit volatile metadata."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11, "axes.labelsize": 11,
        "axes.edgecolor": "#9aa6b2", "axes.labelcolor": "#263648",
        "xtick.color": "#455468", "ytick.color": "#263648",
        "svg.fonttype": "none", "svg.hashsalt": "marksix-figures-v1",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def render_research(source: Path, destination: Path):
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = data["periods"]["new_machine"]
    draws = rows[0]["draws"]
    min_holm = min(row["holm_p"] for row in rows if row["model"] != "uniform")
    fig, ax = plt.subplots(figsize=(11.8, 9.5))
    fig.subplots_adjust(left=0.345, right=0.965, top=0.84, bottom=0.16)
    for index, row in enumerate(rows):
        mean = row["mean_log_gain"]
        low, high = row["mean_95pct_block_ci"]
        color = "#66788a" if row["model"] == "uniform" else "#266390"
        ax.errorbar(mean, index, xerr=[[mean - low], [high - mean]], fmt="o",
                    color=color, ecolor=color, markersize=5.2, elinewidth=1.6,
                    capsize=3.4, capthick=1.4, zorder=3)
    ax.axvline(0, color="#3c4653", linewidth=1.05, linestyle=(0, (4, 4)), zorder=2)
    ax.set_yticks(range(len(rows)), [data["model_labels"][r["model"]] for r in rows])
    ax.set_ylim(len(rows) - 0.4, -0.65)
    limits = [v for row in rows for v in row["mean_95pct_block_ci"]]
    padding = max((max(limits) - min(limits)) * 0.07, 0.005)
    ax.set_xlim(min(limits) - padding, max(limits) + padding)
    ax.xaxis.set_major_locator(MultipleLocator(0.025))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(axis="x", color="#e3e8ee", linewidth=0.8, zorder=1)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_xlabel("Mean log-score gain versus uniform (nats per draw)", labelpad=14)
    for side in ["top", "left", "right"]:
        ax.spines[side].set_visible(False)
    fig.text(0.035, 0.956, "Positive estimates remain uncertain", fontsize=22,
             fontweight="bold", color="#1c3048", va="top")
    fig.text(0.035, 0.910,
             f"Corrected historical study · {draws} newer-machine draws · {data['model_count']} model configurations",
             fontsize=12, color="#455468", va="top")
    fig.text(0.035, 0.872,
             "Points: mean gain   |   Bars: 95% whole-draw block-bootstrap intervals",
             fontsize=11, color="#455468", va="top")
    fig.text(0.035, 0.079,
             f"Smallest Holm-adjusted p-value: {min_holm:.4f}. " +
             ("No improvement established at the 5% familywise level." if min_holm >= 0.05
              else "An adjusted test is below 0.05; retrospective evidence requires caution."),
             fontsize=11, color="#263648", va="top")
    fig.text(0.035, 0.050,
             "Retrospective exploratory comparison. Source: results/research_summary.json.",
             fontsize=10, color="#5b6877", va="top")
    fig.savefig(destination, metadata={
        "Date": None, "Creator": "Matplotlib",
        "Title": f"Mean log-score gain and uncertainty on {draws} newer-machine draws",
        "Description": f"All {data['model_count']} corrected historical models. Whole-draw block bootstrap; smallest Holm-adjusted p-value {min_holm:.4f}. Values from research_summary.json.",
    })
    plt.close(fig)
    destination.write_text("\n".join(line.rstrip() for line in destination.read_text().splitlines()) + "\n")


def render_demo(source: Path, destination: Path):
    summary = json.loads((source.parent / "summary.json").read_text(encoding="utf-8"))
    if summary.get("dataset_kind") != "synthetic_demo":
        raise ValueError("The demo figure requires a synthetic_demo run")
    labels = {
        "uniform": "Uniform baseline",
        "sparse_single_ball": "Single-ball model",
        "spike_slab": "Spike-and-slab model",
        "online_mixture": "Online mixture",
    }
    values = {model: [] for model in labels}
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values[row["model"]].append(float(row["log_gain"]))
    if not values["uniform"] or len({len(v) for v in values.values()}) != 1:
        raise ValueError("Expected paired scores for all four demo models")
    styles = [("#66788a", (0, (4, 3))), ("#266390", "-"),
              ("#b36c23", (0, (6, 2, 1, 2))), ("#3f765f", ":")]
    fig, ax = plt.subplots(figsize=(11.8, 6.3))
    fig.subplots_adjust(left=0.105, right=0.965, top=0.775, bottom=0.225)
    for (model, color_style) in zip(labels, styles):
        color, style = color_style
        cumulative = [0.0, *accumulate(values[model])]
        ax.plot(range(len(cumulative)), cumulative, label=labels[model],
                color=color, linestyle=style, linewidth=2.0)
    ax.set_xlim(0, len(values["uniform"]))
    ax.set_xlabel("Scored draw after the training warmup", labelpad=12)
    ax.set_ylabel("Cumulative log-score gain (nats)", labelpad=12)
    ax.grid(axis="y", color="#e3e8ee", linewidth=0.8)
    ax.tick_params(axis="both", length=0, pad=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncols=4,
              frameon=False, fontsize=10, handlelength=2.5)
    fig.text(0.035, 0.956, "A reproducible demo on fair synthetic draws", fontsize=21,
             fontweight="bold", color="#1c3048", va="top")
    fig.text(0.035, 0.895,
             f"{len(values['uniform'])} scored draws · four models · fixed generator seed",
             fontsize=12, color="#455468", va="top")
    fig.text(0.035, 0.10,
             "One simulated path exercises the pipeline; it does not estimate historical Mark Six performance.",
             fontsize=11, color="#263648", va="top")
    fig.text(0.035, 0.055,
             "Source: saved synthetic demo scores. Higher means a better log score on this generated sequence.",
             fontsize=10, color="#5b6877", va="top")
    fig.savefig(destination, metadata={
        "Date": None, "Creator": "Matplotlib",
        "Title": "Cumulative log-score gains on fair synthetic draws",
        "Description": "Four-model evaluation on a generated fair sequence. This is not a historical Mark Six performance result.",
    })
    plt.close(fig)
    destination.write_text("\n".join(line.rstrip() for line in destination.read_text().splitlines()) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, default=ROOT / "results" / "demo")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    render_research(ROOT / "results" / "research_summary.json",
                    args.output_dir / "research_comparison.svg")
    render_demo(args.demo_dir / "scores.csv", args.output_dir / "demo_comparison.svg")
    print(f"Rendered research_comparison.svg and demo_comparison.svg in {args.output_dir}")


if __name__ == "__main__":
    main()
