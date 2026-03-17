#!/usr/bin/env python3
"""
Analyze transfer-selectivity outputs from run_selectivity_eval.py.

Reads:
  results/experiments/<exp_name>/raw/transfer_eval_records.jsonl

Writes:
  results/experiments/<exp_name>/metrics/summary_by_direction_dataset.csv
  results/experiments/<exp_name>/metrics/selectivity_gaps.csv
  results/experiments/<exp_name>/figures/flip_rate_heatmap.png
  results/experiments/<exp_name>/figures/mean_delta_margin_heatmap.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gcols = ["direction_id", "direction_source_dataset", "direction_component", "direction_label", "target_dataset"]

    for key, sub in df.groupby(gcols, dropna=False):
        direction_id, src_ds, comp, label, tgt_ds = key

        base = sub[sub["baseline_correct"] == True]
        n_base = int(len(base))
        flip_rate = float(base["flip_success"].mean()) if n_base > 0 else 0.0

        rows.append({
            "direction_id": direction_id,
            "direction_source_dataset": src_ds,
            "direction_component": int(comp),
            "direction_label": label,

            "target_dataset": tgt_ds,
            "n_examples": int(len(sub)),
            "n_base": n_base,

            "flip_rate": flip_rate,
            "mean_delta_margin": float(sub["delta_margin"].mean()),
            "mean_adv_margin": float(sub["adv_margin"].mean()),
            "mean_clean_margin": float(sub["clean_margin"].mean()),
        })

    out = pd.DataFrame(rows).sort_values(["direction_id", "target_dataset"])
    return out


def compute_selectivity_gaps(summary: pd.DataFrame) -> pd.DataFrame:
    """
    For each direction:
      flip_rate_gap = flip_rate(target=source_dataset) - mean(flip_rate(other targets))
      delta_margin_gap = mean_delta_margin(target=source_dataset) - mean(other)
    Note: For delta_margin, "more negative" is stronger attack effect.
          So negative gap suggests stronger push on source dataset.
    """
    rows = []
    for d_id, sub in summary.groupby("direction_id"):
        src = sub["direction_source_dataset"].iloc[0]
        if src not in set(sub["target_dataset"]):
            continue

        src_row = sub[sub["target_dataset"] == src].iloc[0]
        oth = sub[sub["target_dataset"] != src]
        if len(oth) == 0:
            continue

        rows.append({
            "direction_id": d_id,
            "direction_source_dataset": src,
            "direction_label": src_row["direction_label"],

            "flip_rate_source": float(src_row["flip_rate"]),
            "flip_rate_others_mean": float(oth["flip_rate"].mean()),
            "flip_rate_gap_source_minus_others": float(src_row["flip_rate"] - oth["flip_rate"].mean()),

            "mean_delta_margin_source": float(src_row["mean_delta_margin"]),
            "mean_delta_margin_others_mean": float(oth["mean_delta_margin"].mean()),
            "delta_margin_gap_source_minus_others": float(src_row["mean_delta_margin"] - oth["mean_delta_margin"].mean()),

            "mean_clean_margin_source": float(src_row["mean_clean_margin"]),
            "mean_adv_margin_source": float(src_row["mean_adv_margin"]),
            "n_base_source": int(src_row["n_base"]),
        })

    return pd.DataFrame(rows).sort_values("direction_id")


def save_heatmap(df_wide: pd.DataFrame, title: str, out_path: Path, cmap: str = "viridis", center=None):
    fig, ax = plt.subplots(figsize=(1.7 * max(4, df_wide.shape[1]), 0.8 * max(3, df_wide.shape[0])))
    mat = df_wide.values.astype(float)

    if center is None:
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
    else:
        vmax = np.nanmax(np.abs(mat))
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(df_wide.shape[1]))
    ax.set_xticklabels(df_wide.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(df_wide.shape[0]))
    ax.set_yticklabels(df_wide.index)

    for i in range(df_wide.shape[0]):
        for j in range(df_wide.shape[1]):
            v = mat[i, j]
            txt = "nan" if np.isnan(v) else f"{v:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white")

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True, help="results/experiments/<exp_name>")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    raw_path = exp_dir / "raw" / "transfer_eval_records.jsonl"
    metrics_dir = exp_dir / "metrics"
    figs_dir = exp_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(raw_path)
    if not rows:
        raise RuntimeError(f"No records found in {raw_path}")

    df = pd.DataFrame(rows)

    needed = [
        "direction_id", "direction_source_dataset", "direction_component", "direction_label",
        "target_dataset", "baseline_correct", "flip_success",
        "clean_margin", "adv_margin", "delta_margin",
    ]
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    summary = summarize(df)
    summary_path = metrics_dir / "summary_by_direction_dataset.csv"
    summary.to_csv(summary_path, index=False)

    gaps = compute_selectivity_gaps(summary)
    gaps_path = metrics_dir / "selectivity_gaps.csv"
    gaps.to_csv(gaps_path, index=False)

    # Figure 1: flip rate heatmap
    flip_wide = summary.pivot(index="direction_id", columns="target_dataset", values="flip_rate")
    save_heatmap(
        flip_wide,
        "Flip Rate (baseline-correct subset)\nrows=source direction, cols=target dataset",
        figs_dir / "flip_rate_heatmap.png",
        cmap="Blues",
        center=None,
    )

    # Figure 2: mean delta margin heatmap
    dmargin_wide = summary.pivot(index="direction_id", columns="target_dataset", values="mean_delta_margin")
    save_heatmap(
        dmargin_wide,
        "Mean Delta Margin (adv - clean)\nrows=source direction, cols=target dataset",
        figs_dir / "mean_delta_margin_heatmap.png",
        cmap="RdBu_r",
        center=0.0,
    )

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {gaps_path}")
    print(f"Wrote: {figs_dir / 'flip_rate_heatmap.png'}")
    print(f"Wrote: {figs_dir / 'mean_delta_margin_heatmap.png'}")

    if len(gaps) > 0:
        print("\nSelectivity gaps (source minus others):")
        print(gaps[[
            "direction_id",
            "flip_rate_gap_source_minus_others",
            "delta_margin_gap_source_minus_others",
            "n_base_source",
        ]].to_string(index=False))


if __name__ == "__main__":
    main()