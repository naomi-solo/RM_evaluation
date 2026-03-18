#!/usr/bin/env python3
"""
analyze_selectivity_eval_v2.py

Analyze transfer-selectivity outputs produced by run_selectivity_eval_v2.py.

Input:
  results/experiments/<exp_name>/raw/transfer_eval_records.jsonl

What this script computes:
  1) Alpha-aware direction×target summaries with bootstrap CIs:
       - flip_rate (baseline-correct subset)
       - mean_delta_margin (all rows)
       - mean_clean_margin / mean_adv_margin
  2) Source-vs-others selectivity gaps per (direction, alpha), with bootstrap CIs:
       - flip_rate_gap = source_target_flip_rate - mean(other_targets_flip_rate)
       - delta_margin_gap = source_target_mean_delta_margin - mean(other_targets_delta_margin)
  3) Clean-margin-bin stratified summaries to control for baseline fragility.
  4) Optional comparison against GLOBAL baseline directions (if present).
  5) Per-alpha visualizations:
       - flip rate heatmaps
       - mean delta margin heatmaps
       - gap error-bar plots (flip and delta-margin gaps)

Outputs:
  results/experiments/<exp_name>/metrics/summary_by_direction_dataset_alpha.csv
  results/experiments/<exp_name>/metrics/selectivity_gaps_by_direction_alpha.csv
  results/experiments/<exp_name>/metrics/clean_margin_stratified_summary.csv
  results/experiments/<exp_name>/metrics/global_baseline_comparison.csv   (optional)

  results/experiments/<exp_name>/figures/flip_rate_heatmap_alpha=<a>.png
  results/experiments/<exp_name>/figures/mean_delta_margin_heatmap_alpha=<a>.png
  results/experiments/<exp_name>/figures/flip_rate_gap_errorbars_alpha=<a>.png
  results/experiments/<exp_name>/figures/delta_margin_gap_errorbars_alpha=<a>.png
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# IO
# ----------------------------

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ----------------------------
# Bootstrap helpers
# ----------------------------

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def bootstrap_mean_ci(
    x: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0
) -> Tuple[float, float, float, int]:
    """
    Returns (point, lo, hi, n) for mean(x) with percentile bootstrap CI.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return np.nan, np.nan, np.nan, 0

    point = float(np.mean(x))
    if n < 2 or n_boot <= 0:
        return point, np.nan, np.nan, n

    rg = _rng(seed)
    idx = np.arange(n)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        s = rg.choice(idx, size=n, replace=True)
        boots[b] = float(np.mean(x[s]))

    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return point, lo, hi, n


def bootstrap_rate_ci(
    x_bool: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0
) -> Tuple[float, float, float, int]:
    """
    Returns (point, lo, hi, n) for mean(bool_array) with bootstrap CI.
    """
    x = np.asarray(x_bool).astype(bool).astype(float)
    return bootstrap_mean_ci(x, n_boot=n_boot, alpha=alpha, seed=seed)


def bootstrap_diff_means_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0
) -> Tuple[float, float, float, int, int]:
    """
    Independent bootstrap for difference in means: mean(x) - mean(y).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    nx = int(x.size)
    ny = int(y.size)

    if nx == 0 or ny == 0:
        return np.nan, np.nan, np.nan, nx, ny

    point = float(np.mean(x) - np.mean(y))
    if nx < 2 or ny < 2 or n_boot <= 0:
        return point, np.nan, np.nan, nx, ny

    rg = _rng(seed)
    ix = np.arange(nx)
    iy = np.arange(ny)
    boots = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        sx = rg.choice(ix, size=nx, replace=True)
        sy = rg.choice(iy, size=ny, replace=True)
        boots[b] = float(np.mean(x[sx]) - np.mean(y[sy]))

    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return point, lo, hi, nx, ny


# ----------------------------
# Core summaries
# ----------------------------

def ensure_clean_margin_bin(df: pd.DataFrame) -> pd.DataFrame:
    if "clean_margin_bin" in df.columns:
        return df

    if "clean_margin" not in df.columns:
        raise KeyError("Missing clean_margin and clean_margin_bin; cannot stratify.")

    cm_abs = df["clean_margin"].abs().astype(float)

    def _bin(v: float) -> str:
        if v < 0.25:
            return "<0.25"
        if v < 0.5:
            return "[0.25,0.5)"
        if v < 1.0:
            return "[0.5,1.0)"
        if v < 2.0:
            return "[1.0,2.0)"
        return ">=2.0"

    df = df.copy()
    df["clean_margin_bin"] = cm_abs.map(_bin)
    return df


def summarize_by_direction_dataset_alpha(
    df: pd.DataFrame,
    n_boot: int,
    ci_alpha: float,
    seed: int
) -> pd.DataFrame:
    rows = []

    gcols = [
        "direction_id",
        "direction_source_dataset",
        "direction_component",
        "direction_label",
        "direction_is_global_baseline",
        "target_dataset",
        "alpha",
    ]

    for key, sub in df.groupby(gcols, dropna=False):
        (
            d_id,
            src_ds,
            comp,
            label,
            is_global,
            tgt_ds,
            a,
        ) = key

        base = sub[sub["baseline_correct"] == True]
        n_examples = int(len(sub))
        n_base = int(len(base))

        # flip rate on baseline-correct subset
        flip_arr = base["flip_success"].to_numpy(dtype=bool)
        fr, fr_lo, fr_hi, _ = bootstrap_rate_ci(
            flip_arr,
            n_boot=n_boot,
            alpha=ci_alpha,
            seed=seed,
        ) if n_base > 0 else (0.0, np.nan, np.nan, 0)

        # delta margin on full subset
        dm = sub["delta_margin"].to_numpy(dtype=float)
        dm_mu, dm_lo, dm_hi, _ = bootstrap_mean_ci(
            dm,
            n_boot=n_boot,
            alpha=ci_alpha,
            seed=seed + 17,
        )

        row = {
            "direction_id": d_id,
            "direction_source_dataset": src_ds,
            "direction_component": int(comp),
            "direction_label": label,
            "direction_is_global_baseline": bool(is_global),

            "target_dataset": tgt_ds,
            "alpha": float(a),

            "n_examples": n_examples,
            "n_base": n_base,

            "flip_rate": float(fr),
            "flip_rate_ci_lo": float(fr_lo) if np.isfinite(fr_lo) else np.nan,
            "flip_rate_ci_hi": float(fr_hi) if np.isfinite(fr_hi) else np.nan,

            "mean_delta_margin": float(dm_mu),
            "mean_delta_margin_ci_lo": float(dm_lo) if np.isfinite(dm_lo) else np.nan,
            "mean_delta_margin_ci_hi": float(dm_hi) if np.isfinite(dm_hi) else np.nan,

            "mean_clean_margin": float(sub["clean_margin"].mean()),
            "mean_adv_margin": float(sub["adv_margin"].mean()),
        }
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["direction_id", "alpha", "target_dataset"])
    return out


def compute_selectivity_gaps_by_alpha(
    summary: pd.DataFrame,
    raw_df: pd.DataFrame,
    n_boot: int,
    ci_alpha: float,
    seed: int
) -> pd.DataFrame:
    """
    For each (direction_id, alpha):
      flip_rate_gap = flip_rate(source target) - mean(flip_rate(other targets))
      delta_margin_gap = mean_delta_margin(source target) - mean(other targets)

    Also computes bootstrapped CIs from raw rows:
      - flip gap uses baseline-correct subsets (bool means)
      - delta margin gap uses full subsets (means)
    """
    rows = []

    # convenient keys
    key_cols = ["direction_id", "alpha"]

    for (d_id, a), subsum in summary.groupby(key_cols, dropna=False):
        src = subsum["direction_source_dataset"].iloc[0]

        # skip if source target missing
        src_rows = subsum[subsum["target_dataset"] == src]
        oth_rows = subsum[subsum["target_dataset"] != src]
        if len(src_rows) == 0 or len(oth_rows) == 0:
            continue

        src_row = src_rows.iloc[0]

        # point estimates from summary table
        flip_gap_point = float(src_row["flip_rate"] - oth_rows["flip_rate"].mean())
        dm_gap_point = float(src_row["mean_delta_margin"] - oth_rows["mean_delta_margin"].mean())

        # raw for CI
        raw_sub = raw_df[(raw_df["direction_id"] == d_id) & (raw_df["alpha"] == a)]

        raw_src = raw_sub[raw_sub["target_dataset"] == src]
        raw_oth = raw_sub[raw_sub["target_dataset"] != src]

        # flip arrays: baseline-correct only
        src_flip = raw_src[raw_src["baseline_correct"] == True]["flip_success"].to_numpy(dtype=bool)
        oth_flip = raw_oth[raw_oth["baseline_correct"] == True]["flip_success"].to_numpy(dtype=bool)

        flip_gap, flip_lo, flip_hi, n_src_base, n_oth_base = bootstrap_diff_means_ci(
            src_flip.astype(float),
            oth_flip.astype(float),
            n_boot=n_boot,
            alpha=ci_alpha,
            seed=seed + 101,
        )

        # delta margin arrays: all rows
        src_dm = raw_src["delta_margin"].to_numpy(dtype=float)
        oth_dm = raw_oth["delta_margin"].to_numpy(dtype=float)

        dm_gap, dm_lo, dm_hi, n_src, n_oth = bootstrap_diff_means_ci(
            src_dm,
            oth_dm,
            n_boot=n_boot,
            alpha=ci_alpha,
            seed=seed + 211,
        )

        rows.append({
            "direction_id": d_id,
            "alpha": float(a),
            "direction_source_dataset": src,
            "direction_label": src_row["direction_label"],
            "direction_is_global_baseline": bool(src_row["direction_is_global_baseline"]),

            "flip_rate_source": float(src_row["flip_rate"]),
            "flip_rate_others_mean": float(oth_rows["flip_rate"].mean()),
            "flip_rate_gap_source_minus_others": float(flip_gap_point),
            "flip_rate_gap_boot": float(flip_gap) if np.isfinite(flip_gap) else np.nan,
            "flip_rate_gap_ci_lo": float(flip_lo) if np.isfinite(flip_lo) else np.nan,
            "flip_rate_gap_ci_hi": float(flip_hi) if np.isfinite(flip_hi) else np.nan,
            "n_base_source": int(n_src_base),
            "n_base_others": int(n_oth_base),

            "mean_delta_margin_source": float(src_row["mean_delta_margin"]),
            "mean_delta_margin_others_mean": float(oth_rows["mean_delta_margin"].mean()),
            "delta_margin_gap_source_minus_others": float(dm_gap_point),
            "delta_margin_gap_boot": float(dm_gap) if np.isfinite(dm_gap) else np.nan,
            "delta_margin_gap_ci_lo": float(dm_lo) if np.isfinite(dm_lo) else np.nan,
            "delta_margin_gap_ci_hi": float(dm_hi) if np.isfinite(dm_hi) else np.nan,
            "n_source": int(n_src),
            "n_others": int(n_oth),
        })

    out = pd.DataFrame(rows).sort_values(["direction_id", "alpha"])
    return out


def stratified_summary(
    df: pd.DataFrame,
    n_boot: int,
    ci_alpha: float,
    seed: int
) -> pd.DataFrame:
    """
    Summaries by clean_margin_bin to support baseline-fragility-controlled views.
    """
    rows = []
    gcols = [
        "direction_id",
        "direction_source_dataset",
        "direction_component",
        "direction_label",
        "direction_is_global_baseline",
        "target_dataset",
        "alpha",
        "clean_margin_bin",
    ]

    for key, sub in df.groupby(gcols, dropna=False):
        (
            d_id,
            src_ds,
            comp,
            label,
            is_global,
            tgt_ds,
            a,
            cm_bin,
        ) = key

        base = sub[sub["baseline_correct"] == True]
        n_examples = int(len(sub))
        n_base = int(len(base))

        fr = 0.0
        fr_lo = np.nan
        fr_hi = np.nan
        if n_base > 0:
            fr, fr_lo, fr_hi, _ = bootstrap_rate_ci(
                base["flip_success"].to_numpy(dtype=bool),
                n_boot=n_boot,
                alpha=ci_alpha,
                seed=seed + 313,
            )

        dm_mu, dm_lo, dm_hi, _ = bootstrap_mean_ci(
            sub["delta_margin"].to_numpy(dtype=float),
            n_boot=n_boot,
            alpha=ci_alpha,
            seed=seed + 419,
        )

        rows.append({
            "direction_id": d_id,
            "direction_source_dataset": src_ds,
            "direction_component": int(comp),
            "direction_label": label,
            "direction_is_global_baseline": bool(is_global),
            "target_dataset": tgt_ds,
            "alpha": float(a),
            "clean_margin_bin": str(cm_bin),

            "n_examples": n_examples,
            "n_base": n_base,

            "flip_rate": float(fr),
            "flip_rate_ci_lo": float(fr_lo) if np.isfinite(fr_lo) else np.nan,
            "flip_rate_ci_hi": float(fr_hi) if np.isfinite(fr_hi) else np.nan,

            "mean_delta_margin": float(dm_mu),
            "mean_delta_margin_ci_lo": float(dm_lo) if np.isfinite(dm_lo) else np.nan,
            "mean_delta_margin_ci_hi": float(dm_hi) if np.isfinite(dm_hi) else np.nan,

            "mean_clean_margin": float(sub["clean_margin"].mean()),
            "mean_adv_margin": float(sub["adv_margin"].mean()),
        })

    out = pd.DataFrame(rows).sort_values(
        ["direction_id", "alpha", "clean_margin_bin", "target_dataset"]
    )
    return out


def compare_against_global(summary: pd.DataFrame) -> pd.DataFrame:
    """
    If global baseline directions exist, compare each non-global direction
    against the mean of global directions on same (target_dataset, alpha).

    Output columns include:
      delta_vs_global_flip_rate
      delta_vs_global_mean_delta_margin
    """
    if "direction_is_global_baseline" not in summary.columns:
        return pd.DataFrame([])

    g = summary[summary["direction_is_global_baseline"] == True].copy()
    ng = summary[summary["direction_is_global_baseline"] == False].copy()

    if len(g) == 0 or len(ng) == 0:
        return pd.DataFrame([])

    g_ref = (
        g.groupby(["target_dataset", "alpha"], dropna=False)
        .agg(
            global_flip_rate=("flip_rate", "mean"),
            global_mean_delta_margin=("mean_delta_margin", "mean"),
            global_n_dirs=("direction_id", "nunique"),
        )
        .reset_index()
    )

    merged = ng.merge(g_ref, on=["target_dataset", "alpha"], how="left")
    merged["delta_vs_global_flip_rate"] = merged["flip_rate"] - merged["global_flip_rate"]
    merged["delta_vs_global_mean_delta_margin"] = (
        merged["mean_delta_margin"] - merged["global_mean_delta_margin"]
    )

    cols = [
        "direction_id",
        "direction_source_dataset",
        "direction_component",
        "direction_label",
        "target_dataset",
        "alpha",
        "flip_rate",
        "global_flip_rate",
        "delta_vs_global_flip_rate",
        "mean_delta_margin",
        "global_mean_delta_margin",
        "delta_vs_global_mean_delta_margin",
        "global_n_dirs",
    ]
    return merged[cols].sort_values(["direction_id", "alpha", "target_dataset"])


# ----------------------------
# Plotting
# ----------------------------

def save_heatmap(
    df_wide: pd.DataFrame,
    title: str,
    out_path: Path,
    cmap: str = "viridis",
    center_zero: bool = False
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(
        figsize=(1.7 * max(4, df_wide.shape[1]), 0.8 * max(3, df_wide.shape[0]))
    )
    mat = df_wide.values.astype(float)

    if center_zero:
        vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(mat, aspect="auto", cmap=cmap)

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


def make_per_alpha_heatmaps(summary: pd.DataFrame, figs_dir: Path):
    alphas = sorted(summary["alpha"].dropna().unique().tolist())

    for a in alphas:
        sub = summary[summary["alpha"] == a].copy()
        if sub.empty:
            continue

        # rows=direction, cols=target_dataset
        flip_wide = sub.pivot(index="direction_id", columns="target_dataset", values="flip_rate")
        save_heatmap(
            flip_wide,
            f"Flip Rate (baseline-correct) | alpha={a:g}",
            figs_dir / f"flip_rate_heatmap_alpha={a:g}.png",
            cmap="Blues",
            center_zero=False,
        )

        dm_wide = sub.pivot(index="direction_id", columns="target_dataset", values="mean_delta_margin")
        save_heatmap(
            dm_wide,
            f"Mean Delta Margin (adv-clean) | alpha={a:g}",
            figs_dir / f"mean_delta_margin_heatmap_alpha={a:g}.png",
            cmap="RdBu_r",
            center_zero=True,
        )

def plot_gap_errorbars(
    gaps: pd.DataFrame,
    figs_dir: Path,
):
    """
    Dot + error bar plots for selectivity gaps.
    One plot per alpha:
      - flip_rate gap
      - delta_margin gap
    """
    figs_dir.mkdir(parents=True, exist_ok=True)

    alphas = sorted(gaps["alpha"].dropna().unique().tolist())

    for a in alphas:
        sub = gaps[gaps["alpha"] == a].copy()
        if sub.empty:
            continue

        # sort for readability (largest effect on top)
        sub = sub.sort_values("delta_margin_gap_source_minus_others", ascending=True)

        y = np.arange(len(sub))
        labels = sub["direction_id"].tolist()

        # ------------------------
        # Flip rate gap
        # ------------------------
        fig, ax = plt.subplots(figsize=(6, 0.5 * len(sub) + 1))

        x = sub["flip_rate_gap_source_minus_others"].values
        lo = sub["flip_rate_gap_ci_lo"].values
        hi = sub["flip_rate_gap_ci_hi"].values

        xerr = np.vstack([x - lo, hi - x])

        ax.errorbar(x, y, xerr=xerr, fmt="o")
        ax.axvline(0.0)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Flip Rate Gap (source − others)")
        ax.set_title(f"Selectivity (Flip Rate Gap) | alpha={a:g}")

        fig.tight_layout()
        fig.savefig(figs_dir / f"flip_rate_gap_errorbars_alpha={a:g}.png", dpi=200)
        plt.close(fig)

        # ------------------------
        # Delta margin gap
        # ------------------------
        fig, ax = plt.subplots(figsize=(6, 0.5 * len(sub) + 1))

        x = sub["delta_margin_gap_source_minus_others"].values
        lo = sub["delta_margin_gap_ci_lo"].values
        hi = sub["delta_margin_gap_ci_hi"].values

        xerr = np.vstack([x - lo, hi - x])

        ax.errorbar(x, y, xerr=xerr, fmt="o")
        ax.axvline(0.0)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Delta Margin Gap (source − others)")
        ax.set_title(f"Selectivity (Delta Margin Gap) | alpha={a:g}")

        fig.tight_layout()
        fig.savefig(figs_dir / f"delta_margin_gap_errorbars_alpha={a:g}.png", dpi=200)
        plt.close(fig)

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True, help="results/experiments/<exp_name>")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--ci_alpha", type=float, default=0.05, help="0.05 => 95% CI")
    ap.add_argument("--seed", type=int, default=0)
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

    required = [
        "direction_id",
        "direction_source_dataset",
        "direction_component",
        "direction_label",
        "target_dataset",
        "baseline_correct",
        "flip_success",
        "clean_margin",
        "adv_margin",
        "delta_margin",
        "alpha",
    ]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    if "direction_is_global_baseline" not in df.columns:
        df["direction_is_global_baseline"] = False

    df["alpha"] = df["alpha"].astype(float)
    df["baseline_correct"] = df["baseline_correct"].astype(bool)
    df["flip_success"] = df["flip_success"].astype(bool)

    df = ensure_clean_margin_bin(df)

    # 1) Main alpha-aware summary
    summary = summarize_by_direction_dataset_alpha(
        df=df,
        n_boot=args.n_boot,
        ci_alpha=args.ci_alpha,
        seed=args.seed,
    )
    summary_path = metrics_dir / "summary_by_direction_dataset_alpha.csv"
    save_csv(summary, summary_path)

    # 2) Selectivity gaps by direction+alpha
    gaps = compute_selectivity_gaps_by_alpha(
        summary=summary,
        raw_df=df,
        n_boot=args.n_boot,
        ci_alpha=args.ci_alpha,
        seed=args.seed + 7,
    )
    gaps_path = metrics_dir / "selectivity_gaps_by_direction_alpha.csv"
    save_csv(gaps, gaps_path)

    # 3) Clean-margin-bin stratified summary
    strat = stratified_summary(
        df=df,
        n_boot=max(300, args.n_boot // 2),
        ci_alpha=args.ci_alpha,
        seed=args.seed + 13,
    )
    strat_path = metrics_dir / "clean_margin_stratified_summary.csv"
    save_csv(strat, strat_path)

    # 4) Compare against global baseline if present
    glb = compare_against_global(summary)
    glb_path = metrics_dir / "global_baseline_comparison.csv"
    if len(glb) > 0:
        save_csv(glb, glb_path)

    # 5) Per-alpha heatmaps
    make_per_alpha_heatmaps(summary, figs_dir)

    # 6) Gap error bar plots (new)
    plot_gap_errorbars(gaps, figs_dir)

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {gaps_path}")
    print(f"Wrote: {strat_path}")
    if len(glb) > 0:
        print(f"Wrote: {glb_path}")
    print(f"Wrote figures to: {figs_dir}")

    # quick console preview
    if len(gaps) > 0:
        show = [
            "direction_id",
            "alpha",
            "flip_rate_gap_source_minus_others",
            "flip_rate_gap_ci_lo",
            "flip_rate_gap_ci_hi",
            "delta_margin_gap_source_minus_others",
            "delta_margin_gap_ci_lo",
            "delta_margin_gap_ci_hi",
            "n_base_source",
            "n_base_others",
        ]
        print("\nTop selectivity rows (by |delta_margin gap|):")
        tmp = gaps.copy()
        tmp["abs_dm_gap"] = tmp["delta_margin_gap_source_minus_others"].abs()
        print(tmp.sort_values("abs_dm_gap", ascending=False)[show].head(15).to_string(index=False))


if __name__ == "__main__":
    main()