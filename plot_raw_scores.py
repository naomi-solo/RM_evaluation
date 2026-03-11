#!/usr/bin/env python3
"""
Make 3 figures (early/mid/late). Each figure is a panel grid over epsilons,
where each panel is a 2D histogram of (Δchosen, Δrejected):

  x = Δchosen   = score_chosen_adv - score_chosen
  y = Δrejected = score_rejected_adv - score_rejected

Layer selection (per bucket) uses rank aggregation across ε:
  - For each ε, rank layers by a metric (default: mean_delta_gap = mean(Δrejected - Δchosen))
  - Aggregate ranks across ε (mean rank); choose best (lowest mean rank) within bucket.

Usage:
  python plot_2d_eps_panels.py --infile results.jsonl --outdir figs \
      --metric mean_delta_gap --only_baseline_correct \
      --bins 50 --panel_cols 5 --use_hexbin

Outputs:
  figs/early_layerX_2d.png
  figs/mid_layerY_2d.png
  figs/late_layerZ_2d.png
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return pd.DataFrame(rows)


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["delta_chosen"] = df["score_chosen_adv"] - df["score_chosen"]
    df["delta_rejected"] = df["score_rejected_adv"] - df["score_rejected"]
    df["delta_gap"] = df["delta_rejected"] - df["delta_chosen"]  # helps rejected, hurts chosen
    df["abs_delta_margin"] = (df["margin_adv"] - df["margin"]).abs()
    return df


def add_depth_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    For a 0..27 layer model:
      early: 0..8
      mid:   9..17
      late:  18..27
    """
    df = df.copy()

    def bucket(layer: int) -> str:
        layer = int(layer)
        if layer <= 8:
            return "early"
        elif layer <= 17:
            return "mid"
        else:
            return "late"

    df["depth_bucket"] = df["layer"].astype(int).apply(bucket)
    return df


def layer_metric_by_eps(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Returns a table indexed by (epsilon, layer) with a column 'score' for ranking.
    """
    g = df.groupby(["epsilon", "layer"], sort=True)

    if metric == "mean_delta_gap":
        s = g["delta_gap"].mean()
    elif metric == "mean_abs_delta_margin":
        s = g["abs_delta_margin"].mean()
    elif metric == "flip_rate":
        # flip_rate = flipped / baseline_correct, per (eps, layer)
        # if baseline_correct count is 0, define flip_rate = 0
        def flip_rate_fn(x: pd.DataFrame) -> float:
            base = int(x["baseline_correct"].sum())
            if base <= 0:
                return 0.0
            # "flipped" should already imply baseline_correct, but keep it safe:
            flips = int((x["flipped"] & x["baseline_correct"]).sum())
            return flips / base

        s = g.apply(flip_rate_fn)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    out = s.reset_index().rename(columns={0: "score"})
    # For groupby.apply in newer pandas, the column name can differ; normalize:
    if "score" not in out.columns:
        # find last column
        out = out.rename(columns={out.columns[-1]: "score"})
    return out


def choose_best_layer_rank_agg(
    df_bucket: pd.DataFrame,
    metric: str,
    eps_values: list[float] | None = None,
    higher_is_better: bool = True,
) -> tuple[int, pd.DataFrame]:
    """
    Rank aggregation across eps:
      - For each eps, rank layers by score (descending if higher_is_better)
      - Aggregate ranks across eps (mean rank)
      - Pick layer with best (lowest mean rank)

    Returns (best_layer, ranking_df), where ranking_df has mean_rank and per-eps ranks.
    """
    metrics = layer_metric_by_eps(df_bucket, metric)

    if eps_values is not None:
        metrics = metrics[metrics["epsilon"].isin(eps_values)].copy()

    if metrics.empty:
        raise RuntimeError("No data available for rank aggregation (check filters / eps range).")

    # Build per-epsilon ranks
    ranks = []
    for eps, sub in metrics.groupby("epsilon", sort=True):
        sub = sub.copy()
        # rank 1 = best
        sub["rank"] = sub["score"].rank(
            ascending=not higher_is_better,
            method="average",
        )
        sub["epsilon"] = float(eps)
        ranks.append(sub[["epsilon", "layer", "rank", "score"]])

    r = pd.concat(ranks, ignore_index=True)

    # Pivot to see ranks by eps
    pivot = r.pivot_table(index="layer", columns="epsilon", values="rank", aggfunc="mean")
    mean_rank = pivot.mean(axis=1)
    ranking = pivot.copy()
    ranking["mean_rank"] = mean_rank
    ranking = ranking.sort_values("mean_rank", ascending=True)

    best_layer = int(ranking.index[0])
    return best_layer, ranking.reset_index()


def _shared_limits(vals_x: np.ndarray, vals_y: np.ndarray, clip_pct: float) -> tuple[float, float, float, float]:
    """
    Shared x/y limits using symmetric percentiles to avoid outliers.
    """
    x = vals_x[np.isfinite(vals_x)]
    y = vals_y[np.isfinite(vals_y)]
    if x.size == 0 or y.size == 0:
        return -1.0, 1.0, -1.0, 1.0

    lo = float(clip_pct)
    hi = float(100.0 - clip_pct)

    xlo, xhi = np.percentile(x, [lo, hi])
    ylo, yhi = np.percentile(y, [lo, hi])

    # Add small padding
    xpad = 0.05 * (xhi - xlo + 1e-9)
    ypad = 0.05 * (yhi - ylo + 1e-9)
    return xlo - xpad, xhi + xpad, ylo - ypad, yhi + ypad


def plot_bucket_2d_panels(
    df_layer: pd.DataFrame,
    bucket: str,
    layer: int,
    eps_values: list[float],
    outpath: Path,
    bins: int,
    panel_cols: int,
    clip_pct: float,
    use_hexbin: bool,
    gridsize: int,
):
    """
    One figure per bucket: panels over eps, each panel is 2D histogram of (delta_chosen, delta_rejected)
    """
    df_layer = df_layer.copy()
    # ensure eps ordering and presence
    eps_values = [float(e) for e in eps_values if float(e) in set(df_layer["epsilon"].astype(float).unique())]
    eps_values = sorted(eps_values)
    if not eps_values:
        raise RuntimeError(f"No eps values found for {bucket} layer={layer} (after filtering).")

    # Shared axis limits across eps (within this bucket/layer)
    x_all = df_layer["delta_chosen"].to_numpy(dtype=float)
    y_all = df_layer["delta_rejected"].to_numpy(dtype=float)
    xmin, xmax, ymin, ymax = _shared_limits(x_all, y_all, clip_pct=clip_pct)

    n_panels = len(eps_values)
    ncols = max(1, int(panel_cols))
    nrows = int(math.ceil(n_panels / ncols))

    fig_w = 3.2 * ncols
    fig_h = 3.0 * nrows + 0.6
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)

    for idx, eps in enumerate(eps_values):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        dfe = df_layer[df_layer["epsilon"].astype(float) == float(eps)]
        x = dfe["delta_chosen"].to_numpy(dtype=float)
        y = dfe["delta_rejected"].to_numpy(dtype=float)

        # Clip to shared bounds for comparability (optional but helps)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]

        ax.set_title(f"ε={eps:g} (n={len(x)})", fontsize=10)
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.axhline(0, linestyle="--", linewidth=1)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        if use_hexbin:
            hb = ax.hexbin(
                x, y,
                gridsize=gridsize,
                extent=(xmin, xmax, ymin, ymax),
                mincnt=1,
            )
        else:
            # 2D histogram via hist2d
            h = ax.hist2d(
                x, y,
                bins=bins,
                range=[[xmin, xmax], [ymin, ymax]],
            )

        ax.set_xlabel("Δchosen", fontsize=9)
        ax.set_ylabel("Δrejected", fontsize=9)

    # Turn off any unused axes
    for idx in range(n_panels, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].axis("off")

    fig.suptitle(f"{bucket} | selected layer={layer} | 2D hist of (Δchosen, Δrejected) by ε", y=0.995)
    fig.tight_layout()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="JSONL produced by run_flip_sweep.py")
    ap.add_argument("--outdir", default="figs", help="Directory to write figures")
    ap.add_argument("--metric", default="mean_delta_gap",
                    choices=["mean_delta_gap", "mean_abs_delta_margin", "flip_rate"],
                    help="Metric to rank layers within each bucket per epsilon, then rank-aggregate across eps.")
    ap.add_argument("--eps_cap_for_selection", type=float, default=None,
                    help="If set, only eps <= cap are used for selecting layers (rank aggregation). "
                         "Plot still uses all eps unless --plot_eps_cap is set.")
    ap.add_argument("--plot_eps_cap", type=float, default=None,
                    help="If set, only eps <= cap are plotted.")
    ap.add_argument("--only_baseline_correct", action="store_true",
                    help="If set, filter to baseline_correct==True before selection and plotting.")
    ap.add_argument("--bins", type=int, default=50, help="Bins for 2D histogram (if not using hexbin).")
    ap.add_argument("--panel_cols", type=int, default=5, help="Number of columns in epsilon panel grid.")
    ap.add_argument("--clip_pct", type=float, default=1.0,
                    help="Percentile clip for shared axis limits (e.g., 1.0 uses 1st..99th).")
    ap.add_argument("--use_hexbin", action="store_true", help="Use hexbin instead of hist2d.")
    ap.add_argument("--gridsize", type=int, default=40, help="Hexbin gridsize (if --use_hexbin).")
    args = ap.parse_args()

    df = load_jsonl(args.infile)
    df = add_deltas(df)
    df = add_depth_bucket(df)

    if args.only_baseline_correct:
        df = df[df["baseline_correct"] == True].copy()

    # Decide eps sets
    eps_all = sorted(df["epsilon"].astype(float).unique().tolist())
    if not eps_all:
        raise RuntimeError("No epsilon values found after filtering.")

    eps_for_selection = eps_all
    if args.eps_cap_for_selection is not None:
        eps_for_selection = [e for e in eps_all if e <= float(args.eps_cap_for_selection)]

    eps_for_plot = eps_all
    if args.plot_eps_cap is not None:
        eps_for_plot = [e for e in eps_all if e <= float(args.plot_eps_cap)]

    # metric direction
    # For all built-in metrics, higher is better.
    higher_is_better = True

    outdir = Path(args.outdir)

    selected = {}
    for bucket in ["early", "mid", "late"]:
        dfb = df[df["depth_bucket"] == bucket].copy()
        if dfb.empty:
            raise RuntimeError(f"No data for bucket={bucket} after filtering.")

        best_layer, ranking = choose_best_layer_rank_agg(
            dfb,
            metric=args.metric,
            eps_values=eps_for_selection,
            higher_is_better=higher_is_better,
        )
        selected[bucket] = (best_layer, ranking)

        print(f"\n[{bucket}] selected layer={best_layer} by rank aggregation across eps")
        # show top 5 with mean rank
        show = ranking[["layer", "mean_rank"]].sort_values("mean_rank").head(5)
        print(show.to_string(index=False))

    # Plot one figure per bucket using the selected layer
    for bucket in ["early", "mid", "late"]:
        layer, _ranking = selected[bucket]
        df_layer = df[(df["depth_bucket"] == bucket) & (df["layer"].astype(int) == int(layer))].copy()

        outpath = outdir / f"{bucket}_layer{layer}_2d.png"
        plot_bucket_2d_panels(
            df_layer=df_layer,
            bucket=bucket,
            layer=int(layer),
            eps_values=eps_for_plot,
            outpath=outpath,
            bins=int(args.bins),
            panel_cols=int(args.panel_cols),
            clip_pct=float(args.clip_pct),
            use_hexbin=bool(args.use_hexbin),
            gridsize=int(args.gridsize),
        )
        print(f"Wrote {outpath}")

    print("\nDone.")


if __name__ == "__main__":
    main()
