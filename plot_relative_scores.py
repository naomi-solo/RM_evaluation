#!/usr/bin/env python3
"""
Plots requested by the whiteboard notes.

A) 2D panels over eps: (relative change chosen, relative change rejected)
   x = (score_chosen_adv - score_chosen) / score_chosen
   y = (score_rejected_adv - score_rejected) / score_rejected

   We produce ONE figure per depth bucket (early/mid/late), and we do it
   separately for each attack mode:
     - attack_mode = "sign_flip"
     - attack_mode = "no_sign_flip"

   Layer selection (per bucket, per mode) uses rank aggregation across ε:
     - For each ε, rank layers by a metric (default: mean_rel_gap = mean(rel_rejected - rel_chosen))
     - Aggregate ranks across ε (mean rank); choose best (lowest mean rank) within bucket.

B) Flip-success-rate (FSR) vs layer depth ℓ, with a separate curve per ε
   (matching the whiteboard sketch), again separately per attack mode.

   This version adds bootstrap uncertainty bands for each curve:
     - Bootstrap over examples within each (epsilon, layer) cell.
     - Plot the point estimate plus a (1-alpha) confidence interval band.

Usage:
  python pick_layer_and_plot.py --infile results.jsonl --outdir figs

Example:
  python pick_layer_and_plot.py --infile results/sweep.jsonl --outdir figs \
      --metric mean_rel_gap --only_baseline_correct --use_hexbin \
      --bootstrap --n_boot 500 --alpha 0.05

Outputs (examples):
  figs/sign_flip/early_layer5_rel2d.png
  figs/sign_flip/mid_layer13_rel2d.png
  figs/sign_flip/late_layer24_rel2d.png
  figs/sign_flip/fsr_by_layer.png

  figs/no_sign_flip/early_layer5_rel2d.png
  ...
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DENOM_EPS = 1e-8


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


REL_FLOOR = 0.1   # stability floor for relative normalization (tune if needed)

def add_relative_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds stable relative-change columns:

      rel_chosen   = (score_chosen_adv - score_chosen) / max(|score_chosen|, REL_FLOOR)
      rel_rejected = (score_rejected_adv - score_rejected) / max(|score_rejected|, REL_FLOOR)

    This avoids blow-ups when baseline scores are near zero or negative,
    while still behaving like a relative change.
    """
    df = df.copy()

    # raw deltas (useful for debugging / sanity checks)
    df["delta_chosen_raw"] = df["score_chosen_adv"] - df["score_chosen"]
    df["delta_rejected_raw"] = df["score_rejected_adv"] - df["score_rejected"]

    # stable denominators: abs(clean) with a floor
    denom_c = df["score_chosen"].astype(float).abs().clip(lower=REL_FLOOR)
    denom_r = df["score_rejected"].astype(float).abs().clip(lower=REL_FLOOR)

    # relative changes
    df["rel_chosen"] = df["delta_chosen_raw"].astype(float) / denom_c
    df["rel_rejected"] = df["delta_rejected_raw"].astype(float) / denom_r
    df["rel_gap"] = df["rel_rejected"] - df["rel_chosen"]

    # Keep legacy names so the rest of the plotting code is unchanged
    df["delta_chosen"] = df["rel_chosen"]
    df["delta_rejected"] = df["rel_rejected"]
    df["delta_gap"] = df["rel_gap"]

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


def flip_rate_group(x: pd.DataFrame) -> float:
    """
    FSR = flips / baseline_correct, where flips is counted only among baseline-correct.
    """
    bc = x["baseline_correct"].astype(bool)
    base = int(bc.sum())
    if base <= 0:
        return 0.0
    flips = int((x["flipped"].astype(bool) & bc).sum())
    return flips / base


def layer_metric_by_eps(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Returns a table indexed by (epsilon, layer) with a column 'score' for ranking.
    """
    g = df.groupby(["epsilon", "layer"], sort=True)

    if metric == "mean_rel_gap":
        s = g["rel_gap"].mean()
    elif metric == "median_rel_gap":
        s = g["rel_gap"].median()
    elif metric == "flip_rate":
        s = g.apply(flip_rate_group)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    out = s.reset_index()
    # The value column name after reset_index can vary; normalize it.
    val_cols = [c for c in out.columns if c not in ("epsilon", "layer")]
    if len(val_cols) != 1:
        raise RuntimeError(f"Unexpected metric table columns: {out.columns.tolist()}")
    out = out.rename(columns={val_cols[0]: "score"})
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
    """
    metrics = layer_metric_by_eps(df_bucket, metric)

    if eps_values is not None:
        metrics = metrics[metrics["epsilon"].isin(eps_values)].copy()

    if metrics.empty:
        raise RuntimeError("No data available for rank aggregation (check filters / eps range).")

    ranks = []
    for eps, sub in metrics.groupby("epsilon", sort=True):
        sub = sub.copy()
        sub["rank"] = sub["score"].rank(
            ascending=not higher_is_better,
            method="average",
        )
        sub["epsilon"] = float(eps)
        ranks.append(sub[["epsilon", "layer", "rank", "score"]])

    r = pd.concat(ranks, ignore_index=True)

    pivot = r.pivot_table(index="layer", columns="epsilon", values="rank", aggfunc="mean")
    mean_rank = pivot.mean(axis=1)

    ranking = pivot.copy()
    ranking["mean_rank"] = mean_rank
    ranking = ranking.sort_values("mean_rank", ascending=True)

    best_layer = int(ranking.index[0])
    return best_layer, ranking.reset_index()


def _shared_limits(vals_x: np.ndarray, vals_y: np.ndarray, clip_pct: float) -> tuple[float, float, float, float]:
    x = vals_x[np.isfinite(vals_x)]
    y = vals_y[np.isfinite(vals_y)]
    if x.size == 0 or y.size == 0:
        return -1.0, 1.0, -1.0, 1.0

    lo = float(clip_pct)
    hi = float(100.0 - clip_pct)

    xlo, xhi = np.percentile(x, [lo, hi])
    ylo, yhi = np.percentile(y, [lo, hi])

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
    One figure per bucket: panels over eps, each panel is 2D histogram of
      (rel_chosen, rel_rejected)
    """
    df_layer = df_layer.copy()

    eps_values = [float(e) for e in eps_values if float(e) in set(df_layer["epsilon"].astype(float).unique())]
    eps_values = sorted(eps_values)
    if not eps_values:
        raise RuntimeError(f"No eps values found for {bucket} layer={layer} (after filtering).")

    x_all = df_layer["rel_chosen"].to_numpy(dtype=float)
    y_all = df_layer["rel_rejected"].to_numpy(dtype=float)
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
        x = dfe["rel_chosen"].to_numpy(dtype=float)
        y = dfe["rel_rejected"].to_numpy(dtype=float)

        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]

        ax.set_title(f"ε={eps:g} (n={len(x)})", fontsize=10)
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.axhline(0, linestyle="--", linewidth=1)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        if use_hexbin:
            ax.hexbin(
                x, y,
                gridsize=gridsize,
                extent=(xmin, xmax, ymin, ymax),
                mincnt=1,
            )
        else:
            ax.hist2d(
                x, y,
                bins=bins,
                range=[[xmin, xmax], [ymin, ymax]],
                cmin=1
            )

        ax.set_xlabel("Δchosen / chosen", fontsize=9)
        ax.set_ylabel("Δrejected / rejected", fontsize=9)

    for idx in range(n_panels, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].axis("off")

    fig.suptitle(f"{bucket} | selected layer={layer} | relative changes by ε", y=0.995)
    fig.tight_layout()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _bootstrap_flip_rate(values_baseline_correct: np.ndarray, values_flipped: np.ndarray, n_boot: int, alpha: float, rng: np.random.Generator):
    # Filter to baseline-correct only
    mask = values_baseline_correct
    flipped_bc = values_flipped[mask]
    
    n = len(flipped_bc)
    if n == 0:
        return 0.0, np.nan, np.nan, 0
    
    # Point estimate
    point = float(flipped_bc.sum()) / n
    
    if n_boot <= 0 or n < 2:
        return point, np.nan, np.nan, n
    
    boots = np.empty(n_boot, dtype=float)
    idx = np.arange(n)
    for b in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        boots[b] = float(flipped_bc[samp].sum()) / n  # denominator is constant
    
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return point, lo, hi, n


def plot_fsr_by_layer(
    df_mode: pd.DataFrame,
    outpath: Path,
    eps_values: list[float],
    layers: list[int] | None = None,
    bootstrap: bool = True,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
):
    """
    Flip Success Rate (FSR) vs layer depth ℓ, with one curve per epsilon.
    Matches the whiteboard sketch: multiple epsilon curves over the layer axis.

    FSR for a given (layer, eps) is:
      flipped / baseline_correct

    If bootstrap=True, add (1-alpha) bootstrap CI bands per curve.
    """
    df_mode = df_mode.copy()
    df_mode["layer"] = df_mode["layer"].astype(int)
    df_mode["epsilon"] = df_mode["epsilon"].astype(float)
    df_mode["baseline_correct"] = df_mode["baseline_correct"].astype(bool)
    df_mode["flipped"] = df_mode["flipped"].astype(bool)

    if layers is None:
        layers = sorted(df_mode["layer"].unique().tolist())
    else:
        layers = [int(x) for x in layers]

    eps_values = sorted([float(e) for e in eps_values if float(e) in set(df_mode["epsilon"].unique())])
    if not eps_values or not layers:
        raise RuntimeError("No eps or layers found for FSR plot.")

    rng = np.random.default_rng(int(seed))

    # Compute point + CI per (eps, layer)
    records = []
    for eps in eps_values:
        dfe = df_mode[df_mode["epsilon"] == float(eps)]
        for layer in layers:
            cell = dfe[dfe["layer"] == int(layer)]
            bc = cell["baseline_correct"].to_numpy(dtype=bool)
            fl = cell["flipped"].to_numpy(dtype=bool)

            if bootstrap:
                point, lo, hi, base = _bootstrap_flip_rate(bc, fl, n_boot=n_boot, alpha=alpha, rng=rng)
            else:
                base = int(bc.sum())
                if base == 0:
                    point, lo, hi = 0.0, np.nan, np.nan
                else:
                    point = float((fl & bc).sum()) / base
                    lo, hi = np.nan, np.nan

            records.append({"epsilon": float(eps), "layer": int(layer), "fsr": point, "lo": lo, "hi": hi, "n_base": base})

    fr = pd.DataFrame.from_records(records)
    piv = fr.pivot_table(index="layer", columns="epsilon", values="fsr", aggfunc="mean").reindex(index=layers)
    piv_lo = fr.pivot_table(index="layer", columns="epsilon", values="lo", aggfunc="mean").reindex(index=layers)
    piv_hi = fr.pivot_table(index="layer", columns="epsilon", values="hi", aggfunc="mean").reindex(index=layers)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for eps in eps_values:
        if eps not in piv.columns:
            continue
        y = piv[eps].to_numpy(dtype=float)
        ax.plot(piv.index.values, y, marker="o", linewidth=1.6, label=f"ε={eps:g}")

        if bootstrap:
            lo = piv_lo[eps].to_numpy(dtype=float)
            hi = piv_hi[eps].to_numpy(dtype=float)
            m = np.isfinite(lo) & np.isfinite(hi)
            if m.any():
                ax.fill_between(piv.index.values, lo, hi, alpha=0.18)

    ax.set_xlabel("layer ℓ")
    ax.set_ylabel("flip success rate (FSR)")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(layers)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)

    title = "FSR vs layer (curves by ε)"
    if bootstrap:
        title += f" | bootstrap {(1.0-alpha)*100:.0f}% CI (n_boot={n_boot})"
    ax.set_title(title, fontsize=11)

    ax.legend(title="epsilon", fontsize=8, title_fontsize=9, ncols=3)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="JSONL produced by run_flip_sweep.py")
    ap.add_argument("--outdir", default="figs", help="Directory to write figures")

    ap.add_argument("--metric", default="mean_rel_gap",
                    choices=["mean_rel_gap", "median_rel_gap", "flip_rate"],
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

    ap.add_argument("--modes", nargs="*", default=None,
                    help='Attack modes to plot (default: both). Example: --modes sign_flip no_sign_flip')

    # Bootstrap options for FSR
    ap.add_argument("--bootstrap", action="store_true", help="Add bootstrap CI bands to the FSR plot.")
    ap.add_argument("--n_boot", type=int, default=500, help="Number of bootstrap resamples for FSR CIs.")
    ap.add_argument("--alpha", type=float, default=0.05, help="Alpha for CI bands (e.g., 0.05 => 95% CI).")
    ap.add_argument("--boot_seed", type=int, default=0, help="RNG seed for bootstrap.")

    args = ap.parse_args()

    df = load_jsonl(args.infile)

    # attack_mode: if missing, infer from sign_flip boolean if present
    if "attack_mode" not in df.columns:
        if "sign_flip" in df.columns:
            df["attack_mode"] = df["sign_flip"].apply(lambda x: "sign_flip" if bool(x) else "no_sign_flip")
        else:
            df["attack_mode"] = "sign_flip"

    df = add_relative_changes(df)
    df = add_depth_bucket(df)

    if args.only_baseline_correct:
        df = df[df["baseline_correct"] == True].copy()

    outdir = Path(args.outdir)

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

    # For built-in metrics here, higher is better.
    higher_is_better = True

    modes = args.modes
    if modes is None or len(modes) == 0:
        modes = ["sign_flip", "no_sign_flip"]
    modes = list(dict.fromkeys(modes))  # unique, preserve order

    for mode in modes:
        dfm = df[df["attack_mode"] == mode].copy()
        if dfm.empty:
            print(f"[warn] No rows found for attack_mode={mode}; skipping.")
            continue

        mode_dir = outdir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        # ---- FSR plot (per mode) ----
        try:
            plot_fsr_by_layer(
                df_mode=dfm,
                outpath=mode_dir / "fsr_by_layer.png",
                eps_values=eps_for_plot,
                bootstrap=bool(args.bootstrap),
                n_boot=int(args.n_boot),
                alpha=float(args.alpha),
                seed=int(args.boot_seed),
            )
            print(f"Wrote {mode_dir / 'fsr_by_layer.png'}")
        except Exception as e:
            print(f"[warn] Failed to plot FSR for mode={mode}: {e}")

        # ---- Select + plot 2D panels per depth bucket (per mode) ----
        selected = {}
        for bucket in ["early", "mid", "late"]:
            dfb = dfm[dfm["depth_bucket"] == bucket].copy()
            if dfb.empty:
                raise RuntimeError(f"No data for bucket={bucket} mode={mode} after filtering.")

            best_layer, ranking = choose_best_layer_rank_agg(
                dfb,
                metric=args.metric,
                eps_values=eps_for_selection,
                higher_is_better=higher_is_better,
            )
            selected[bucket] = (best_layer, ranking)

            show = ranking[["layer", "mean_rank"]].sort_values("mean_rank").head(5)
            print(f"\n[{mode} | {bucket}] selected layer={best_layer} by rank-agg across eps")
            print(show.to_string(index=False))

        for bucket in ["early", "mid", "late"]:
            layer, _ranking = selected[bucket]
            df_layer = dfm[(dfm["depth_bucket"] == bucket) & (dfm["layer"].astype(int) == int(layer))].copy()

            outpath = mode_dir / f"{bucket}_layer{layer}_rel2d.png"
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
