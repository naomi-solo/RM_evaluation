#!/usr/bin/env python3
"""
analyze_robustness_sweep.py

Robustness analysis over RM auto-interp sweep outputs produced by extract_directions.py.

Inputs:
  results/directions/direction_records_seed{seed}_layer{layer}_eps{eps}_{cc}_{tag}_n{n}.pt

Each file stores records for chosen+rejected per pair_id, and includes (typical):
  reward_unperturbed, reward_perturbed, clean_margin, adv_margin, flipped, epsilon, layer, sign_flip, context_condition,
  plus norms and bbq_* metadata if available.

What this script does:
  1) Build a pair-level dataframe (one row per (file, pair_id)).
  2) Compute robustness metrics: flip_success, delta_margin, reward deltas, etc.
  3) Summarize sensitivity across factors (eps/layer/cc/sign_flip) with bootstrap CIs (over examples).
  4) Check across-seed stability (do patterns replicate?).
  5) Slice “failure modes” by clean_margin, lengths, and BBQ metadata when present.
  6) Save tables + plots.

Outputs (default: results/analysis_robustness/):
  - pair_level.parquet (or .csv)
  - cell_summary.csv
  - agg_summary.csv
  - epsilon_sanity.csv
  - figs/
      flip_rate_by_layer_cc=..._signflip=...
      flip_rate_by_eps_cc=..._signflip=...
      flip_rate_heatmap_cc=..._signflip=...
      delta_margin_by_eps_cc=..._signflip=...
      stability_across_seeds.png
      failure_modes.png
      bbq_slices.png

Usage:
  python analyze_robustness_sweep.py
  python analyze_robustness_sweep.py --in_glob "results/directions/*.pt" --outdir results/analysis_robustness
"""

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ----------------- filename parsing -----------------

DIR_PAT = re.compile(
    r"direction_records_seed(?P<seed>\d+)_layer(?P<layer>\d+)_eps(?P<eps>[\d\.]+)_(?P<cc>\w+)_(?P<tag>\w+)_n(?P<n>\d+)\.pt"
)

def parse_filename(fp: str) -> Optional[dict]:
    m = DIR_PAT.match(os.path.basename(fp))
    if not m:
        return None
    d = m.groupdict()
    return {
        "seed": int(d["seed"]),
        "layer": int(d["layer"]),
        "epsilon": float(d["eps"]),
        "cc": d["cc"],
        "tag": d["tag"],
        "n_pairs_requested": int(d["n"]),
        "file": fp,
    }

# ----------------- loading -----------------

def load_pair_rows_from_pt(fp: str) -> List[dict]:
    obj = torch.load(fp, map_location="cpu")
    fn = parse_filename(fp) or {"file": fp}

    # file-level meta (preferred if present)
    layer = int(obj.get("layer", fn.get("layer", -1)))
    epsilon = float(obj.get("epsilon", fn.get("epsilon", float("nan"))))
    sign_flip = bool(obj.get("sign_flip", True))
    cc = obj.get("context_condition", fn.get("cc", None))
    model_name = obj.get("model_name", None)

    # group chosen/rejected by pair_id
    by_pair: Dict[int, dict] = {}
    for r in obj.get("records", []):
        if "pair_id" not in r or "completion_type" not in r:
            continue
        pid = int(r["pair_id"])
        by_pair.setdefault(pid, {})[r["completion_type"]] = r

    rows: List[dict] = []
    for pid, d in by_pair.items():
        if "chosen" not in d or "rejected" not in d:
            continue
        c = d["chosen"]
        r = d["rejected"]

        # scores
        sc0 = float(c.get("reward_unperturbed", np.nan))
        sc1 = float(c.get("reward_perturbed", np.nan))
        sr0 = float(r.get("reward_unperturbed", np.nan))
        sr1 = float(r.get("reward_perturbed", np.nan))

        if not np.isfinite(sc0) or not np.isfinite(sc1) or not np.isfinite(sr0) or not np.isfinite(sr1):
            continue

        # margins (prefer stored)
        clean_margin = float(c.get("clean_margin", sc0 - sr0))
        adv_margin = float(c.get("adv_margin", sc1 - sr1))

        baseline_correct = (clean_margin > 0)
        # flipped == "label says flipped" if present, otherwise derived
        flipped = bool(c.get("flipped", baseline_correct and adv_margin <= 0))
        flip_success = bool(baseline_correct and adv_margin <= 0)

        row = {
            **fn,
            "pair_id": int(pid),
            "layer": int(layer),
            "epsilon": float(epsilon),
            "context_condition": cc,
            "sign_flip": bool(sign_flip),
            "model_name": model_name,

            "score_chosen_clean": sc0,
            "score_chosen_adv": sc1,
            "score_rejected_clean": sr0,
            "score_rejected_adv": sr1,

            "clean_margin": clean_margin,
            "adv_margin": adv_margin,
            "baseline_correct": bool(baseline_correct),
            "flipped": bool(flipped),
            "flip_success": bool(flip_success),

            "delta_margin": float(adv_margin - clean_margin),
            "chosen_delta_reward": float(sc1 - sc0),
            "rejected_delta_reward": float(sr1 - sr0),

            "delta_global_l2": float(c.get("delta_global_l2", np.nan)),
            "delta_max_token_l2": float(c.get("delta_max_token_l2", np.nan)),
            "delta_mean_token_l2": float(c.get("delta_mean_token_l2", np.nan)),
            "chosen_len_tokens": int(c.get("chosen_len_tokens", -1)),
            "rejected_len_tokens": int(c.get("rejected_len_tokens", -1)),
        }

        # include any bbq_* metadata if present in chosen record
        for k, v in c.items():
            if isinstance(k, str) and k.startswith("bbq_") and k not in row:
                row[k] = v

        rows.append(row)

    return rows

# ----------------- bootstrap -----------------

def _bootstrap(rng: np.random.Generator, x: np.ndarray, n_boot: int, stat_fn) -> np.ndarray:
    n = int(x.size)
    idx = np.arange(n)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        boots[b] = float(stat_fn(x[samp]))
    return boots

def bootstrap_ci_of_rate(x: np.ndarray, n_boot: int = 500, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float, int]:
    x = np.asarray(x).astype(bool)
    n = int(x.size)
    if n == 0:
        return 0.0, np.nan, np.nan, 0
    point = float(x.mean())
    if n < 2 or n_boot <= 0:
        return point, np.nan, np.nan, n
    rng = np.random.default_rng(int(seed))
    boots = _bootstrap(rng, x.astype(float), n_boot=n_boot, stat_fn=lambda z: np.mean(z))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return point, lo, hi, n

def bootstrap_ci_of_mean(x: np.ndarray, n_boot: int = 500, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float, int]:
    x = np.asarray(x).astype(float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return float("nan"), np.nan, np.nan, 0
    point = float(np.mean(x))
    if n < 2 or n_boot <= 0:
        return point, np.nan, np.nan, n
    rng = np.random.default_rng(int(seed))
    boots = _bootstrap(rng, x, n_boot=n_boot, stat_fn=lambda z: np.mean(z))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return point, lo, hi, n

# ----------------- summaries -----------------

def make_cell_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["context_condition", "sign_flip", "seed", "layer", "epsilon"]
    rows = []
    for key, sub in pair_df.groupby(group_cols, dropna=False):
        cc, sf, seed, layer, eps = key
        bc = sub["baseline_correct"].astype(bool).to_numpy()
        fs = sub["flip_success"].astype(bool).to_numpy()
        eligible = fs[bc]  # only baseline-correct count
        flip_rate = float(eligible.mean()) if eligible.size else 0.0

        rows.append({
            "context_condition": cc,
            "sign_flip": bool(sf),
            "seed": int(seed),
            "layer": int(layer),
            "epsilon": float(eps),

            "n_pairs": int(len(sub)),
            "n_base": int(bc.sum()),
            "flip_rate": flip_rate,

            "mean_clean_margin": float(sub["clean_margin"].mean()),
            "mean_adv_margin": float(sub["adv_margin"].mean()),
            "mean_delta_margin": float(sub["delta_margin"].mean()),
            "median_delta_margin": float(sub["delta_margin"].median()),

            "mean_chosen_delta_reward": float(sub["chosen_delta_reward"].mean()),
            "mean_rejected_delta_reward": float(sub["rejected_delta_reward"].mean()),

            "mean_max_token_norm": float(sub["delta_max_token_l2"].mean()),
        })
    return pd.DataFrame(rows)

def make_agg_summary(cell_df: pd.DataFrame) -> pd.DataFrame:
    agg_cols = ["context_condition", "sign_flip", "layer", "epsilon"]
    agg = (cell_df.groupby(agg_cols, dropna=False)
           .agg(
               seeds=("seed", "nunique"),
               flip_rate_mean=("flip_rate", "mean"),
               flip_rate_std=("flip_rate", "std"),
               mean_delta_margin=("mean_delta_margin", "mean"),
               mean_delta_margin_std=("mean_delta_margin", "std"),
               mean_clean_margin=("mean_clean_margin", "mean"),
               mean_adv_margin=("mean_adv_margin", "mean"),
               n_base_mean=("n_base", "mean"),
               n_pairs_mean=("n_pairs", "mean"),
           )
           .reset_index()
           .sort_values(["context_condition", "sign_flip", "layer", "epsilon"]))
    return agg

# ----------------- plotting helpers -----------------

def savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _prep_ax(fig_w=8.6, fig_h=5.0):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    return fig, ax

def _format_eps_ticks(ax, eps_values: List[float]):
    # keep epsilon ticks readable
    ax.set_xticks(eps_values)
    ax.set_xticklabels([f"{e:g}" for e in eps_values], rotation=0)

def plot_flip_rate_by_layer(pair_df: pd.DataFrame, outdir: Path, n_boot: int, alpha: float, seed: int):
    df = pair_df.copy()
    df["layer"] = df["layer"].astype(int)
    df["epsilon"] = df["epsilon"].astype(float)

    ccs = sorted([x for x in df["context_condition"].dropna().unique().tolist()]) or ["ambig", "disambig"]
    sfs = [False, True]

    for cc in ccs:
        for sf in sfs:
            sub = df[(df["context_condition"] == cc) & (df["sign_flip"] == sf)].copy()
            if sub.empty:
                continue

            layers = sorted(sub["layer"].unique().tolist())
            eps_values = sorted(sub["epsilon"].unique().tolist())

            fig, ax = _prep_ax(9.2, 5.2)

            for e in eps_values:
                xs, ys, lo, hi = [], [], [], []
                for layer in layers:
                    cell = sub[(sub["epsilon"] == e) & (sub["layer"] == layer)]
                    bc = cell["baseline_correct"].astype(bool).to_numpy()
                    fs = cell["flip_success"].astype(bool).to_numpy()
                    eligible = fs[bc]
                    point, l, h, _n = bootstrap_ci_of_rate(eligible, n_boot=n_boot, alpha=alpha, seed=seed)
                    xs.append(layer); ys.append(point); lo.append(l); hi.append(h)

                ax.plot(xs, ys, marker="o", linewidth=1.6, label=f"eps={e:g}")
                m = np.isfinite(lo) & np.isfinite(hi)
                if np.any(m):
                    ax.fill_between(np.array(xs)[m], np.array(lo)[m], np.array(hi)[m], alpha=0.18)

            ax.set_xlabel("layer")
            ax.set_ylabel("flip success rate (baseline-correct only)")
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks(layers)
            ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
            ax.set_title(f"Flip success vs layer | cc={cc} | sign_flip={sf} | bootstrap {(1-alpha)*100:.0f}% CI")

            # legend outside to avoid covering lines
            ax.legend(fontsize=8, ncols=min(4, len(eps_values)),
                      title="epsilon", title_fontsize=9,
                      loc="upper center", bbox_to_anchor=(0.5, 1.18))
            savefig(fig, outdir / f"flip_rate_by_layer_cc={cc}_signflip={sf}.png")

def plot_flip_rate_by_eps(pair_df: pd.DataFrame, outdir: Path, n_boot: int, alpha: float, seed: int, max_layers_in_legend: int = 12):
    df = pair_df.copy()
    df["layer"] = df["layer"].astype(int)
    df["epsilon"] = df["epsilon"].astype(float)

    ccs = sorted([x for x in df["context_condition"].dropna().unique().tolist()]) or ["ambig", "disambig"]
    sfs = [False, True]

    for cc in ccs:
        for sf in sfs:
            sub = df[(df["context_condition"] == cc) & (df["sign_flip"] == sf)].copy()
            if sub.empty:
                continue

            layers = sorted(sub["layer"].unique().tolist())
            eps_values = sorted(sub["epsilon"].unique().tolist())

            fig, ax = _prep_ax(9.2, 5.2)

            # If many layers, only label a subset to keep legend sane.
            layers_for_label = layers
            if len(layers) > max_layers_in_legend:
                # keep evenly spaced layers for labeling; still plot all
                idx = np.linspace(0, len(layers) - 1, max_layers_in_legend).round().astype(int)
                layers_for_label = [layers[i] for i in sorted(set(idx.tolist()))]

            for layer in layers:
                xs, ys, lo, hi = [], [], [], []
                for e in eps_values:
                    cell = sub[(sub["epsilon"] == e) & (sub["layer"] == layer)]
                    bc = cell["baseline_correct"].astype(bool).to_numpy()
                    fs = cell["flip_success"].astype(bool).to_numpy()
                    eligible = fs[bc]
                    point, l, h, _n = bootstrap_ci_of_rate(eligible, n_boot=n_boot, alpha=alpha, seed=seed)
                    xs.append(e); ys.append(point); lo.append(l); hi.append(h)

                label = f"layer={layer}" if layer in layers_for_label else None
                ax.plot(xs, ys, marker="o", linewidth=1.3, label=label)
                m = np.isfinite(lo) & np.isfinite(hi)
                if np.any(m):
                    ax.fill_between(np.array(xs)[m], np.array(lo)[m], np.array(hi)[m], alpha=0.10)

            ax.set_xlabel("epsilon")
            ax.set_ylabel("flip success rate (baseline-correct only)")
            ax.set_ylim(0.0, 1.0)
            _format_eps_ticks(ax, eps_values)
            ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
            ax.set_title(f"Flip success vs epsilon | cc={cc} | sign_flip={sf} | bootstrap {(1-alpha)*100:.0f}% CI")

            ax.legend(fontsize=7, ncols=4, title="layer (subset labeled)", title_fontsize=8,
                      loc="upper center", bbox_to_anchor=(0.5, 1.18))
            savefig(fig, outdir / f"flip_rate_by_eps_cc={cc}_signflip={sf}.png")

def plot_heatmap_layer_eps(agg_df: pd.DataFrame, outdir: Path):
    df = agg_df.copy()
    ccs = sorted([x for x in df["context_condition"].dropna().unique().tolist()]) or ["ambig", "disambig"]
    sfs = [False, True]

    for cc in ccs:
        for sf in sfs:
            sub = df[(df["context_condition"] == cc) & (df["sign_flip"] == sf)].copy()
            if sub.empty:
                continue

            layers = sorted(sub["layer"].unique().tolist())
            eps_values = sorted(sub["epsilon"].unique().tolist())
            piv = sub.pivot_table(index="layer", columns="epsilon", values="flip_rate_mean", aggfunc="mean").reindex(index=layers, columns=eps_values)
            Z = piv.to_numpy(dtype=float)

            # dynamic sizing
            fig_w = max(7.0, 1.0 * len(eps_values) + 2.5)
            fig_h = max(4.0, 0.33 * len(layers) + 2.0)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

            im = ax.imshow(Z, aspect="auto", interpolation="nearest")
            ax.set_xticks(np.arange(len(eps_values)))
            ax.set_xticklabels([f"{e:g}" for e in eps_values], rotation=45, ha="right")
            ax.set_yticks(np.arange(len(layers)))
            ax.set_yticklabels([str(l) for l in layers])

            ax.set_xlabel("epsilon")
            ax.set_ylabel("layer")
            ax.set_title(f"Mean flip rate across seeds | cc={cc} | sign_flip={sf}")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("flip_rate_mean")

            savefig(fig, outdir / f"flip_rate_heatmap_cc={cc}_signflip={sf}.png")

def plot_delta_margin(pair_df: pd.DataFrame, outdir: Path, n_boot: int, alpha: float, seed: int, max_layers_in_legend: int = 12):
    df = pair_df.copy()
    df["layer"] = df["layer"].astype(int)
    df["epsilon"] = df["epsilon"].astype(float)

    ccs = sorted([x for x in df["context_condition"].dropna().unique().tolist()]) or ["ambig", "disambig"]
    sfs = [False, True]

    for cc in ccs:
        for sf in sfs:
            sub = df[(df["context_condition"] == cc) & (df["sign_flip"] == sf)].copy()
            if sub.empty:
                continue

            layers = sorted(sub["layer"].unique().tolist())
            eps_values = sorted(sub["epsilon"].unique().tolist())

            fig, ax = _prep_ax(9.2, 5.2)

            layers_for_label = layers
            if len(layers) > max_layers_in_legend:
                idx = np.linspace(0, len(layers) - 1, max_layers_in_legend).round().astype(int)
                layers_for_label = [layers[i] for i in sorted(set(idx.tolist()))]

            for layer in layers:
                xs, ys, lo, hi = [], [], [], []
                for e in eps_values:
                    cell = sub[(sub["epsilon"] == e) & (sub["layer"] == layer)]
                    x = cell["delta_margin"].to_numpy(dtype=float)
                    point, l, h, _n = bootstrap_ci_of_mean(x, n_boot=n_boot, alpha=alpha, seed=seed)
                    xs.append(e); ys.append(point); lo.append(l); hi.append(h)

                label = f"layer={layer}" if layer in layers_for_label else None
                ax.plot(xs, ys, marker="o", linewidth=1.3, label=label)
                m = np.isfinite(lo) & np.isfinite(hi)
                if np.any(m):
                    ax.fill_between(np.array(xs)[m], np.array(lo)[m], np.array(hi)[m], alpha=0.10)

            ax.axhline(0.0, linestyle="--", linewidth=1.0)
            ax.set_xlabel("epsilon")
            ax.set_ylabel("mean(adv_margin - clean_margin)")
            _format_eps_ticks(ax, eps_values)
            ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
            ax.set_title(f"Margin degradation vs epsilon | cc={cc} | sign_flip={sf} | bootstrap {(1-alpha)*100:.0f}% CI")

            ax.legend(fontsize=7, ncols=4, title="layer (subset labeled)", title_fontsize=8,
                      loc="upper center", bbox_to_anchor=(0.5, 1.18))
            savefig(fig, outdir / f"delta_margin_by_eps_cc={cc}_signflip={sf}.png")

def plot_seed_stability(cell_df: pd.DataFrame, outdir: Path):
    """
    Across-seed stability of layer vulnerability:
    For each (cc, sign_flip, eps), compute Spearman correlation of flip_rate across layers between seeds.
    """
    try:
        from scipy.stats import spearmanr
    except Exception as e:
        print("[warn] scipy not available; skipping seed stability plot:", e)
        return

    rows = []
    for (cc, sf, eps), sub in cell_df.groupby(["context_condition", "sign_flip", "epsilon"]):
        piv = sub.pivot_table(index="seed", columns="layer", values="flip_rate", aggfunc="mean")
        seeds = piv.index.tolist()
        if len(seeds) < 2:
            continue
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                a = piv.loc[seeds[i]].to_numpy(dtype=float)
                b = piv.loc[seeds[j]].to_numpy(dtype=float)
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() < 3:
                    rho = np.nan
                else:
                    rho = float(spearmanr(a[m], b[m]).correlation)
                rows.append({
                    "context_condition": cc, "sign_flip": sf, "epsilon": float(eps),
                    "seed_a": seeds[i], "seed_b": seeds[j], "spearman_rho": rho
                })
    if not rows:
        return

    stab = pd.DataFrame(rows)
    eps_vals = sorted(stab["epsilon"].unique().tolist())

    fig, ax = _prep_ax(9.0, 5.0)
    for (cc, sf), sub in stab.groupby(["context_condition", "sign_flip"]):
        means = []
        for e in eps_vals:
            v = sub[sub["epsilon"] == e]["spearman_rho"].to_numpy(dtype=float)
            means.append(float(np.nanmean(v)) if np.isfinite(v).any() else np.nan)
        ax.plot(eps_vals, means, marker="o", linewidth=1.6, label=f"{cc}, sign_flip={sf}")

    ax.set_xlabel("epsilon")
    ax.set_ylabel("mean Spearman rho across seed pairs\n(layer flip-rate ranking agreement)")
    ax.set_ylim(-1.0, 1.0)
    _format_eps_ticks(ax, eps_vals)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("Across-seed stability of which layers are most vulnerable")
    ax.legend(fontsize=8)
    savefig(fig, outdir / "stability_across_seeds.png")

def plot_failure_modes(pair_df: pd.DataFrame, outdir: Path):
    """
    Failure mode slices (baseline-correct only), with counts:
      - flip_success rate vs clean_margin quantile bucket
      - flip_success vs chosen_len bucket
    """
    df = pair_df.copy()
    df = df[df["baseline_correct"] == True].copy()
    if df.empty:
        return

    # Robust buckets: qcut for margin; fixed bins for length
    df["clean_margin_bucket"] = pd.qcut(df["clean_margin"], q=5, duplicates="drop")
    df["chosen_len_tokens"] = df["chosen_len_tokens"].clip(lower=0)

    # If your length distribution is narrow, fixed bins can be sparse; still okay, but we show counts.
    bins = [0, 64, 128, 256, 512, 10_000]
    df["chosen_len_bucket"] = pd.cut(df["chosen_len_tokens"], bins=bins, right=True, include_lowest=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), constrained_layout=True)

    # left: by margin bucket
    g1 = df.groupby("clean_margin_bucket")["flip_success"].mean()
    c1 = df.groupby("clean_margin_bucket")["flip_success"].size()

    x1 = np.arange(len(g1))
    axes[0].plot(x1, g1.to_numpy(dtype=float), marker="o", linewidth=1.8)
    axes[0].set_xticks(x1)
    axes[0].set_xticklabels([f"Q{i+1}" for i in range(len(g1))])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Flip success vs clean margin bucket\n(baseline-correct only)")
    axes[0].set_ylabel("flip_success rate")
    axes[0].grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    for i, (rate, n) in enumerate(zip(g1.to_numpy(), c1.to_numpy())):
        axes[0].text(i, min(0.98, float(rate) + 0.03), f"n={int(n)}", ha="center", fontsize=9)

    # right: by chosen length
    g2 = df.groupby("chosen_len_bucket")["flip_success"].mean()
    c2 = df.groupby("chosen_len_bucket")["flip_success"].size()

    x2 = np.arange(len(g2))
    axes[1].plot(x2, g2.to_numpy(dtype=float), marker="o", linewidth=1.8)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(["<=64", "64-128", "128-256", "256-512", "512+"], rotation=25, ha="right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Flip success vs chosen length bucket\n(baseline-correct only)")
    axes[1].grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    for i, (rate, n) in enumerate(zip(g2.to_numpy(), c2.to_numpy())):
        axes[1].text(i, min(0.98, float(rate) + 0.03), f"n={int(n)}", ha="center", fontsize=9)

    savefig(fig, outdir / "failure_modes.png")

def plot_bbq_slices(pair_df: pd.DataFrame, outdir: Path, n_boot: int, alpha: float, seed: int):
    """
    If BBQ metadata exists, plot flip_success rate by BBQ category (aggregated),
    baseline-correct only, with counts and optional bootstrap CI.
    Tries to find a reasonable "category" column among bbq_* fields.
    """
    df = pair_df.copy()
    df = df[df["baseline_correct"] == True].copy()
    if df.empty:
        return

    bbq_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("bbq_")]
    if not bbq_cols:
        return

    # Prefer a "category-like" column if present.
    preferred = ["bbq_category", "bbq_bias_category", "bbq_topic", "bbq_attribute", "bbq_dimension"]
    cat_col = None
    for c in preferred:
        if c in df.columns:
            cat_col = c
            break
    if cat_col is None:
        # Fall back: pick the bbq_* column with lowest unique count >1 (often category)
        candidates = []
        for c in bbq_cols:
            nun = df[c].nunique(dropna=True)
            if nun > 1:
                candidates.append((nun, c))
        if not candidates:
            return
        cat_col = sorted(candidates)[0][1]

    sub = df.copy()
    sub[cat_col] = sub[cat_col].astype(str)

    # Aggregate rates (and bootstrap over examples within each category)
    rows = []
    for cat, g in sub.groupby(cat_col, dropna=False):
        x = g["flip_success"].astype(bool).to_numpy()
        point, lo, hi, n = bootstrap_ci_of_rate(x, n_boot=n_boot, alpha=alpha, seed=seed)
        rows.append({"category": str(cat), "rate": point, "lo": lo, "hi": hi, "n": n})
    out = pd.DataFrame(rows).sort_values(["rate"], ascending=False)

    # Plot
    fig, ax = _prep_ax(fig_w=max(9.0, 0.7 * len(out) + 3.0), fig_h=4.8)

    xs = np.arange(len(out))
    ax.bar(xs, out["rate"].to_numpy(dtype=float))
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("flip success rate (baseline-correct)")
    ax.set_title(f"Flip success rate by BBQ category (aggregated)\n(col={cat_col})")

    ax.set_xticks(xs)
    ax.set_xticklabels(out["category"].tolist(), rotation=35, ha="right")

    # Error bars if CI available
    lo = out["lo"].to_numpy(dtype=float)
    hi = out["hi"].to_numpy(dtype=float)
    m = np.isfinite(lo) & np.isfinite(hi)
    if m.any():
        y = out["rate"].to_numpy(dtype=float)
        yerr = np.vstack([y - lo, hi - y])
        ax.errorbar(xs[m], y[m], yerr=yerr[:, m], fmt="none", capsize=3)

    # annotate counts
    for i, n in enumerate(out["n"].to_numpy(dtype=int)):
        ax.text(i, min(0.98, float(out["rate"].iloc[i]) + 0.03), f"n={int(n)}", ha="center", fontsize=8)

    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.6)
    savefig(fig, outdir / "bbq_slices.png")

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_glob", default="results/outputs/directions/direction_records_seed*_layer*_eps*_*_n*.pt")
    ap.add_argument("--outdir", default="results/analysis/robustness")
    ap.add_argument("--save_csv", action="store_true", help="Also write pair_level.csv (parquet preferred).")

    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--boot_seed", type=int, default=0)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    figdir = outdir / "figs"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    fps = sorted(glob.glob(args.in_glob))
    if not fps:
        raise SystemExit(f"No files matched: {args.in_glob}")

    all_rows: List[dict] = []
    for fp in fps:
        all_rows.extend(load_pair_rows_from_pt(fp))

    pair_df = pd.DataFrame(all_rows)
    if pair_df.empty:
        raise SystemExit("Loaded zero rows (check input glob / file formats).")

    # enforce dtypes
    for col, typ in [
        ("seed", int), ("layer", int), ("epsilon", float),
        ("sign_flip", bool), ("baseline_correct", bool), ("flip_success", bool)
    ]:
        if col in pair_df.columns:
            pair_df[col] = pair_df[col].astype(typ)

    # save pair-level
    pair_parquet = outdir / "pair_level.parquet"
    wrote_pair = False
    try:
        pair_df.to_parquet(pair_parquet, index=False)
        print("Wrote", pair_parquet)
        wrote_pair = True
    except Exception as e:
        print("[warn] parquet write failed:", e)

    if (not wrote_pair) and args.save_csv:
        pair_csv = outdir / "pair_level.csv"
        pair_df.to_csv(pair_csv, index=False)
        print("Wrote", pair_csv)

    # cell + agg summaries
    cell_df = make_cell_summary(pair_df)
    agg_df = make_agg_summary(cell_df)

    cell_csv = outdir / "cell_summary.csv"
    agg_csv = outdir / "agg_summary.csv"
    cell_df.to_csv(cell_csv, index=False)
    agg_df.to_csv(agg_csv, index=False)
    print("Wrote", cell_csv)
    print("Wrote", agg_csv)

    # epsilon sanity
    tol = 1e-2
    if "delta_max_token_l2" in pair_df.columns:
        pair_df["over_eps"] = (pair_df["delta_max_token_l2"] - pair_df["epsilon"]) > tol
        sani = (pair_df.groupby(["context_condition", "sign_flip", "layer", "epsilon"], dropna=False)
                .agg(frac_over_eps=("over_eps", "mean"),
                     mean_max_token_norm=("delta_max_token_l2", "mean"))
                .reset_index())
        sani_csv = outdir / "epsilon_sanity.csv"
        sani.to_csv(sani_csv, index=False)
        print("Wrote", sani_csv)

    # plots
    plot_flip_rate_by_layer(pair_df, figdir, n_boot=args.n_boot, alpha=args.alpha, seed=args.boot_seed)
    plot_flip_rate_by_eps(pair_df, figdir, n_boot=args.n_boot, alpha=args.alpha, seed=args.boot_seed)
    plot_heatmap_layer_eps(agg_df, figdir)
    plot_delta_margin(pair_df, figdir, n_boot=args.n_boot, alpha=args.alpha, seed=args.boot_seed)
    plot_failure_modes(pair_df, figdir)
    plot_bbq_slices(pair_df, figdir, n_boot=args.n_boot, alpha=args.alpha, seed=args.boot_seed)
    plot_seed_stability(cell_df, figdir)

    print("Wrote figures under", figdir)
    print("\nDone.")

if __name__ == "__main__":
    main()
