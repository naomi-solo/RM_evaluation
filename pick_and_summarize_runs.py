#!/usr/bin/env python3
"""
pick_and_summarize_pca_with_labels.py

Meeting-friendly "AUTO-INTERP" summary:
  For a small subset of PCA runs, report:
    - dominant PCA components (by mean |score| and winner%)
    - their LABEL + explanation (from results/labels/*.jsonl)
    - EVR (explained variance ratio) when available
    - label status diagnostics (missing / MANUAL / PARSE_ERROR)

Auto-picks a SMALL subset of runs (default):
  - max epsilon available
  - prefer cc=disambig, tag=flip, mode=chosen
  - layers closest to [1, 13, 21]
  - optionally also include mid-layer ambig

Inputs:
  PCA:    results/pca/pca_seed{seed}_layer{layer}_eps{eps}_k{k}_{cc}_{tag}_{mode}_n{n}.pt
  Labels: results/labels/component_labels_seed{seed}_layer{layer}_eps{eps}_K{K}_{cc}_{tag}_{mode}_n{n}.jsonl

Usage:
  python pick_and_summarize_pca_with_labels.py
  python pick_and_summarize_pca_with_labels.py --seed 0 --include_mid_ambig --topk 4
  python pick_and_summarize_pca_with_labels.py --prefer_cc ambig --prefer_tag noflip --prefer_mode rejected

Env overrides:
  PCA_GLOB="results/pca/*.pt"
  LABEL_GLOB="results/labels/*.jsonl"
"""

import argparse
import glob
import os
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


PCA_PAT = re.compile(
    r"pca_seed(?P<seed>\d+)_layer(?P<layer>\d+)_eps(?P<eps>[\d\.]+)_k(?P<k>\d+)_(?P<cc>\w+)_(?P<tag>\w+)_(?P<mode>\w+)_n(?P<n>\d+)\.pt$"
)

LBL_PAT = re.compile(
    r"component_labels_seed(?P<seed>\d+)_layer(?P<layer>\d+)_eps(?P<eps>[\d\.]+)_K(?P<K>\d+)_(?P<cc>\w+)_(?P<tag>\w+)_(?P<mode>\w+)_n(?P<n>\d+)\.jsonl$"
)


@dataclass(frozen=True)
class Meta:
    seed: int
    layer: int
    eps: float
    k: int
    cc: str
    tag: str
    mode: str
    n: int
    path: str


def parse_meta(path: str) -> Optional[Meta]:
    m = PCA_PAT.search(os.path.basename(path))
    if not m:
        return None
    d = m.groupdict()
    return Meta(
        seed=int(d["seed"]),
        layer=int(d["layer"]),
        eps=float(d["eps"]),
        k=int(d["k"]),
        cc=str(d["cc"]),
        tag=str(d["tag"]),
        mode=str(d["mode"]),
        n=int(d["n"]),
        path=path,
    )


def index_labels(label_glob: str) -> Dict[Tuple[int,int,float,str,str,str,int,int], str]:
    """
    Index by (seed, layer, eps, cc, tag, mode, K, n_records)
    Note: label file's `n` is len(records) in PCA (chosen/rejected separately),
          and your label_components.py writes n=len(records).
    """
    idx: Dict[Tuple[int,int,float,str,str,str,int,int], str] = {}
    for lp in sorted(glob.glob(label_glob)):
        m = LBL_PAT.search(os.path.basename(lp))
        if not m:
            continue
        d = m.groupdict()
        key = (
            int(d["seed"]),
            int(d["layer"]),
            float(d["eps"]),
            str(d["cc"]),
            str(d["tag"]),
            str(d["mode"]),
            int(d["K"]),
            int(d["n"]),
        )
        idx[key] = lp
    return idx


def load_label_rows(jsonl_path: str) -> Dict[int, Dict[str, str]]:
    """
    Return {component -> {"label":..., "explanation":..., "keywords":..., "negatives":...}}
    Only keep fields relevant for a meeting summary.
    """
    out: Dict[int, Dict[str, str]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            k = int(row.get("component"))
            out[k] = {
                "label": str(row.get("label", "")),
                "explanation": str(row.get("explanation", "")),
            }
    return out


def normalize_abs(scores: torch.Tensor) -> torch.Tensor:
    mag = scores.abs()
    denom = mag.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return mag / denom


def dominance_stats(scores: torch.Tensor):
    """
    Returns:
      mean_abs (K,), winner_freq (K,), mean_share (K,)
    """
    scores = scores.float()
    N, K = scores.shape

    mean_abs = scores.abs().mean(dim=0)  # (K,)
    winners = torch.argmax(scores.abs(), dim=1)
    winner_counts = torch.bincount(winners, minlength=K).float()
    winner_freq = winner_counts / max(1, N)

    pct = normalize_abs(scores)
    mean_share = pct.mean(dim=0)

    return (
        mean_abs.cpu().numpy(),
        winner_freq.cpu().numpy(),
        mean_share.cpu().numpy(),
    )


def score_run(m: Meta, prefer_cc: str, prefer_tag: str, prefer_mode: str, target_eps: float, target_layer: int):
    cc_pen = 0 if m.cc == prefer_cc else 1
    tag_pen = 0 if m.tag == prefer_tag else 1
    mode_pen = 0 if m.mode == prefer_mode else 1
    eps_dist = abs(m.eps - target_eps)
    layer_dist = abs(m.layer - target_layer)
    # primary: match prefs; secondary: eps closeness; tertiary: layer closeness
    return (cc_pen + tag_pen + mode_pen, eps_dist, layer_dist)


def pick_one(metas: List[Meta], target_layer: int, prefer_cc: str, prefer_tag: str, prefer_mode: str, target_eps: float, seed: Optional[int]):
    cands = metas
    if seed is not None:
        cands = [x for x in cands if x.seed == seed]
    if not cands:
        return None
    return sorted(cands, key=lambda m: score_run(m, prefer_cc, prefer_tag, prefer_mode, target_eps, target_layer))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca_glob", default=os.getenv("PCA_GLOB", "results/pca/*.pt"))
    ap.add_argument("--label_glob", default=os.getenv("LABEL_GLOB", "results/labels/*.jsonl"))

    ap.add_argument("--seed", type=int, default=None, help="If set, only summarize this seed (e.g., 0).")
    ap.add_argument("--topk", type=int, default=4, help="How many dominant components to show per run.")
    ap.add_argument("--layers", nargs="*", type=int, default=[1, 13, 21], help="Target layers for early/mid/late.")
    ap.add_argument("--include_mid_ambig", action="store_true")

    ap.add_argument("--prefer_cc", default="disambig")
    ap.add_argument("--prefer_tag", default="flip")
    ap.add_argument("--prefer_mode", default="chosen")

    args = ap.parse_args()

    # discover PCA runs
    metas: List[Meta] = []
    for fp in sorted(glob.glob(args.pca_glob)):
        m = parse_meta(fp)
        if m:
            metas.append(m)
    if not metas:
        raise SystemExit(f"No PCA files found for glob: {args.pca_glob}")

    # choose max eps as default “strong attack” setting for meeting
    eps_all = sorted({m.eps for m in metas})
    target_eps = float(max(eps_all))

    # index label files
    lbl_index = index_labels(args.label_glob)

    # pick small subset
    selected: List[Meta] = []
    for L in args.layers:
        pick = pick_one(metas, L, args.prefer_cc, args.prefer_tag, args.prefer_mode, target_eps, args.seed)
        if pick:
            selected.append(pick)

    if args.include_mid_ambig:
        mid_layer = int(args.layers[len(args.layers)//2]) if args.layers else 13
        pickA = pick_one(metas, mid_layer, "ambig", args.prefer_tag, args.prefer_mode, target_eps, args.seed)
        if pickA and all(pickA.path != s.path for s in selected):
            selected.append(pickA)

    # de-dup
    selected = list({m.path: m for m in selected}.values())
    selected = sorted(selected, key=lambda x: (x.cc, x.tag, x.mode, x.layer, x.seed, x.eps))

    print("=" * 100)
    print("AUTO-INTERP PCA SUMMARY (dominant components + labels)")
    print(f"PCA glob:    {args.pca_glob}")
    print(f"Label glob:  {args.label_glob}")
    print(f"Seed filter: {args.seed}")
    print(f"Target eps:  {target_eps:g} (max found)")
    print(f"Prefs: cc={args.prefer_cc}, tag={args.prefer_tag}, mode={args.prefer_mode}")
    print("Selected runs:")
    for m in selected:
        print(f"  seed={m.seed} layer={m.layer} eps={m.eps:g} cc={m.cc} tag={m.tag} mode={m.mode} k={m.k} n={m.n} | {os.path.basename(m.path)}")
    print("=" * 100)

    for meta in selected:
        obj = torch.load(meta.path, map_location="cpu")
        scores: torch.Tensor = obj["scores"]
        N, K = scores.shape

        evr = obj.get("explained_var_ratio", None)
        evr_np = evr.float().cpu().numpy() if isinstance(evr, torch.Tensor) else None

        # Find matching labels file.
        # IMPORTANT: label_components.py uses n_default = len(records), not n_pairs.
        records = obj.get("records", None)
        n_records = len(records) if isinstance(records, list) else int(meta.n)

        lbl_key = (meta.seed, meta.layer, meta.eps, meta.cc, meta.tag, meta.mode, int(K), int(n_records))
        lbl_path = lbl_index.get(lbl_key)

        labels = load_label_rows(lbl_path) if lbl_path else {}

        mean_abs, winner_freq, mean_share = dominance_stats(scores)
        order = np.argsort(-mean_abs)[: min(args.topk, K)]

        print("\n" + "-" * 100)
        print(f"RUN: seed={meta.seed} layer={meta.layer} eps={meta.eps:g} cc={meta.cc} tag={meta.tag} mode={meta.mode} | N={N} K={K}")
        if not lbl_path:
            print("⚠️  LABEL FILE NOT FOUND for this run.")
            print("    This usually means: label_components.py didn't run OR filename mismatch.")
            print(f"    Expected key: seed={meta.seed}, layer={meta.layer}, eps={meta.eps}, cc={meta.cc}, tag={meta.tag}, mode={meta.mode}, K={K}, n={n_records}")
        else:
            print(f"Labels: {os.path.basename(lbl_path)}")

        print("\nTop components (by Mean |score|) with labels:")
        for j in order:
            lab = labels.get(int(j), {}).get("label", "(missing)")
            expl = labels.get(int(j), {}).get("explanation", "")

            # label status flags
            status = ""
            if lab.strip().lower() in ("manual",):
                status = "  [MANUAL]"
            if "PARSE_ERROR" in lab:
                status = "  [PARSE_ERROR]"
            if lab == "(missing)":
                status = "  [NO_LABEL]"

            evr_str = f"{100*evr_np[j]:5.2f}%" if evr_np is not None and np.isfinite(evr_np[j]) else " n/a "
            print(
                f"  comp {int(j):02d} | "
                f"Mean|score|={mean_abs[j]:.4f} | "
                f"Winner%={100*winner_freq[j]:6.2f}% | "
                f"Mean share={100*mean_share[j]:6.2f}% | "
                f"EVR={evr_str} | "
                f"Label: {lab}{status}"
            )
            if expl.strip():
                # keep explanation short for console
                short = expl.replace("\n", " ").strip()
                if len(short) > 180:
                    short = short[:180] + "…"
                print(f"           ↳ {short}")

        # quick “how concentrated is PCA?” stat for narrative
        top1 = float(np.max(winner_freq))
        top3 = float(np.sort(winner_freq)[::-1][: min(3, K)].sum())
        print(f"\nConcentration: top1 winner%={100*top1:.1f}%, top3 winner% sum={100*top3:.1f}%")

        # label coverage sanity
        labeled = sum(1 for k in range(K) if k in labels)
        manual = sum(1 for k in range(K) if labels.get(k, {}).get("label", "").strip().lower() == "manual")
        parse_err = sum(1 for k in range(K) if "PARSE_ERROR" in labels.get(k, {}).get("label", ""))
        print(f"Label coverage: {labeled}/{K} labeled | MANUAL={manual} | PARSE_ERROR={parse_err}")

    print("\nDone.")


if __name__ == "__main__":
    main()
