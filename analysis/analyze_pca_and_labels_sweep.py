#!/usr/bin/env python3
"""
analyze_pca_and_labels_sweep.py

Sweep-aware PCA + autointerp-label analysis for your RM auto-interp pipeline.

It auto-discovers:
  - PCA files:    results/pca/pca_seed{seed}_layer{layer}_eps{eps}_k{k}_{cc}_{tag}_{mode}_n{n}.pt
  - Label files:  results/labels/component_labels_seed{seed}_layer{layer}_eps{eps}_K{K}_{cc}_{tag}_{mode}_n{n}.jsonl

And produces:
  1) Console-friendly summaries per run:
     - EVR per component
     - label coverage / PARSE_ERROR rate
     - dataset-level component dominance stats:
         mean |score|, winner%, mean share (normalized |score|)
     - component correlations with scalars:
         corr(score_k, reward_delta), corr(score_k, clean_margin), etc.
  2) Human inspection dumps:
     - top/bottom examples per component (with prompt/completion and reward delta)
  3) Cross-seed stability:
     - component matching across seeds using cosine similarity of component vectors (Hungarian if scipy installed, else greedy)
     - matched score stability (Pearson correlation of per-example scores for matched components)
     - matched label similarity (token Jaccard of labels)

Usage:
  python analyze_pca_and_labels_sweep.py

Common:
  python analyze_pca_and_labels_sweep.py --pca_glob "results/pca/*.pt" --labels_glob "results/labels/*.jsonl" \
     --outdir results/analysis_pca --topM 12 --botM 6 --dump_examples --max_runs 20

Notes:
- This script expects each PCA .pt to include:
    "components" (K,D), "scores" (N,K), "records" (list of dict records)
  which matches your pca_directions.py output.
- It does NOT require passing seed0/seed1 paths manually; it groups runs by config.
"""

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# Optional (Hungarian matching). If missing, we do greedy.
try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:
    linear_sum_assignment = None


# -------------------------
# Filename parsing
# -------------------------

PCA_PAT = re.compile(
    r"pca_seed(?P<seed>\d+)_layer(?P<layer>\d+)_eps(?P<eps>[\d\.]+)_k(?P<k>\d+)_(?P<cc>\w+)_(?P<tag>\w+)_(?P<mode>\w+)_n(?P<n>\d+)\.pt$"
)

LBL_PAT = re.compile(
    r"component_labels_seed(?P<seed>\d+)_layer(?P<layer>\d+)_eps(?P<eps>[\d\.]+)_K(?P<K>\d+)_(?P<cc>\w+)_(?P<tag>\w+)_(?P<mode>\w+)_n(?P<n>\d+)\.jsonl$"
)

def parse_pca_name(path: str) -> Optional[dict]:
    m = PCA_PAT.search(os.path.basename(path))
    if not m:
        return None
    d = m.groupdict()
    return {
        "seed": int(d["seed"]),
        "layer": int(d["layer"]),
        "epsilon": float(d["eps"]),
        "k": int(d["k"]),
        "cc": d["cc"],
        "tag": d["tag"],
        "mode": d["mode"],
        "n": int(d["n"]),
        "path": path,
    }

def parse_label_name(path: str) -> Optional[dict]:
    m = LBL_PAT.search(os.path.basename(path))
    if not m:
        return None
    d = m.groupdict()
    return {
        "seed": int(d["seed"]),
        "layer": int(d["layer"]),
        "epsilon": float(d["eps"]),
        "K": int(d["K"]),
        "cc": d["cc"],
        "tag": d["tag"],
        "mode": d["mode"],
        "n": int(d["n"]),
        "path": path,
    }


# -------------------------
# Label + record preview
# -------------------------

def load_labels(jsonl_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Returns {component_index: row_dict} where row_dict includes label/explanation/etc.
    """
    out: Dict[int, Dict[str, Any]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            k = int(row.get("component", row.get("k", row.get("idx", -1))))
            if k < 0:
                continue
            out[k] = row
    return out

def preview_record(rec: Any, n_prompt: int = 160, n_comp: int = 160) -> str:
    if isinstance(rec, str):
        return rec.replace("\n", " ")[: (n_prompt + n_comp)]
    if not isinstance(rec, dict):
        return str(rec)[: (n_prompt + n_comp)]

    p = (rec.get("prompt", "") or "").replace("\n", " ").strip()[:n_prompt]
    c = (rec.get("completion", "") or "").replace("\n", " ").strip()[:n_comp]
    ct = rec.get("completion_type", "unknown")

    r0 = rec.get("reward_unperturbed", None)
    r1 = rec.get("reward_perturbed", None)
    if isinstance(r0, (int, float)) and isinstance(r1, (int, float)):
        dr = float(r1) - float(r0)
        reward = f"clean={float(r0):+.3f} adv={float(r1):+.3f} Δ={dr:+.3f}"
    else:
        reward = "reward=?"

    cm = rec.get("clean_margin", None)
    am = rec.get("adv_margin", None)
    mline = ""
    if isinstance(cm, (int,float)) and isinstance(am, (int,float)):
        mline = f" | margin clean={float(cm):+.3f} adv={float(am):+.3f} Δ={float(am)-float(cm):+.3f}"

    return f"[{ct}] {reward}{mline}\nPROMPT: {p}\nCOMP: {c}"


# -------------------------
# PCA helpers
# -------------------------

def normalize_abs(scores: torch.Tensor) -> torch.Tensor:
    mag = scores.abs()
    denom = mag.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return mag / denom

def token_jaccard(a: str, b: str) -> float:
    def tok(s: str) -> set:
        return set(re.findall(r"[A-Za-z0-9_]+", (s or "").lower()))
    sa, sb = tok(a), tok(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def l2_norm_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A2 = l2_norm_rows(A)
    B2 = l2_norm_rows(B)
    return A2 @ B2.T

def match_components(sim: np.ndarray) -> List[Tuple[int,int,float]]:
    """
    sim: (K,K) cosine similarity. Match indices across seeds.
    Returns list (i0, i1, sim).
    """
    K0, K1 = sim.shape
    if linear_sum_assignment is not None and K0 == K1:
        r, c = linear_sum_assignment(-sim)
        pairs = [(int(i), int(j), float(sim[i,j])) for i,j in zip(r,c)]
        pairs.sort(key=lambda t: t[0])
        return pairs

    # greedy fallback
    used_r, used_c = set(), set()
    flat = [(i,j,float(sim[i,j])) for i in range(K0) for j in range(K1)]
    flat.sort(key=lambda t: t[2], reverse=True)
    out = []
    for i,j,s in flat:
        if i in used_r or j in used_c:
            continue
        used_r.add(i); used_c.add(j)
        out.append((i,j,s))
        if len(out) >= min(K0,K1):
            break
    out.sort(key=lambda t: t[0])
    return out

def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom < 1e-12:
        return float("nan")
    return float((x @ y) / denom)


# -------------------------
# Scalar extraction for correlations
# -------------------------

SCALAR_FIELDS_DEFAULT = [
    "reward_delta",          # reward_perturbed - reward_unperturbed
    "clean_margin",
    "adv_margin",
    "margin_delta",          # adv_margin - clean_margin
    "chosen_len_tokens",
    "rejected_len_tokens",
    "delta_global_l2",
    "delta_max_token_l2",
    "delta_mean_token_l2",
    "flipped",
]

def extract_scalar_vector(records: List[dict], field: str) -> np.ndarray:
    out = []
    for r in records:
        if field == "reward_delta":
            a = r.get("reward_unperturbed", None)
            b = r.get("reward_perturbed", None)
            if isinstance(a,(int,float)) and isinstance(b,(int,float)):
                out.append(float(b) - float(a))
            else:
                out.append(np.nan)
        elif field == "margin_delta":
            a = r.get("clean_margin", None)
            b = r.get("adv_margin", None)
            if isinstance(a,(int,float)) and isinstance(b,(int,float)):
                out.append(float(b) - float(a))
            else:
                out.append(np.nan)
        else:
            v = r.get(field, np.nan)
            if isinstance(v, bool):
                out.append(1.0 if v else 0.0)
            elif isinstance(v, (int,float)):
                out.append(float(v))
            else:
                out.append(np.nan)
    return np.asarray(out, dtype=float)


# -------------------------
# Outputs
# -------------------------

def dump_top_bottom_examples(
    outpath: Path,
    scores: torch.Tensor,
    records: List[dict],
    labels: Dict[int, Dict[str, Any]],
    comp: int,
    topM: int,
    botM: int,
):
    s = scores[:, comp].float()
    top_vals, top_idx = torch.topk(s, k=min(topM, s.numel()))
    bot_vals, bot_idx = torch.topk(-s, k=min(botM, s.numel()))

    lbl = labels.get(comp, {}).get("label", f"component_{comp}")
    expl = labels.get(comp, {}).get("explanation", "")

    with open(outpath, "w", encoding="utf-8") as f:
        f.write(f"Component {comp}: {lbl}\n")
        if expl:
            f.write(f"Explanation: {expl}\n")
        f.write("="*90 + "\n\n")

        f.write("POSITIVE END (top scores)\n")
        f.write("-"*90 + "\n")
        for rank, (v, i) in enumerate(zip(top_vals.tolist(), top_idx.tolist()), 1):
            f.write(f"\n[{rank:02d}] score={v:+.4f} id={records[i].get('pair_id', i)}\n")
            f.write(preview_record(records[i]) + "\n")

        f.write("\n\nNEGATIVE END (bottom scores)\n")
        f.write("-"*90 + "\n")
        for rank, (v, i) in enumerate(zip((-bot_vals).tolist(), bot_idx.tolist()), 1):
            f.write(f"\n[{rank:02d}] score={v:+.4f} id={records[i].get('pair_id', i)}\n")
            f.write(preview_record(records[i]) + "\n")


# -------------------------
# Run-level analysis
# -------------------------

@dataclass(frozen=True)
class RunKey:
    layer: int
    epsilon: float
    cc: str
    tag: str
    mode: str
    k: int
    n: int

@dataclass
class RunData:
    seed: int
    key: RunKey
    pca_path: str
    labels_path: Optional[str]
    components: torch.Tensor     # (K,D)
    scores: torch.Tensor         # (N,K)
    evr: Optional[torch.Tensor]  # (K,)
    records: List[dict]
    labels: Dict[int, Dict[str, Any]]


def load_run(pca_path: str, labels_index: Dict[Tuple[int,int,float,str,str,str,int,int], str]) -> RunData:
    meta = parse_pca_name(pca_path)
    if meta is None:
        raise ValueError(f"Unrecognized PCA filename: {pca_path}")

    obj = torch.load(pca_path, map_location="cpu")

    components = obj["components"]
    scores = obj["scores"]
    records = obj.get("records", None)
    if records is None:
        raise KeyError(f"{pca_path} missing 'records' (needed for interpretation)")

    evr = obj.get("explained_var_ratio", None)

    seed = int(meta["seed"])
    key = RunKey(
        layer=int(meta["layer"]),
        epsilon=float(meta["epsilon"]),
        cc=str(meta["cc"]),
        tag=str(meta["tag"]),
        mode=str(meta["mode"]),
        k=int(meta["k"]),
        n=int(meta["n"]),
    )

    # Find label file that matches this run (seed must match too)
    lbl_path = labels_index.get((seed, key.layer, key.epsilon, key.cc, key.tag, key.mode, key.k, len(records)), None)
    labels = load_labels(lbl_path) if lbl_path else {}

    return RunData(
        seed=seed,
        key=key,
        pca_path=pca_path,
        labels_path=lbl_path,
        components=components,
        scores=scores,
        evr=evr,
        records=records,
        labels=labels,
    )


def run_summary_table(run: RunData) -> List[dict]:
    scores = run.scores
    K = scores.shape[1]

    pct = normalize_abs(scores)           # (N,K)
    mean_share = pct.mean(dim=0)          # (K,)
    mean_abs = scores.abs().mean(dim=0)   # (K,)

    winners = torch.argmax(scores.abs(), dim=1)
    winner_counts = torch.bincount(winners, minlength=K).float()
    winner_freq = winner_counts / max(1, scores.shape[0])

    evr = run.evr.float().tolist() if isinstance(run.evr, torch.Tensor) else [None]*K

    rows = []
    for k in range(K):
        lbl_row = run.labels.get(k, {})
        label = lbl_row.get("label", f"component_{k}")
        rows.append({
            "seed": run.seed,
            "layer": run.key.layer,
            "epsilon": run.key.epsilon,
            "cc": run.key.cc,
            "tag": run.key.tag,
            "mode": run.key.mode,
            "K": run.key.k,
            "N": int(scores.shape[0]),
            "component": k,
            "label": label,
            "label_is_parse_error": (label == "PARSE_ERROR"),
            "mean_abs_score": float(mean_abs[k]),
            "winner_freq": float(winner_freq[k]),
            "mean_share": float(mean_share[k]),
            "evr": (float(evr[k]) if evr[k] is not None else np.nan),
        })
    rows.sort(key=lambda r: (-r["mean_abs_score"], -r["winner_freq"]))
    return rows


def run_component_correlations(run: RunData, scalar_fields: List[str]) -> List[dict]:
    scores = run.scores.float()
    K = scores.shape[1]
    recs = run.records

    out = []
    for field in scalar_fields:
        y = extract_scalar_vector(recs, field)
        for k in range(K):
            x = scores[:, k].numpy()
            out.append({
                "seed": run.seed,
                "layer": run.key.layer,
                "epsilon": run.key.epsilon,
                "cc": run.key.cc,
                "tag": run.key.tag,
                "mode": run.key.mode,
                "component": k,
                "label": run.labels.get(k, {}).get("label", f"component_{k}"),
                "field": field,
                "pearson_corr": pearson(x, y),
            })
    return out


# -------------------------
# Cross-seed stability (within each config key)
# -------------------------

def compare_two_runs(a: RunData, b: RunData) -> Tuple[List[dict], List[dict]]:
    """
    Returns:
      - per-component matched stability rows
      - run-level stability summary rows
    """
    Va = a.components.float().numpy()
    Vb = b.components.float().numpy()

    sim = cosine_sim_matrix(Va, Vb)
    pairs = match_components(sim)

    # score stability for matched comps
    Sa = a.scores.float().numpy()
    Sb = b.scores.float().numpy()
    N = min(Sa.shape[0], Sb.shape[0])

    comp_rows = []
    for i0, i1, cs in pairs:
        corr = pearson(Sa[:N, i0], Sb[:N, i1])
        la = a.labels.get(i0, {}).get("label", f"component_{i0}")
        lb = b.labels.get(i1, {}).get("label", f"component_{i1}")
        lsim = token_jaccard(la, lb)

        comp_rows.append({
            "layer": a.key.layer,
            "epsilon": a.key.epsilon,
            "cc": a.key.cc,
            "tag": a.key.tag,
            "mode": a.key.mode,
            "K": a.key.k,
            "N": N,

            "seed_a": a.seed,
            "seed_b": b.seed,
            "comp_a": i0,
            "comp_b": i1,
            "cosine_components": float(cs),
            "score_corr": float(corr),
            "label_a": la,
            "label_b": lb,
            "label_token_jaccard": float(lsim),
        })

    cos_vals = np.array([r["cosine_components"] for r in comp_rows], dtype=float)
    sc_vals = np.array([r["score_corr"] for r in comp_rows], dtype=float)

    run_rows = [{
        "layer": a.key.layer,
        "epsilon": a.key.epsilon,
        "cc": a.key.cc,
        "tag": a.key.tag,
        "mode": a.key.mode,
        "K": a.key.k,
        "N": N,
        "seed_a": a.seed,
        "seed_b": b.seed,
        "mean_cosine": float(np.nanmean(cos_vals)) if cos_vals.size else np.nan,
        "min_cosine": float(np.nanmin(cos_vals)) if cos_vals.size else np.nan,
        "mean_score_corr": float(np.nanmean(sc_vals)) if sc_vals.size else np.nan,
        "min_score_corr": float(np.nanmin(sc_vals)) if sc_vals.size else np.nan,
        "mean_label_jaccard": float(np.nanmean([r["label_token_jaccard"] for r in comp_rows])) if comp_rows else np.nan,
    }]

    return comp_rows, run_rows


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca_glob", default="results/outputs/pca/*.pt")
    ap.add_argument("--labels_glob", default="results/outputs/labels/*.jsonl")
    ap.add_argument("--outdir", default="results/analysis/pca")

    ap.add_argument("--max_runs", type=int, default=0, help="If >0, only analyze first N PCA files (debug).")
    ap.add_argument("--dump_examples", action="store_true")
    ap.add_argument("--topM", type=int, default=12)
    ap.add_argument("--botM", type=int, default=6)
    ap.add_argument("--scalar_fields", nargs="*", default=SCALAR_FIELDS_DEFAULT)

    ap.add_argument("--print_examples", type=int, default=3, help="Print first N examples with top contributing comps (per run).")
    ap.add_argument("--topk_mix", type=int, default=5, help="Top-k components to show per example using normalized |score| share.")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "runs").mkdir(exist_ok=True)
    (outdir / "examples").mkdir(exist_ok=True)

    # index labels by (seed, layer, eps, cc, tag, mode, K, n_records)
    labels_index: Dict[Tuple[int,int,float,str,str,str,int,int], str] = {}
    for lp in sorted(glob.glob(args.labels_glob)):
        m = parse_label_name(lp)
        if not m:
            continue
        labels_index[(m["seed"], m["layer"], m["epsilon"], m["cc"], m["tag"], m["mode"], m["K"], m["n"])] = lp

    pca_files = sorted(glob.glob(args.pca_glob))
    if args.max_runs and args.max_runs > 0:
        pca_files = pca_files[: args.max_runs]

    if not pca_files:
        raise SystemExit(f"No PCA files matched: {args.pca_glob}")

    # load runs and group by config key (excluding seed)
    runs_by_key: Dict[RunKey, List[RunData]] = {}
    all_runs: List[RunData] = []
    for fp in pca_files:
        meta = parse_pca_name(fp)
        if not meta:
            continue
        run = load_run(fp, labels_index)
        runs_by_key.setdefault(run.key, []).append(run)
        all_runs.append(run)

    # 1) per-run tables
    summary_rows: List[dict] = []
    corr_rows: List[dict] = []

    for run in all_runs:
        rows = run_summary_table(run)
        summary_rows.extend(rows)

        corr_rows.extend(run_component_correlations(run, args.scalar_fields))

        # Per-run human-readable file
        run_dir = outdir / "runs" / f"seed{run.seed}_layer{run.key.layer}_eps{run.key.epsilon}_{run.key.cc}_{run.key.tag}_{run.key.mode}_k{run.key.k}_n{run.key.n}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Write component summary as JSONL (easy to grep)
        with open(run_dir / "component_summary.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        # Print a few “mixture share” examples like your draft (optional, quick sanity)
        if args.print_examples and args.print_examples > 0:
            scores = run.scores
            pct = normalize_abs(scores)
            labels_simple = {k: v.get("label", f"component_{k}") for k, v in run.labels.items()}

            n_print = min(args.print_examples, scores.shape[0])
            with open(run_dir / "example_mixture_preview.txt", "w", encoding="utf-8") as f:
                f.write(f"PCA: {run.pca_path}\n")
                f.write(f"Labels: {run.labels_path}\n\n")
                for i in range(n_print):
                    f.write("="*90 + "\n")
                    f.write(f"Example {i}\n")
                    f.write(preview_record(run.records[i]) + "\n\n")

                    vals, idx = torch.topk(pct[i], k=min(args.topk_mix, pct.shape[1]))
                    f.write("Top component contributions (by normalized |score| share):\n")
                    for v, j in zip(vals.tolist(), idx.tolist()):
                        lab = labels_simple.get(j, f"component_{j}")
                        raw = float(scores[i, j].item())
                        sign = "+" if raw >= 0 else "-"
                        f.write(f"  {lab:35s}  {sign}  {v*100:6.2f}%   raw_score={raw:+.4f}\n")
                    f.write("\n")

        # Dump top/bottom examples per component (this is the “what does it mean?” artifact)
        if args.dump_examples:
            exdir = outdir / "examples" / f"seed{run.seed}_layer{run.key.layer}_eps{run.key.epsilon}_{run.key.cc}_{run.key.tag}_{run.key.mode}_k{run.key.k}_n{run.key.n}"
            exdir.mkdir(parents=True, exist_ok=True)
            for k in range(run.scores.shape[1]):
                dump_top_bottom_examples(
                    outpath=exdir / f"component_{k:02d}.txt",
                    scores=run.scores,
                    records=run.records,
                    labels=run.labels,
                    comp=k,
                    topM=args.topM,
                    botM=args.botM,
                )

    # write sweep-wide summary tables
    import pandas as pd  # keep local, but pandas is common in your env

    summary_df = pd.DataFrame(summary_rows)
    corr_df = pd.DataFrame(corr_rows)

    summary_df.to_csv(outdir / "component_summary_all_runs.csv", index=False)
    corr_df.to_csv(outdir / "component_correlations_all_runs.csv", index=False)

    # 2) label coverage summary per run
    cov_rows = []
    for run in all_runs:
        K = run.scores.shape[1]
        labels_present = sum([1 for k in range(K) if k in run.labels])
        parse_err = sum([1 for k in range(K) if run.labels.get(k, {}).get("label", "") == "PARSE_ERROR"])
        cov_rows.append({
            "seed": run.seed, "layer": run.key.layer, "epsilon": run.key.epsilon, "cc": run.key.cc,
            "tag": run.key.tag, "mode": run.key.mode, "K": K, "N": int(run.scores.shape[0]),
            "labels_present": int(labels_present),
            "labels_missing": int(K - labels_present),
            "parse_error": int(parse_err),
            "labels_path": run.labels_path or "",
            "pca_path": run.pca_path,
        })
    cov_df = pd.DataFrame(cov_rows)
    cov_df.to_csv(outdir / "label_coverage_by_run.csv", index=False)

    # 3) cross-seed stability per config key
    comp_stab_rows: List[dict] = []
    run_stab_rows: List[dict] = []

    for key, runs in runs_by_key.items():
        if len(runs) < 2:
            continue
        runs = sorted(runs, key=lambda r: r.seed)
        # compare all pairs of seeds for this config
        for i in range(len(runs)):
            for j in range(i+1, len(runs)):
                a, b = runs[i], runs[j]
                cr, rr = compare_two_runs(a, b)
                comp_stab_rows.extend(cr)
                run_stab_rows.extend(rr)

    if comp_stab_rows:
        pd.DataFrame(comp_stab_rows).to_csv(outdir / "seed_stability_component_matches.csv", index=False)
        pd.DataFrame(run_stab_rows).to_csv(outdir / "seed_stability_run_summary.csv", index=False)

    # 4) brief console report (so it feels like your draft, not “useless CLI”)
    print("\n" + "="*90)
    print("WROTE:")
    print(f"  {outdir / 'component_summary_all_runs.csv'}")
    print(f"  {outdir / 'component_correlations_all_runs.csv'}")
    print(f"  {outdir / 'label_coverage_by_run.csv'}")
    if comp_stab_rows:
        print(f"  {outdir / 'seed_stability_component_matches.csv'}")
        print(f"  {outdir / 'seed_stability_run_summary.csv'}")
    if args.dump_examples:
        print(f"  {outdir / 'examples'}/... (top/bottom examples per component)")
    print("="*90)

    # Show “top components” for a handful of runs (meeting-friendly)
    # pick 5 runs and show their top-3 components by mean_abs_score
    show_runs = all_runs[: min(5, len(all_runs))]
    for run in show_runs:
        sub = summary_df[
            (summary_df["seed"] == run.seed) &
            (summary_df["layer"] == run.key.layer) &
            (summary_df["epsilon"] == run.key.epsilon) &
            (summary_df["cc"] == run.key.cc) &
            (summary_df["tag"] == run.key.tag) &
            (summary_df["mode"] == run.key.mode)
        ].copy()
        sub = sub.sort_values("mean_abs_score", ascending=False).head(3)
        print(f"\nRun: seed={run.seed} layer={run.key.layer} eps={run.key.epsilon} cc={run.key.cc} tag={run.key.tag} mode={run.key.mode}")
        for _, r in sub.iterrows():
            print(f"  comp {int(r['component']):2d}: {r['label'][:45]:45s} | mean|score|={r['mean_abs_score']:.4f} | winner%={100*r['winner_freq']:.1f}% | evr={r['evr']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
