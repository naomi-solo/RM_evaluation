#summarize_component_contributions.py
import os
import json
import torch
from typing import Dict, List, Tuple, Any


def load_labels(jsonl_path: str) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            labels[int(row["component"])] = row.get("label", f"component_{row['component']}")
    return labels


def preview_record(rec: Any, n_prompt: int = 140, n_comp: int = 140) -> str:
    """
    Handles new record format (dict) and legacy string.
    """
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

    return f"[{ct}] {reward}\nPROMPT: {p}\nCOMP: {c}"


def normalize_abs(scores: torch.Tensor) -> torch.Tensor:
    """
    Per-example normalized ABS(score) so rows sum to 1.
    Useful as a "mixture share" view, not explained variance.
    """
    mag = scores.abs()
    denom = mag.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return mag / denom


def topk_contributors_for_example(
    pct_row: torch.Tensor,
    raw_scores_row: torch.Tensor,
    labels: Dict[int, str],
    k: int = 5,
) -> List[Tuple[str, float, int, float]]:
    """
    Returns list of (label, percent_share, component_index, raw_score)
    sorted strongest->weakest by share.
    """
    vals, idx = torch.topk(pct_row, k=min(k, pct_row.numel()))
    out = []
    for v, j in zip(vals.tolist(), idx.tolist()):
        out.append((
            labels.get(j, f"component_{j}"),
            float(v),
            int(j),
            float(raw_scores_row[j].item()),
        ))
    return out


def main():
    # ---- Inputs ----
    pca_path = os.getenv("PCA_PATH", "results/outputs/pca/pca_layer14_eps8.0_k10_ambig_flip_chosen_n200.pt")
    labels_path = os.getenv("LABELS_PATH", "results/outputs/labels/component_labels_layer14_eps8.0_K10_ambig_flip_chosen.jsonl")

    n_print = int(os.getenv("N_PRINT", "5"))
    topk = int(os.getenv("TOPK", "5"))

    obj = torch.load(pca_path, map_location="cpu")
    scores: torch.Tensor = obj["scores"]  # (N, K)

    records = obj.get("records", None)
    if records is None:
        records = obj.get("texts", None)
    if records is None:
        raise KeyError("Expected PCA file to include 'records' (or legacy 'texts').")

    labels = load_labels(labels_path)

    # ---- Per-example normalized shares (for per-example display + mean share stats) ----
    pct = normalize_abs(scores)  # (N, K)

    print("=" * 80)
    print("PCA file:", pca_path)
    print("Labels:", labels_path)
    print("N examples:", scores.shape[0], "K comps:", scores.shape[1])
    print("NOTE: 'Mean share' is mean(normalized |score|) per example; not explained variance.")
    print("=" * 80)

    # ---- Print a few examples ----
    for i in range(min(n_print, pct.shape[0])):
        print("\n" + "=" * 80)
        print(f"Example {i}")
        print(preview_record(records[i]))

        contribs = topk_contributors_for_example(pct[i], scores[i], labels, k=topk)
        print("\nTop component contributions (by |score| share):")
        for lab, share, j, raw in contribs:
            sign = "+" if raw >= 0 else "-"
            print(f"{lab:35s}  {sign}  {share*100:6.2f}%   raw_score={raw:+.4f}")

    # ---- Dataset-level summaries ----
    mean_share = pct.mean(dim=0)            # (K,)
    mean_abs = scores.abs().mean(dim=0)     # (K,)

    # Winner frequency based on absolute score directly:
    # "Which component is biggest (in |score|) for each example?"
    winners = torch.argmax(scores.abs(), dim=1)  # (N,)
    winner_counts = torch.bincount(winners, minlength=scores.shape[1]).float()
    winner_freq = winner_counts / max(1, scores.shape[0])

    print("\n" + "=" * 80)
    print("Dataset-level summaries (sorted by Mean |score|)")
    print("=" * 80)

    rows = []
    for k in range(scores.shape[1]):
        lab = labels.get(k, f"component_{k}")
        rows.append((
            lab,
            float(mean_abs[k]),     # primary
            float(winner_freq[k]),  # secondary
            float(mean_share[k]),   # tertiary
        ))

    # Sort by mean absolute score (your request)
    rows.sort(key=lambda x: -x[1])

    print(f"{'Component':35s} | {'Mean |score|':>12s} | {'Winner %':>9s} | {'Mean share':>10s}")
    print("-" * 80)
    for lab, mabs, win, share in rows:
        print(f"{lab:35s} | {mabs:12.4f} | {win*100:8.2f}% | {share*100:9.2f}%")

    # Optional: show explained variance ratio too
    evr = obj.get("explained_var_ratio", None)
    if isinstance(evr, torch.Tensor):
        evr = evr.float()
        print("\nExplained variance ratio (PCA property, not per-example):")
        for k, v in enumerate(evr.tolist()):
            lab = labels.get(k, f"component_{k}")
            print(f"{lab:35s} → {v*100:6.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
