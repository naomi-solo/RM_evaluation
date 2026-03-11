#summarize_pca_run.py
#!/usr/bin/env python3

import os
import json
import torch

REL_FLOOR = 1e-12


def load_labels(jsonl_path):
    labels = {}
    if not os.path.exists(jsonl_path):
        return labels

    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            labels[int(row["component"])] = row.get("label", f"component_{row['component']}")
    return labels


def normalize_abs(scores):
    mag = scores.abs()
    denom = mag.sum(dim=1, keepdim=True).clamp_min(REL_FLOOR)
    return mag / denom


def main():

    pca_path = os.getenv("PCA_PATH")
    labels_path = os.getenv("LABELS_PATH")

    if not pca_path:
        raise ValueError("Set PCA_PATH")

    obj = torch.load(pca_path, map_location="cpu")
    scores = obj["scores"]  # (N, K)

    labels = load_labels(labels_path) if labels_path else {}

    pct = normalize_abs(scores)

    mean_abs = scores.abs().mean(dim=0)
    mean_share = pct.mean(dim=0)

    winners = torch.argmax(scores.abs(), dim=1)
    winner_counts = torch.bincount(winners, minlength=scores.shape[1]).float()
    winner_freq = winner_counts / max(1, scores.shape[0])

    evr = obj.get("explained_var_ratio", None)

    rows = []
    for k in range(scores.shape[1]):
        rows.append({
            "component": k,
            "label": labels.get(k, f"component_{k}"),
            "mean_abs": float(mean_abs[k]),
            "winner_freq": float(winner_freq[k]),
            "mean_share": float(mean_share[k]),
            "evr": float(evr[k]) if evr is not None else None,
        })

    rows.sort(key=lambda r: -r["mean_abs"])

    print("\n" + "=" * 90)
    print("PCA:", pca_path)
    if labels_path:
        print("Labels:", labels_path)
    print("=" * 90)

    header = f"{'Comp':>4s} | {'Label':35s} | {'Mean |score|':>12s} | {'Winner %':>9s} | {'Mean share %':>13s}"
    if evr is not None:
        header += " | EVR %"
    print(header)
    print("-" * 90)

    for r in rows:
        line = (
            f"{r['component']:4d} | "
            f"{r['label'][:35]:35s} | "
            f"{r['mean_abs']:12.4f} | "
            f"{r['winner_freq']*100:8.2f}% | "
            f"{r['mean_share']*100:12.2f}%"
        )
        if r["evr"] is not None:
            line += f" | {r['evr']*100:6.2f}%"
        print(line)

    print()


if __name__ == "__main__":
    main()
