# pca_from_sweep.py

import os
import json
import torch
from pca_directions import pca_torch

# =============================
# CONFIG
# =============================

RUN_DIR = "sweep_results/run_eps01_deltas_20260227_203203"
DATASETS = ["bbq", "gsm_mc", "math_mc", "mmlu", "sgxs"]

REGIME = "shared_flip"
EPS = 0.1
K = 20

OUT_DIR = "results/pca_from_sweep"
os.makedirs(OUT_DIR, exist_ok=True)


# =============================
# Helpers
# =============================

def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def convert_records(records, dataset_name):
    out = []

    for i, r in enumerate(records):

        if "delta_h_chosen_eos" not in r:
            continue

        direction = torch.tensor(
            r["delta_h_chosen_eos"],
            dtype=torch.float32
        )

        rec = {
            "pair_id": f"{dataset_name}_{i}",
            "prompt": r.get("prompt", ""),
            "completion": r.get("chosen", ""),
            "completion_type": "chosen",
            "reward_unperturbed": r.get("clean_chosen"),
            "reward_perturbed": r.get("adv_chosen"),
            "perturbation_direction": direction,
            "dataset": dataset_name,
        }

        out.append(rec)

    return out


def save_pca(dataset_name, X, records):
    comps, scores, evr, mean = pca_torch(X, k=min(K, X.shape[0]))

    out_path = os.path.join(
        OUT_DIR,
        f"pca_{dataset_name}_eps{EPS}_{REGIME}_k{K}.pt"
    )

    torch.save({
        "components": comps.half(),
        "scores": scores.float(),
        "explained_var_ratio": evr.float(),
        "mean": mean.float(),
        "records": records,
        "ids": [r["pair_id"] for r in records],
        "dataset": dataset_name,
        "epsilon": EPS,
        "regime": REGIME,
    }, out_path)

    print(f"\nSaved PCA → {out_path}")
    print("Top explained variance:", evr[:5].tolist())


# =============================
# MAIN
# =============================

def main():

    all_records = []
    all_vectors = []

    for dataset in DATASETS:

        print("\n==========================")
        print("DATASET:", dataset)
        print("==========================")

        path = os.path.join(
            RUN_DIR,
            dataset,
            f"{REGIME}_eps_{EPS}.jsonl"
        )

        if not os.path.exists(path):
            print("Missing:", path)
            continue

        records_raw = load_jsonl(path)
        records = convert_records(records_raw, dataset)

        print("Loaded records:", len(records))

        if len(records) == 0:
            continue

        X = torch.stack(
            [r["perturbation_direction"] for r in records],
            dim=0
        )

        # Per-dataset PCA
        save_pca(dataset, X, records)

        # Accumulate for global
        all_records.extend(records)
        all_vectors.append(X)

    # =============================
    # GLOBAL PCA
    # =============================

    if len(all_vectors) > 0:

        print("\n==========================")
        print("GLOBAL PCA")
        print("==========================")

        X_global = torch.cat(all_vectors, dim=0)

        print("Global matrix shape:", X_global.shape)

        save_pca("GLOBAL", X_global, all_records)


if __name__ == "__main__":
    main()