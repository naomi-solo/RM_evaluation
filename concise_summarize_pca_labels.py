# concise_summarize_pca_labels.py

import os
import json
import torch
import numpy as np

PCA_DIR = "results/pca_from_sweep"
LABEL_DIR = "results/labels_from_sweep"


def parse_pca_filename(fname):
    # Example:
    # pca_gsm_mc_eps0.1_shared_flip_k20.pt

    base = fname.replace(".pt", "")
    base = base[len("pca_"):]  # remove prefix

    # Remove trailing _kXX
    base = base[:base.rfind("_k")]

    # Now:
    # gsm_mc_eps0.1_shared_flip

    idx = base.rfind("_eps")
    dataset = base[:idx]
    rest = base[idx + 4:]  # skip "_eps"

    # rest = "0.1_shared_flip"
    eps, regime = rest.split("_", 1)

    return dataset, eps, regime


def summarize_one(pca_path):

    fname = os.path.basename(pca_path)
    dataset, eps, regime = parse_pca_filename(fname)

    label_path = os.path.join(
        LABEL_DIR,
        f"labels_{dataset}_eps{eps}_{regime}.jsonl"
    )

    if not os.path.exists(label_path):
        print("Missing labels for:", fname)
        print("Expected:", label_path)
        return

    pca_obj = torch.load(pca_path, map_location="cpu")
    scores = pca_obj["scores"]
    evr = pca_obj["explained_var_ratio"]

    labels = []
    with open(label_path, "r") as f:
        for line in f:
            labels.append(json.loads(line))

    print("\n" + "="*90)
    print(f"{dataset} | eps={eps} | regime={regime}")
    print("="*90)
    print(f"{'Comp':>4} | {'Label':<35} | {'EVR %':>7} | {'Mean |score|':>14}")
    print("-"*90)

    for k, lab in enumerate(labels):

        mean_abs = float(np.mean(np.abs(scores[:, k].numpy())))
        evr_pct = float(evr[k]) * 100

        print(f"{k:>4} | {lab['label'][:35]:<35} | {evr_pct:7.2f} | {mean_abs:14.4f}")


def main():

    for fname in os.listdir(PCA_DIR):
        if fname.endswith(".pt"):
            summarize_one(os.path.join(PCA_DIR, fname))


if __name__ == "__main__":
    main()