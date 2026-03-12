# label_from_sweep.py

import os
import json
import torch
from label_components import build_label_prompt, call_openai

PCA_DIR = "results/pca_from_sweep"
OUT_DIR = "results/labels_from_sweep"
os.makedirs(OUT_DIR, exist_ok=True)

USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

TOP_M = 12
BOT_M = 6


def label_file(pca_path):

    obj = torch.load(pca_path, map_location="cpu")

    scores = obj["scores"]
    records = obj["records"]
    ids = obj["ids"]

    dataset = obj["dataset"]
    eps = obj["epsilon"]
    regime = obj["regime"]
    evr = obj["explained_var_ratio"]

    K = scores.shape[1]

    out_path = os.path.join(
        OUT_DIR,
        f"labels_{dataset}_eps{eps}_{regime}.jsonl"
    )

    with open(out_path, "w") as f:

        for k in range(K):

            col = scores[:, k]

            top_vals, top_idx = torch.topk(col, k=min(TOP_M, col.shape[0]))
            bot_vals, bot_idx = torch.topk(-col, k=min(BOT_M, col.shape[0]))

            top_recs = [records[i] for i in top_idx.tolist()]
            bot_recs = [records[i] for i in bot_idx.tolist()]

            prompt = build_label_prompt(k, top_recs, bot_recs)

            if USE_OPENAI:
                raw = call_openai(prompt, model=MODEL)
                clean = raw.replace("```json", "").replace("```", "").strip()

                try:
                    data = json.loads(clean)
                except:
                    data = {
                        "label": "PARSE_ERROR",
                        "explanation": raw,
                        "keywords": [],
                        "negatives": [],
                    }
            else:
                print("\n" + "="*80)
                print(prompt)
                data = {
                    "label": "MANUAL",
                    "explanation": "",
                    "keywords": [],
                    "negatives": [],
                }

            row = {
                "component": k,
                "dataset": dataset,
                "epsilon": eps,
                "regime": regime,
                "top_ids": [ids[i] for i in top_idx.tolist()],
                "top_scores": [float(v) for v in top_vals.tolist()],
                "bottom_ids": [ids[i] for i in bot_idx.tolist()],
                "bottom_scores": [float(col[i]) for i in bot_idx.tolist()],
                "explained_var_ratio": float(evr[k]),
                **data
            }

            f.write(json.dumps(row) + "\n")

    print("Saved labels →", out_path)


def main():

    for fname in os.listdir(PCA_DIR):
        if not fname.endswith(".pt"):
            continue

        label_file(os.path.join(PCA_DIR, fname))


if __name__ == "__main__":
    main()