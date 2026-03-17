#!/usr/bin/env python3
"""
Run topic-selectivity transfer experiment:

For each selected PCA direction v (from one source dataset/component),
apply h' = h + alpha * v at a target RM layer for every mixed-eval example
across datasets, then measure:
  - clean_margin
  - adv_margin
  - delta_margin
  - flip_success (on baseline-correct subset)

Outputs one raw JSONL in ONE experiment folder:
  results/experiments/<exp_name>/raw/transfer_eval_records.jsonl
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from src.core.rm_utils import load_rm_and_tokenizer
from src.core.layer_attack_direction import format_chat, tokenize
from src.data.load_pairs import load_pairs


ALL_DATASETS = ["bbq", "gsm_mc", "math_mc", "mmlu", "sgxs"]


def parse_component_specs(spec: str) -> List[Tuple[str, int]]:
    """
    "sgxs:0,gsm_mc:0,math_mc:1" -> [("sgxs",0), ("gsm_mc",0), ("math_mc",1)]
    """
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        ds, comp = chunk.split(":")
        out.append((ds.strip(), int(comp)))
    if not out:
        raise ValueError("No component specs parsed.")
    return out


def load_labels_map(label_path: str) -> Dict[int, dict]:
    m = {}
    with open(label_path, "r") as f:
        for line in f:
            row = json.loads(line)
            c = int(row.get("component", -1))
            if c >= 0:
                m[c] = row
    return m


def find_single(pattern: str) -> str:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matches pattern: {pattern}")
    # choose latest lexicographically (usually highest k / latest naming)
    return matches[-1]


def load_direction(
    source_dataset: str,
    component_idx: int,
    pca_dir: str,
    labels_dir: str,
    eps_tag: str = "0.1",
) -> dict:
    """
    Expects files like:
      pca_<dataset>_eps0.1_shared_flip_k20.pt
      labels_<dataset>_eps0.1_shared_flip.jsonl
    """
    pca_path = find_single(os.path.join(pca_dir, f"pca_{source_dataset}_eps{eps_tag}_shared_flip_k*.pt"))
    label_path = os.path.join(labels_dir, f"labels_{source_dataset}_eps{eps_tag}_shared_flip.jsonl")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing labels file: {label_path}")

    obj = torch.load(pca_path, map_location="cpu")
    comps = obj["components"]  # (K, D)
    if component_idx < 0 or component_idx >= comps.shape[0]:
        raise IndexError(f"Component {component_idx} out of range for {pca_path}; K={comps.shape[0]}")

    v = comps[component_idx].float().cpu()
    v_norm = torch.linalg.norm(v).item()
    if v_norm <= 0:
        raise RuntimeError(f"Zero-norm component vector for {source_dataset}:{component_idx}")
    v = v / v_norm

    labels_map = load_labels_map(label_path)
    lab_row = labels_map.get(component_idx, {})
    label = lab_row.get("label", f"{source_dataset}_comp{component_idx}")

    return {
        "source_dataset": source_dataset,
        "component": component_idx,
        "label": label,
        "vector": v,  # unit vector
        "pca_path": pca_path,
        "label_path": label_path,
    }


@torch.no_grad()
def score_with_direction(
    tok,
    rm,
    device: str,
    layer: int,
    prompt: str,
    response: str,
    direction_vec: torch.Tensor,   # (D,), CPU float
    alpha: float,
    max_length: int,
) -> Tuple[float, float]:
    """
    Returns (clean_score, adv_score) for one (prompt,response),
    with perturbation injected at residual stream output of `layer`:
       h' = h + alpha * v
    """
    text = format_chat(tok, prompt, response)
    inputs = tokenize(tok, text, device=device, max_length=max_length)

    clean_score = float(rm(**inputs).logits.squeeze())

    layer_mod = rm.model.layers[layer]
    delta = (alpha * direction_vec).to(device=device)

    def hook_fn(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        d = delta.to(dtype=h.dtype).view(1, 1, -1)
        h2 = h + d
        return (h2,) + out[1:] if isinstance(out, tuple) else h2

    handle = layer_mod.register_forward_hook(hook_fn)
    try:
        adv_score = float(rm(**inputs).logits.squeeze())
    finally:
        handle.remove()

    return clean_score, adv_score


def build_mixed_eval(
    datasets: List[str],
    n_per_dataset: int,
    seed: int,
    split: str,
    holdout_size: int,
) -> List[dict]:
    rows = []
    for ds in datasets:
        data = load_pairs(
            dataset_name=ds,
            n_total=n_per_dataset,
            seed=seed,
            split=split,
            holdout_size=holdout_size,
            context_condition="ambig",  # used by BBQ loader; harmless for others
        )
        for i, ex in enumerate(data):
            rows.append(
                {
                    "dataset": ds,
                    "pair_id_local": i,
                    "prompt": ex["prompt"],
                    "chosen": ex["chosen"],
                    "rejected": ex["rejected"],
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--out_root", default="results/experiments")

    ap.add_argument("--pca_dir", default="archive/results/pca_from_sweep")
    ap.add_argument("--labels_dir", default="archive/results/labels_from_sweep")
    ap.add_argument("--eps_tag", default="0.1")

    ap.add_argument("--component_specs", required=True,
                    help='Comma list, e.g. "sgxs:0,gsm_mc:0,math_mc:0,bbq:0,mmlu:0"')

    ap.add_argument("--datasets", default="bbq,gsm_mc,math_mc,mmlu,sgxs")
    ap.add_argument("--n_per_dataset", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="holdout", choices=["train", "holdout"])
    ap.add_argument("--holdout_size", type=int, default=100)

    ap.add_argument("--model", default="Skywork/Skywork-Reward-V2-Qwen3-0.6B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=2048)

    args = ap.parse_args()

    exp_dir = Path(args.out_root) / args.exp_name
    raw_dir = exp_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    for ds in datasets:
        if ds not in ALL_DATASETS:
            raise ValueError(f"Unknown dataset '{ds}'. Valid: {ALL_DATASETS}")

    directions = []
    for ds, comp in parse_component_specs(args.component_specs):
        directions.append(
            load_direction(
                source_dataset=ds,
                component_idx=comp,
                pca_dir=args.pca_dir,
                labels_dir=args.labels_dir,
                eps_tag=args.eps_tag,
            )
        )

    mixed = build_mixed_eval(
        datasets=datasets,
        n_per_dataset=args.n_per_dataset,
        seed=args.seed,
        split=args.split,
        holdout_size=args.holdout_size,
    )

    tok, rm, dev = load_rm_and_tokenizer(args.model, device=args.device)

    out_path = raw_dir / "transfer_eval_records.jsonl"
    n_total = len(mixed) * len(directions)
    done = 0

    with out_path.open("w") as f:
        for d in directions:
            vec = d["vector"]
            direction_id = f"{d['source_dataset']}:c{d['component']}"

            for ex in mixed:
                prompt = ex["prompt"]
                chosen = ex["chosen"]
                rejected = ex["rejected"]

                cc0, cc1 = score_with_direction(
                    tok, rm, dev, args.layer, prompt, chosen, vec, args.alpha, args.max_length
                )
                cr0, cr1 = score_with_direction(
                    tok, rm, dev, args.layer, prompt, rejected, vec, args.alpha, args.max_length
                )

                clean_margin = cc0 - cr0
                adv_margin = cc1 - cr1
                delta_margin = adv_margin - clean_margin

                baseline_correct = bool(clean_margin > 0)
                flip_success = bool(baseline_correct and adv_margin <= 0)

                row = {
                    "direction_id": direction_id,
                    "direction_source_dataset": d["source_dataset"],
                    "direction_component": d["component"],
                    "direction_label": d["label"],
                    "direction_pca_path": d["pca_path"],
                    "direction_label_path": d["label_path"],

                    "target_dataset": ex["dataset"],
                    "pair_id_local": ex["pair_id_local"],

                    "alpha": float(args.alpha),
                    "layer": int(args.layer),
                    "model_name": args.model,

                    "score_chosen_clean": float(cc0),
                    "score_chosen_adv": float(cc1),
                    "score_rejected_clean": float(cr0),
                    "score_rejected_adv": float(cr1),

                    "clean_margin": float(clean_margin),
                    "adv_margin": float(adv_margin),
                    "delta_margin": float(delta_margin),
                    "baseline_correct": baseline_correct,
                    "flip_success": flip_success,
                }
                f.write(json.dumps(row) + "\n")

                done += 1
                if done % 25 == 0 or done == n_total:
                    print(f"[{done}/{n_total}] direction={direction_id} target={ex['dataset']}")

    # small run metadata
    meta = {
        "exp_name": args.exp_name,
        "out_path": str(out_path),
        "n_directions": len(directions),
        "n_targets": len(mixed),
        "n_records": len(directions) * len(mixed),
        "datasets": datasets,
        "n_per_dataset": args.n_per_dataset,
        "split": args.split,
        "seed": args.seed,
        "holdout_size": args.holdout_size,
        "alpha": args.alpha,
        "layer": args.layer,
        "model": args.model,
        "component_specs": args.component_specs,
        "pca_dir": args.pca_dir,
        "labels_dir": args.labels_dir,
    }
    with (exp_dir / "run_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone.")
    print(f"Raw records: {out_path}")
    print(f"Meta: {exp_dir / 'run_meta.json'}")


if __name__ == "__main__":
    main()