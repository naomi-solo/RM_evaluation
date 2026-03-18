#!/usr/bin/env python3
"""
run_selectivity_eval_v2.py

Topic-selectivity transfer experiment with:
- multi-component support
- alpha sweep
- optional GLOBAL baseline directions
- extra fields for clean-margin-controlled downstream analysis

For each selected PCA direction v and each alpha:
  h' = h + alpha * v
at the chosen RM layer for every mixed-eval pair.

Writes:
  results/experiments/<exp_name>/raw/transfer_eval_records.jsonl
  results/experiments/<exp_name>/run_meta.json
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
GLOBAL_ALIASES = {"global", "GLOBAL", "all", "ALL"}


def parse_component_specs(spec: str) -> List[Tuple[str, int]]:
    """
    "sgxs:0,gsm_mc:1,global:0" -> [("sgxs",0), ("gsm_mc",1), ("global",0)]
    """
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Bad component spec '{chunk}'. Expected dataset:component")
        ds, comp = chunk.split(":", 1)
        out.append((ds.strip(), int(comp)))
    if not out:
        raise ValueError("No component specs parsed.")
    return out


def parse_alphas(alpha: float, alphas: str) -> List[float]:
    """
    Priority:
      1) --alphas "0.25,0.5,1.0,2.0"
      2) --alpha 1.0
    """
    if alphas is None or not alphas.strip():
        return [float(alpha)]

    vals = []
    for x in alphas.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    if not vals:
        raise ValueError("No valid alphas parsed from --alphas")
    return vals


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
    return matches[-1]


def _is_global_name(source_dataset: str) -> bool:
    return source_dataset in GLOBAL_ALIASES


def load_direction(
    source_dataset: str,
    component_idx: int,
    pca_dir: str,
    labels_dir: str,
    eps_tag: str = "0.1",
) -> dict:
    """
    Dataset-specific expected files:
      pca_<dataset>_eps{eps_tag}_shared_flip_k*.pt
      labels_<dataset>_eps{eps_tag}_shared_flip.jsonl

    GLOBAL expected files:
      pca_GLOBAL_eps{eps_tag}_shared_flip_k*.pt
      labels_GLOBAL_eps{eps_tag}_shared_flip.jsonl
    """
    is_global = _is_global_name(source_dataset)
    ds_key = "GLOBAL" if is_global else source_dataset

    pca_pattern = os.path.join(pca_dir, f"pca_{ds_key}_eps{eps_tag}_shared_flip_k*.pt")
    pca_path = find_single(pca_pattern)

    label_path = os.path.join(labels_dir, f"labels_{ds_key}_eps{eps_tag}_shared_flip.jsonl")
    labels_map = {}
    if os.path.exists(label_path):
        labels_map = load_labels_map(label_path)

    obj = torch.load(pca_path, map_location="cpu")
    comps = obj["components"]  # (K, D)
    if component_idx < 0 or component_idx >= comps.shape[0]:
        raise IndexError(f"Component {component_idx} out of range for {pca_path}; K={comps.shape[0]}")

    v = comps[component_idx].float().cpu()
    v_norm = torch.linalg.norm(v).item()
    if v_norm <= 0:
        raise RuntimeError(f"Zero-norm component vector for {source_dataset}:{component_idx}")
    v = v / v_norm

    lab_row = labels_map.get(component_idx, {})
    default_label = f"{ds_key}_comp{component_idx}"
    label = lab_row.get("label", default_label)

    return {
        "source_dataset": source_dataset,
        "source_dataset_key": ds_key,
        "component": component_idx,
        "label": label,
        "vector": v,  # unit vector
        "pca_path": pca_path,
        "label_path": label_path if os.path.exists(label_path) else None,
        "is_global_baseline": bool(is_global),
    }


@torch.no_grad()
def score_with_direction_from_inputs(
    rm,
    inputs,
    layer: int,
    direction_vec: torch.Tensor,
    alpha: float,
) -> Tuple[float, float]:

    clean_score = float(rm(**inputs).logits.squeeze())

    layer_mod = rm.model.layers[layer]
    delta = (alpha * direction_vec).to(device=inputs["input_ids"].device)

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


def clean_margin_bin(clean_margin_abs: float) -> str:
    """
    Bins for downstream clean-margin-controlled analysis.
    """
    if clean_margin_abs < 0.25:
        return "<0.25"
    if clean_margin_abs < 0.5:
        return "[0.25,0.5)"
    if clean_margin_abs < 1.0:
        return "[0.5,1.0)"
    if clean_margin_abs < 2.0:
        return "[1.0,2.0)"
    return ">=2.0"


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
            # no context_condition passed
        )
        for i, ex in enumerate(data):
            rows.append(
                {
                    "dataset": ds,
                    "pair_id_local": i,
                    "pair_uid": f"{ds}:{i}",
                    "prompt": ex["prompt"],
                    "chosen": ex["chosen"],
                    "rejected": ex["rejected"],
                    "meta": ex.get("meta", {}),
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

    ap.add_argument(
        "--component_specs",
        required=True,
        help='Comma list, e.g. "sgxs:0,gsm_mc:0,math_mc:0,bbq:0,mmlu:0,global:0"',
    )

    ap.add_argument("--datasets", default="bbq,gsm_mc,math_mc,mmlu,sgxs")
    ap.add_argument("--n_per_dataset", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="holdout", choices=["train", "holdout"])
    ap.add_argument("--holdout_size", type=int, default=100)

    ap.add_argument("--model", default="Skywork/Skywork-Reward-V2-Qwen3-0.6B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layer", type=int, default=14)

    # Backward-compatible alpha + new alphas sweep
    ap.add_argument("--alpha", type=float, default=1.0, help="Used when --alphas is not set.")
    ap.add_argument("--alphas", default=None, help='Comma list, e.g. "0.25,0.5,1.0,2.0"')

    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--store_text", action="store_true", help="Store prompt/chosen/rejected in each row.")

    args = ap.parse_args()

    alphas = parse_alphas(args.alpha, args.alphas)

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

    print("Pre-tokenizing all examples...")

    cached = []
    
    for ex in mixed:
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]
    
        text_c = format_chat(tok, prompt, chosen)
        text_r = format_chat(tok, prompt, rejected)
    
        inputs_c = tokenize(tok, text_c, device=dev, max_length=args.max_length)
        inputs_r = tokenize(tok, text_r, device=dev, max_length=args.max_length)
    
        cached.append({
            "dataset": ex["dataset"],
            "pair_id_local": ex["pair_id_local"],
            "pair_uid": ex["pair_uid"],
            "inputs_chosen": inputs_c,
            "inputs_rejected": inputs_r,
            "meta": ex.get("meta", {}),
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    
    print(f"Cached {len(cached)} examples.")

    out_path = raw_dir / "transfer_eval_records.jsonl"
    n_total = len(mixed) * len(directions) * len(alphas)
    done = 0

    with out_path.open("w") as f:
        for d in directions:
            vec = d["vector"]
            direction_id = f"{d['source_dataset']}:c{d['component']}"

            for alpha in alphas:
                alpha = float(alpha)

                for ex in cached:

                    cc0, cc1 = score_with_direction_from_inputs(
                        rm,
                        ex["inputs_chosen"],
                        args.layer,
                        vec,
                        alpha,
                    )
                
                    cr0, cr1 = score_with_direction_from_inputs(
                        rm,
                        ex["inputs_rejected"],
                        args.layer,
                        vec,
                        alpha,
                    )

                    clean_margin = cc0 - cr0
                    adv_margin = cc1 - cr1
                    delta_margin = adv_margin - clean_margin

                    baseline_correct = bool(clean_margin > 0)
                    flip_success = bool(baseline_correct and adv_margin <= 0)

                    cm_abs = abs(float(clean_margin))
                    cm_bin = clean_margin_bin(cm_abs)

                    row = {
                        "direction_id": direction_id,
                        "direction_source_dataset": d["source_dataset"],
                        "direction_source_dataset_key": d["source_dataset_key"],
                        "direction_component": d["component"],
                        "direction_label": d["label"],
                        "direction_pca_path": d["pca_path"],
                        "direction_label_path": d["label_path"],
                        "direction_is_global_baseline": d["is_global_baseline"],

                        "target_dataset": ex["dataset"],
                        "pair_id_local": ex["pair_id_local"],
                        "pair_uid": ex["pair_uid"],

                        "alpha": alpha,
                        "layer": int(args.layer),
                        "model_name": args.model,

                        "score_chosen_clean": float(cc0),
                        "score_chosen_adv": float(cc1),
                        "score_rejected_clean": float(cr0),
                        "score_rejected_adv": float(cr1),

                        "clean_margin": float(clean_margin),
                        "adv_margin": float(adv_margin),
                        "delta_margin": float(delta_margin),

                        "clean_margin_abs": float(cm_abs),
                        "clean_margin_bin": cm_bin,

                        "baseline_correct": baseline_correct,
                        "flip_success": flip_success,
                    }

                    # carry some source pair metadata for analysis
                    if isinstance(ex.get("meta"), dict):
                        for k, v in ex["meta"].items():
                            row[f"pair_meta_{k}"] = v

                    if args.store_text:
                        row["prompt"] = prompt
                        row["chosen"] = chosen
                        row["rejected"] = rejected

                    f.write(json.dumps(row) + "\n")

                    done += 1
                    if done % 50 == 0 or done == n_total:
                        print(
                            f"[{done}/{n_total}] "
                            f"direction={direction_id} alpha={alpha} target={ex['dataset']}"
                        )

    meta = {
        "exp_name": args.exp_name,
        "out_path": str(out_path),
        "n_directions": len(directions),
        "n_targets": len(mixed),
        "n_alphas": len(alphas),
        "n_records": len(directions) * len(mixed) * len(alphas),

        "datasets": datasets,
        "n_per_dataset": args.n_per_dataset,
        "split": args.split,
        "seed": args.seed,
        "holdout_size": args.holdout_size,

        "alphas": alphas,
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