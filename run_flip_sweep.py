import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from synthetic_data_load import load_pairs
from rm_utils import load_rm_and_tokenizer
from layer_attack import AttackConfig, attack_pair_margin_pgd

MODEL_NAME = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
ATTACK_TARGET = "residual_stream"
TOKEN_SCOPE = "all"
ATTACK_MODES = [True, False]  # [sign_flip, no_sign_flip]

def parse_args():
    p = argparse.ArgumentParser(
        description="Latent adversarial perturbations on a reward model (pairwise flip test)."
    )
    p.add_argument(
        "--layers",
        type=str,
        nargs="+",
        required=True,
        help="Layers to perturb. Example: --layers 0 1 2 ... 27  OR  --layers all",
    )
    p.add_argument(
        "--eps",
        type=float,
        nargs="+",
        required=True,
        help="L2 perturbation radii (e.g. 0.5 1 2 4 8)",
    )
    p.add_argument("--steps", type=int, default=6, help="PGD steps (default: 6)")
    p.add_argument("--out", type=str, default="results/sweep.jsonl", help="Output JSONL")
    p.add_argument("--split", type=str, default="harmless", choices=["harmless", "helpful"])
    p.add_argument("--n", type=int, default=200, help="Number of examples (default: 200)")
    p.add_argument("--max_length", type=int, default=512, help="Tokenizer max_length (default: 512)")
    p.add_argument("--max_tokens", type=int, default=512, help="Skip examples where either seq > this (default: 512)")
    p.add_argument("--per_token_l2", action="store_true", help="Use per-token L2 constraint")
    return p.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_layers(layer_args, num_layers: int):
    if len(layer_args) == 1 and layer_args[0].lower() == "all":
        return list(range(num_layers))
    return [int(x) for x in layer_args]

def _fmt_eta(seconds: float) -> str:
    """Format ETA in human-readable format"""
    seconds = max(0.0, float(seconds))
    if seconds >= 3600:
        return f"{seconds/3600:.2f}h"
    if seconds >= 60:
        return f"{seconds/60:.1f}m"
    return f"{seconds:.0f}s"

def main():
    args = parse_args()
    set_seed(SEED)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n========== CONFIG ==========")
    print(f"Model:        {MODEL_NAME}")
    print(f"Split:        {args.split}")
    print(f"Requested n:  {args.n}")
    print(f"Epsilons:     {args.eps}")
    print(f"PGD steps:    {args.steps}")
    print(f"max_length:   {args.max_length}")
    print(f"max_tokens:   {args.max_tokens} (skip long pairs)")
    print(f"per_token_l2: {args.per_token_l2}")
    print(f"Device:       {DEVICE}")
    print(f"Out:          {args.out}")
    print("============================\n")

    print("Loading reward model...")
    tokenizer, rm, device = load_rm_and_tokenizer(MODEL_NAME, device=DEVICE)
    
    for p in rm.parameters():
        p.requires_grad_(False)
    rm.eval()

    try:
        num_layers = len(rm.model.layers)
    except Exception as e:
        raise RuntimeError(
            "Could not read rm.model.layers. If this model structure differs, "
            "we need to update layer access."
        ) from e

    layers = parse_layers(args.layers, num_layers=num_layers)
    print(f"Using layers: {layers}")

    print("Loading dataset...")
    dataset = load_pairs(split=args.split, n=args.n, seed=SEED)

    summary = {
        (("sign_flip" if sign_flip else "no_sign_flip"), int(layer), float(eps)): {
            "n": 0,
            "n_base_correct": 0,
            "n_flipped": 0,
            "n_skipped": 0,
        }
        for sign_flip in ATTACK_MODES
        for layer in layers
        for eps in args.eps
    }

    baseline_rows = []
    print("Scoring baselines (and filtering long examples)...")
    kept = 0
    skipped = 0
    t0 = time.time()
    
    for ex_id, ex in enumerate(dataset):
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]
        
        txt_c = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}],
            tokenize=False
        )
        txt_r = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}],
            tokenize=False
        )
        
        in_c = tokenizer(txt_c, return_tensors="pt", truncation=True, max_length=args.max_length)
        in_r = tokenizer(txt_r, return_tensors="pt", truncation=True, max_length=args.max_length)
        
        len_c = int(in_c["input_ids"].shape[1])
        len_r = int(in_r["input_ids"].shape[1])
        
        if len_c > args.max_tokens or len_r > args.max_tokens:
            skipped += 1
            continue
        
        in_c = {k: v.to(device) for k, v in in_c.items()}
        in_r = {k: v.to(device) for k, v in in_r.items()}
        
        with torch.no_grad():
            s_c0 = float(rm(**in_c).logits.squeeze().float().cpu().item())
            s_r0 = float(rm(**in_r).logits.squeeze().float().cpu().item())
        
        baseline_correct = (s_c0 > s_r0)
        margin_0 = s_c0 - s_r0
        
        baseline_rows.append({
            "example_id": kept,
            "split": args.split,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "score_chosen": s_c0,
            "score_rejected": s_r0,
            "margin": margin_0,
            "baseline_correct": baseline_correct,
            "chosen_len_tokens": len_c,
            "rejected_len_tokens": len_r,
            "in_c": in_c,
            "in_r": in_r,
        })
        kept += 1
    
    print(f"Baseline done. kept={kept} skipped={skipped} (by max_tokens/max_length) in {time.time()-t0:.1f}s\n")
    
    if kept == 0:
        raise RuntimeError("No examples left after filtering. Increase --max_tokens or --max_length.")
    
    print("Running sweep...\n")
    total_jobs = kept * len(layers) * len(args.eps) * len(ATTACK_MODES)
    done_jobs = 0
    start = time.time()
    
    LOG_EVERY = 25
    
    with out_path.open("w") as f:
        for base in baseline_rows:
            ex_id = base["example_id"]
            prompt = base["prompt"]
            chosen = base["chosen"]
            rejected = base["rejected"]
            in_c = base["in_c"]
            in_r = base["in_r"]
            
            for layer in layers:
                for eps in args.eps:
                    for sign_flip in ATTACK_MODES:
                        step_size = float(eps) / max(int(args.steps), 1)
                        cfg = AttackConfig(
                            layer=int(layer),
                            epsilon=float(eps),
                            pgd_steps=int(args.steps),
                            step_size=float(step_size),
                            max_length=int(args.max_length),
                            per_token_l2=bool(args.per_token_l2),
                            sign_flip=bool(sign_flip),
                        )
                        
                        out = attack_pair_margin_pgd(
                            tokenizer=tokenizer,
                            rm_model=rm,
                            prompt=prompt,
                            chosen=chosen,
                            rejected=rejected,
                            cfg=cfg,
                            device=device,
                            attack_mask=None,
                            in_c=in_c,
                            in_r=in_r,
                        )
                        
                        flipped = bool(base["baseline_correct"] and out["adv_margin"] <= 0)
                        
                        row = {
                            "example_id": int(ex_id),
                            "split": args.split,
                            "layer": int(layer),
                            "epsilon": float(eps),
                            "pgd_steps": int(args.steps),
                            "step_size": float(step_size),
                            "per_token_l2": bool(args.per_token_l2),
                            "attack_target": ATTACK_TARGET,
                            "token_scope": TOKEN_SCOPE,
                            "score_chosen": float(base["score_chosen"]),
                            "score_rejected": float(base["score_rejected"]),
                            "margin": float(base["margin"]),
                            "baseline_correct": bool(base["baseline_correct"]),
                            "score_chosen_adv": float(out["adv_chosen"]),
                            "score_rejected_adv": float(out["adv_rejected"]),
                            "margin_adv": float(out["adv_margin"]),
                            "flipped": bool(flipped),
                            "delta_global_l2": float(out["delta_global_l2"]),
                            "delta_max_token_l2": float(out["delta_max_token_l2"]),
                            "chosen_len_tokens": int(out["chosen_len_tokens"]),
                            "rejected_len_tokens": int(out["rejected_len_tokens"]),
                        }
                        row["attack_mode"] = ("sign_flip" if sign_flip else "no_sign_flip")
                        
                        f.write(json.dumps(row) + "\n")
                        
                        key = (("sign_flip" if sign_flip else "no_sign_flip"), int(layer), float(eps))
                        summary[key]["n"] += 1
                        if base["baseline_correct"]:
                            summary[key]["n_base_correct"] += 1
                            if flipped:
                                summary[key]["n_flipped"] += 1
                        
                        done_jobs += 1
                        
                        if done_jobs % LOG_EVERY == 0 or done_jobs == total_jobs:
                            elapsed = time.time() - start
                            rate = done_jobs / max(elapsed, 1e-9)
                            remaining = total_jobs - done_jobs
                            eta = remaining / max(rate, 1e-9)
                            pct = 100.0 * done_jobs / max(total_jobs, 1)
                            
                            print(
                                f"[{done_jobs}/{total_jobs}] {pct:6.2f}% | "
                                f"{rate:6.2f} jobs/s | ETA {_fmt_eta(eta)}"
                            )
        f.flush()
    
    print("\n========== SUMMARY (flip_rate = flipped / baseline_correct) ==========")
    for (mode, layer, eps), s in sorted(summary.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        nb = s["n_base_correct"]
        nf = s["n_flipped"]
        flip_rate = (nf / nb) if nb > 0 else 0.0
        print(
            f"mode={mode:<13} "
            f"layer={layer:02d} "
            f"eps={eps:<6} "
            f"n={s['n']:<5} "
            f"base_correct={nb:<5} "
            f"flipped={nf:<5} "
            f"flip_rate={flip_rate:.3f}"
        )
    print("===============================================================\n")
    print("Done. Wrote:", out_path)

if __name__ == "__main__":
    main()