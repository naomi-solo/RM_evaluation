import json
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np

from layer_attack_direction import AttackConfig, attack_pair_margin_pgd
from rm_utils import load_rm_and_tokenizer
from load_pairs import load_pairs


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"

DATASETS = [
    "bbq",
    "gsm_mc",
    "math_mc",
    "mmlu",
    "sgxs",
]

N_TOTAL = 100
HOLDOUT_SIZE = 25
SEED = 42

EPSILONS = [0.01, 0.02, 0.05, 0.1, 0.5]

PERTURBATION_REGIMES = [
    dict(name="shared_noflip", perturbation_mode="shared", sign_flip=False),
    dict(name="shared_flip", perturbation_mode="shared", sign_flip=True),
    dict(name="separate", perturbation_mode="separate", sign_flip=False),
]

PGD_STEPS = 8
STEP_SIZE = 1.0

# Timestamped experiment folder
timestamp = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"sweep_results/run_{timestamp}")


# ============================================================
# Helpers
# ============================================================

def save_jsonl(path, rows, append=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_existing(path):
    if not path.exists():
        return []
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def summarize(results):
    shifts = [r["adv_margin"] - r["clean_margin"] for r in results]
    flip_rate = np.mean([r["flipped"] for r in results])
    return flip_rate, float(np.mean(shifts)), float(np.mean(np.abs(shifts)))


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.time()

    print(f"\nLoading model: {MODEL_NAME}")
    tok, rm, device = load_rm_and_tokenizer(MODEL_NAME, device="cuda")
    print("Model loaded.\n")

    for dataset in DATASETS:

        print(f"\n============================")
        print(f"DATASET: {dataset}")
        print(f"============================")

        train_data = load_pairs(
            dataset_name=dataset,
            n_total=N_TOTAL,
            seed=SEED,
            split="train",
            holdout_size=HOLDOUT_SIZE,
        )

        print(f"Train examples: {len(train_data)}")

        for regime in PERTURBATION_REGIMES:

            for eps in EPSILONS:

                save_path = (
                    OUTPUT_DIR
                    / dataset
                    / f"{regime['name']}_eps_{eps}.jsonl"
                )

                # ---------------------------------------
                # Resume logic
                # ---------------------------------------

                existing_rows = load_existing(save_path)
                n_done = len(existing_rows)

                if n_done >= len(train_data):
                    print(f"✓ Skipping (already complete): {save_path}")
                    continue

                print(f"\n--- Regime: {regime['name']} | eps: {eps}")
                print(f"Resuming at example {n_done}/{len(train_data)}")

                cfg = AttackConfig(
                    layer=len(rm.model.layers) // 2,
                    epsilon=eps,
                    pgd_steps=PGD_STEPS,
                    step_size=STEP_SIZE,
                    perturbation_mode=regime["perturbation_mode"],
                    sign_flip=regime["sign_flip"],
                )

                results = existing_rows.copy()

                pbar = tqdm(train_data[n_done:], leave=False)

                for ex in pbar:

                    out = attack_pair_margin_pgd(
                        tokenizer=tok,
                        rm_model=rm,
                        prompt=ex["prompt"],
                        chosen=ex["chosen"],
                        rejected=ex["rejected"],
                        cfg=cfg,
                        device=device,
                        return_delta=True,
                    )

                    out["dataset"] = dataset
                    out["regime"] = regime["name"]
                    out["epsilon"] = eps
                    out["prompt"] = ex["prompt"]
                    out["chosen"] = ex["chosen"]
                    out["rejected"] = ex["rejected"]

                    if out.get("delta_h_chosen_eos") is not None:
                        out["delta_h_chosen_eos"] = out["delta_h_chosen_eos"].tolist()

                    if out.get("delta_h_rejected_eos") is not None:
                        out["delta_h_rejected_eos"] = out["delta_h_rejected_eos"].tolist()

                    results.append(out)

                    # Save incrementally every 10 examples
                    if len(results) % 10 == 0:
                        save_jsonl(save_path, results[-10:], append=True)

                    if len(results) % 25 == 0:
                        flip_rate, _, mean_abs = summarize(results)
                        pbar.set_postfix({
                            "flip": f"{flip_rate:.2f}",
                            "mean|Δ|": f"{mean_abs:.4f}"
                        })

                # Final save of remaining
                save_jsonl(save_path, results[n_done:], append=True)

                flip_rate, mean_shift, mean_abs_shift = summarize(results)

                print(f"Saved → {save_path}")
                print(f"Flip rate      : {flip_rate:.3f}")
                print(f"Mean Δmargin   : {mean_shift:.5f}")
                print(f"Mean |Δmargin| : {mean_abs_shift:.5f}")

    total_time = time.time() - total_start

    print("\n=================================")
    print("SWEEP COMPLETE")
    print(f"Total time: {total_time/60:.1f} min")
    print("=================================")


if __name__ == "__main__":
    main()