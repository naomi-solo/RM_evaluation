import json
import time
from pathlib import Path
from tqdm import tqdm

from layer_attack_direction import AttackConfig, attack_pair_margin_pgd
from rm_utils import load_rm_and_tokenizer
from load_pairs import load_pairs

# ======================================
# CONFIG
# ======================================

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

EPS = 0.1
PGD_STEPS = 8
STEP_SIZE = 1.0

timestamp = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"sweep_results/run_eps01_deltas_{timestamp}")

# ======================================

def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

# ======================================

def main():

    print(f"\nLoading model: {MODEL_NAME}")
    tok, rm, device = load_rm_and_tokenizer(MODEL_NAME, device="cuda")
    print("Model loaded.\n")

    for dataset in DATASETS:

        print("\n==========================")
        print("DATASET:", dataset)
        print("==========================")

        data = load_pairs(
            dataset_name=dataset,
            n_total=N_TOTAL,
            seed=SEED,
            split="train",
            holdout_size=HOLDOUT_SIZE,
        )

        cfg = AttackConfig(
            layer=len(rm.model.layers) // 2,
            epsilon=EPS,
            pgd_steps=PGD_STEPS,
            step_size=STEP_SIZE,
            perturbation_mode="shared",
            sign_flip=True,
        )

        results = []

        pbar = tqdm(data)

        for ex in pbar:

            out = attack_pair_margin_pgd(
                tokenizer=tok,
                rm_model=rm,
                prompt=ex["prompt"],
                chosen=ex["chosen"],
                rejected=ex["rejected"],
                cfg=cfg,
                device=device,
                return_delta=True,   # CRITICAL
            )

            out["dataset"] = dataset
            out["regime"] = "shared_flip"
            out["epsilon"] = EPS
            out["prompt"] = ex["prompt"]
            out["chosen"] = ex["chosen"]
            out["rejected"] = ex["rejected"]

            # Convert tensors to lists
            out["delta_h_chosen_eos"] = out["delta_h_chosen_eos"].tolist()
            out["delta_h_rejected_eos"] = out["delta_h_rejected_eos"].tolist()

            results.append(out)

        save_path = OUTPUT_DIR / dataset / f"shared_flip_eps_{EPS}.jsonl"
        save_jsonl(save_path, results)

        print("Saved →", save_path)

    print("\nDone.")

if __name__ == "__main__":
    main()