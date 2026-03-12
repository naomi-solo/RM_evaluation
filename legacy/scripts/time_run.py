import time
import torch

from rm_utils import load_rm_and_tokenizer
from load_bbq import load_bbq_pairs_all_categories
from layer_attack_direction import AttackConfig, attack_pair_margin_pgd


def main():
    N_PAIRS = 20   # ← change if you want

    MODEL = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"

    print(f"Loading model: {MODEL}")
    tok, rm, dev = load_rm_and_tokenizer(MODEL, device="cuda")

    layer = len(rm.model.layers) // 2

    cfg = AttackConfig(
        layer=layer,
        epsilon=8.0,
        pgd_steps=8,
        step_size=1.0,
        max_length=2048,
        per_token_l2=True,
        sign_flip=True,
    )

    print(f"Loading {N_PAIRS} BBQ pairs...")
    data = load_bbq_pairs_all_categories(
        n_total=N_PAIRS,
        seed=0,
        context_condition="ambig",   # ← keep fixed for timing
        include_instruction=True,
        quiet=True,
    )

    print("Starting timing...")
    t0 = time.time()

    for i, ex in enumerate(data):
        prompt, chosen, rejected = ex["prompt"], ex["chosen"], ex["rejected"]

        _ = attack_pair_margin_pgd(
            tok,
            rm,
            prompt,
            chosen,
            rejected,
            cfg,
            device=dev,
            return_delta=True,
        )

        print(f"{i+1}/{len(data)}")

    t1 = time.time()
    total = t1 - t0

    print("\n==============================")
    print(f"Total time: {total:.2f} seconds")
    print(f"Per pair:   {total / N_PAIRS:.2f} seconds")
    print("==============================")


if __name__ == "__main__":
    main()
