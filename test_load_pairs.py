if __name__ == "__main__":
    from rm_utils import load_rm_and_tokenizer
    from synthetic_data_load import load_pairs
    from layer_attack_direction import AttackConfig, attack_pair_margin_pgd

    MODEL = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    tok, rm, dev = load_rm_and_tokenizer(MODEL, device="cuda")

    ex = load_pairs(split="harmless", n=1, seed=0)[0]
    prompt, chosen, rejected = ex["prompt"], ex["chosen"], ex["rejected"]

    layers = rm.model.layers
    mid = len(layers) // 2

    configs = [
        AttackConfig(layer=mid, epsilon=0.0, pgd_steps=4,
                     perturbation_mode="shared", sign_flip=False),

        AttackConfig(layer=mid, epsilon=0.0, pgd_steps=4,
                     perturbation_mode="shared", sign_flip=True),

        AttackConfig(layer=mid, epsilon=0.0, pgd_steps=4,
                     perturbation_mode="separate"),
    ]

    for cfg in configs:
        print("\n---")
        print("mode:", cfg.perturbation_mode, "| sign_flip:", cfg.sign_flip)

        out = attack_pair_margin_pgd(
            tok, rm, prompt, chosen, rejected,
            cfg, device=dev
        )

        print("clean_margin:", round(out["clean_margin"], 4))
        print("adv_margin  :", round(out["adv_margin"], 4))
        print("flipped     :", out["flipped"])