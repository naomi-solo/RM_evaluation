from rm_utils import load_rm_and_tokenizer
from hh_data import load_hh_pairs
from lat_attack import AttackConfig, attack_pair_margin_pgd

def main():
    tok, rm, dev = load_rm_and_tokenizer("Skywork/Skywork-Reward-V2-Qwen3-0.6B", device="cuda")

    ex = load_hh_pairs(split="harmless", n=1, seed=0)[0]
    prompt, chosen, rejected = ex["prompt"], ex["chosen"], ex["rejected"]

    mid = len(rm.model.layers) // 2
    cfg = AttackConfig(layer=mid, epsilon=4.0, pgd_steps=4, step_size=1.0)

    out = attack_pair_margin_pgd(tok, rm, prompt, chosen, rejected, cfg, device=dev)

    print("clean_margin:", out["clean_margin"])
    print("adv_margin:  ", out["adv_margin"])
    print("flipped:     ", out["flipped"])
    print("delta_max_token_l2:", out["delta_max_token_l2"])

    # sanity checks
    assert out["delta_max_token_l2"] <= cfg.epsilon + 1e-3, "delta exceeded epsilon"
    assert isinstance(out["adv_margin"], float)
    assert out["delta_max_token_l2"] <= cfg.epsilon + 1e-3
    assert abs(out["adv_margin"] - out["clean_margin"]) > 1e-4


if __name__ == "__main__":
    main()
