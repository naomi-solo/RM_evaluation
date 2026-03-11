import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from rm_utils import load_rm_and_tokenizer
from synthetic_data_load import load_pairs
from layer_attack_direction import AttackConfig, attack_pair_margin_pgd


def main():
    MODEL = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    tok, rm, dev = load_rm_and_tokenizer(MODEL, device="cuda")

    ex = load_pairs(split="harmless", n=1, seed=0)[0]
    prompt, chosen, rejected = ex["prompt"], ex["chosen"], ex["rejected"]

    layer = len(rm.model.layers) // 2
    cfg = AttackConfig(
        layer=layer,
        epsilon=2.0,       # small & fast
        pgd_steps=2,       # small & fast
        step_size=1.0,
        max_length=512,
        per_token_l2=True,
        sign_flip=True,
    )

    out = attack_pair_margin_pgd(tok, rm, prompt, chosen, rejected, cfg, device=dev, return_delta=True)

    # Must have both deltas now
    assert "delta_h_chosen_eos" in out and out["delta_h_chosen_eos"] is not None
    assert "delta_h_rejected_eos" in out and out["delta_h_rejected_eos"] is not None

    dc = out["delta_h_chosen_eos"]
    dr = out["delta_h_rejected_eos"]

    assert isinstance(dc, torch.Tensor) and isinstance(dr, torch.Tensor)
    assert dc.dim() == 1 and dr.dim() == 1
    assert dc.shape == dr.shape, f"chosen/rejected delta shapes differ: {dc.shape} vs {dr.shape}"
    assert dc.device.type == "cpu" and dr.device.type == "cpu"
    assert dc.dtype == torch.float16 and dr.dtype == torch.float16

    print("OK layer_attack_direction")
    print("layer:", out["layer"], "eps:", out["epsilon"], "D:", dc.numel())
    print("clean_margin:", out["clean_margin"], "adv_margin:", out["adv_margin"], "flipped:", out["flipped"])


if __name__ == "__main__":
    main()
