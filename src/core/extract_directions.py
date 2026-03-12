# extract_directions.py
import os
import torch

from datasets import disable_progress_bar
from transformers.utils import logging as hf_logging

from src.core.rm_utils import load_rm_and_tokenizer
from src.data.load_pairs import load_pairs
from src.core.layer_attack_direction import AttackConfig, attack_pair_margin_pgd


SCHEMA_VERSION = 1

os.makedirs("results/outputs/directions", exist_ok=True)

def _make_record(
    *,
    pair_id: int,
    prompt: str,
    completion: str,
    completion_type: str,  # "chosen" | "rejected"
    reward_unperturbed: float,
    reward_perturbed: float,
    perturbation_direction: torch.Tensor,  # (D,) float16 CPU
    perturbation_type: str,
    cfg: AttackConfig,
    context_condition: str,
    extra: dict,
) -> dict:
    """
    Builds a fully self-describing observation record.
    Everything needed to interpret/plot/filter this datapoint is inside.
    """
    if completion_type not in ("chosen", "rejected"):
        raise ValueError(f"completion_type must be chosen/rejected, got {completion_type}")

    if not isinstance(perturbation_direction, torch.Tensor):
        raise TypeError("perturbation_direction must be a torch.Tensor")
    if perturbation_direction.dim() != 1:
        raise ValueError(f"perturbation_direction must be shape (D,), got {tuple(perturbation_direction.shape)}")
    if perturbation_direction.device.type != "cpu":
        # We want these files to be portable (no GPU needed to load)
        perturbation_direction = perturbation_direction.to("cpu")
    if perturbation_direction.dtype != torch.float16:
        perturbation_direction = perturbation_direction.to(dtype=torch.float16)

    rec = {
        "schema_version": SCHEMA_VERSION,

        # Core fields you requested
        "prompt": prompt,
        "completion": completion,
        "completion_type": completion_type,
        "reward_unperturbed": float(reward_unperturbed),
        "reward_perturbed": float(reward_perturbed),
        "perturbation_direction": perturbation_direction,   # Δh = h_adv - h_clean
        "perturbation_type": perturbation_type,
        "epsilon": float(cfg.epsilon),

        # Strongly recommended metadata (so it's actually self-describing)
        "pair_id": int(pair_id),
        "layer": int(cfg.layer),
        "sign_flip": bool(cfg.sign_flip),
        "pgd_steps": int(cfg.pgd_steps),
        "step_size": float(cfg.step_size),
        "per_token_l2": bool(cfg.per_token_l2),
        "max_length": int(cfg.max_length),
        "context_condition": context_condition,
    }

    # Any extras from the attack output (lengths, norms, etc.)
    # Keep these flat to make analysis easier later.
    for k, v in (extra or {}).items():
        rec[k] = v

    return rec


def run_one(
    sign_flip: bool,
    context_condition: str,
    n_total: int = 200,
    perturbation_type: str = "pgd",
):
    # Make it quiet
    disable_progress_bar()
    hf_logging.set_verbosity_error()
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    MODEL = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    tok, rm, dev = load_rm_and_tokenizer(MODEL, device="cuda")

    layer = len(rm.model.layers) // 2
    # ---- read config from env (defaults keep old behavior) ----
    seed = int(os.getenv("SEED", "0"))
    eps = float(os.getenv("EPS", "8.0"))
    n_total = int(os.getenv("N_TOTAL", str(n_total)))
    layer_env = os.getenv("LAYER", None)
    
    # compute layer
    if layer_env is not None:
        layer = int(layer_env)
    else:
        layer = len(rm.model.layers) // 2
    
    cfg = AttackConfig(
        layer=layer,
        epsilon=eps,
        pgd_steps=int(os.getenv("PGD_STEPS", "8")),
        step_size=float(os.getenv("STEP_SIZE", "1.0")),
        max_length=int(os.getenv("MAX_LENGTH", "2048")),
        per_token_l2=bool(int(os.getenv("PER_TOKEN", "1"))),
        sign_flip=sign_flip,
    )
    
    dataset_name = os.getenv("DATASET", "bbq")
    split = os.getenv("SPLIT", "train")
    holdout_size = int(os.getenv("HOLDOUT", "100"))
        
    data = load_pairs(
        dataset_name=dataset_name,
        n_total=n_total,
        seed=seed,
        split=split,
        holdout_size=holdout_size,
        context_condition=context_condition,
    )

    records = []

    for i, ex in enumerate(data):
        prompt, chosen, rejected = ex["prompt"], ex["chosen"], ex["rejected"]

        out = attack_pair_margin_pgd(
            tok,
            rm,
            prompt,
            chosen,
            rejected,
            cfg,
            device=dev,
            return_delta=True,
        )

        # Minimal extra metadata from attack output that’s often useful
        # (kept in each record so each record is self-contained).
        common_extra = {
            "clean_margin": float(out["clean_margin"]),
            "adv_margin": float(out["adv_margin"]),
            "flipped": bool(out["flipped"]),
            "chosen_len_tokens": int(out["chosen_len_tokens"]),
            "rejected_len_tokens": int(out["rejected_len_tokens"]),
            "delta_global_l2": float(out["delta_global_l2"]),
            "delta_max_token_l2": float(out["delta_max_token_l2"]),
            "delta_mean_token_l2": float(out["delta_mean_token_l2"]),
            "S_attack": int(out["S_attack"]),
            "D_hidden": int(out["D_hidden"]),
        }

        common_extra["model_name"] = MODEL
        common_extra.update({f"bbq_{k}": v for k, v in ex.get("meta", {}).items()})


        # --- record for chosen completion ---
        records.append(
            _make_record(
                pair_id=i,
                prompt=prompt,
                completion=chosen,
                completion_type="chosen",
                reward_unperturbed=out["clean_chosen"],
                reward_perturbed=out["adv_chosen"],
                perturbation_direction=out["delta_h_chosen_eos"],
                perturbation_type=perturbation_type,
                cfg=cfg,
                context_condition=context_condition,
                extra=common_extra,
            )
        )

        # --- record for rejected completion ---
        records.append(
            _make_record(
                pair_id=i,
                prompt=prompt,
                completion=rejected,
                completion_type="rejected",
                reward_unperturbed=out["clean_rejected"],
                reward_perturbed=out["adv_rejected"],
                perturbation_direction=out["delta_h_rejected_eos"],
                perturbation_type=perturbation_type,
                cfg=cfg,
                context_condition=context_condition,
                extra=common_extra,
            )
        )

        if (i + 1) % 10 == 0:
            print(f"[{context_condition} | sign_flip={sign_flip}] {i+1}/{len(data)}")

    tag = "flip" if sign_flip else "noflip"
    save_path = (
        f"results/directions/"
        f"{dataset_name}_{split}_seed{seed}_layer{cfg.layer}_"
        f"eps{cfg.epsilon}_{context_condition}_{tag}_n{n_total}.pt"
    )


    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "records": records,
            "n_pairs": int(n_total),
            "n_records": int(len(records)),  # should be 2*n_pairs

            # File-level metadata (nice for quick inspection)
            "layer": int(cfg.layer),
            "epsilon": float(cfg.epsilon),
            "sign_flip": bool(sign_flip),
            "context_condition": context_condition,
            "perturbation_type": perturbation_type,
            "model_name": MODEL,
        },
        save_path,
    )

    # quick sanity check
    D = records[0]["perturbation_direction"].numel() if records else None
    print("saved:", save_path, "n_records:", len(records), "D:", D)


def main():
    cc = os.getenv("CC", "ambig")  # ambig|disambig
    sign_flip = bool(int(os.getenv("SIGN_FLIP", "1")))  # 0/1
    n_total = int(os.getenv("N_TOTAL", "200"))
    run_one(sign_flip=sign_flip, context_condition=cc, n_total=n_total, perturbation_type="pgd")



if __name__ == "__main__":
    main()
