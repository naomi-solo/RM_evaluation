# layer_attack_direction.py
#
# Updated: supports perturbation_mode = "shared" | "separate"

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch


# -----------------------------
# Utilities
# -----------------------------

def format_chat(tokenizer, prompt: str, response: str) -> str:
    conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    return tokenizer.apply_chat_template(conv, tokenize=False)


def tokenize(tokenizer, text: str, device: str, max_length: int) -> Dict[str, torch.Tensor]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    return {k: v.to(device) for k, v in inputs.items()}


# -----------------------------
# Attack Config
# -----------------------------

@dataclass
class AttackConfig:
    layer: int
    epsilon: float = 8.0
    pgd_steps: int = 8
    step_size: float = 1.0
    max_length: int = 2048
    per_token_l2: bool = True
    grad_eps: float = 1e-12

    # shared-mode only
    sign_flip: bool = True

    # NEW
    perturbation_mode: str = "shared"  # "shared" | "separate"


# -----------------------------
# Adversary
# -----------------------------

class GDAdversary:
    def __init__(self, hidden_shape: torch.Size, device: str, epsilon: float, per_token_l2: bool = True):
        self.device = device
        self.epsilon = float(epsilon)
        self.per_token_l2 = bool(per_token_l2)

        self.attack = torch.zeros(hidden_shape, device=device, dtype=torch.float32, requires_grad=True)

        if self.attack.dim() == 2:
            self.attack_mask = torch.ones((hidden_shape[0],), device=device, dtype=torch.bool)
        else:
            self.attack_mask = torch.ones((hidden_shape[0], hidden_shape[1]), device=device, dtype=torch.bool)

    def set_mask(self, mask: torch.Tensor) -> None:
        self.attack_mask = mask.to(self.device).bool()

    @torch.no_grad()
    def clip_attack(self) -> None:
        eps = self.epsilon
        if eps <= 0:
            self.attack.zero_()
            return

        a = self.attack

        if self.per_token_l2:
            norms = torch.linalg.norm(a, dim=-1, ord=2)
            scales = torch.clamp(eps / (norms + 1e-12), max=1.0)
            a.mul_(scales.unsqueeze(-1))
        else:
            flat = a.view(a.shape[0], -1)
            norms = torch.linalg.norm(flat, dim=1, ord=2)
            scales = torch.clamp(eps / (norms + 1e-12), max=1.0)
            a.mul_(scales.view(-1, 1, 1))

    def apply(self, h: torch.Tensor, sign: float = 1.0) -> torch.Tensor:
        a = (sign * self.attack).to(dtype=h.dtype)
    
        # 2D case (S, D)
        if h.dim() == 2:
            Sh = h.shape[0]
            a_use = a[:Sh]
            m_use = self.attack_mask[:Sh]
            return torch.where(m_use.unsqueeze(1), h + a_use, h)
    
        # 3D case (B, S, D)
        Bh, Sh, Dh = h.shape
        Ba, Sa, Da = a.shape
    
        if Da != Dh:
            raise RuntimeError("Hidden dim mismatch.")
    
        # slice to match runtime sequence length
        a_use = a[:, :Sh, :]
        m_use = self.attack_mask[:, :Sh]
    
        return torch.where(m_use.unsqueeze(2), h + a_use, h)


# -----------------------------
# Hooks
# -----------------------------

def _capture_hidden_shape_at_layer(rm_model, layer_module, inputs):
    captured = {"shape": None}

    def hook_fn(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["shape"] = h.shape
        return out

    handle = layer_module.register_forward_hook(hook_fn)
    try:
        _ = rm_model(**inputs)
        return captured["shape"]
    finally:
        handle.remove()


def _forward_with_adversary(rm_model, layer_module, inputs, adversary, sign, capture=None):
    def hook_fn(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h2 = adversary.apply(h, sign=sign)
        if capture is not None:
            capture["h"] = h2.detach()
        return (h2,) + out[1:] if isinstance(out, tuple) else h2

    handle = layer_module.register_forward_hook(hook_fn)
    try:
        out = rm_model(**inputs)
        return out.logits.squeeze()
    finally:
        handle.remove()


def _forward_capture_layer_output(rm_model, layer_module, inputs, capture):
    def hook_fn(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        capture["h"] = h.detach()
        return out

    handle = layer_module.register_forward_hook(hook_fn)
    try:
        out = rm_model(**inputs)
        return out.logits.squeeze()
    finally:
        handle.remove()


# -----------------------------
# Main Attack
# -----------------------------

def attack_pair_margin_pgd(
    tokenizer,
    rm_model,
    prompt,
    chosen,
    rejected,
    cfg: AttackConfig,
    device="cuda",
    attack_mask=None,
    in_c=None,
    in_r=None,
    return_delta=True,
):

    rm_model.eval()

    if cfg.perturbation_mode not in {"shared", "separate"}:
        raise ValueError("perturbation_mode must be 'shared' or 'separate'")

    layers = rm_model.model.layers
    layer_module = layers[cfg.layer]

    if in_c is None:
        in_c = tokenize(tokenizer, format_chat(tokenizer, prompt, chosen), device, cfg.max_length)
    if in_r is None:
        in_r = tokenize(tokenizer, format_chat(tokenizer, prompt, rejected), device, cfg.max_length)

    with torch.no_grad():
        sc_clean = float(rm_model(**in_c).logits.squeeze())
        sr_clean = float(rm_model(**in_r).logits.squeeze())

    clean_margin = sc_clean - sr_clean

    shape_c = _capture_hidden_shape_at_layer(rm_model, layer_module, in_c)
    shape_r = _capture_hidden_shape_at_layer(rm_model, layer_module, in_r)

    S = max(shape_c[1], shape_r[1])
    D = shape_c[-1]

    len_c = in_c["input_ids"].shape[1]
    len_r = in_r["input_ids"].shape[1]

    base_mask = torch.zeros((1, S), device=device, dtype=torch.bool)
    base_mask[0, :max(len_c, len_r)] = True

    if cfg.perturbation_mode == "shared":
        adv = GDAdversary(torch.Size([1, S, D]), device, cfg.epsilon, cfg.per_token_l2)
    else:
        adv_c = GDAdversary(torch.Size([1, S, D]), device, cfg.epsilon, cfg.per_token_l2)
        adv_r = GDAdversary(torch.Size([1, S, D]), device, cfg.epsilon, cfg.per_token_l2)

    mask_c = base_mask.clone()
    mask_c[0, len_c:] = False
    mask_r = base_mask.clone()
    mask_r[0, len_r:] = False

    # ---------------- PGD ----------------

    with torch.enable_grad():
        for _ in range(cfg.pgd_steps):

            rm_model.zero_grad(set_to_none=True)

            if cfg.perturbation_mode == "shared":

                adv.attack.grad = None

                adv.set_mask(mask_c)
                sc = _forward_with_adversary(rm_model, layer_module, in_c, adv, +1.0)

                adv.set_mask(mask_r)
                rej_sign = -1.0 if cfg.sign_flip else +1.0
                sr = _forward_with_adversary(rm_model, layer_module, in_r, adv, rej_sign)

                loss = -(sr - sc)
                loss.backward()

                g = adv.attack.grad
                step = g / (torch.linalg.norm(g.view(1, -1), dim=1).view(-1, 1, 1) + cfg.grad_eps)
                adv.attack.data -= cfg.step_size * step
                adv.clip_attack()

            else:  # separate

                adv_c.attack.grad = None
                adv_r.attack.grad = None

                adv_c.set_mask(mask_c)
                sc = _forward_with_adversary(rm_model, layer_module, in_c, adv_c, +1.0)

                adv_r.set_mask(mask_r)
                sr = _forward_with_adversary(rm_model, layer_module, in_r, adv_r, +1.0)

                loss = -(sr - sc)
                loss.backward()

                for adv_i in (adv_c, adv_r):
                    g = adv_i.attack.grad
                    step = g / (torch.linalg.norm(g.view(1, -1), dim=1).view(-1, 1, 1) + cfg.grad_eps)
                    adv_i.attack.data -= cfg.step_size * step
                    adv_i.clip_attack()

    # ---------------- Final Eval ----------------

    with torch.no_grad():
        if cfg.perturbation_mode == "shared":

            adv.set_mask(mask_c)
            sc_adv = float(_forward_with_adversary(rm_model, layer_module, in_c, adv, +1.0))

            adv.set_mask(mask_r)
            rej_sign = -1.0 if cfg.sign_flip else +1.0
            sr_adv = float(_forward_with_adversary(rm_model, layer_module, in_r, adv, rej_sign))

        else:

            adv_c.set_mask(mask_c)
            sc_adv = float(_forward_with_adversary(rm_model, layer_module, in_c, adv_c, +1.0))

            adv_r.set_mask(mask_r)
            sr_adv = float(_forward_with_adversary(rm_model, layer_module, in_r, adv_r, +1.0))

    adv_margin = sc_adv - sr_adv
    flipped = (clean_margin > 0) and (adv_margin <= 0)

    # ---------------- Output ----------------

        # ---------------- Delta extraction (EOS) ----------------

    def _eos_delta_for_input(inps, adv_obj, mask, sign):
        """
        Returns Δh at EOS token: h_adv - h_clean, as (D,) float16 CPU tensor.
        """
        cap_clean = {}
        _forward_capture_layer_output(rm_model, layer_module, inps, cap_clean)
        h_clean = cap_clean["h"]

        cap_adv = {}
        adv_obj.set_mask(mask)
        _forward_with_adversary(rm_model, layer_module, inps, adv_obj, sign, capture=cap_adv)
        h_adv = cap_adv["h"]

        seq_len = int(inps["input_ids"].shape[1])
        eos_idx = seq_len - 1

        # h is (B,S,D) for Qwen blocks
        v_clean = h_clean[0, eos_idx, :].float()
        v_adv = h_adv[0, eos_idx, :].float()

        return (v_adv - v_clean).to("cpu", dtype=torch.float16)

    delta_h_chosen = None
    delta_h_rejected = None

    if return_delta:
        if cfg.perturbation_mode == "shared":
            # chosen always +δ
            delta_h_chosen = _eos_delta_for_input(in_c, adv, mask_c, +1.0)

            # rejected uses -δ if sign_flip else +δ
            rej_sign = -1.0 if cfg.sign_flip else +1.0
            delta_h_rejected = _eos_delta_for_input(in_r, adv, mask_r, rej_sign)

        else:
            # separate: each has its own +δ
            delta_h_chosen = _eos_delta_for_input(in_c, adv_c, mask_c, +1.0)
            delta_h_rejected = _eos_delta_for_input(in_r, adv_r, mask_r, +1.0)
            
    out = dict(
        layer=cfg.layer,
        epsilon=cfg.epsilon,
        perturbation_mode=cfg.perturbation_mode,
        sign_flip=cfg.sign_flip if cfg.perturbation_mode == "shared" else None,
        clean_margin=clean_margin,
        adv_margin=adv_margin,
        flipped=flipped,
        clean_chosen=sc_clean,
        clean_rejected=sr_clean,
        adv_chosen=sc_adv,
        adv_rejected=sr_adv,
    )

    if return_delta:
        out["delta_h_chosen_eos"] = delta_h_chosen
        out["delta_h_rejected_eos"] = delta_h_rejected
        
    return out