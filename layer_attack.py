# lat_attack.py
#
# Repo-faithful latent adversarial attack for a *reward model* (RM).
#
# This file intentionally mirrors the structure and mechanics of the
# aengusl/latent-adversarial-training repo:
#   - an adversary object that stores a perturbation tensor ("attack")
#   - an attack mask (optional) to select token positions to perturb
#   - a projection / clipping step to enforce an L2 constraint
#   - a PGD loop: forward -> loss -> backward -> step -> clip
#   - perturbations are injected into activations during the forward pass
#
# Key adaptation for your experiment:
#   - Objective is NOT token-level CE loss. Instead we optimize an RM margin:
#       maximize  score(prompt, rejected) - score(prompt, chosen)
#     i.e. loss = -(sr - sc)
#
# We keep the rest (hook + PGD + projection) as close as possible in spirit
# to the repo code, but implemented with plain PyTorch forward hooks so you
# don't need their full hook framework.
#
# If you later want to use their CustomHook/add_hooks utilities verbatim,
# we can swap the hook installation to those functions—this file keeps the
# same conceptual architecture without pulling in the whole package.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


# -----------------------------
# Utilities: locating layers
# -----------------------------


def format_chat(tokenizer, prompt: str, response: str) -> str:
    conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    return tokenizer.apply_chat_template(conv, tokenize=False)


def tokenize(tokenizer, text: str, device: str, max_length: int) -> Dict[str, torch.Tensor]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    return {k: v.to(device) for k, v in inputs.items()}


# -----------------------------
# Repo-faithful adversary object
# -----------------------------

@dataclass
class AttackConfig:
    layer: int
    epsilon: float = 8.0
    pgd_steps: int = 8
    step_size: float = 1.0
    max_length: int = 2048
    # repo-like: constrain per-token L2 (vector norm over hidden dim) by default
    per_token_l2: bool = True
    grad_eps: float = 1e-12

    # If True: chosen gets +δ, rejected gets -δ (stronger / paired)
    # If False: both chosen and rejected get +δ (single-δ threat model)
    sign_flip: bool = True


class GDAdversary:
    """
    Mirrors the repo idea: an object holding a perturbation tensor that is injected
    into activations, with an attack_mask and a clip/projection operation.

    In the aengusl repo:
      - self.attack is an nn.Parameter
      - clip_attack enforces the epsilon constraint
      - forward adds perturbation on masked positions only

    Here we keep the same semantics but don't require torch.nn.Module; we just
    store a tensor and update it manually to avoid optimizer boilerplate.
    """

    def __init__(self, hidden_shape: torch.Size, device: str, epsilon: float, per_token_l2: bool = True):
        # hidden_shape is usually (B, S, D) or (S, D)
        self.device = device
        self.epsilon = float(epsilon)
        self.per_token_l2 = bool(per_token_l2)

        # Always store attack in float32 for stable optimization
        self.attack = torch.zeros(hidden_shape, device=device, dtype=torch.float32, requires_grad=True)

        # attack_mask: True means perturb this position
        # default: perturb everything
        if self.attack.dim() == 2:
            # (S, D) -> mask is (S,)
            self.attack_mask = torch.ones((hidden_shape[0],), device=device, dtype=torch.bool)
        else:
            # (B, S, D) -> mask is (B, S)
            self.attack_mask = torch.ones((hidden_shape[0], hidden_shape[1]), device=device, dtype=torch.bool)

    def set_mask(self, mask: torch.Tensor) -> None:
        """
        mask should be:
          - (S,) if attack is (S, D)
          - (B, S) if attack is (B, S, D)
        """
        mask = mask.to(self.device).bool()
        self.attack_mask = mask

    @torch.no_grad()
    def clip_attack(self) -> None:
        """
        Repo-faithful clipping:
          - if per_token_l2: each token vector ||attack[b, t, :]||_2 <= epsilon
          - else: global L2 over (S*D) per batch element
        """
        eps = self.epsilon
        if eps <= 0:
            self.attack.zero_()
            return

        a = self.attack

        if a.dim() == 2:
            # (S, D)
            if self.per_token_l2:
                norms = torch.linalg.norm(a, dim=1, ord=2)  # (S,)
                scales = torch.clamp(eps / (norms + 1e-12), max=1.0)
                a.mul_(scales.unsqueeze(1))
            else:
                n = torch.linalg.norm(a.reshape(-1), ord=2)
                scale = min(1.0, eps / (n.item() + 1e-12))
                a.mul_(scale)
            return

        # (B, S, D)
        if self.per_token_l2:
            norms = torch.linalg.norm(a, dim=2, ord=2)  # (B, S)
            scales = torch.clamp(eps / (norms + 1e-12), max=1.0)  # (B, S)
            a.mul_(scales.unsqueeze(2))
        else:
            flat = a.view(a.shape[0], -1)  # (B, S*D)
            norms = torch.linalg.norm(flat, dim=1, ord=2)  # (B,)
            scales = torch.clamp(eps / (norms + 1e-12), max=1.0)  # (B,)
            a.mul_(scales.view(-1, 1, 1))

    def apply(self, h: torch.Tensor, sign: float = +1.0) -> torch.Tensor:
        """
        Add perturbation to hidden states h on masked positions.
        
        Robust to seq-len mismatches: slices/pads attack + mask to match h at runtime.
        """
        a = (sign * self.attack).to(dtype=h.dtype)
        
        # ---------- 2D case: (S, D) ----------
        if h.dim() == 2:
            Sh, Dh = h.shape
            Sa, Da = a.shape
            if Dh != Da:
                raise RuntimeError(f"Hidden dim mismatch: h has D={Dh}, attack has D={Da}")
        
            m = self.attack_mask
            # align lengths
            if Sa < Sh:
                # pad attack with zeros
                pad = torch.zeros((Sh - Sa, Da), device=a.device, dtype=a.dtype)
                a_use = torch.cat([a, pad], dim=0)
                m_use = torch.cat([m, torch.zeros((Sh - Sa,), device=m.device, dtype=torch.bool)], dim=0)
            else:
                a_use = a[:Sh]
                m_use = m[:Sh]
        
            return torch.where(m_use.unsqueeze(1), h + a_use, h)
        
        # ---------- 3D case: (B, S, D) ----------
        if h.dim() != 3:
            raise RuntimeError(f"Expected h dim 2 or 3, got {h.dim()}")
        
        Bh, Sh, Dh = h.shape
        Ba, Sa, Da = a.shape
        if Dh != Da:
            raise RuntimeError(f"Hidden dim mismatch: h has D={Dh}, attack has D={Da}")
        if Ba != Bh:
            # We expect batch=1 in this project; if you later batch, you can expand here.
            raise RuntimeError(f"Batch mismatch: h has B={Bh}, attack has B={Ba}")
        
        m = self.attack_mask  # (B,S)
        # align seq length
        if Sa < Sh:
            pad = torch.zeros((Bh, Sh - Sa, Da), device=a.device, dtype=a.dtype)
            a_use = torch.cat([a, pad], dim=1)
            m_pad = torch.zeros((Bh, Sh - Sa), device=m.device, dtype=torch.bool)
            m_use = torch.cat([m, m_pad], dim=1)
        else:
            a_use = a[:, :Sh, :]
            m_use = m[:, :Sh]
        
        return torch.where(m_use.unsqueeze(2), h + a_use, h)



# -----------------------------
# Hook installation (minimal)
# -----------------------------

def _capture_hidden_shape_at_layer(
    rm_model: torch.nn.Module, layer_module: torch.nn.Module, inputs: Dict[str, torch.Tensor]
) -> torch.Size:
    """
    Run a forward pass with a hook that captures the layer output shape.
    """
    captured = {"shape": None}

    def hook_fn(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["shape"] = h.shape
        return out

    handle = layer_module.register_forward_hook(hook_fn)
    try:
        _ = rm_model(**inputs)
        if captured["shape"] is None:
            raise RuntimeError("Failed to capture hidden shape.")
        return captured["shape"]
    finally:
        handle.remove()


def _forward_with_adversary(
    rm_model: torch.nn.Module,
    layer_module: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    adversary: GDAdversary,
    sign: float
) -> torch.Tensor:
    """
    Forward pass where we inject the adversary into the layer output.
    Returns RM scalar score tensor.
    """

    def hook_fn(_m, _inp, out):
        if isinstance(out, tuple):
            h = out[0]
            return (adversary.apply(h, sign=sign),) + out[1:]
        else:
            return adversary.apply(out, sign=sign)

    handle = layer_module.register_forward_hook(hook_fn)
    try:
        out = rm_model(**inputs)
        return out.logits.squeeze()
    finally:
        handle.remove()


# -----------------------------
# PGD on pairwise RM margin
# -----------------------------

def attack_pair_margin_pgd(
    tokenizer,
    rm_model: torch.nn.Module,
    prompt: str,
    chosen: str,
    rejected: str,
    cfg: AttackConfig,
    device: str = "cuda",
    attack_mask: Optional[torch.Tensor] = None,
    in_c: Optional[Dict[str, torch.Tensor]] = None,
    in_r: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """
    Repo-faithful version of the attack:
      - ONE adversary (one δ) shared across both chosen & rejected forwards
      - objective: maximize (score_rejected - score_chosen)
      - PGD loop with clipping after each step

    attack_mask (optional):
      - If provided, must match adversary mask shape:
          (S,) for single example w/ hidden (S,D), or
          (1,S) for hidden (1,S,D)
      - If None, perturb all tokens (simplest)
    
    in_c, in_r (optional):
      - Pre-tokenized inputs for chosen and rejected.
      - If provided, skips tokenization (significant speedup in sweeps).
      - If None, will tokenize prompt+chosen/rejected internally.
    """
    rm_model.eval()

    layers = rm_model.model.layers
    if cfg.layer < 0 or cfg.layer >= len(layers):
        raise ValueError(f"layer={cfg.layer} out of range (0..{len(layers)-1})")
    layer_module = layers[cfg.layer]

    # Tokenize only if not provided (allows pre-tokenization for speedup)
    if in_c is None:
        txt_c = format_chat(tokenizer, prompt, chosen)
        in_c = tokenize(tokenizer, txt_c, device=device, max_length=cfg.max_length)
    
    if in_r is None:
        txt_r = format_chat(tokenizer, prompt, rejected)
        in_r = tokenize(tokenizer, txt_r, device=device, max_length=cfg.max_length)

    # Clean scores (no perturbation)
    with torch.no_grad():
        sc_clean = float(rm_model(**in_c).logits.squeeze().float().cpu().item())
        sr_clean = float(rm_model(**in_r).logits.squeeze().float().cpu().item())
    clean_margin = sc_clean - sr_clean

    # Build adversary sized to chosen hidden shape (chosen & rejected should share hidden dim;
    # seq len may differ, but we need one shared δ. We'll align to the LONGER seq and mask out
    # beyond each example length.
    # To keep things simple and repo-faithful, we create δ with shape of the LONGER seq among the two.
    # Then for the shorter input we only apply δ up to its seq_len.
    shape_c = _capture_hidden_shape_at_layer(rm_model, layer_module, in_c)
    shape_r = _capture_hidden_shape_at_layer(rm_model, layer_module, in_r)

    # Expect (B,S,D) typically. We handle both 2D and 3D.
    def _seq_len(shape: torch.Size) -> int:
        if len(shape) == 2:
            return shape[0]
        return shape[1]

    def _hidden_dim(shape: torch.Size) -> int:
        return shape[-1]

    if _hidden_dim(shape_c) != _hidden_dim(shape_r):
        raise RuntimeError("Chosen and rejected hidden dims differ; unexpected for same model.")

    S = max(_seq_len(shape_c), _seq_len(shape_r))
    D = _hidden_dim(shape_c)

    # We'll use (1,S,D) adversary for simplicity (batch size 1)
    adv = GDAdversary(hidden_shape=torch.Size([1, S, D]), device=device, epsilon=cfg.epsilon, per_token_l2=cfg.per_token_l2)

    # Masking logic:
    # - default: all tokens up to each input length are attackable
    # - if user provides attack_mask, we AND it with the "within length" mask
    len_c = int(in_c["input_ids"].shape[1])
    len_r = int(in_r["input_ids"].shape[1])

    base_mask = torch.zeros((1, S), device=device, dtype=torch.bool)
    base_mask[0, :max(len_c, len_r)] = True

    # if provided, normalize to (1,S)
    if attack_mask is not None:
        m = attack_mask.to(device).bool()
        if m.dim() == 1:
            m = m.unsqueeze(0)
        if m.shape != (1, S):
            raise ValueError(f"attack_mask must have shape (S,) or (1,S) with S={S}, got {tuple(m.shape)}")
        base_mask = base_mask & m

    adv.set_mask(base_mask)

    # Pre-compute masks for each sequence (avoid cloning in tight PGD loop)
    mask_c = base_mask.clone()
    mask_c[0, len_c:] = False
    
    mask_r = base_mask.clone()
    mask_r[0, len_r:] = False

    # PGD loop: maximize (sr - sc) => minimize loss = -(sr - sc)
    rm_model.eval()

    with torch.enable_grad():
        for _ in range(cfg.pgd_steps):
            rm_model.zero_grad(set_to_none=True)
            if adv.attack.grad is not None:
                adv.attack.grad.zero_()
    
            # Forward with SAME adversary for both sequences.
            # For each forward we need to ensure we only apply δ up to that seq len.
    
            # chosen
            adv.set_mask(mask_c)
            sc = _forward_with_adversary(rm_model, layer_module, in_c, adv, sign=+1.0)
    
            # rejected
            adv.set_mask(mask_r)
            rej_sign = (-1.0 if cfg.sign_flip else +1.0)
            sr = _forward_with_adversary(rm_model, layer_module, in_r, adv, sign=rej_sign)
    
            loss = -(sr - sc)
            loss.backward()
    
            with torch.no_grad():
                g = adv.attack.grad
                if g is None:
                    raise RuntimeError("No gradient flowed to adversary.attack")
    
                # PGD-style update: take a step in direction of gradient descent on loss
                # Use L2-normalized step over the *active* positions (roughly repo spirit).
                # For simplicity, normalize over full tensor; mask zeros won't matter much.
                g_flat = g.view(g.shape[0], -1)
                g_norm = torch.linalg.norm(g_flat, dim=1, ord=2)  # (B,)
                step = g / (g_norm.view(-1, 1, 1) + cfg.grad_eps)
    
                adv.attack.add_(-cfg.step_size * step)
                adv.clip_attack()

    with torch.no_grad():
        # Final attacked scores
        adv.set_mask(mask_c)
        sc_adv = float(_forward_with_adversary(rm_model, layer_module, in_c, adv, sign=+1.0).detach().float().cpu().item())

        adv.set_mask(mask_r)
        rej_sign = (-1.0 if cfg.sign_flip else +1.0)
        sr_adv = float(_forward_with_adversary(rm_model, layer_module, in_r, adv, sign=rej_sign).detach().float().cpu().item())

    adv_margin = sc_adv - sr_adv
    flipped = (clean_margin > 0) and (adv_margin <= 0)

    # norms (report both global and per-token max norm for interpretability)
    with torch.no_grad():
        a = adv.attack.detach().float()[0]  # (S,D)
        global_l2 = float(torch.linalg.norm(a.reshape(-1), ord=2).cpu().item())
        per_tok = torch.linalg.norm(a, dim=1, ord=2)
        max_tok = float(per_tok.max().cpu().item())
        mean_tok = float(per_tok.mean().cpu().item())

    return {
        "layer": cfg.layer,
        "epsilon": cfg.epsilon,
        "pgd_steps": cfg.pgd_steps,
        "step_size": cfg.step_size,
        "per_token_l2": cfg.per_token_l2,
        "clean_chosen": sc_clean,
        "clean_rejected": sr_clean,
        "clean_margin": clean_margin,
        "adv_chosen": sc_adv,
        "adv_rejected": sr_adv,
        "adv_margin": adv_margin,
        "flipped": bool(flipped),
        "delta_global_l2": global_l2,
        "delta_max_token_l2": max_tok,
        "delta_mean_token_l2": mean_tok,
        "chosen_len_tokens": len_c,
        "rejected_len_tokens": len_r,
        "S_attack": S,
        "D_hidden": D,
        "sign_flip": bool(cfg.sign_flip),
    }


# -----------------------------
# Minimal self-test
# -----------------------------
if __name__ == "__main__":
    from rm_utils import load_rm_and_tokenizer
    from hh_data import load_hh_pairs

    MODEL = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    tok, rm, dev = load_rm_and_tokenizer(MODEL, device="cuda")

    ex = load_hh_pairs(split="harmless", n=1, seed=0)[0]
    prompt, chosen, rejected = ex["prompt"], ex["chosen"], ex["rejected"]

    layers = rm.model.layers
    mid = len(layers) // 2

    cfg = AttackConfig(layer=mid, epsilon=8.0, pgd_steps=8, step_size=1.0, max_length=2048, per_token_l2=True)
    out = attack_pair_margin_pgd(tok, rm, prompt, chosen, rejected, cfg, device=dev)
    print(out)