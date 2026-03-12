# rm_utils.py
#
# Skywork reward model wrapper for research use.
# Provides:
#   - load_rm_and_tokenizer()
#   - score_single(prompt, response)
#   - score_pair(prompt, chosen, rejected)
#   - score_batch(prompts, responses) - for efficiency
#   - score_single_grad(prompt, response) - grad-enabled (returns tensor)
#   - score_pair_grad(prompt, chosen, rejected) - grad-enabled (returns tensors)

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_rm_and_tokenizer(model_name: str, device: Optional[str] = None):
    """
    Loads the Skywork RM and tokenizer.
    Auto-detects device if not specified (MPS/CUDA/CPU).

    Returns: (tokenizer, model, device)
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Fix missing pad token (some models need this)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Auto-detect device if not specified
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float32  # MPS works best with float32
        elif torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16  # CUDA works best with bfloat16
        else:
            device = "cpu"
            dtype = torch.float32
    else:
        # Manual device specified
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    model.eval()
    print(f"Model loaded on {device.upper()} ({dtype})")
    return tokenizer, model, device


def _format_conversation(tokenizer, prompt: str, response: str) -> str:
    """Helper to format a conversation using the model's chat template."""
    conv = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(conv, tokenize=False)
    return text


@torch.no_grad()
def score_single(
    tokenizer,
    model,
    prompt: str,
    response: str,
    device: str = "cuda",
    max_length: int = 4096,
) -> float:
    """
    Returns the scalar reward for one (prompt, response) pair.
    Higher scores = better responses.
    """
    text = _format_conversation(tokenizer, prompt, response)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model(**inputs)
    score = out.logits[0][0].float().item()

    return score


def score_single_grad(
    tokenizer,
    model,
    prompt: str,
    response: str,
    device: str = "cuda",
    max_length: int = 2048,
) -> torch.Tensor:
    """
    Gradient-enabled reward score for one (prompt, response) pair.

    Returns: scalar tensor on `device` (NOT a Python float).
    Use this inside PGD/LAT where you need backprop through the score.
    """
    text = _format_conversation(tokenizer, prompt, response)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model(**inputs)
    score = out.logits.squeeze()  # scalar tensor
    return score


@torch.no_grad()
def score_pair(
    tokenizer,
    model,
    prompt: str,
    chosen: str,
    rejected: str,
    device: str = "cuda",
    max_length: int = 2048,
) -> Tuple[float, float]:
    """
    Scores both (prompt, chosen) and (prompt, rejected).
    Returns: (score_chosen, score_rejected)
    """
    score_c = score_single(tokenizer, model, prompt, chosen, device, max_length)
    score_r = score_single(tokenizer, model, prompt, rejected, device, max_length)
    return score_c, score_r


def score_pair_grad(
    tokenizer,
    model,
    prompt: str,
    chosen: str,
    rejected: str,
    device: str = "cuda",
    max_length: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gradient-enabled pair scoring.

    Returns: (score_chosen, score_rejected) as scalar tensors.
    """
    score_c = score_single_grad(tokenizer, model, prompt, chosen, device, max_length)
    score_r = score_single_grad(tokenizer, model, prompt, rejected, device, max_length)
    return score_c, score_r


@torch.no_grad()
def score_batch(
    tokenizer,
    model,
    prompts: List[str],
    responses: List[str],
    device: str = "cuda",
    max_length: int = 2048,
    batch_size: int = 8,
) -> List[float]:
    """
    Batch scoring for efficiency. Processes in chunks to avoid OOM.
    Returns: List of scores (same length as inputs)
    """
    assert len(prompts) == len(responses), "Prompts and responses must have same length"

    all_scores = []

    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_responses = responses[i:i + batch_size]

        # Format all conversations
        texts = [
            _format_conversation(tokenizer, p, r)
            for p, r in zip(batch_prompts, batch_responses)
        ]

        # Tokenize batch with padding
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get scores
        out = model(**inputs)
        batch_scores = out.logits.squeeze(-1).float().cpu().tolist()

        # Handle single item (becomes float instead of list)
        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]

        all_scores.extend(batch_scores)

    return all_scores


# Example usage:
if __name__ == "__main__":
    # Load model (auto-detects device)
    tokenizer, model, device = load_rm_and_tokenizer(
        "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    )

    # Score single response
    score = score_single(
        tokenizer, model,
        prompt="What is 2+2?",
        response="The answer is 4.",
        device=device
    )
    print(f"Single score: {score:.4f}")

    # Compare two responses
    score_good, score_bad = score_pair(
        tokenizer, model,
        prompt="What is 2+2?",
        chosen="The answer is 4.",
        rejected="I don't know.",
        device=device
    )
    print(f"Good response: {score_good:.4f}, Bad response: {score_bad:.4f}")

    # Gradient-enabled example
    model.zero_grad(set_to_none=True)
    s = score_single_grad(tokenizer, model, "What is 2+2?", "The answer is 4.", device=device)
    # You normally wouldn't backprop into the model here; this is just a demo.
    s.backward()
    print("Grad-enabled scoring works (backward succeeded).")

    # Batch scoring
    prompts = ["What is 2+2?", "What is the capital of France?", "Explain gravity"]
    responses = ["4", "Paris", "Gravity is a fundamental force"]
    scores = score_batch(tokenizer, model, prompts, responses, device=device)
    print(f"Batch scores: {scores}")
