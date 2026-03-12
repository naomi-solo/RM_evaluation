# hh_data.py
#
# Minimal loader for Anthropic HH dataset pairs (prompt, chosen, rejected).
# Designed to feed the RM scoring + latent attack pipeline.
#
# Usage:
#   data = load_hh_pairs(split="harmless", n=500, seed=0)
#   for ex in data:
#       prompt, chosen, rejected = ex["prompt"], ex["chosen"], ex["rejected"]

from __future__ import annotations

from typing import Dict, List, Optional
import random

from datasets import load_dataset

import re


def _extract_prompt(text: str) -> str:
    if text is None:
        return ""

    t = text.strip()  # important: removes leading \n\n
    human_marker = "Human:"
    asst_marker = "Assistant:"

    # Find Human:
    h = t.find(human_marker)
    if h != -1:
        t2 = t[h + len(human_marker):].strip()
    else:
        t2 = t

    # Split on first Assistant:
    a = t2.find(asst_marker)
    if a != -1:
        prompt = t2[:a].strip()
    else:
        prompt = t2.strip()

    return prompt


def _extract_response(text: str) -> str:
    if text is None:
        return ""

    t = text.strip()
    asst_marker = "Assistant:"

    a = t.find(asst_marker)
    if a == -1:
        return t.strip()

    # Keep EVERYTHING after the first Assistant:
    return t[a + len(asst_marker):].strip()



def load_hh_pairs(
    split: str = "harmless",
    n: Optional[int] = None,
    seed: int = 0,
    shuffle: bool = True,
) -> List[Dict[str, str]]:
    """
    Load HH pairs from Anthropic/hh-rlhf and return a list of dicts:
      {"prompt": ..., "chosen": ..., "rejected": ...}

    Args:
      split: "harmless" or "helpful" (HHH corresponds to "harmless")
      n: optionally limit to first n examples after shuffling
      seed: random seed for shuffling/subsampling
      shuffle: whether to shuffle before truncating

    Returns:
      List of examples with prompt/chosen/rejected strings.
      - prompt: Just the human's question (no "Assistant:" marker)
      - chosen: The preferred assistant response
      - rejected: The dispreferred assistant response
    """
    if split not in {"harmless", "helpful"}:
        raise ValueError("split must be 'harmless' or 'helpful'")

    data_dir = "harmless-base" if split == "harmless" else "helpful-base"
    print(f"Loading Anthropic/hh-rlhf data_dir={data_dir} split=train ...")
    ds = load_dataset("Anthropic/hh-rlhf", data_dir=data_dir, split="train")
    print(f"Loaded {len(ds)} examples")

    idxs = list(range(len(ds)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(idxs)

    if n is not None:
        idxs = idxs[:n]
        print(f"Using {len(idxs)} examples (shuffled with seed={seed})")


    out: List[Dict[str, str]] = []
    for i in idxs:
        ex = ds[i]
        if "chosen" not in ex or "rejected" not in ex:
            raise KeyError(f"Example missing chosen/rejected keys: {ex.keys()}")
        chosen_full = ex["chosen"]
        rejected_full = ex["rejected"]

        # Extract prompt from chosen (both should have same prompt)
        prompt = _extract_prompt(chosen_full)
        chosen = _extract_response(chosen_full)
        rejected = _extract_response(rejected_full)

        out.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    return out


if __name__ == "__main__":
    # Quick self-test
    print("Testing HH data loader...\n")
    data = load_hh_pairs(split="harmless", n=3, seed=0)
    
    for j, ex in enumerate(data):
        print(f"\n{'='*60}")
        print(f"Example {j}")
        print(f"{'='*60}")
        print(f"PROMPT ({len(ex['prompt'])} chars):")
        print("  " + ex["prompt"][:200].replace("\n", "\\n"))
        print(f"\nCHOSEN ({len(ex['chosen'])} chars):")
        print("  " + ex["chosen"][:200].replace("\n", "\\n"))
        print(f"\nREJECTED ({len(ex['rejected'])} chars):")
        print("  " + ex["rejected"][:200].replace("\n", "\\n"))
        
        # Sanity check: prompt shouldn't contain "Assistant:"
        if "Assistant:" in ex["prompt"]:
            print("\n⚠️  WARNING: Prompt contains 'Assistant:' - extraction may be wrong!")