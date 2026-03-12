# synthetic_data.py
#
# Clean synthetic dataset loader.
# EXACT SAME interface + output format as HH loader:
#   load_pairs(...) -> List[{"prompt","chosen","rejected"}]

from __future__ import annotations

from typing import Dict, List, Optional
import json
import random
from pathlib import Path


def load_pairs(
    split: str = "harmless",   # kept for API compatibility; ignored
    n: Optional[int] = None,
    seed: int = 0,
    shuffle: bool = True,
    path: str = "data/synthetic_hh_like.jsonl",
) -> List[Dict[str, str]]:
    """
    Load synthetic pairwise preference data.

    Args:
      split: ignored (kept so caller code doesn't change)
      n: optionally limit number of examples
      seed: random seed for shuffling
      shuffle: shuffle before truncation
      path: path to JSONL or JSON file

    Returns:
      List of dicts with keys: prompt, chosen, rejected
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Synthetic dataset not found: {path}")

    # Read file
    data: List[Dict[str, str]] = []
    if path.suffix == ".jsonl":
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                if not all(k in ex for k in ("prompt", "chosen", "rejected")):
                    raise KeyError(f"Example missing keys: {ex.keys()}")
                data.append({
                    "prompt": ex["prompt"].strip(),
                    "chosen": ex["chosen"].strip(),
                    "rejected": ex["rejected"].strip(),
                })
    elif path.suffix == ".json":
        with path.open("r") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise ValueError("Top-level JSON must be a list")
        for ex in items:
            if not all(k in ex for k in ("prompt", "chosen", "rejected")):
                raise KeyError(f"Example missing keys: {ex.keys()}")
            data.append({
                "prompt": ex["prompt"].strip(),
                "chosen": ex["chosen"].strip(),
                "rejected": ex["rejected"].strip(),
            })
    else:
        raise ValueError("Dataset must be .jsonl or .json")

    # Shuffle / truncate
    idxs = list(range(len(data)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(idxs)
    if n is not None:
        idxs = idxs[:n]

    return [data[i] for i in idxs]


if __name__ == "__main__":
    # quick sanity check
    exs = load_pairs(n=3)
    for i, ex in enumerate(exs):
        print(f"\nExample {i}")
        print("PROMPT:", ex["prompt"])
        print("CHOSEN:", ex["chosen"])
        print("REJECTED:", ex["rejected"])
