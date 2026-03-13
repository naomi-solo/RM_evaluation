from src.data.load_bbq import load_bbq_pairs_all_categories
import random
import json
import os


# ============================================================
# 🔹 Helpers
# ============================================================

def _split_list_deterministic(items, holdout_size: int, seed: int):
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    holdout = items[:holdout_size]
    train = items[holdout_size:]
    return train, holdout


def _load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# ============================================================
# 🔹 MATH Loader
# ============================================================

def load_mc_pairs(
    *,
    subset: str,   # "gsm8k-mc" | "math-mc" | "pythonio-mc"
    data_dir: str,
    n_total: int,
    seed: int,
    split: str,
    holdout_size: int,
    include_instruction: bool = True,
):
    """
    GSM8K-MC and MATH-MC loader using MC-Evaluation dataset.
    First candidate is correct answer.
    """

    rng = random.Random(seed)

    # Path: MC-Evaluation/gsm8k-mc/train.jsonl
    path = os.path.join(data_dir, "data", f"{subset}", "train.jsonl")
    rows = _load_jsonl(path)

    # Deterministic split
    train_rows, holdout_rows = _split_list_deterministic(
        rows,
        holdout_size=holdout_size,
        seed=seed,
    )

    rows = holdout_rows if split == "holdout" else train_rows

    out = []

    for ex in rows:

        question = ex["Question"]
    
        # Collect choices in fixed order
        choices = [ex["A"], ex["B"], ex["C"], ex["D"]]
    
        correct_letter = ex["Answer"].strip().upper()
        letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    
        if correct_letter not in letter_to_idx:
            continue
    
        correct_idx = letter_to_idx[correct_letter]
    
        # Pick random wrong
        wrong_indices = [i for i in range(4) if i != correct_idx]
        rejected_idx = rng.choice(wrong_indices)
    
        correct = choices[correct_idx]
        rejected = choices[rejected_idx]
    
        prompt = question.strip()
        if include_instruction:
            prompt = f"{prompt}\n\nAnswer briefly."
    
        out.append(
            {
                "prompt": prompt,
                "chosen": str(correct),
                "rejected": str(rejected),
                "meta": {
                    "dataset": "gsm_mc",
                    "correct_idx": correct_idx,
                    "rejected_idx": rejected_idx,
                },
            }
        )
    
        if len(out) >= n_total:
            break

    return out

# ============================================================
# 🔹 MMLU Loader
# ============================================================

from datasets import load_dataset


def load_mmlu_pairs(
    *,
    n_total: int,
    seed: int,
    split: str,
    holdout_size: int,
    include_instruction: bool = True,
):
    rng = random.Random(seed)

    ds = load_dataset("cais/mmlu", "all", split="test")

    rows = [dict(ex) for ex in ds]

    train_rows, holdout_rows = _split_list_deterministic(
        rows,
        holdout_size=holdout_size,
        seed=seed,
    )

    rows = holdout_rows if split == "holdout" else train_rows

    out = []

    for ex in rows:
        question = ex["question"]
        choices = ex["choices"]
        correct_idx = int(ex["answer"])

        if not isinstance(choices, list) or len(choices) < 2:
            continue

        wrong_indices = [i for i in range(len(choices)) if i != correct_idx]
        rejected_idx = rng.choice(wrong_indices)

        prompt = question.strip()
        if include_instruction:
            prompt = f"{prompt}\n\nAnswer briefly."

        out.append(
            {
                "prompt": prompt,
                "chosen": str(choices[correct_idx]),
                "rejected": str(choices[rejected_idx]),
                "meta": {
                    "dataset": "mmlu",
                    "subject": ex.get("subject"),
                },
            }
        )

        if len(out) >= n_total:
            break

    return out

# ============================================================
# 🔹 SGXS Loader
# ============================================================


def load_sgxs_pairs(
    *,
    n_total: int,
    seed: int,
    split: str,
    holdout_size: int,
    include_instruction: bool = False,
):
    rng = random.Random(seed)

    ds = load_dataset("walledai/SGXSTest", split="train")

    rows = [dict(ex) for ex in ds]

    # Group by identical prompt pairs? No.
    # Dataset already alternates safe/unsafe rows.

    # Split deterministically
    train_rows, holdout_rows = _split_list_deterministic(
        rows,
        holdout_size=holdout_size,
        seed=seed,
    )

    rows = holdout_rows if split == "holdout" else train_rows

    out = []

    for ex in rows:

        prompt = ex["prompt"]
        label = ex["label"].strip().lower()

        if label not in ["safe", "unsafe"]:
            continue

        # For RM we need chosen vs rejected.
        if label == "safe":
            chosen = "safe"
            rejected = "unsafe"
        else:
            chosen = "unsafe"
            rejected = "safe"

        out.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "meta": {
                    "dataset": "sgxs",
                    "category": ex.get("category"),
                },
            }
        )

        if len(out) >= n_total:
            break

    return out
    
# ============================================================
# 🔹 Main Router
# ============================================================

def load_pairs(
    dataset_name: str,
    n_total: int,
    seed: int,
    split: str = "train",
    holdout_size: int = 100,
    **kwargs,
):
    split = split.lower()

    # ---------------------------------------------------
    # BBQ
    # ---------------------------------------------------
    if dataset_name == "bbq":

        pool_size = max(n_total + holdout_size + 50, 500)

        full_data = load_bbq_pairs_all_categories(
            n_total=pool_size,
            seed=seed,
            context_condition=kwargs.get("context_condition"),
            include_instruction=True,
            quiet=True,
        )

        train, holdout = _split_list_deterministic(
            full_data,
            holdout_size=holdout_size,
            seed=seed,
        )

        data = holdout if split == "holdout" else train
        return data[:n_total]

    # ---------------------------------------------------
    # MATH MC
    # ---------------------------------------------------
    if dataset_name in ["gsm_mc", "math_mc"]:

        data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../MC-Evaluation")
        )

        if not os.path.exists(data_dir):
            raise RuntimeError(
                f"MC-Evaluation dataset not found at {data_dir}. "
                "Expected it next to the repo in ~/lat/MC-Evaluation."
            )
    
        subset_map = {
            "gsm_mc": "gsm8k-mc",
            "math_mc": "math-mc",
        }
    
        return load_mc_pairs(
            subset=subset_map[dataset_name],
            data_dir=data_dir,
            n_total=n_total,
            seed=seed,
            split=split,
            holdout_size=holdout_size,
            include_instruction=True,
        )

    # ---------------------------------------------------
    # MMLU
    # ---------------------------------------------------
    if dataset_name == "mmlu":

        return load_mmlu_pairs(
            n_total=n_total,
            seed=seed,
            split=split,
            holdout_size=holdout_size,
            include_instruction=True,
        )

    # ---------------------------------------------------
    # SGXS
    # ---------------------------------------------------
    if dataset_name == "sgxs":

        return load_sgxs_pairs(
            n_total=n_total,
            seed=seed,
            split=split,
            holdout_size=holdout_size,
            include_instruction=False,
        )
    # ---------------------------------------------------
    # Unknown dataset
    # ---------------------------------------------------
    raise ValueError(f"Unknown dataset: {dataset_name}")