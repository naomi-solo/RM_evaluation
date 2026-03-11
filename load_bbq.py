# bbq_to_pairs.py
from __future__ import annotations

import random
from typing import Dict, List, Optional

from datasets import load_dataset, concatenate_datasets
from datasets import disable_progress_bar


# All category folder names present in heegyu/bbq parquet branch
ALL_CATEGORIES = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Race_x_SES",
    "Race_x_gender",
    "Religion",
    "SES",
    "Sexual_orientation",
]


def _load_one_category_parquet(category: str):
    path = f"hf://datasets/heegyu/bbq@refs/convert/parquet/{category}/test/0000.parquet"
    return load_dataset("parquet", data_files=path, split="train")


def _get_answers(ex: Dict) -> List[str]:
    return [ex["ans0"], ex["ans1"], ex["ans2"]]


def _get_answer_groups(ex: Dict) -> Optional[Dict[int, str]]:
    """
    Try to extract per-answer group tags from ex["answer_info"].

    Example:
      "answer_info": {
        "ans0": ["grandfather", "old"],
        "ans1": ["grandson", "nonOld"],
        "ans2": ["Can't be determined", "unknown"]
      }
    Returns {0:"old", 1:"nonOld", 2:"unknown"} or None if missing.
    """
    ai = ex.get("answer_info")
    if not isinstance(ai, dict):
        return None

    out = {}
    for i, k in enumerate(["ans0", "ans1", "ans2"]):
        v = ai.get(k)
        if isinstance(v, (list, tuple)) and len(v) >= 2 and isinstance(v[1], str):
            out[i] = v[1]
        else:
            return None
    return out


def _get_bias_target_group(ex: Dict) -> Optional[str]:
    """
    Try to read bias-target group from additional_metadata.stereotyped_groups[0]
    Example: ["old"] -> "old"
    """
    md = ex.get("additional_metadata")
    if not isinstance(md, dict):
        return None
    sg = md.get("stereotyped_groups")
    if isinstance(sg, list) and sg and isinstance(sg[0], str):
        return sg[0]
    return None


def _pick_rejected_ambig(ex: Dict, chosen_idx: int, rng: random.Random) -> int:
    """
    Ambiguous: best rejected for bias directions = bias-target PERSON if inferable.
    Otherwise pick any PERSON wrong answer.
    """
    groups = _get_answer_groups(ex)
    bias_target = _get_bias_target_group(ex)

    wrong = [i for i in range(3) if i != chosen_idx]

    # Prefer bias-target person (not "unknown") if metadata supports it
    if groups is not None and bias_target is not None:
        for i in wrong:
            if groups.get(i) == bias_target and groups.get(i) != "unknown":
                return i

    # Otherwise prefer person (non-unknown)
    if groups is not None:
        person_wrong = [i for i in wrong if groups.get(i) != "unknown"]
        if person_wrong:
            return rng.choice(person_wrong)

    # Fallback: random wrong
    return rng.choice(wrong)


def _pick_rejected_disambig(ex: Dict, chosen_idx: int) -> int:
    """
    Disambiguated: best rejected for bias directions = the OTHER PERSON (never UNKNOWN).
    """
    groups = _get_answer_groups(ex)
    wrong = [i for i in range(3) if i != chosen_idx]

    if groups is not None:
        person_wrong = [i for i in wrong if groups.get(i) != "unknown"]
        if person_wrong:
            # should be exactly one other person
            return person_wrong[0]

    # Fallback heuristic: avoid "unknown-ish" strings
    answers = _get_answers(ex)
    def looks_unknown(s: str) -> bool:
        s2 = (s or "").strip().lower()
        return any(x in s2 for x in ["unknown", "can't be determined", "cannot be determined", "undetermined", "not known"])

    non_unknown_wrong = [i for i in wrong if not looks_unknown(answers[i])]
    if non_unknown_wrong:
        return non_unknown_wrong[0]

    return wrong[0]

def load_bbq_pairs_all_categories(
    n_total: int = 200,
    seed: int = 0,
    context_condition: Optional[str] = None,  # "ambig" or "disambig" or None
    include_instruction: bool = True,
    quiet: bool = True,
) -> List[Dict]:
    """
    Quiet + faster version: no giant concat + repeated filter spam.
    """
    if quiet:
        disable_progress_bar()

    rng = random.Random(seed)

    # Balance per category
    k = len(ALL_CATEGORIES)
    n_per = max(1, n_total // k)

    cc = context_condition.strip().lower() if context_condition else None

    per_cat_rows: Dict[str, List[Dict]] = {}
    for cat in ALL_CATEGORIES:
        ds = _load_one_category_parquet(cat)

        # Only add category col if missing (parquet often already has it)
        if "category" not in ds.column_names:
            ds = ds.add_column("category", [cat] * len(ds))

        # Filter once, inside the shard
        if cc is not None:
            ds = ds.filter(lambda ex: str(ex.get("context_condition", "")).strip().lower() == cc)

        # Shuffle and take up to n_per (+ a little extra for top-up)
        ds = ds.shuffle(seed=seed)

        take = min(len(ds), n_per + 5)  # small cushion to help fill n_total without re-filtering
        ds = ds.select(range(take))

        per_cat_rows[cat] = list(ds)

    pairs: List[Dict] = []

    # First pass: exactly n_per from each category (if available)
    for cat in ALL_CATEGORIES:
        rows = per_cat_rows.get(cat, [])
        for ex in rows[: min(n_per, len(rows))]:
            context = ex["context"]
            question = ex["question"]
            answers = _get_answers(ex)

            label = int(ex["label"])
            chosen_idx = label
            chosen = answers[chosen_idx]

            cc_ex = str(ex.get("context_condition", "")).strip().lower()
            if cc_ex == "ambig":
                rej_idx = _pick_rejected_ambig(ex, chosen_idx, rng)
            else:  # disambig (or anything else)
                rej_idx = _pick_rejected_disambig(ex, chosen_idx)

            rejected = answers[rej_idx]

            prompt = (
                f"{context}\n\nQuestion: {question}\n\nAnswer briefly."
                if include_instruction
                else f"{context}\n\nQuestion: {question}"
            )

            pairs.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "meta": {
                        "category": ex.get("category", cat),
                        "context_condition": ex.get("context_condition"),
                        "question_polarity": ex.get("question_polarity"),
                        "label": label,
                        "chosen_idx": chosen_idx,
                        "rejected_idx": rej_idx,
                        "example_id": ex.get("example_id"),
                        "question_index": ex.get("question_index"),
                    },
                }
            )

    # Top up to reach n_total from the extra cushion rows
    if len(pairs) < n_total:
        need = n_total - len(pairs)
        cats = ALL_CATEGORIES[:]
        rng.shuffle(cats)

        for cat in cats:
            if need <= 0:
                break
            rows = per_cat_rows.get(cat, [])
            extras = rows[n_per:]  # from the cushion
            for ex in extras:
                if need <= 0:
                    break

                context = ex["context"]
                question = ex["question"]
                answers = _get_answers(ex)

                label = int(ex["label"])
                chosen_idx = label
                chosen = answers[chosen_idx]

                cc_ex = str(ex.get("context_condition", "")).strip().lower()
                if cc_ex == "ambig":
                    rej_idx = _pick_rejected_ambig(ex, chosen_idx, rng)
                else:
                    rej_idx = _pick_rejected_disambig(ex, chosen_idx)

                rejected = answers[rej_idx]
                prompt = (
                    f"{context}\n\nQuestion: {question}\n\nAnswer briefly."
                    if include_instruction
                    else f"{context}\n\nQuestion: {question}"
                )

                pairs.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "meta": {
                            "category": ex.get("category", cat),
                            "context_condition": ex.get("context_condition"),
                            "question_polarity": ex.get("question_polarity"),
                            "label": label,
                            "chosen_idx": chosen_idx,
                            "rejected_idx": rej_idx,
                            "example_id": ex.get("example_id"),
                            "question_index": ex.get("question_index"),
                        },
                    }
                )
                need -= 1

    return pairs[:n_total]
