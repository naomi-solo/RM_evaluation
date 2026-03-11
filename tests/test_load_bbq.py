from collections import Counter
import re

from datasets import load_dataset

from load_bbq import load_bbq_pairs_all_categories, ALL_CATEGORIES

UNKNOWN_RE = re.compile(r"\b(unknown|can't be determined|cannot be determined|undetermined|not known)\b", re.I)

def looks_unknown(s: str) -> bool:
    return bool(UNKNOWN_RE.search((s or "").strip()))

def load_raw_index_for_context(context_condition: str):
    """
    Load all category parquet shards (same as loader), filter to context_condition,
    and build a lookup:
      (category, example_id, question_index) -> raw row dict
    """
    cc = context_condition.strip().lower()
    index = {}

    for cat in ALL_CATEGORIES:
        path = f"hf://datasets/heegyu/bbq@refs/convert/parquet/{cat}/test/0000.parquet"
        ds = load_dataset("parquet", data_files=path, split="train")
        # ensure category matches folder
        # (some parquet files already have category column, but keep stable)
        # filter to requested context_condition
        ds = ds.filter(lambda ex: str(ex.get("context_condition", "")).strip().lower() == cc)

        for ex in ds:
            key = (cat, int(ex["example_id"]), int(ex["question_index"]))
            index[key] = ex

    return index

def get_bias_target_answer_idx(raw_ex):
    """
    Return the index (0/1/2) of the bias-target PERSON answer if uniquely determinable.
    Otherwise return None.

    Uses:
      raw_ex["additional_metadata"]["stereotyped_groups"][0]  -> target group (e.g., "old")
      raw_ex["answer_info"]["ans{i}"][1]                      -> group tag per answer (e.g., "old", "nonOld", "unknown")
    """
    md = raw_ex.get("additional_metadata")
    ai = raw_ex.get("answer_info")

    if not isinstance(md, dict) or not isinstance(ai, dict):
        return None

    sg = md.get("stereotyped_groups")
    if not (isinstance(sg, list) and len(sg) > 0 and isinstance(sg[0], str)):
        return None
    target_group = sg[0]

    groups = {}
    for i, k in enumerate(["ans0", "ans1", "ans2"]):
        v = ai.get(k)
        if not (isinstance(v, (list, tuple)) and len(v) >= 2 and isinstance(v[1], str)):
            return None
        groups[i] = v[1]

    # candidates: answers whose group tag equals target_group AND are not unknown
    candidates = [i for i, g in groups.items() if g == target_group and g != "unknown"]

    if len(candidates) == 1:
        return candidates[0]
    return None

def answer_text_to_idx(raw_ex, text: str):
    """
    Map an answer string to ans index 0/1/2 by exact match.
    Returns None if not found or ambiguous.
    """
    answers = [raw_ex["ans0"], raw_ex["ans1"], raw_ex["ans2"]]
    matches = [i for i, a in enumerate(answers) if a == text]
    if len(matches) == 1:
        return matches[0]
    return None

def test_split(context_condition: str, n_total: int = 220, seed: int = 0):
    pairs = load_bbq_pairs_all_categories(
        n_total=n_total,
        seed=seed,
        context_condition=context_condition,
        include_instruction=True,
    )

    assert len(pairs) == n_total, f"Expected {n_total} pairs, got {len(pairs)}"

    # 1) Verify all examples are of the requested context_condition
    cc = [p["meta"]["context_condition"] for p in pairs]
    bad = [x for x in cc if (str(x).strip().lower() != context_condition)]
    assert len(bad) == 0, f"Found {len(bad)} examples not in context_condition={context_condition}"

    # 2) Verify categories are present and ~balanced
    cats = [p["meta"]["category"] for p in pairs]
    c = Counter(cats)
    print(f"\n[{context_condition}] Category counts:", dict(c))

    k = len(ALL_CATEGORIES)
    base = n_total // k
    slack = 3  # top-up and uneven availability can skew slightly
    for cat in ALL_CATEGORIES:
        if cat in c:
            assert abs(c[cat] - base) <= slack, f"Category {cat} count {c[cat]} far from base {base}"

    # 3) Verify prompt is RM-style (not MC)
    for p in pairs[:20]:
        prompt = p["prompt"]
        assert "Question:" in prompt, "Prompt missing 'Question:'"
        assert "Options:" not in prompt, "Prompt accidentally contains multiple-choice formatting"
        assert "Answer briefly." in prompt, "Prompt missing instruction 'Answer briefly.'"

    # 4) Verify chosen matches label index (by stored indices)
    for p in pairs:
        m = p["meta"]
        assert m["chosen_idx"] == m["label"], "chosen_idx != label (chosen not based on label)"
        assert p["chosen"] != p["rejected"], "chosen == rejected"

    # 5) Basic rejected rules (heuristic unknown detection)
    unknown_rej = sum(looks_unknown(p["rejected"]) for p in pairs)
    rate = unknown_rej / len(pairs)
    print(f"[{context_condition}] rejected looks UNKNOWN rate: {rate:.3%} ({unknown_rej}/{len(pairs)})")
    assert rate < 0.02, f"Too many rejected answers look UNKNOWN in {context_condition}: {rate:.3%}"

    # 6) Stronger disambig-only check: chosen should usually not be UNKNOWN
    if context_condition == "disambig":
        unknown_chosen = sum(looks_unknown(p["chosen"]) for p in pairs)
        rate_c = unknown_chosen / len(pairs)
        print(f"[disambig] chosen looks UNKNOWN rate: {rate_c:.3%} ({unknown_chosen}/{len(pairs)})")
        assert rate_c < 0.02, f"Too many chosen answers look UNKNOWN in disambig: {rate_c:.3%}"

    # 7) NEW: Ambig bias-target rejected check (when metadata supports a unique target person)
    if context_condition == "ambig":
        raw_index = load_raw_index_for_context("ambig")

        checked = 0
        enforced = 0
        violations = 0
        missing = 0

        for p in pairs:
            m = p["meta"]
            key = (m["category"], int(m["example_id"]), int(m["question_index"]))
            raw_ex = raw_index.get(key)
            if raw_ex is None:
                missing += 1
                continue

            target_idx = get_bias_target_answer_idx(raw_ex)
            if target_idx is None:
                checked += 1
                continue  # cannot uniquely determine target person

            rej_idx = answer_text_to_idx(raw_ex, p["rejected"])
            if rej_idx is None:
                checked += 1
                continue  # ambiguous mapping; skip hard assert

            enforced += 1
            if rej_idx != target_idx:
                violations += 1

            checked += 1

        print(
            f"[ambig] bias-target enforcement: enforced={enforced}, "
            f"violations={violations}, missing_raw={missing}, total_pairs={len(pairs)}"
        )

        # We expect zero violations when enforcement applies.
        # If this fails, your ambig rejection picker isn't choosing the bias-target person.
        assert violations == 0, f"Found {violations} ambig examples where rejected != bias-target person (when determinable)."

        # Also ensure we actually tested a meaningful number of cases
        assert enforced >= 20, (
            f"Only enforced {enforced} bias-target checks; too low. "
            f"Try increasing n_total or check metadata parsing."
        )

    # Show a few examples
    print(f"\n[{context_condition}] Sample rows:")
    for p in pairs[:3]:
        print("-" * 80)
        print("category:", p["meta"]["category"], "| polarity:", p["meta"]["question_polarity"])
        print("context_condition:", p["meta"]["context_condition"])
        print("chosen:", p["chosen"])
        print("rejected:", p["rejected"])
        print("prompt:\n", p["prompt"][:350])

    return True


def main():
    test_split("ambig", n_total=330, seed=0)    # bigger n_total helps enforcement coverage
    test_split("disambig", n_total=330, seed=0)
    print("\n✅ bbq_to_pairs tests passed for ambig + disambig (including bias-target checks).")


if __name__ == "__main__":
    main()
