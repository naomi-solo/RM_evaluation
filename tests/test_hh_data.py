from hh_data import load_hh_pairs

def main():
    data = load_hh_pairs(split="harmless", n=5, seed=0)

    for i, ex in enumerate(data):
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        print("\n" + "=" * 70)
        print(f"Example {i}")
        print("=" * 70)
        print("PROMPT (first 200 chars):")
        print(prompt[:200].replace("\n", "\\n"))

        # Sanity checks
        assert "Assistant:" not in prompt, "Prompt still contains 'Assistant:'"
        assert len(prompt) > 0, "Empty prompt"
        assert len(chosen) > 0, "Empty chosen response"
        assert len(rejected) > 0, "Empty rejected response"

        # Often chosen/rejected are different; not guaranteed but usually true
        if chosen.strip() == rejected.strip():
            print("⚠️ chosen == rejected (rare but possible)")

        print("\nCHOSEN (first 120 chars):")
        print(chosen[:120].replace("\n", "\\n"))
        print("\nREJECTED (first 120 chars):")
        print(rejected[:120].replace("\n", "\\n"))

    print("\n✅ hh_data looks sane.")

if __name__ == "__main__":
    main()
