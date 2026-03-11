from hh_data import load_hh_pairs

def main():
    data = load_hh_pairs(split="harmless", n=30, seed=0)

    same = 0
    checked = 0

    for i, ex in enumerate(data):
        c = ex["chosen"].strip()
        r = ex["rejected"].strip()

        checked += 1
        if c == r:
            same += 1

        # print a few where they're the same / different
        if i < 5 or c == r:
            print("\n" + "="*70)
            print(f"Example {i}")
            print("="*70)
            print("PROMPT:", ex["prompt"][:150].replace("\n", "\\n"))
            print("\nCHOSEN (first 200):", c[:200].replace("\n", "\\n"))
            print("\nREJECTED (first 200):", r[:200].replace("\n", "\\n"))
            print("\nchosen==rejected?", c == r)

    print(f"\nSame chosen/rejected: {same}/{checked} = {same/checked:.3f}")

if __name__ == "__main__":
    main()
