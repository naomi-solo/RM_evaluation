import numpy as np
from hh_data import load_hh_pairs
from rm_utils import load_rm_and_tokenizer, score_pair

MODEL = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"

def main():
    tok, rm, dev = load_rm_and_tokenizer(MODEL, device="cuda")
    data = load_hh_pairs(split="harmless", n=500, seed=0)

    margins = []
    for ex in data:
        sc, sr = score_pair(tok, rm, ex["prompt"], ex["chosen"], ex["rejected"], device=dev)
        margins.append(sc - sr)

    margins = np.array(margins)
    acc = (margins > 0).mean()

    for thr in [0.05, 0.1, 0.25, 0.5, 1.0]:
        near = (np.abs(margins) < thr).mean()
        print(f"near-tie |m|<{thr:>4}: {near:.3f}")

    print("\n--- summary ---")
    print(f"n={len(margins)}  accuracy={acc:.3f}")
    print(f"mean margin={margins.mean():+.3f}")
    print(f"median margin={np.median(margins):+.3f}")
    print(f"10/90 pct={np.quantile(margins,0.1):+.3f} / {np.quantile(margins,0.9):+.3f}")

if __name__ == "__main__":
    main()
