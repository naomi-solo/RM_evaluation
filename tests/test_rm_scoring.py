from hh_data import load_hh_pairs
from rm_utils import load_rm_and_tokenizer, score_pair

MODEL = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"

def main():
    tokenizer, model, device = load_rm_and_tokenizer(MODEL, device="cuda")

    data = load_hh_pairs(split="harmless", n=200, seed=0)

    correct = 0
    margins = []

    for i, ex in enumerate(data):
        s_c, s_r = score_pair(
            tokenizer, model,
            ex["prompt"], ex["chosen"], ex["rejected"],
            device=device
        )
        m = s_c - s_r
        margins.append(m)
        correct += int(s_c > s_r)
        # print(f"{i:02d} | chosen={s_c:+.4f} rejected={s_r:+.4f} margin={m:+.4f}")

    print("\n" + "-" * 70)
    print(f"Baseline accuracy: {correct}/{len(data)} = {correct/len(data):.3f}")
    print(f"Mean margin: {sum(margins)/len(margins):+.4f}")
    print(f"Min/Max margin: {min(margins):+.4f} / {max(margins):+.4f}")

if __name__ == "__main__":
    main()
