# pca_directions.py
import os
import torch


def pca_torch(X: torch.Tensor, k: int = 10):
    """
    PCA via SVD on centered data.
    X: (N, D) float tensor on CPU
    Returns:
      components: (k, D)
      scores: (N, k)
      explained_var_ratio: (k,)
      mean: (D,)
    """
    X = X.float()
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean

    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)

    components = Vh[:k, :]          # (k, D)
    scores = Xc @ components.T      # (N, k)

    eigvals = (S**2) / max(1, (Xc.shape[0] - 1))
    total = eigvals.sum()
    explained = eigvals[:k] / (total + 1e-12)

    return components, scores, explained, mean.squeeze(0)


def _preview_record(rec: dict) -> str:
    p = (rec.get("prompt", "") or "").replace("\n", " ").strip()
    c = (rec.get("completion", "") or "").replace("\n", " ").strip()
    ct = rec.get("completion_type", "")
    r0 = rec.get("reward_unperturbed", None)
    r1 = rec.get("reward_perturbed", None)

    # keep it short but informative
    head = f"[{ct}] clean={r0:.3f} adv={r1:.3f}" if isinstance(r0, float) and isinstance(r1, float) else f"[{ct}]"
    s = f"{head} | {p} || {c}"
    return s[:110]


def _select_records(records: list, mode: str) -> list:
    if mode not in ("chosen", "rejected", "both"):
        raise ValueError(f"MODE must be chosen|rejected|both, got {mode}")

    if mode == "both":
        use = records
    else:
        use = [r for r in records if r.get("completion_type") == mode]

    if len(use) == 0:
        raise RuntimeError(f"No records selected for MODE={mode}. Check your input file.")
    return use

os.makedirs("results/pca", exist_ok=True)

def main():
    cc_env = os.getenv("CC", None)      # "ambig" | "disambig"
    tag_env = os.getenv("TAG", None)    # "flip" | "noflip"

    seed_default = int(os.getenv("SEED", "0"))

    # Settings (match extract_directions.py)
    layer_default = int(os.getenv("LAYER", "14"))
    eps_default = float(os.getenv("EPS", "8.0"))
    n_default = int(os.getenv("N", "200"))
    k_default = int(os.getenv("K", "10"))
    mode = os.getenv("MODE", "chosen")  # chosen|rejected|both

    cc = cc_env if cc_env is not None else "ambig"
    tag = tag_env if tag_env is not None else "flip"
    
    in_path = f"results/directions/direction_records_seed{seed_default}_layer{layer_default}_eps{eps_default}_{cc}_{tag}_n{n_default}.pt"
    if not os.path.exists(in_path):
        print("missing (skipping):", in_path)
        return
    obj = torch.load(in_path, map_location="cpu")
    if "records" not in obj:
        raise KeyError(f"Expected 'records' in {in_path}. Did you run the new extractor?")
    records_all = obj["records"]
    records = _select_records(records_all, mode=mode)

    X = torch.stack([r["perturbation_direction"] for r in records], dim=0)  # (N, D)
    ids = [int(r.get("pair_id", i)) for i, r in enumerate(records)]

    layer = int(obj.get("layer", layer_default))
    eps = float(obj.get("epsilon", eps_default))

    k_eff = min(k_default, X.shape[0], X.shape[1])
    comps, scores, evr, mean = pca_torch(X, k=k_eff)
    out_path = f"results/pca/pca_seed{seed_default}_layer{layer}_eps{eps}_k{k_eff}_{cc}_{tag}_{mode}_n{X.shape[0]}.pt"
    torch.save(
        {
            "components": comps.half(),
            "scores": scores.float(),
            "explained_var_ratio": evr.float(),
            "mean": mean.float(),
            "records": records,   # keep full records for labeling/inspection
            "ids": ids,
            "layer": layer,
            "epsilon": eps,
            "in_path": in_path,
            "context_condition": cc,
            "tag": tag,
            "mode": mode,
            "requested_k": int(k_default),
            "k": int(k_eff),

        },
        out_path,
    )
    print("loaded:", in_path)
    print("saved:", out_path)
    print("explained_var_ratio:", evr.tolist())


    # quick peek: top examples for component 0
    topk = 8
    k0 = 0
    vals, idx = torch.topk(scores[:, k0], k=min(topk, scores.shape[0]))
    print(f"\n[{cc} | {tag} | mode={mode}] Top {len(idx)} examples for component {k0}:")
    for rank, (v, j) in enumerate(zip(vals.tolist(), idx.tolist()), 1):
        print(f"{rank:02d} score={v:+.4f} id={ids[j]}  {_preview_record(records[j])}")
    print()


if __name__ == "__main__":
    main()
