import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import glob
import torch
import subprocess


def main():
    os.makedirs("results/pca", exist_ok=True)

    # Find the tiny directions file created by test_extract_directions_small
    paths = glob.glob("results/directions/direction_records_*_ambig_flip_n2.pt")
    assert paths, "No directions file found. Run test_extract_directions_small.py first."
    in_path = sorted(paths)[-1]

    obj = torch.load(in_path, map_location="cpu")
    layer = obj["layer"]
    eps = obj["epsilon"]
    n_pairs = obj["n_pairs"]

    # Run PCA script for chosen, rejected, both
    for mode in ["chosen", "rejected", "both"]:
        env = os.environ.copy()
        env["LAYER"] = str(layer)
        env["EPS"] = str(eps)
        env["N"] = str(n_pairs)
        env["K"] = "3"
        env["MODE"] = mode

        # run pca_directions.py as a subprocess
        subprocess.check_call([sys.executable, "pca_directions.py"], env=env)

    # Check PCA outputs exist
    # chosen/rejected: N should be n_pairs
    # both: N should be 2*n_pairs
    expected = [
        f"results/pca/pca_layer{layer}_eps{eps}_k2_ambig_flip_chosen_n{n_pairs}.pt",
        f"results/pca/pca_layer{layer}_eps{eps}_k2_ambig_flip_rejected_n{n_pairs}.pt",
        f"results/pca/pca_layer{layer}_eps{eps}_k3_ambig_flip_both_n{2*n_pairs}.pt",
    ]

    for p in expected:
        assert os.path.exists(p), f"Missing PCA output: {p}"

        o = torch.load(p, map_location="cpu")
        assert "components" in o and "scores" in o and "records" in o
        assert o["scores"].dim() == 2
        assert len(o["records"]) == o["scores"].shape[0]

    print("OK pca_directions")
    print("generated:", "\n  " + "\n  ".join(expected))


if __name__ == "__main__":
    main()
