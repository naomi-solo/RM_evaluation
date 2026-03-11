import os
import glob
import json
import torch
import subprocess
import sys

# Make repo root importable (if needed later)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

def main():
    os.makedirs("results/labels", exist_ok=True)

    # Use the chosen PCA file produced by the PCA test (note: k can be 2 here)
    pca_paths = glob.glob("results/pca/pca_layer*_eps*_k*_ambig_flip_chosen_n2.pt")
    assert pca_paths, "No chosen PCA file found. Run test_pca_directions_small.py first."
    pca_path = sorted(pca_paths)[-1]

    obj = torch.load(pca_path, map_location="cpu")
    layer = int(obj["layer"])
    eps = float(obj["epsilon"])
    n_default = int(obj["scores"].shape[0])     # should be 2
    k_default = int(obj["scores"].shape[1])     # should be 2 (effective k)
    mode = obj.get("mode", "chosen")

    env = os.environ.copy()
    env["LAYER"] = str(layer)
    env["EPS"] = str(eps)
    env["N"] = str(n_default)
    env["K"] = str(k_default)
    env["MODE"] = str(mode)
    env.pop("OPENAI_API_KEY", None)  # force manual mode

    subprocess.check_call([sys.executable, "label_components.py"], env=env)

    out = f"results/labels/component_labels_layer{layer}_eps{eps}_K{k_default}_ambig_flip_{mode}.jsonl"
    assert os.path.exists(out), f"Missing labels output: {out}"

    # Validate JSONL has one line per component
    with open(out) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert len(lines) == k_default, f"Expected {k_default} lines, got {len(lines)}"

    # Basic JSON validation
    row0 = json.loads(lines[0])
    assert "component" in row0 and "label" in row0
    assert row0["label"] in ("MANUAL", "PARSE_ERROR") or isinstance(row0["label"], str)

    print("OK label_components")
    print("used PCA:", pca_path)
    print("wrote:", out)

if __name__ == "__main__":
    main()
