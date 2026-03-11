import os
import glob
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def main():
    # Find *any* chosen PCA file for ambig_flip_n2
    pca_paths = glob.glob(os.path.join(ROOT, "results/pca/pca_layer*_eps*_k*_ambig_flip_chosen_n2.pt"))
    assert pca_paths, "No PCA file found in results/pca/. Run test_pca_directions_small.py first."
    pca_path = sorted(pca_paths)[-1]

    # Find *any* matching labels file (K can vary)
    label_paths = glob.glob(os.path.join(ROOT, "results/labels/component_labels_layer*_eps*_K*_ambig_flip_chosen.jsonl"))
    assert label_paths, "No labels JSONL found in results/labels/. Run test_label_components_small.py first."
    labels_path = sorted(label_paths)[-1]

    env = os.environ.copy()
    env["PCA_PATH"] = pca_path
    env["LABELS_PATH"] = labels_path
    env["N_PRINT"] = "2"
    env["TOPK"] = "3"

    subprocess.check_call([sys.executable, os.path.join(ROOT, "summarize_component_contributions.py")], env=env)

    print("OK summary script")
    print("used PCA:", pca_path)
    print("used labels:", labels_path)

if __name__ == "__main__":
    main()
