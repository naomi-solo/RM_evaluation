import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import os
import torch
from extract_directions import run_one


def main():
    os.makedirs("results/directions", exist_ok=True)

    # Tiny run (2 pairs -> 4 records)
    run_one(sign_flip=True, context_condition="ambig", n_total=2, perturbation_type="pgd")

    # Find the expected output file
    files = [f for f in os.listdir("results/directions") if "ambig_flip_n2" in f and f.startswith("direction_records_")]
    assert files, "No direction_records file found in results/directions/ (did extract save there?)"

    path = os.path.join("results/directions", sorted(files)[-1])
    obj = torch.load(path, map_location="cpu")

    assert "records" in obj, "Missing 'records' in saved file"
    records = obj["records"]
    assert len(records) == 4, f"Expected 4 records, got {len(records)}"

    # Check record fields + tensor properties
    required = [
        "schema_version",
        "prompt",
        "completion",
        "completion_type",
        "reward_unperturbed",
        "reward_perturbed",
        "perturbation_direction",
        "perturbation_type",
        "epsilon",
        "pair_id",
        "layer",
        "context_condition",
    ]
    for k in required:
        assert k in records[0], f"Missing key {k} in record"

    types = {r["completion_type"] for r in records}
    assert types == {"chosen", "rejected"}, f"Expected chosen+rejected types, got {types}"

    v = records[0]["perturbation_direction"]
    assert isinstance(v, torch.Tensor) and v.dim() == 1
    assert v.device.type == "cpu"
    assert v.dtype == torch.float16

    print("OK extract_directions")
    print("file:", path)
    print("n_records:", len(records), "D:", v.numel(), "types:", types)


if __name__ == "__main__":
    main()
