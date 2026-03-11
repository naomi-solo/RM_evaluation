#!/usr/bin/env python3
"""
run_label_all_pca.py

Find all PCA outputs in results/pca/*.pt, parse metadata from filenames,
and run label_components.py with the correct env vars for each file.

This only does labeling (cheap). It does NOT rerun extraction or PCA.

Usage:
  python run_label_all_pca.py
  python run_label_all_pca.py --glob "results/pca/*.pt" --model "gpt-4o-mini"
  python run_label_all_pca.py --dry_run
"""

import argparse
import glob
import os
import re
import subprocess
from typing import Optional, Dict

PCA_PAT = re.compile(
    r"pca_seed(?P<seed>\d+)_layer(?P<layer>\d+)_eps(?P<eps>[\d\.]+)_k(?P<k>\d+)_(?P<cc>\w+)_(?P<tag>\w+)_(?P<mode>\w+)_n(?P<n>\d+)\.pt$"
)

def parse_pca_filename(path: str) -> Optional[Dict[str, str]]:
    m = PCA_PAT.match(os.path.basename(path))
    if not m:
        return None
    d = m.groupdict()
    # These map directly to what label_components.py reads:
    # SEED, LAYER, EPS, K, CC, TAG, MODE, N
    return {
        "SEED": d["seed"],
        "LAYER": d["layer"],
        "EPS": d["eps"],
        "K": d["k"],
        "CC": d["cc"],
        "TAG": d["tag"],
        "MODE": d["mode"],
        "N": d["n"],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/pca/*.pt")
    ap.add_argument("--label_script", default="label_components.py")
    ap.add_argument("--model", default="gpt-4o-mini", help="Passed via env LABEL_MODEL to label_components.py")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip if the expected labels jsonl already exists.")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set in this environment.")

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"No PCA files matched: {args.glob}")

    ok = 0
    skip = 0
    bad = 0

    for p in paths:
        meta = parse_pca_filename(p)
        if meta is None:
            print(f"[skip] filename didn't match pattern: {p}")
            skip += 1
            continue

        # This matches how label_components.py constructs pca_path internally.
        # Label output path pattern in label_components.py:
        # results/labels/component_labels_seed{seed}_layer{layer}_eps{eps}_K{K}_{cc}_{tag}_{mode}_n{len(records)}.jsonl
        # In that script it uses n_default (from env N) for both PCA path and output.
        out_jsonl = (
            f"results/labels/component_labels_seed{meta['SEED']}_layer{meta['LAYER']}_"
            f"eps{meta['EPS']}_K{meta['K']}_{meta['CC']}_{meta['TAG']}_{meta['MODE']}_n{meta['N']}.jsonl"
        )

        if args.skip_existing and os.path.exists(out_jsonl):
            print(f"[skip existing] {os.path.basename(out_jsonl)}")
            skip += 1
            continue

        env = os.environ.copy()
        env.update(meta)
        env["LABEL_MODEL"] = args.model  # (optional) you can read this in label_components.py if you want

        cmd = ["python", args.label_script]

        print(f"\n=== Labeling: {os.path.basename(p)}")
        print("ENV:", " ".join([f"{k}={meta[k]}" for k in ["SEED","LAYER","EPS","K","CC","TAG","MODE","N"]]))
        print("OUT:", out_jsonl)

        if args.dry_run:
            ok += 1
            continue

        proc = subprocess.run(cmd, env=env, text=True)
        if proc.returncode == 0:
            ok += 1
        else:
            bad += 1
            print(f"[error] labeling failed for {p} (rc={proc.returncode})")

    print("\nDone.")
    print("OK:", ok)
    print("Skipped:", skip)
    print("Failed:", bad)

if __name__ == "__main__":
    main()
