#!/usr/bin/env python3
"""Preprocess raw dataset CSVs into the cached splits used by every run.

python scripts/preprocess.py --dataset cicids2017 edge_iiotset ciciot2023
       [--raw PATH ...] [--root data] [--taxonomy family|fine] [--max-rows 600000]
       [--class-floor 10000] [--min-class-count 100] [--max-files N] [--split-seed 0]

Raw files: <root>/<name>/raw/**/*.csv, or --raw with files / folders / zip
archives (one dataset per call). Sources are read in place, never modified;
their paths and SHA-256 go into the metadata. Output: <root>/<name>/processed/.
Then: python scripts/make_dataset_cards.py and python scripts/leakage_audit.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.preprocess import DEFAULT_PARAMS, SPECS, preprocess

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", required=True, choices=sorted(SPECS))
    ap.add_argument("--raw", nargs="+", default=None, help="raw CSV files, folders or zip archives")
    ap.add_argument("--note", default=None, help="provenance note recorded in the dataset card")
    ap.add_argument("--root", default="data")
    ap.add_argument("--taxonomy", default=DEFAULT_PARAMS["taxonomy"], choices=["family", "fine"])
    ap.add_argument("--max-rows", type=int, default=DEFAULT_PARAMS["max_rows"])
    ap.add_argument("--class-floor", type=int, default=DEFAULT_PARAMS["class_floor"])
    ap.add_argument("--min-class-count", type=int, default=DEFAULT_PARAMS["min_class_count"])
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--split-seed", type=int, default=DEFAULT_PARAMS["split_seed"])
    a = ap.parse_args()
    if a.raw and len(a.dataset) > 1:
        ap.error("--raw applies to a single --dataset")
    for name in a.dataset:
        out = os.path.join(a.root, name, "processed")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"preprocess_{a.taxonomy}.log"), "w", encoding="utf-8") as logf:
            def log(msg):
                print(msg, flush=True)
                logf.write(msg + "\n")
            preprocess(name, a.root, log=log, raw=a.raw, note=a.note, taxonomy=a.taxonomy, max_rows=a.max_rows,
                       class_floor=a.class_floor, min_class_count=a.min_class_count,
                       max_files=a.max_files, split_seed=a.split_seed)
