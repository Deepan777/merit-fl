#!/usr/bin/env python3
"""Freeze the v2 training settings per dataset from the validation calibration (group cal_hfl):
the learning-rate x local-epoch setting with the highest mean macro_f1_last3 over the calibration
seeds (100-102). Writes configs/v2/<dataset>.yaml and reports/v2_calibration.md.

python scripts/v2_freeze.py
"""
import json
import os
import sys

import pandas as pd
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.v2.plan import CAL_GRID, DATASETS  # noqa: E402


def main():
    r = pd.read_csv(os.path.join(REPO, "experiments", "v2_registry.csv"))
    r = r[(r.group == "cal_hfl") & (r.status == "done")].copy()
    r["set"] = r.run_id.str.extract(r"/set(\d+)/")[0].astype(int)
    md = ["# v2 calibration (validation split, trace seeds 100-102)", "",
          "Energy-gated random participation without forwarding; holdout macro-F1, mean of the last three "
          "checkpoints; mean over seeds (min-max).", "",
          "| dataset | " + " | ".join(f"lr {g['fl.lr']}, epochs {g['fl.local_epochs']}" for g in CAL_GRID) + " | chosen |",
          "|---|" + "---|" * (len(CAL_GRID) + 1)]
    os.makedirs(os.path.join(REPO, "configs", "v2"), exist_ok=True)
    for d in DATASETS:
        g = r[r.dataset == d].groupby("set").macro_f1_last3.agg(["mean", "min", "max", "count"])
        best = int(g["mean"].idxmax())
        cells = [f"{g.loc[i, 'mean']:.3f} ({g.loc[i, 'min']:.3f}-{g.loc[i, 'max']:.3f}, n={int(g.loc[i, 'count'])})"
                 if i in g.index else "-" for i in range(len(CAL_GRID))]
        md.append(f"| {d} | " + " | ".join(cells) + f" | set {best} |")
        frozen = {"data": {"name": d}, "fl": {"lr": CAL_GRID[best]["fl.lr"], "local_epochs": CAL_GRID[best]["fl.local_epochs"]},
                  "calibration_set": best,
                  "provenance": f"scripts/v2_freeze.py from cal_hfl (validation, seeds 100-102); mean macro_f1_last3 "
                                f"{g.loc[best, 'mean']:.4f}"}
        with open(os.path.join(REPO, "configs", "v2", f"{d}.yaml"), "w", encoding="utf-8") as fh:
            fh.write("# Frozen by scripts/v2_freeze.py from the validation calibration. Do not edit by hand.\n")
            yaml.safe_dump(frozen, fh, sort_keys=False)
    open(os.path.join(REPO, "reports", "v2_calibration.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
