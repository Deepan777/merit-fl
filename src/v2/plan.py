"""v2 run plan: every hierarchical-FL run as one registry row (experiments/v2_registry.csv).

Calibration (validation split, trace seeds 100-102):
  cal_hfl      learning rate x local epochs, energy-gated random participation without forwarding
Final (test split, trace seeds 42-51; frozen configs/v2/<dataset>.yaml):
  test_hfl     participation (random / speed / residence / oracle) x forwarding (off / on); the
               oracle never needs forwarding
  test_round   round length 15 s and 60 s (30 s = test_hfl), seeds 42-46, three arms
Containment experiments (C1) do not train and are run by scripts/v2_containment.py on the detectors
of the test_hfl random/no-forwarding runs.
"""
import itertools
import json

DATASETS = ("5g_nidd", "ciciot2023", "cicids2017")
CAL_SEEDS = (100, 101, 102)
TEST_SEEDS = tuple(range(42, 52))
SWEEP_SEEDS = tuple(range(42, 47))
CAL_GRID = [{"fl.lr": lr, "fl.local_epochs": ep} for lr, ep in itertools.product((0.001, 0.003), (1, 3))]
ARMS = [("random", False), ("random", True), ("speed", False), ("speed", True),
        ("residence", False), ("residence", True), ("oracle", False)]
ROUND_ARMS = [("random", False), ("residence", True), ("oracle", False)]


def arm_name(sel, fw):
    return f"{sel}{'+fw' if fw else ''}"


def plan_rows():
    rows = []
    for d in DATASETS:
        for i, ov in enumerate(CAL_GRID):
            for s in CAL_SEEDS:
                rows.append({"run_id": f"cal_hfl/{d}/random/set{i}/s{s}", "group": "cal_hfl", "dataset": d,
                             "arm": "random", "seed": s,
                             "overrides": json.dumps({"eval.split": "val", **ov})})
        for sel, fw in ARMS:
            for s in TEST_SEEDS:
                rows.append({"run_id": f"test_hfl/{d}/{arm_name(sel, fw)}/set0/s{s}", "group": "test_hfl",
                             "dataset": d, "arm": arm_name(sel, fw), "seed": s,
                             "overrides": json.dumps({"fl.selection": sel, "fl.forwarding": fw})})
        for i, rs in enumerate((15, 60)):
            for sel, fw in ROUND_ARMS:
                for s in SWEEP_SEEDS:
                    rows.append({"run_id": f"test_round/{d}/{arm_name(sel, fw)}/set{i}/s{s}", "group": "test_round",
                                 "dataset": d, "arm": arm_name(sel, fw), "seed": s,
                                 "overrides": json.dumps({"fl.selection": sel, "fl.forwarding": fw, "fl.round_s": rs})})
    return rows
