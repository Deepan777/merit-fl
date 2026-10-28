#!/usr/bin/env python3
"""C1 containment experiments (src/v2/containment.py) on trained v2 detectors.

cal  (validation split, trace seeds 100-102; detectors of the frozen calibration setting): choose
     the CUSUM reference value k and threshold h, and the v1-trust quarantine level q, so that at most
     FA_TARGET of the honest devices are falsely contained per hour (averaged over runs); among the
     CUSUM settings that meet it, the one containing the most persistent 5% attackers with carry-over.
     Writes configs/v2/containment.yaml.
test (test split, trace seeds 42-51; detectors of the test_hfl random runs): every scenario and method
     with the frozen parameters. Writes results/v2/containment_test/<dataset>.json (per run, scenario,
     method: outcomes per attacker and honest device).

python scripts/v2_containment.py cal
python scripts/v2_containment.py test
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.evaluation.metrics import predict_proba  # noqa: E402
from src.models.fib import build_model  # noqa: E402
from src.v2.config import load_v2  # noqa: E402
from src.v2.containment import METHODS, contain, evidence, outcomes, simulate_traffic  # noqa: E402
from src.v2.hfl import load_trace  # noqa: E402
from src.v2.network import compute_links, edge_grid  # noqa: E402
from src.v2.plan import CAL_SEEDS, DATASETS, TEST_SEEDS  # noqa: E402
from src.v2.traffic import V2Data  # noqa: E402
from src.utils.common import env_info  # noqa: E402

FLOWS_PER_S, WINDOW_S, FA_TARGET = 2, 30, 0.01
TRUST = {"tau0": 0.5, "eps": 0.2, "delta": 0.85}               # v1 values
SCENARIOS = {                                                   # name: (kind, par, token strategy)
    "persistent_0.02": ("persistent", 0.02, None), "persistent_0.05": ("persistent", 0.05, None),
    "persistent_0.1": ("persistent", 0.1, None), "persistent_0.2": ("persistent", 0.2, None),
    "handover_timed": ("handover", (0.8, 0.01), None), "on_off_0.2": ("onoff", 0.2, None),
    "stealthy_0.005": ("persistent", 0.005, None),
    "persistent_0.05_omit": ("persistent", 0.05, "omit"), "persistent_0.05_forge": ("persistent", 0.05, "forge")}
# token scenarios reuse the attackers and traffic of their base scenario, so token comparisons are paired
TOKEN_BASE = {"persistent_0.05_omit": "persistent_0.05", "persistent_0.05_forge": "persistent_0.05"}
K_GRID = (0.005, 0.01, 0.02, 0.04)
H_GRID = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2, 1.6)
Q_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)
REG = os.path.join(REPO, "experiments", "v2_registry.csv")


def detector_scores(run_dir, cfg, seed):
    data = V2Data(cfg, seed)
    model = build_model(cfg["model"]["name"], data.n_features, data.K)
    model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu"))
    torch.set_num_threads(cfg["threads"])
    pr = predict_proba(model, data.X_stream, torch.device("cpu"))
    score = 1.0 - pr[:, data.benign]
    sb = score[data.y_stream == data.benign]
    sa = {k: score[data.y_stream == k] for k in data.attack_classes if (data.y_stream == k).any()}
    return sb, sa, data


def attackers_for(scen_i, seed, vclass, families, rng_base):
    rng = np.random.default_rng([seed, scen_i, rng_base])
    pick = []
    for c, n in (("veh", 5), ("ped", 3), ("static", 2)):
        pick += list(rng.choice(np.flatnonzero(vclass == c), n, replace=False))
    return {int(u): {"start": int(rng.integers(600, 1800)), "family": int(rng.choice(families)),
                     "vclass": str(vclass[u])} for u in pick}


def run_dataset_seed(dataset, seed, run_dir, cfg, params_list, scenarios):
    """Returns rows: one per (scenario, params, method) with per-device outcomes."""
    sb, sa, data = detector_scores(run_dir, cfg, seed)
    pos, vclass, side = load_trace(cfg, seed)
    links = compute_links(pos, edge_grid(side, cfg["net"]["edges_per_side"]), seed)
    out = []
    names = list(scenarios)
    for name, (kind, par, tok) in scenarios.items():
        base = TOKEN_BASE.get(name, name)
        si = names.index(base if base in names else name)
        att = attackers_for(si, seed, vclass, list(sa), 3)
        for u in att:
            att[u].update(kind=kind, par=par)
        rng = np.random.default_rng([seed, si, 5])
        tr = simulate_traffic(links, sb, sa, att, FLOWS_PER_S, WINDOW_S, rng)
        yhat, x = evidence(tr)
        honest = [u for u in range(len(vclass)) if u not in att]
        token = {u: (tok, a["start"] // WINDOW_S) for u, a in att.items()} if tok else None
        for p in params_list:
            for m in p["methods"]:
                first = contain(m, tr, yhat, x, {**TRUST, **p}, token=token)
                res = outcomes(first, tr, att, WINDOW_S, honest)
                out.append({"dataset": dataset, "seed": seed, "scenario": name, "method": m,
                            "params": {k: p[k] for k in ("k", "h", "h_reset", "q") if k in p}, **res,
                            "p_fa": float((sb >= 0.5).mean()),
                            "p_det": float(np.mean([(v >= 0.5).mean() for v in sa.values()]))})
    return out


def detectors(group, seeds, dataset, arm="random", set_idx=None, reg=None):
    r = pd.read_csv(reg or REG)
    r = r[(r.group == group) & (r.dataset == dataset) & (r.arm == arm) & (r.status == "done") & r.seed.isin(seeds)]
    if set_idx is not None:
        r = r[r.run_id.str.contains(f"/set{set_idx}/")]
    return {int(x.seed): x.run_dir for x in r.itertuples()}


def summarize(rows):
    t = []
    for r in rows:
        att, hon = pd.DataFrame(r["att"]), pd.DataFrame(r["hon"])
        t.append({"dataset": r["dataset"], "seed": r["seed"], "scenario": r["scenario"], "method": r["method"],
                  **r["params"], "contained": att.contained.mean(), "delay_s": att.delay_s.mean(),
                  "exposure": att.exposure.sum() / max(att.attack_total.sum(), 1),
                  "false": hon.contained.mean(), "blocked_benign": hon.blocked_benign.sum()})
    return pd.DataFrame(t)


def cal():
    params = [{"k": k, "h": h, "methods": ["cusum_carry", "cusum_reset"]} for k in K_GRID for h in H_GRID]
    params += [{"q": q, "methods": ["trust_carry"]} for q in Q_GRID]
    scen = {k: SCENARIOS[k] for k in ("persistent_0.05", "persistent_0.02", "handover_timed")}
    rows = []
    for d in DATASETS:
        frozen = os.path.join(REPO, "configs", "v2", f"{d}.yaml")
        set_idx = yaml.safe_load(open(frozen, encoding="utf-8")).get("calibration_set")
        for seed, rd in detectors("cal_hfl", CAL_SEEDS, d, set_idx=set_idx).items():
            cfg = load_v2(frozen, {"seed": seed, "eval.split": "val"})
            rows += run_dataset_seed(d, seed, rd, cfg, params, scen)
            print("cal", d, seed, flush=True)
    t = summarize(rows)
    cus = t[t.method == "cusum_carry"].groupby(["k", "h"]).agg(false=("false", "mean"), det=("contained", "mean")).reset_index()
    t = t.copy()
    p05 = t[(t.method == "cusum_carry") & (t.scenario == "persistent_0.05")].groupby(["k", "h"]).contained.mean()
    cus["det05"] = [p05[(k, h)] for k, h in zip(cus.k, cus.h)]
    ok = cus[cus.false <= FA_TARGET].sort_values(["det05", "h"], ascending=[False, True])
    best = ok.iloc[0] if len(ok) else cus.sort_values("false").iloc[0]
    tr = t[t.method == "trust_carry"].groupby("q").agg(false=("false", "mean"), det=("contained", "mean")).reset_index()
    # reset recalibrated to the same budget: same k, the lowest h whose false containment meets it
    res = t[(t.method == "cusum_reset") & (t.k == best.k)].groupby("h")["false"].mean()
    ok_r = res[res <= FA_TARGET]
    h_reset = float(ok_r.index.min()) if len(ok_r) else float(res.idxmin())
    okq = tr[tr.false <= FA_TARGET].sort_values("q")
    q = float(okq.q.iloc[0]) if len(okq) else float(tr.sort_values("false").q.iloc[0])
    out = {"k": float(best.k), "h": float(best.h), "h_reset": h_reset, "q": q, "fa_target": FA_TARGET,
           "flows_per_s": FLOWS_PER_S,
           "window_s": WINDOW_S, "trust": TRUST,
           "calibration": {"cusum_false": float(best.false), "cusum_det_persistent_0.05": float(best.det05),
                           "reset_cal_false": float(res[h_reset]),
                           "trust_q_false": float(tr[tr.q == q].false.iloc[0])}}
    os.makedirs(os.path.join(REPO, "configs", "v2"), exist_ok=True)
    yaml.safe_dump(out, open(os.path.join(REPO, "configs", "v2", "containment.yaml"), "w", encoding="utf-8"), sort_keys=False)
    cus.to_csv(os.path.join(REPO, "reports", "v2_containment_calibration.csv"), index=False)
    print(json.dumps(out, indent=1))


def test(reg=None, arm="random"):
    """arm: the test_hfl arm whose detectors score the traffic (random = pre-registered; residence+fw =
    R6 analysis R5, both parts of MERIT-FL together, written to containment_test_<arm>)."""
    p = yaml.safe_load(open(os.path.join(REPO, "configs", "v2", "containment.yaml"), encoding="utf-8"))
    params = [{"k": p["k"], "h": p["h"], "h_reset": p["h_reset"], "q": p["q"], "methods": list(METHODS)}]
    sub = "containment_test" if arm == "random" else f"containment_test_{arm.replace('+', '_')}"
    os.makedirs(os.path.join(REPO, "results", "v2", sub), exist_ok=True)
    for d in DATASETS:
        frozen = os.path.join(REPO, "configs", "v2", f"{d}.yaml")
        rows = []
        for seed, rd in detectors("test_hfl", TEST_SEEDS, d, arm=arm, reg=reg).items():
            cfg = load_v2(frozen, {"seed": seed})
            rows += run_dataset_seed(d, seed, rd, cfg, params, SCENARIOS)
            print("test", d, seed, flush=True)
        json.dump(rows, open(os.path.join(REPO, "results", "v2", sub, f"{d}.json"), "w",
                             encoding="utf-8"), default=float)
        json.dump({"params": p, "detectors": detectors("test_hfl", TEST_SEEDS, d, arm=arm, reg=reg), **env_info()},
                  open(os.path.join(REPO, "results", "v2", sub, f"{d}.env.json"), "w", encoding="utf-8"),
                  indent=1, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cal", "test"])
    ap.add_argument("--registry", default=None, help="registry to read detectors from (default: this checkout)")
    ap.add_argument("--arm", default="random", help="test_hfl arm of the detectors (R6: residence+fw)")
    a = ap.parse_args()
    if a.cmd == "cal":
        os.environ["MERIT_FL_FORBID_TEST"] = "1"
        cal()
    else:
        test(a.registry, a.arm)
