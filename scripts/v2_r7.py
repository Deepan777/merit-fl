#!/usr/bin/env python3
"""R7: self-calibrated evidence and attacks on the evidence the analysis plan; post hoc).

Statistic llrh: binomial log-likelihood-ratio CUSUM against the median hourly flagged share of the device's
type (src/v2/robust.hourly_type_reference); threshold h = ln(ARL0) from the budget alone (no calibration).

cal    validation: choose delta per world, dataset and budget (carry-over, persistent 5%) -> configs/v2/r7.yaml
test   test: every scenario (R6's seven + clustered + adaptive) x placement (carry, memory, reset, carry_loss)
       -> results/v2/r7_test/<world>_<dataset>.json
curves test, persistent 5%: thresholds swept (matched-rate operating curves) -> results/v2/r7_curves/

python scripts/v2_r7.py cal | test [--worlds W1 ...] | curves [--worlds ...]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import v2_robust as V  # noqa: E402
from src.v2 import robust as R  # noqa: E402
from src.v2.config import load_v2  # noqa: E402
from src.v2.plan import CAL_SEEDS, DATASETS, TEST_SEEDS  # noqa: E402

WORLDS = ("W0", "W1", "W2", "W3", "W4")
LINKS_OF = {"W0": "W0", "W1": "W1", "W2": "W2", "W3": "W3", "W4": "W1"}          # W4 = W1 + within-type jitter
BUDGETS = {"b01": 0.001, "b1": 0.01}
WIN_PER_H = 3600 // V.WINDOW_S
H_THEORY = {b: float(np.log(WIN_PER_H / B)) for b, B in BUDGETS.items()}           # ln(ARL0 in windows)
DELTAS = (0.02, 0.05, 0.1)
PLACE = ("carry", "memory", "reset", "carry_loss")
LOSS = 0.2
SCEN = dict(V.SCENARIOS, clustered_5=("cluster", 5), clustered_10=("cluster", 10),
            adaptive_0_8=("adaptive", 0.8), adaptive_1_2=("adaptive", 1.2))
W_BURN = V.BURN_H * WIN_PER_H
FOLLOW = V.FOLLOW_MIN * 60 // V.WINDOW_S


def devices7(seed, wname, U, prof):
    lam, p_ben, t = V.devices(seed, "W0" if wname == "W0" else "W1", U, prof)
    if wname == "W4":
        rng = np.random.default_rng([seed, 4, 98])
        p_ben = np.clip(p_ben * np.exp(rng.normal(0, 0.3, size=U)), 0, 1)
    return lam, p_ben, t


def attackers7(scen, seed, vclass, prof, dev_type, p_ben, delta):
    kind, par = SCEN[scen]
    if kind == "cluster":
        rng = np.random.default_rng([seed, 100 + par, 3])
        types = [t for t in np.unique(dev_type) if (dev_type == t).sum() >= par]
        t = int(rng.choice(types))
        pick = rng.choice(np.flatnonzero(dev_type == t), par, replace=False)
        fams = list(range(len(prof["fam"])))
        out = {}
        for u in pick:
            f = int(rng.choice(fams))
            out[int(u)] = {"start": int(rng.integers(V.ATTACK_H * 3600, V.ATTACK_H * 3600 + 1800)),
                           "p_det": float(prof["fam_rate"][f]), "vclass": str(vclass[u]), "kind": "persistent", "par": 0.05}
        return out
    base = "persistent_0.05" if kind == "adaptive" else scen
    att = V.attackers(list(V.SCENARIOS).index(base), seed, vclass, prof)
    for u, a in att.items():
        if kind == "adaptive":
            rho = R.llr_floor(float(p_ben[u]), a["p_det"], delta)
            a.update(kind="persistent", par=float(min(1.0, max(0.0, par * rho))), floor=float(rho))
        else:
            a.update(kind=kind, par=par)
    return att


def run7(wname, d, split, seed, rep, scen, cfg, delta):
    prof = V.profile(split, d, seed)
    node, ho, outage, vclass, n_nodes = V.world_links(seed, LINKS_OF[wname], cfg)
    U = node.shape[1]
    lam, p_ben, dev_type = devices7(seed, wname, U, prof)
    att = attackers7(scen, seed, vclass, prof, dev_type, p_ben, delta)
    rng = np.random.default_rng([seed, rep, list(SCEN).index(scen), WORLDS.index(wname), 177])
    tr = R.simulate(node, ho, outage, lam, p_ben, att, V.WINDOW_S, rng)
    p0, valid = R.hourly_type_reference(tr, dev_type)
    return tr, p0, valid, att, vclass, n_nodes


def alarms7(tr, p0, valid, delta, placement, n_nodes, hs, att_sw, seed):
    inc = R.llr_increment(tr, p0, valid, delta)
    ok = tr["seen"] > 0
    return R.run_alarms("llr", delta, inc, ok, tr["node"], tr["ho"], placement, n_nodes, hs, W_BURN, att_sw,
                        loss=LOSS, rng=np.random.default_rng([seed, 55]))


def cal():
    rows, chosen = [], {}
    hs = [H_THEORY["b01"], H_THEORY["b1"]]
    for wname in WORLDS:
        for d in DATASETS:
            acc = {dl: {"ev": np.zeros(2), "hrs": 0.0, "cont": np.zeros(2), "n": 0} for dl in DELTAS}
            for seed in CAL_SEEDS:
                cfg = load_v2(None, {"seed": seed, "eval.split": "val"})
                for rep in range(3):
                    # persistent 5%: the traffic does not depend on delta, so it is simulated once
                    tr, p0, valid, att, vclass, n_nodes = run7(wname, d, "val", seed, rep, "persistent_0.05", cfg, DELTAS[0])
                    honest = np.ones(tr["seen"].shape[1], bool)
                    honest[list(att)] = False
                    att_sw = {u: a["start"] // V.WINDOW_S for u, a in att.items()}
                    for dl in DELTAS:
                        al, first = alarms7(tr, p0, valid, dl, "carry", n_nodes, hs, att_sw, seed)
                        acc[dl]["ev"] += al[:, honest].sum(1)
                        acc[dl]["hrs"] += honest.sum() * (tr["seen"].shape[0] - W_BURN) / WIN_PER_H
                        for u, sw in att_sw.items():
                            acc[dl]["cont"] += first[:, u] < sw + FOLLOW
                        acc[dl]["n"] += len(att)
                print("cal7", wname, d, seed, flush=True)
            for i, (b, B) in enumerate(BUDGETS.items()):
                cand = []
                for dl in DELTAS:
                    rate = acc[dl]["ev"][i] / acc[dl]["hrs"]
                    cont = acc[dl]["cont"][i] / acc[dl]["n"]
                    rows.append({"world": wname, "dataset": d, "budget": b, "delta": dl, "val_rate": float(rate),
                                 "val_contained": float(cont)})
                    cand.append((rate <= B, cont, dl, rate))
                ok = [c for c in cand if c[0]]
                best = max(ok, key=lambda c: (c[1], -c[2])) if ok else max(cand, key=lambda c: c[2])
                chosen.setdefault(wname, {}).setdefault(d, {})[b] = {"delta": best[2], "h": H_THEORY[b],
                                                                     "val_rate": float(best[3]), "val_contained": float(best[1]),
                                                                     "feasible": bool(best[0])}
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(REPO, "reports", "v2_r7_calibration.csv"), index=False)
    out = {"provenance": "scripts/v2_r7.py cal (validation, traces 100-102, 3 replicates); delta only, h by theory",
           "h_theory": H_THEORY, "budgets": BUDGETS, "loss": LOSS, "chosen": chosen}
    yaml.safe_dump(json.loads(json.dumps(out)), open(os.path.join(REPO, "configs", "v2", "r7.yaml"), "w", encoding="utf-8"),
                   sort_keys=False)
    print("wrote configs/v2/r7.yaml")


def test(only=None):
    from src.utils.common import env_info
    conf = yaml.safe_load(open(os.path.join(REPO, "configs", "v2", "r7.yaml"), encoding="utf-8"))
    out_dir = os.path.join(REPO, "results", "v2", "r7_test")
    os.makedirs(out_dir, exist_ok=True)
    for wname in WORLDS:
        if only and wname not in only:
            continue
        for d in DATASETS:
            out_p = os.path.join(out_dir, f"{wname}_{d}.json")
            if os.path.exists(out_p):
                continue
            ch = conf["chosen"][wname][d]
            res, t0 = [], time.time()
            for seed in TEST_SEEDS:
                cfg = load_v2(None, {"seed": seed})
                for rep in range(2):
                    for scen in SCEN:
                        for b in BUDGETS:
                            dl, h = ch[b]["delta"], ch[b]["h"]
                            tr, p0, valid, att, vclass, n_nodes = run7(wname, d, "test", seed, rep, scen, cfg, dl)
                            U = tr["seen"].shape[1]
                            honest = np.ones(U, bool)
                            honest[list(att)] = False
                            att_sw = {u: a["start"] // V.WINDOW_S for u, a in att.items()}
                            Wn = tr["seen"].shape[0]
                            hrs_dev = (Wn - W_BURN) / WIN_PER_H
                            for pl in PLACE:
                                al, first = alarms7(tr, p0, valid, dl, pl, n_nodes, [h], att_sw, seed)
                                A = []
                                for u, a in att.items():
                                    sw = att_sw[u]
                                    f = int(first[0, u])
                                    okc = f < sw + FOLLOW
                                    upto = f if okc else min(sw + FOLLOW, Wn)
                                    A.append({"u": u, "vclass": a["vclass"], "contained": bool(okc),
                                              "delay_w": (f - sw + 1) if okc else None, "par": a.get("par"),
                                              "floor": a.get("floor"), "exposure": int(tr["attacks"][sw:upto, u].sum()),
                                              "attack_total": int(tr["attacks"][sw:sw + FOLLOW, u].sum()),
                                              "residence_w": int(R.longest_residence(tr["node"], u, sw, min(sw + FOLLOW, Wn)))})
                                res.append({"world": wname, "dataset": d, "seed": seed, "rep": rep, "scenario": scen,
                                            "budget": b, "placement": pl, "delta": dl, "h": h,
                                            "honest_events": int(al[0, honest].sum()), "honest_hours": float(honest.sum() * hrs_dev),
                                            "honest_by_class": {c: [int(al[0, honest & (vclass == c)].sum()),
                                                                    float((honest & (vclass == c)).sum() * hrs_dev)]
                                                                for c in ("veh", "ped", "static")},
                                            "att": A})
                print("test7", wname, d, seed, f"{time.time() - t0:.0f}s", flush=True)
            json.dump(res, open(out_p, "w", encoding="utf-8"), default=float)
            json.dump({"config": ch, **env_info()}, open(out_p.replace(".json", ".env.json"), "w", encoding="utf-8"),
                      indent=1, default=str)


def curves(only=None):
    from src.utils.common import env_info
    conf = yaml.safe_load(open(os.path.join(REPO, "configs", "v2", "r7.yaml"), encoding="utf-8"))
    grid = V.GRID["llr"]
    out_dir = os.path.join(REPO, "results", "v2", "r7_curves")
    os.makedirs(out_dir, exist_ok=True)
    for wname in WORLDS:
        if only and wname not in only:
            continue
        for d in DATASETS:
            out_p = os.path.join(out_dir, f"{wname}_{d}.json")
            if os.path.exists(out_p):
                continue
            dl = conf["chosen"][wname][d]["b01"]["delta"]
            acc = {pl: {"ev": [], "hours": [], "cont": [], "n_att": []} for pl in ("carry", "memory", "reset")}
            for seed in TEST_SEEDS:
                cfg = load_v2(None, {"seed": seed})
                per = {pl: {"ev": np.zeros(len(grid)), "hours": 0.0, "cont": np.zeros(len(grid)), "n_att": 0} for pl in acc}
                for rep in range(2):
                    tr, p0, valid, att, vclass, n_nodes = run7(wname, d, "test", seed, rep, "persistent_0.05", cfg, dl)
                    honest = np.ones(tr["seen"].shape[1], bool)
                    honest[list(att)] = False
                    att_sw = {u: a["start"] // V.WINDOW_S for u, a in att.items()}
                    for pl in acc:
                        al, first = alarms7(tr, p0, valid, dl, pl, n_nodes, grid, att_sw, seed)
                        per[pl]["ev"] += al[:, honest].sum(1)
                        per[pl]["hours"] += honest.sum() * (tr["seen"].shape[0] - W_BURN) / WIN_PER_H
                        for u, sw in att_sw.items():
                            per[pl]["cont"] += first[:, u] < sw + FOLLOW
                        per[pl]["n_att"] += len(att)
                for pl in acc:
                    for f_ in ("ev", "hours", "cont", "n_att"):
                        v = per[pl][f_]
                        acc[pl][f_].append(v.tolist() if isinstance(v, np.ndarray) else v)
                print("curves7", wname, d, seed, flush=True)
            json.dump({"grid": grid.tolist(), "delta": dl, "seeds": list(TEST_SEEDS), "rules": acc}, open(out_p, "w", encoding="utf-8"))
            json.dump(env_info(), open(out_p.replace(".json", ".env.json"), "w", encoding="utf-8"), indent=1, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cal", "test", "curves"])
    ap.add_argument("--worlds", nargs="*")
    a = ap.parse_args()
    if a.cmd == "cal":
        os.environ["MERIT_FL_FORBID_TEST"] = "1"
        cal()
    elif a.cmd == "test":
        test(a.worlds)
    else:
        curves(a.worlds)
