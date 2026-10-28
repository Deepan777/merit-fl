#!/usr/bin/env python3
"""R6 robustness experiment of C1 (post-hoc; the analysis plan, src/v2/robust.py).

cache  detector scores of every calibration (validation) and test detector on its run's stream pool:
       benign flag rate, benign types (k-means clusters of the benign stream flows) with their flag
       rates, attack-family detection rates -> results/v2/robust/cache/<split>_<dataset>_s<seed>.npz
cal    validation (traces 100-102, 3 traffic replicates): per world, dataset and rule, the lowest
       threshold within the steady-state budget, then the parameter with the most persistent 5%
       attackers contained -> configs/v2/robust.yaml, reports/v2_robust_calibration.csv
test   test (traces 42-51, 2 replicates), every scenario, every rule at its calibrated setting
       -> results/v2/robust_test/<world>_<dataset>.json (+ .env.json)

python scripts/v2_robust.py cache --registry experiments/v2_registry.csv
python scripts/v2_robust.py cal
python scripts/v2_robust.py test
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.v2 import robust as R  # noqa: E402
from src.v2.config import load_v2  # noqa: E402
from src.v2.hfl import load_trace  # noqa: E402
from src.v2.network import compute_links, edge_grid  # noqa: E402
from src.v2.plan import CAL_SEEDS, DATASETS, TEST_SEEDS  # noqa: E402

K_HOURS, BURN_H, WINDOW_S, ATTACK_H, FOLLOW_MIN = 6, 1, 30, 4, 60
N_TYPES, RATE_RANGE = 6, (0.2, 4.0)
BUDGETS = {"b01": 0.001, "b1": 0.01}                                     # per honest device-hour
WORLDS = {"W0": dict(hetero=False, block=1, hyst=None), "W1": dict(hetero=True, block=1, hyst=None),
          "W2": dict(hetero=True, block=2, hyst=None), "W3": dict(hetero=True, block=1, hyst=(6.0, 5))}
SCENARIOS = {"persistent_0.02": ("persistent", 0.02), "persistent_0.05": ("persistent", 0.05),
             "persistent_0.1": ("persistent", 0.1), "persistent_0.2": ("persistent", 0.2),
             "handover_timed": ("handover", (0.8, 0.01)), "on_off_0.2": ("onoff", 0.2),
             "stealthy_0.005": ("persistent", 0.005)}
GRID = {"share": np.geomspace(0.02, 30, 400), "llr": np.geomspace(0.05, 2000, 400),
        "ewma": np.geomspace(1e-4, 0.6, 400)}
CACHE = os.path.join(REPO, "results", "v2", "robust", "cache")
OUT = os.path.join(REPO, "results", "v2", "robust_test")


# ------------------------------------------------------------------ detector profiles
def cache(reg):
    import torch  # noqa: F401
    from sklearn.cluster import MiniBatchKMeans
    from v2_containment import detector_scores, detectors
    os.makedirs(CACHE, exist_ok=True)
    for split, group, seeds, set_idx in (("val", "cal_hfl", CAL_SEEDS, 3), ("test", "test_hfl", TEST_SEEDS, None)):
        for d in DATASETS:
            frozen = os.path.join(REPO, "configs", "v2", f"{d}.yaml")
            for seed, rd in detectors(group, seeds, d, set_idx=set_idx, reg=reg).items():
                out = os.path.join(CACHE, f"{split}_{d}_s{seed}.npz")
                if os.path.exists(out):
                    continue
                cfg = load_v2(frozen, {"seed": seed, "eval.split": split})
                sb, sa, data = detector_scores(os.path.join(os.path.dirname(reg), "..", rd), cfg, seed)
                Xb = data.X_stream[data.y_stream == data.benign]
                rng = np.random.default_rng(seed)
                sub = rng.choice(len(Xb), size=min(20000, len(Xb)), replace=False)
                km = MiniBatchKMeans(n_clusters=N_TYPES, random_state=seed, n_init=3, batch_size=4096).fit(Xb[sub])
                lab = km.predict(Xb)
                flag = sb >= 0.5
                np.savez(out, p_fa=flag.mean(), type_rate=np.array([flag[lab == t].mean() if (lab == t).any() else flag.mean()
                                                                    for t in range(N_TYPES)]),
                         type_share=np.bincount(lab, minlength=N_TYPES) / len(lab),
                         fam=np.array(list(sa)), fam_rate=np.array([(v >= 0.5).mean() for v in sa.values()]))
                print("cached", split, d, seed, flush=True)


def profile(split, d, seed):
    z = np.load(os.path.join(CACHE, f"{split}_{d}_s{seed}.npz"))
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ worlds and runs
_links = {}


def world_links(seed, wname, cfg):
    key = (seed, wname)
    if key not in _links:
        for k in [k for k in _links if k[1] != wname]:          # keep one world in memory
            del _links[k]
        pos, vclass, side = load_trace(cfg, seed)
        n_side = cfg["net"]["edges_per_side"]
        w = WORLDS[wname]
        kw = {} if w["hyst"] is None else {"hyst_db": w["hyst"][0], "ttt_s": w["hyst"][1]}
        L = compute_links(pos, edge_grid(side, n_side), seed, **kw)
        ch = R.chain_links({"serving": L["serving"], "outage": L["outage"]}, K_HOURS)
        node, ho = R.edge_view(ch, n_side, w["block"])
        _links[key] = (node.astype(np.int16), ho, ch["outage"], vclass, (n_side // w["block"]) ** 2)
    return _links[key]


def devices(seed, wname, U, prof):
    rng = np.random.default_rng([seed, list(WORLDS).index(wname), 99])
    if not WORLDS[wname]["hetero"]:
        return np.full(U, 2.0), np.full(U, float(prof["p_fa"])), np.zeros(U, int)
    t = rng.integers(N_TYPES, size=U)
    lam = np.exp(rng.uniform(np.log(RATE_RANGE[0]), np.log(RATE_RANGE[1]), size=U))
    return lam, prof["type_rate"][t].astype(float), t


def attackers(si, seed, vclass, prof):
    rng = np.random.default_rng([seed, si, 3])
    pick = []
    for c, n in (("veh", 5), ("ped", 3), ("static", 2)):
        pick += list(rng.choice(np.flatnonzero(vclass == c), n, replace=False))
    fams = list(range(len(prof["fam"])))
    out = {}
    for u in pick:
        f = int(rng.choice(fams))
        out[int(u)] = {"start": int(rng.integers(ATTACK_H * 3600, ATTACK_H * 3600 + 1800)),
                       "p_det": float(prof["fam_rate"][f]), "vclass": str(vclass[u])}
    return out


def run(wname, d, split, seed, rep, scen, cfg):
    prof = profile(split, d, seed)
    node, ho, outage, vclass, n_nodes = world_links(seed, wname, cfg)
    U = node.shape[1]
    lam, p_ben, dev_type = devices(seed, wname, U, prof)
    si = list(SCENARIOS).index(scen)
    kind, par = SCENARIOS[scen]
    att = attackers(si, seed, vclass, prof)
    for a in att.values():
        a.update(kind=kind, par=par)
    rng = np.random.default_rng([seed, rep, si, list(WORLDS).index(wname), 77])
    tr = R.simulate(node, ho, outage, lam, p_ben, att, WINDOW_S, rng)
    yhat, ok, x_share, p0 = R.evidence(tr, dev_type)
    return tr, (yhat, ok, x_share, p0), att, vclass, n_nodes


def alarms_for(stat, par, placement, tr, ev, n_nodes, hs, att_sw):
    yhat, ok, x_share, p0 = ev
    inc = R.increments(stat, par, tr, yhat, ok, x_share, p0)
    return R.run_alarms(stat, par, inc, ok, tr["node"], tr["ho"], placement, n_nodes, hs,
                        BURN_H * 3600 // WINDOW_S, att_sw)


def rules():
    return [(s, p, pl) for s, pars in R.STATS.items() for p in pars for pl in R.PLACEMENTS]


# ------------------------------------------------------------------ calibration (validation)
def cal():
    cfg0 = load_v2(None, {"eval.split": "val"})
    W = K_HOURS * 3600 // WINDOW_S
    w_burn = BURN_H * 3600 // WINDOW_S
    rows, chosen = [], {}
    for wname in WORLDS:
        for d in DATASETS:
            acc = {r: {"ev": np.zeros(400), "hrs": np.zeros(400), "cont": np.zeros(400), "n_att": 0} for r in rules()}
            for seed in CAL_SEEDS:
                cfg = load_v2(None, {"seed": seed, "eval.split": "val"})
                for rep in range(3):
                    tr, ev, att, vclass, n_nodes = run(wname, d, "val", seed, rep, "persistent_0.05", cfg)
                    honest = np.ones(tr["seen"].shape[1], bool)
                    honest[list(att)] = False
                    att_sw = {u: a["start"] // WINDOW_S for u, a in att.items()}
                    for r in rules():
                        stat, par, pl = r
                        al, first = alarms_for(stat, par, pl, tr, ev, n_nodes, GRID[stat], att_sw)
                        acc[r]["ev"] += al[:, honest].sum(1)
                        acc[r]["hrs"] += honest.sum() * (W - w_burn) * WINDOW_S / 3600
                        for u, sw in att_sw.items():
                            acc[r]["cont"] += first[:, u] < sw + FOLLOW_MIN * 60 // WINDOW_S
                        acc[r]["n_att"] += len(att)
                print("cal", wname, d, seed, flush=True)
            for r, a in acc.items():
                haz = a["ev"] / np.maximum(a["hrs"], 1e-9)
                for b, B in BUDGETS.items():
                    okb = np.flatnonzero(haz <= B)
                    i = int(okb.min()) if len(okb) else len(GRID[r[0]]) - 1
                    rows.append({"world": wname, "dataset": d, "stat": r[0], "par": r[1], "placement": r[2], "budget": b,
                                 "h": float(GRID[r[0]][i]), "hazard": float(haz[i]), "events": int(a["ev"][i]),
                                 "hours": float(a["hrs"][i]), "contained": float(a["cont"][i] / a["n_att"])})
            del acc
    t = pd.DataFrame(rows)
    os.makedirs(os.path.join(REPO, "reports"), exist_ok=True)
    t.to_csv(os.path.join(REPO, "reports", "v2_robust_calibration.csv"), index=False)
    for (wname, d, stat, pl, b), g in t.groupby(["world", "dataset", "stat", "placement", "budget"]):
        best = g.sort_values(["contained", "par"], ascending=[False, True]).iloc[0]
        chosen.setdefault(wname, {}).setdefault(d, {}).setdefault(b, {})[f"{stat}/{pl}"] = {
            "par": float(best.par), "h": float(best.h), "val_hazard": float(best.hazard), "val_contained": float(best.contained)}
    out = {"provenance": "scripts/v2_robust.py cal (validation, traces 100-102, 3 replicates); post-hoc R6",
           "budgets": BUDGETS, "k_hours": K_HOURS, "burn_in_h": BURN_H, "attack_start_h": ATTACK_H,
           "follow_min": FOLLOW_MIN, "worlds": WORLDS, "chosen": chosen}
    yaml.safe_dump(json.loads(json.dumps(out)), open(os.path.join(REPO, "configs", "v2", "robust.yaml"), "w", encoding="utf-8"),
                   sort_keys=False)
    print("wrote configs/v2/robust.yaml")


# ------------------------------------------------------------------ test
def test(only_worlds=None):
    from src.utils.common import env_info
    conf = yaml.safe_load(open(os.path.join(REPO, "configs", "v2", "robust.yaml"), encoding="utf-8"))
    W = K_HOURS * 3600 // WINDOW_S
    w_burn = BURN_H * 3600 // WINDOW_S
    follow = FOLLOW_MIN * 60 // WINDOW_S
    os.makedirs(OUT, exist_ok=True)
    for wname in WORLDS:
        if only_worlds and wname not in only_worlds:
            continue
        for d in DATASETS:
            out_p = os.path.join(OUT, f"{wname}_{d}.json")
            if os.path.exists(out_p):
                print("exists", out_p)
                continue
            ch = conf["chosen"][wname][d]
            res = []
            t0 = time.time()
            for seed in TEST_SEEDS:
                cfg = load_v2(None, {"seed": seed})
                for rep in range(2):
                    for scen in SCENARIOS:
                        tr, ev, att, vclass, n_nodes = run(wname, d, "test", seed, rep, scen, cfg)
                        U = tr["seen"].shape[1]
                        honest = np.ones(U, bool)
                        honest[list(att)] = False
                        att_sw = {u: a["start"] // WINDOW_S for u, a in att.items()}
                        hrs_dev = (W - w_burn) * WINDOW_S / 3600
                        for b, rules_b in ch.items():
                            for key, v in rules_b.items():
                                stat, pl = key.split("/")
                                al, first = alarms_for(stat, v["par"], pl, tr, ev, n_nodes, [v["h"]], att_sw)
                                e, h = int(al[0, honest].sum()), float(honest.sum() * hrs_dev)
                                cls = {c: [int(al[0, honest & (vclass == c)].sum()), float((honest & (vclass == c)).sum() * hrs_dev)]
                                       for c in ("veh", "ped", "static")}
                                A = []
                                for u, a in att.items():
                                    sw = att_sw[u]
                                    f = int(first[0, u])
                                    ok = f < sw + follow
                                    upto = f if ok else min(sw + follow, W)
                                    A.append({"u": u, "vclass": a["vclass"], "contained": bool(ok),
                                              "delay_w": (f - sw + 1) if ok else None,
                                              "exposure": int(tr["attacks"][sw:upto, u].sum()),
                                              "exposure_blocking": int((tr["attacks"] - tr["attack_flagged"])[sw:upto, u].sum()),
                                              "attack_total": int(tr["attacks"][sw:sw + follow, u].sum()),
                                              "residence_w": int(R.longest_residence(tr["node"], u, sw, min(sw + follow, W)))})
                                res.append({"world": wname, "dataset": d, "seed": seed, "rep": rep, "scenario": scen,
                                            "budget": b, "rule": key, "par": v["par"], "h": v["h"],
                                            "honest_events": int(e), "honest_hours": float(h), "honest_by_class": cls,
                                            "att": A})
                print("test", wname, d, seed, f"{time.time() - t0:.0f}s", flush=True)
            json.dump(res, open(out_p, "w", encoding="utf-8"), default=float)
            json.dump({"config": conf["chosen"][wname][d], **env_info()}, open(out_p.replace(".json", ".env.json"), "w",
                                                                               encoding="utf-8"), indent=1, default=str)


def curves(only_worlds=None):
    """Amendment A2: operating curves on test (persistent 5%), every rule at its validation-chosen parameter,
    thresholds swept over GRID. Per seed: honest alarms after the burn-in and attackers contained, per threshold.
    -> results/v2/robust_curves/<world>_<dataset>.json"""
    from src.utils.common import env_info
    conf = yaml.safe_load(open(os.path.join(REPO, "configs", "v2", "robust.yaml"), encoding="utf-8"))
    W = K_HOURS * 3600 // WINDOW_S
    w_burn = BURN_H * 3600 // WINDOW_S
    follow = FOLLOW_MIN * 60 // WINDOW_S
    out_dir = os.path.join(REPO, "results", "v2", "robust_curves")
    os.makedirs(out_dir, exist_ok=True)
    for wname in WORLDS:
        if only_worlds and wname not in only_worlds:
            continue
        for d in DATASETS:
            out_p = os.path.join(out_dir, f"{wname}_{d}.json")
            if os.path.exists(out_p):
                continue
            ch = conf["chosen"][wname][d]["b01"]
            acc = {k: {"ev": [], "hours": [], "cont": [], "n_att": []} for k in ch}
            for seed in TEST_SEEDS:
                cfg = load_v2(None, {"seed": seed})
                per = {k: {"ev": np.zeros(len(GRID[k.split('/')[0]])), "hours": 0.0, "cont": np.zeros(len(GRID[k.split('/')[0]])), "n_att": 0}
                       for k in ch}
                for rep in range(2):
                    tr, ev, att, vclass, n_nodes = run(wname, d, "test", seed, rep, "persistent_0.05", cfg)
                    honest = np.ones(tr["seen"].shape[1], bool)
                    honest[list(att)] = False
                    att_sw = {u: a["start"] // WINDOW_S for u, a in att.items()}
                    for key, v in ch.items():
                        stat, pl = key.split("/")
                        al, first = alarms_for(stat, v["par"], pl, tr, ev, n_nodes, GRID[stat], att_sw)
                        per[key]["ev"] += al[:, honest].sum(1)
                        per[key]["hours"] += honest.sum() * (W - w_burn) * WINDOW_S / 3600
                        for u, sw in att_sw.items():
                            per[key]["cont"] += first[:, u] < sw + follow
                        per[key]["n_att"] += len(att)
                for key in ch:
                    for f_ in ("ev", "hours", "cont", "n_att"):
                        val = per[key][f_]
                        acc[key][f_].append(val.tolist() if isinstance(val, np.ndarray) else val)
                print("curves", wname, d, seed, flush=True)
            json.dump({"grid": {s: GRID[s].tolist() for s in GRID}, "par": {k: v["par"] for k, v in ch.items()},
                       "seeds": list(TEST_SEEDS), "rules": acc}, open(out_p, "w", encoding="utf-8"))
            json.dump(env_info(), open(out_p.replace(".json", ".env.json"), "w", encoding="utf-8"), indent=1, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "cal", "test", "curves"])
    ap.add_argument("--registry", default=os.path.join(REPO, "experiments", "v2_registry.csv"))
    ap.add_argument("--worlds", nargs="*")
    a = ap.parse_args()
    if a.cmd == "cache":
        cache(a.registry)
    elif a.cmd == "cal":
        os.environ["MERIT_FL_FORBID_TEST"] = "1"
        cal()
    elif a.cmd == "curves":
        curves(a.worlds)
    else:
        test(a.worlds)
