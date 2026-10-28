#!/usr/bin/env python3
"""R8: confirmation on fresh traces and limits of the self-calibrated statistic the analysis plan.

Fresh SUMO traces 52-61 (6 h); trace s uses the flag rates of test detector s - 10. Two protocols: chained (first
hour repeated six times, as R6/R7) and unchained (the full 6-h trace).

cal       validation (traces 100-102, chained): delta of llrd per world, dataset and budget (R7 rule); validation
          thresholds of the ablation (W1, b01) -> configs/v2/r8.yaml
main      A/B: R7's scenarios + whole-type compromise x placements x {llrh, llrd} -> results/v2/r8_main/<proto>_<w>_<d>.json.gz
ablation  C: reference (per-window / hourly) x threshold (validation / theory), chained W1 -> results/v2/r8_ablation/
sweep     D/E: within-type spread, beta-binomial and burst flags (unchained, W1 links) -> results/v2/r8_sweep/
curves    F: matched-rate comparators (unchained, W0 and W1) -> results/v2/r8_curves/
rates     edge changes per device-hour and returns within 30 s per world, class and protocol -> results/v2/r8_rates.json

python scripts/v2_r8.py cal | main --proto chained|unchained [--worlds W1 ...] | ablation | sweep | curves | rates
"""
import argparse
import gzip
import json
import os
import sys
import time

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import v2_r7 as R7  # noqa: E402
import v2_robust as V  # noqa: E402
from src.v2 import robust as R  # noqa: E402
from src.v2.config import load_v2  # noqa: E402
from src.v2.hfl import load_trace  # noqa: E402
from src.v2.network import compute_links, edge_grid  # noqa: E402
from src.v2.plan import CAL_SEEDS, DATASETS  # noqa: E402

TRACE_SEEDS = tuple(range(52, 62))
DET_OFFSET = 10                                   # trace s -> test detector s - 10
HORIZON = 6 * 3600
WORLDS, LINKS_OF, BUDGETS, H_THEORY, DELTAS = R7.WORLDS, R7.LINKS_OF, R7.BUDGETS, R7.H_THEORY, R7.DELTAS
PLACE = R7.PLACE
STATS = ("llrh", "llrd")
LAG, HIST, M_PSEUDO = 120, 360, 200.0
Z_UCB, CEIL = 2.0, 0.5                           # plan amendment A1 (validation only, before any R8 test run)
SCEN = dict(R7.SCEN, whole_0_6=("whole", 0.6), whole_1_0=("whole", 1.0))
SIGMAS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)
VARIANTS = ([(f"sigma_{s:g}", {"sigma": s}) for s in SIGMAS] +
            [(f"betabin_{r:g}", {"flag": {"kind": "betabin", "rho": r}}) for r in (0.01, 0.03, 0.1)] +
            [(f"burst_{m:g}", {"flag": {"kind": "burst", "m": m, "p_in": 1 / 240, "p_stay": 0.75}}) for m in (3, 10)])
W_BURN, FOLLOW, WIN_PER_H = R7.W_BURN, R7.FOLLOW, R7.WIN_PER_H
OUT = os.path.join(REPO, "results", "v2")
LOWER = float(np.log(0.01))                       # TRW lower barrier (declared benign, restart)


def dump(obj, path):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, default=float)


# ------------------------------------------------------------------ worlds
_links = {}


def world_links(seed, wname, cfg, proto):
    """Edge node per second, edge changes, outage, classes and node count for a trace seed under a protocol.
    chained: the first hour repeated six times (R6/R7); unchained: the full 6-h trace."""
    lw = LINKS_OF[wname]
    key = (seed, lw, proto)
    if key not in _links:
        _links.clear()
        horizon = 3600 if proto == "chained" else HORIZON
        cfg = {**cfg, "net": {**cfg["net"], "horizon_s": horizon}}
        pos, vclass, side = load_trace(cfg, seed)
        n_side = cfg["net"]["edges_per_side"]
        w = V.WORLDS[lw]
        kw = {} if w["hyst"] is None else {"hyst_db": w["hyst"][0], "ttt_s": w["hyst"][1]}
        L = compute_links(pos, edge_grid(side, n_side), seed, **kw)
        links = {"serving": L["serving"], "outage": L["outage"]}
        if proto == "chained":
            links = R.chain_links(links, V.K_HOURS)
        node, ho = R.edge_view(links, n_side, w["block"])
        _links[key] = (node.astype(np.int16), ho, links["outage"], vclass, (n_side // w["block"]) ** 2)
    return _links[key]


def devices(seed, wname, U, prof, sigma=None):
    """R7's devices; W4 is sigma = 0.3. The jitter draws are the same for every sigma (common random numbers)."""
    lam, p_ben, t = V.devices(seed, "W0" if wname == "W0" else "W1", U, prof)
    if sigma is None:
        sigma = 0.3 if wname == "W4" else 0.0
    if sigma > 0:
        z = np.random.default_rng([seed, 4, 98]).normal(0, 1, size=U)
        p_ben = np.clip(p_ben * np.exp(sigma * z), 0, 1)
    return lam, p_ben, t


def attackers(scen, seed, vclass, prof, dev_type, p_ben, delta):
    kind, par = SCEN[scen]
    if kind != "whole":
        return R7.attackers7(scen, seed, vclass, prof, dev_type, p_ben, delta)
    rng = np.random.default_rng([seed, 200 + int(round(100 * par)), 3])
    t = int(rng.choice(np.unique(dev_type)))
    members = np.flatnonzero(dev_type == t)
    pick = rng.choice(members, int(np.ceil(par * len(members))), replace=False)
    out = {}
    for u in pick:
        f = int(rng.integers(len(prof["fam"])))
        out[int(u)] = {"start": int(rng.integers(V.ATTACK_H * 3600, V.ATTACK_H * 3600 + 1800)),
                       "p_det": float(prof["fam_rate"][f]), "vclass": str(vclass[u]), "kind": "persistent", "par": 0.05}
    return out


def run(wname, d, split, seed, rep, scen, cfg, proto, delta, sigma=None, flag=None, tag=0):
    det = seed - DET_OFFSET if split == "test" else seed
    prof = V.profile(split, d, det)
    node, ho, outage, vclass, n_nodes = world_links(seed, wname, cfg, proto)
    U = node.shape[1]
    lam, p_ben, dev_type = devices(seed, wname, U, prof, sigma)
    att = attackers(scen, seed, vclass, prof, dev_type, p_ben, delta)
    rng = np.random.default_rng([seed, rep, list(SCEN).index(scen), WORLDS.index(wname), 188,
                                 0 if proto == "chained" else 1, tag])
    tr = R.simulate(node, ho, outage, lam, p_ben, att, V.WINDOW_S, rng, flag)
    ph, vh = R.hourly_type_reference(tr, dev_type)
    pd_, vd = R.device_reference(tr, ph, vh, LAG, HIST, M_PSEUDO, z=Z_UCB, ceiling=CEIL)
    return tr, {"llrh": (ph, vh), "llrd": (pd_, vd)}, att, vclass, n_nodes, dev_type


def alarms(tr, ref, delta, placement, n_nodes, hs, att_sw, seed):
    p0, valid = ref
    inc = R.llr_increment(tr, p0, valid, delta)
    return R.run_alarms("llr", delta, inc, tr["seen"] > 0, tr["node"], tr["ho"], placement, n_nodes, hs, W_BURN, att_sw,
                        loss=R7.LOSS, rng=np.random.default_rng([seed, 55]))


def contained(first, att_sw, i=0):
    return {u: bool(first[i, u] < sw + FOLLOW) for u, sw in att_sw.items()}


def ablation_alarms(tr, dev_type, refs, ref, delta, n_nodes, hs, att_sw, seed):
    """Ablation arms: 'window' is R6's statistic (per-window type median, R6's increment with its 1e-3 probability
    floor); 'hourly' is R7's llrh."""
    if ref == "hourly":
        return alarms(tr, refs["llrh"], delta, "carry", n_nodes, hs, att_sw, seed)
    yhat, ok, x_share, p0 = R.evidence(tr, dev_type)
    inc = R.increments("llr", delta, tr, yhat, ok, x_share, p0)
    return R.run_alarms("llr", delta, inc, ok, tr["node"], tr["ho"], "carry", n_nodes, hs, W_BURN, att_sw)


# ------------------------------------------------------------------ validation
def cal_ablation():
    """Validation thresholds of the ablation only (W1, b01, carry): the lowest grid threshold within the budget."""
    r7 = yaml.safe_load(open(os.path.join(REPO, "configs", "v2", "r7.yaml"), encoding="utf-8"))["chosen"]
    p_conf = os.path.join(REPO, "configs", "v2", "r8.yaml")
    conf = yaml.safe_load(open(p_conf, encoding="utf-8"))
    grid = V.GRID["llr"]
    for d in DATASETS:
        dl = r7["W1"][d]["b01"]["delta"]
        ab = {ref: {"ev": np.zeros(len(grid)), "hrs": 0.0} for ref in ("window", "hourly")}
        for seed in CAL_SEEDS:
            cfg = load_v2(None, {"seed": seed, "eval.split": "val"})
            for rep in range(3):
                tr, refs, att, vclass, n_nodes, dev_type = run("W1", d, "val", seed, rep, "persistent_0.05", cfg, "chained", dl)
                honest = np.ones(tr["seen"].shape[1], bool)
                honest[list(att)] = False
                att_sw = {u: a["start"] // V.WINDOW_S for u, a in att.items()}
                for ref in ab:
                    al, _ = ablation_alarms(tr, dev_type, refs, ref, dl, n_nodes, grid, att_sw, seed)
                    ab[ref]["ev"] += al[:, honest].sum(1)
                    ab[ref]["hrs"] += honest.sum() * (tr["seen"].shape[0] - W_BURN) / WIN_PER_H
            print("calabl8", d, seed, flush=True)
        for ref in ab:
            rate = ab[ref]["ev"] / ab[ref]["hrs"]
            okh = np.flatnonzero(rate <= BUDGETS["b01"])
            conf["ablation"][d][ref] = {"delta": dl, "h_val": float(grid[okh.min()]) if len(okh) else float(grid[-1]),
                                        "h_theory": H_THEORY["b01"], "val_feasible": bool(len(okh))}
    conf["provenance"] += "; ablation thresholds: scripts/v2_r8.py calabl (window arm = R6's increment)"
    yaml.safe_dump(conf, open(p_conf, "w", encoding="utf-8"), sort_keys=False)
    print("updated ablation in configs/v2/r8.yaml")


def cal():
    r7 = yaml.safe_load(open(os.path.join(REPO, "configs", "v2", "r7.yaml"), encoding="utf-8"))["chosen"]
    hs = [H_THEORY["b01"], H_THEORY["b1"]]
    grid = V.GRID["llr"]
    chosen, abl, rows = {}, {}, []
    for wname in WORLDS:
        for d in DATASETS:
            acc = {dl: {"ev": np.zeros(2), "hrs": 0.0, "cont": np.zeros(2), "n": 0} for dl in DELTAS}
            ab = {ref: {"ev": np.zeros(len(grid)), "hrs": 0.0} for ref in ("window", "hourly")}
            dl_w1 = r7["W1"][d]["b01"]["delta"]
            for seed in CAL_SEEDS:
                cfg = load_v2(None, {"seed": seed, "eval.split": "val"})
                for rep in range(3):
                    tr, refs, att, vclass, n_nodes, dev_type = run(wname, d, "val", seed, rep, "persistent_0.05", cfg,
                                                                  "chained", DELTAS[0])
                    honest = np.ones(tr["seen"].shape[1], bool)
                    honest[list(att)] = False
                    att_sw = {u: a["start"] // V.WINDOW_S for u, a in att.items()}
                    hrs = honest.sum() * (tr["seen"].shape[0] - W_BURN) / WIN_PER_H
                    for dl in DELTAS:
                        al, first = alarms(tr, refs["llrd"], dl, "carry", n_nodes, hs, att_sw, seed)
                        acc[dl]["ev"] += al[:, honest].sum(1)
                        acc[dl]["hrs"] += hrs
                        for u, sw in att_sw.items():
                            acc[dl]["cont"] += first[:, u] < sw + FOLLOW
                        acc[dl]["n"] += len(att)
                    if wname == "W1":
                        for ref in ("window", "hourly"):
                            al, _ = ablation_alarms(tr, dev_type, refs, ref, dl_w1, n_nodes, grid, att_sw, seed)
                            ab[ref]["ev"] += al[:, honest].sum(1)
                            ab[ref]["hrs"] += hrs
                print("cal8", wname, d, seed, flush=True)
            for i, (b, B) in enumerate(BUDGETS.items()):
                cand = []
                for dl in DELTAS:
                    rate, cont = acc[dl]["ev"][i] / acc[dl]["hrs"], acc[dl]["cont"][i] / acc[dl]["n"]
                    rows.append({"world": wname, "dataset": d, "budget": b, "delta": dl, "val_rate": float(rate),
                                 "val_contained": float(cont)})
                    cand.append((rate <= B, cont, dl, rate))
                ok = [c for c in cand if c[0]]
                best = max(ok, key=lambda c: (c[1], -c[2])) if ok else max(cand, key=lambda c: c[2])
                chosen.setdefault(wname, {}).setdefault(d, {})[b] = {"delta": best[2], "h": H_THEORY[b], "val_rate": float(best[3]),
                                                                     "val_contained": float(best[1]), "feasible": bool(best[0])}
            if wname == "W1":
                for ref in ab:
                    rate = ab[ref]["ev"] / ab[ref]["hrs"]
                    okh = np.flatnonzero(rate <= BUDGETS["b01"])
                    abl.setdefault(d, {})[ref] = {"delta": dl_w1, "h_val": float(grid[okh.min()]) if len(okh) else float(grid[-1]),
                                                  "h_theory": H_THEORY["b01"]}
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(REPO, "reports", "v2_r8_calibration.csv"), index=False)
    out = {"provenance": "scripts/v2_r8.py cal (validation, traces 100-102 chained, 3 replicates)",
           "llrd": {"lag": LAG, "hist": HIST, "m": M_PSEUDO, "z": Z_UCB, "ceiling": CEIL, "chosen": chosen}, "ablation": abl}
    yaml.safe_dump(json.loads(json.dumps(out)), open(os.path.join(REPO, "configs", "v2", "r8.yaml"), "w", encoding="utf-8"),
                   sort_keys=False)
    print("wrote configs/v2/r8.yaml")


def configs():
    r7 = yaml.safe_load(open(os.path.join(REPO, "configs", "v2", "r7.yaml"), encoding="utf-8"))["chosen"]
    r8 = yaml.safe_load(open(os.path.join(REPO, "configs", "v2", "r8.yaml"), encoding="utf-8"))
    return r7, r8


def honest_record(al_row, honest, vclass, hrs_dev):
    return {"events": int(al_row[honest].sum()), "hours": float(honest.sum() * hrs_dev),
            "by_class": {c: [int(al_row[honest & (vclass == c)].sum()), float((honest & (vclass == c)).sum() * hrs_dev)]
                         for c in ("veh", "ped", "static")}}


# ------------------------------------------------------------------ A/B
def main(proto, only=None):
    from src.utils.common import env_info
    r7, r8 = configs()
    out_dir = os.path.join(OUT, "r8_main")
    os.makedirs(out_dir, exist_ok=True)
    for wname in WORLDS:
        if only and wname not in only:
            continue
        for d in DATASETS:
            out_p = os.path.join(out_dir, f"{proto}_{wname}_{d}.json.gz")
            if os.path.exists(out_p):
                continue
            dl = {("llrh", b): r7[wname][d][b]["delta"] for b in BUDGETS}
            dl.update({("llrd", b): r8["llrd"]["chosen"][wname][d][b]["delta"] for b in BUDGETS})
            res, t0 = [], time.time()
            for seed in TRACE_SEEDS:
                cfg = load_v2(None, {"seed": seed})
                for rep in range(2):
                    for scen in SCEN:
                        # adaptive attackers depend on llrh's delta, which may differ by budget: simulate per budget
                        sims = [(b,) for b in BUDGETS] if SCEN[scen][0] == "adaptive" else [tuple(BUDGETS)]
                        for bs in sims:
                            tr, refs, att, vclass, n_nodes, _ = run(wname, d, "test", seed, rep, scen, cfg, proto,
                                                                    dl[("llrh", bs[0])])
                            honest = np.ones(tr["seen"].shape[1], bool)
                            honest[list(att)] = False
                            att_sw = {u: a["start"] // V.WINDOW_S for u, a in att.items()}
                            Wn = tr["seen"].shape[0]
                            hrs_dev = (Wn - W_BURN) / WIN_PER_H
                            above = refs["llrd"][0] >= CEIL                   # unmonitored by the ceiling (A1)
                            unmon = float(above[W_BURN:, honest].mean())
                            for b in bs:
                                for stat in STATS:
                                    for pl in PLACE:
                                        al, first = alarms(tr, refs[stat], dl[(stat, b)], pl, n_nodes, [H_THEORY[b]], att_sw, seed)
                                        A = []
                                        for u, a in att.items():
                                            sw, f = att_sw[u], int(first[0, u])
                                            okc = f < sw + FOLLOW
                                            A.append({"u": u, "vclass": a["vclass"], "contained": bool(okc),
                                                      "delay_w": (f - sw + 1) if okc else None, "floor": a.get("floor"),
                                                      "unmon": bool(above[min(sw, Wn - 1), u])})
                                        res.append({"proto": proto, "world": wname, "dataset": d, "seed": seed, "rep": rep,
                                                    "scenario": scen, "budget": b, "stat": stat, "placement": pl,
                                                    "delta": dl[(stat, b)], "h": H_THEORY[b], "unmonitored": unmon,
                                                    "honest": honest_record(al[0], honest, vclass, hrs_dev), "att": A})
                print("main8", proto, wname, d, seed, f"{time.time() - t0:.0f}s", flush=True)
            dump(res, out_p)
            json.dump({"deltas": {f"{k[0]}/{k[1]}": v for k, v in dl.items()}, **env_info()},
                      open(out_p.replace(".json.gz", ".env.json"), "w", encoding="utf-8"), indent=1, default=str)


# ------------------------------------------------------------------ C
def ablation():
    from src.utils.common import env_info
    r7, r8 = configs()
    out_dir = os.path.join(OUT, "r8_ablation")
    os.makedirs(out_dir, exist_ok=True)
    for d in DATASETS:
        out_p = os.path.join(out_dir, f"W1_{d}.json.gz")
        if os.path.exists(out_p):
            continue
        ab = r8["ablation"][d]
        res = []
        for seed in TRACE_SEEDS:
            cfg = load_v2(None, {"seed": seed})
            for rep in range(2):
                tr, refs, att, vclass, n_nodes, dev_type = run("W1", d, "test", seed, rep, "persistent_0.05", cfg, "chained",
                                                               ab["hourly"]["delta"])
                honest = np.ones(tr["seen"].shape[1], bool)
                honest[list(att)] = False
                att_sw = {u: a["start"] // V.WINDOW_S for u, a in att.items()}
                hrs_dev = (tr["seen"].shape[0] - W_BURN) / WIN_PER_H
                for ref in ("window", "hourly"):
                    hs = [ab[ref]["h_val"], ab[ref]["h_theory"]]
                    al, first = ablation_alarms(tr, dev_type, refs, ref, ab[ref]["delta"], n_nodes, hs, att_sw, seed)
                    for i, rule in enumerate(("validation", "theory")):
                        c = contained(first, att_sw, i)
                        res.append({"dataset": d, "seed": seed, "rep": rep, "reference": ref, "threshold": rule, "h": hs[i],
                                    "events": int(al[i, honest].sum()), "hours": float(honest.sum() * hrs_dev),
                                    "contained": int(sum(c.values())), "n_att": len(c)})
            print("ablation8", d, seed, flush=True)
        dump(res, out_p)
        json.dump(env_info(), open(out_p.replace(".json.gz", ".env.json"), "w", encoding="utf-8"), indent=1, default=str)


# ------------------------------------------------------------------ D/E
def sweep(only=None):
    from src.utils.common import env_info
    r7, r8 = configs()
    out_dir = os.path.join(OUT, "r8_sweep")
    os.makedirs(out_dir, exist_ok=True)
    hs = [H_THEORY["b01"], H_THEORY["b1"]]
    for vname, v in VARIANTS:
        if only and vname not in only:
            continue
        for d in DATASETS:
            out_p = os.path.join(out_dir, f"{vname}_{d}.json.gz")
            if os.path.exists(out_p):
                continue
            res, t0 = [], time.time()
            for seed in TRACE_SEEDS:
                cfg = load_v2(None, {"seed": seed})
                for rep in range(2):
                    tr, refs, att, vclass, n_nodes, _ = run("W1", d, "test", seed, rep, "persistent_0.05", cfg, "unchained",
                                                            DELTAS[0], sigma=v.get("sigma", 0.0), flag=v.get("flag"))
                    honest = np.ones(tr["seen"].shape[1], bool)
                    honest[list(att)] = False
                    att_sw = {u: a["start"] // V.WINDOW_S for u, a in att.items()}
                    hrs = float(honest.sum() * (tr["seen"].shape[0] - W_BURN) / WIN_PER_H)
                    unmon = float((refs["llrd"][0] >= CEIL)[W_BURN:, honest].mean())
                    for stat in STATS:
                        for dl in DELTAS:
                            for pl in ("carry", "memory", "reset"):
                                al, first = alarms(tr, refs[stat], dl, pl, n_nodes, hs, att_sw, seed)
                                res.append({"variant": vname, "dataset": d, "seed": seed, "rep": rep, "stat": stat, "delta": dl,
                                            "placement": pl, "events": al[:, honest].sum(1).tolist(), "hours": hrs,
                                            "contained": [int(sum(contained(first, att_sw, i).values())) for i in range(2)],
                                            "n_att": len(att_sw), "unmonitored": unmon if stat == "llrd" else 0.0})
                print("sweep8", vname, d, seed, f"{time.time() - t0:.0f}s", flush=True)
            dump(res, out_p)
            json.dump({"variant": v, **env_info()}, open(out_p.replace(".json.gz", ".env.json"), "w", encoding="utf-8"),
                      indent=1, default=str)


# ------------------------------------------------------------------ F
def curves(only=None):
    from src.utils.common import env_info
    r7, r8 = configs()
    out_dir = os.path.join(OUT, "r8_curves")
    os.makedirs(out_dir, exist_ok=True)
    g_llr, g_frac = V.GRID["llr"], V.GRID["ewma"]
    for wname in ("W0", "W1"):
        if only and wname not in only:
            continue
        for d in DATASETS:
            out_p = os.path.join(out_dir, f"{wname}_{d}.json.gz")
            if os.path.exists(out_p):
                continue
            dh, dd = r7[wname][d]["b01"]["delta"], r8["llrd"]["chosen"][wname][d]["b01"]["delta"]
            acc = {}
            for seed in TRACE_SEEDS:
                cfg = load_v2(None, {"seed": seed})
                for rep in range(2):
                    tr, refs, att, vclass, n_nodes, _ = run(wname, d, "test", seed, rep, "persistent_0.05", cfg, "unchained", dh)
                    honest = np.ones(tr["seen"].shape[1], bool)
                    honest[list(att)] = False
                    att_sw = {u: a["start"] // V.WINDOW_S for u, a in att.items()}
                    hrs = float(honest.sum() * (tr["seen"].shape[0] - W_BURN) / WIN_PER_H)
                    ok = tr["seen"] > 0
                    pooled = R.hourly_pooled_reference(tr)
                    ph, vh = refs["llrh"]
                    yhat = np.where(ok, tr["flagged"] / np.maximum(tr["seen"], 1), 0.0)
                    rules = {}
                    for pl in ("carry", "reset"):
                        rules[f"llrh/{pl}"] = lambda pl=pl: alarms(tr, refs["llrh"], dh, pl, n_nodes, g_llr, att_sw, seed)
                        inc_s = R.llr_increment(tr, pooled[0], pooled[1], dh)
                        rules[f"sprt/{pl}"] = lambda pl=pl, inc_s=inc_s: R.run_alarms(
                            "sprt", LOWER, inc_s, ok, tr["node"], tr["ho"], pl, n_nodes, g_llr, W_BURN, att_sw)
                        for lam in (0.9, 0.98):
                            rules[f"beta{lam:g}/{pl}"] = lambda pl=pl, lam=lam: R.run_alarms_beta(
                                tr["flagged"], tr["seen"], ph, ok & vh, lam, tr["node"], tr["ho"], pl, g_frac, W_BURN, att_sw)
                        for lam in (0.02, 0.05, 0.1):
                            inc_e = np.where(vh, yhat - ph, 0.0)
                            rules[f"ewma{lam:g}/{pl}"] = lambda pl=pl, lam=lam, inc_e=inc_e: R.run_alarms(
                                "ewma", lam, inc_e, ok & vh, tr["node"], tr["ho"], pl, n_nodes, g_frac, W_BURN, att_sw)
                    rules["llrd/carry"] = lambda: alarms(tr, refs["llrd"], dd, "carry", n_nodes, g_llr, att_sw, seed)
                    for name, fn in rules.items():
                        al, first = fn()
                        a = acc.setdefault(name, {"ev": [], "hours": [], "cont": [], "n_att": []})
                        if rep == 0:
                            for k in a:
                                a[k].append(np.zeros(len(g_llr) if name.startswith(("llr", "sprt")) else len(g_frac))
                                            if k in ("ev", "cont") else 0.0)
                        a["ev"][-1] = a["ev"][-1] + al[:, honest].sum(1)
                        a["hours"][-1] += hrs
                        a["cont"][-1] = a["cont"][-1] + sum(first[:, u] < sw + FOLLOW for u, sw in att_sw.items())
                        a["n_att"][-1] += len(att_sw)
                print("curves8", wname, d, seed, flush=True)
            for a in acc.values():
                for k in ("ev", "cont"):
                    a[k] = [np.asarray(x).tolist() for x in a[k]]
            dump({"grid_llr": g_llr.tolist(), "grid_frac": g_frac.tolist(), "delta_h": dh, "delta_d": dd,
                  "seeds": list(TRACE_SEEDS), "rules": acc}, out_p)
            json.dump(env_info(), open(out_p.replace(".json.gz", ".env.json"), "w", encoding="utf-8"), indent=1, default=str)


# ------------------------------------------------------------------ rates
def rates():
    out = {}
    for proto in ("chained", "unchained"):
        for wname in ("W0", "W2", "W3"):                      # W1 and W4 share W0's links
            acc = {c: {"changes": 0, "returns": 0, "hours": 0.0} for c in ("veh", "ped", "static")}
            for seed in TRACE_SEEDS:
                cfg = load_v2(None, {"seed": seed})
                node, ho, _, vclass, _ = world_links(seed, wname, cfg, proto)
                T = node.shape[0]
                for u in range(node.shape[1]):
                    t = np.flatnonzero(ho[:, u])
                    a = acc[str(vclass[u])]
                    a["changes"] += len(t)
                    a["hours"] += T / 3600
                    if len(t) >= 2:
                        back = node[t[1:], u] == node[t[:-1] - 1, u]
                        a["returns"] += int((back & (np.diff(t) <= 30)).sum())
            out[f"{proto}/{wname}"] = {c: {"per_h": a["changes"] / a["hours"],
                                            "return_30s": a["returns"] / max(a["changes"], 1)} for c, a in acc.items()}
            print("rates8", proto, wname, out[f"{proto}/{wname}"], flush=True)
    json.dump(out, open(os.path.join(OUT, "r8_rates.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cal", "calabl", "main", "ablation", "sweep", "curves", "rates"])
    ap.add_argument("--proto", choices=["chained", "unchained"], default="chained")
    ap.add_argument("--worlds", nargs="*")
    a = ap.parse_args()
    if a.cmd in ("cal", "calabl"):
        os.environ["MERIT_FL_FORBID_TEST"] = "1"
        cal() if a.cmd == "cal" else cal_ablation()
    elif a.cmd == "main":
        main(a.proto, a.worlds)
    elif a.cmd == "ablation":
        ablation()
    elif a.cmd == "sweep":
        sweep(a.worlds)
    elif a.cmd == "curves":
        curves(a.worlds)
    else:
        rates()
