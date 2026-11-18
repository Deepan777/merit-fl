#!/usr/bin/env python3
"""R10: real-map mobility, validity of a locally calibrated threshold, and a second real-traffic source
(reports/stats/ANALYSIS_PLAN_V2_R10.md). No statistic, parameter or threshold rule is tuned here: everything is frozen
from R7-R9.

A  rates       edge changes per device-hour on the real-map traces 62-71 -> results/v2/r10_rates.json
   main        R8's unchained protocol on the real-map traces (W0-W3, persistent 5%, 0.1%/h) -> results/v2/r10_map/
B  split       R9's matched-rate replay restricted to each half of the 5G-NIDD capture sessions -> results/v2/r10_split/
C  hostprep    CIC-IDS2017 flows with host addresses: check against the official files, features, device-days
               -> results/v2/r10_hosts/data/
   hostflags   detectors cross-fitted by day (seeds 0-1), window flows and flags per device-day -> r10_hosts/flags/
   hostdescribe, hostreplay   R9's description and derived-threshold replay on the host flags -> r10_hosts/
   hostdays    matched-rate replay restricted to each day -> results/v2/r10_days/

python scripts/v2_r10.py rates | main [--worlds W0 ...] | split | hostprep | hostflags | hostdescribe | hostreplay | hostdays
"""
import argparse
import json
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import v2_r7 as R7  # noqa: E402
import v2_r8 as R8  # noqa: E402
import v2_r9 as R9  # noqa: E402
import v2_robust as V  # noqa: E402
from src.v2 import robust as R  # noqa: E402
from src.v2.config import load_v2  # noqa: E402
from src.v2.plan import DATASETS  # noqa: E402

OUT = os.path.join(REPO, "results", "v2")
MAP_SEEDS, MAP_DET_OFFSET = tuple(range(62, 72)), 20          # real-map trace s -> test detector s - 20
FRESH_SEEDS = tuple(range(52, 62))                            # R8's fresh grid traces (replays)
WORLDS_A, PLACE = ("W0", "W1", "W2", "W3"), ("carry", "memory", "reset")
SCEN_A, BUDGET_A = "persistent_0.05", "b01"
SPLIT_SEED = 10
HOSTS = os.path.join(OUT, "r10_hosts")
HOST_ZIP = os.path.join(REPO, "data", "cicids2017", "raw_hosts", "GeneratedLabelledFlows.zip")
HOST_META = os.path.join(REPO, "data", "cicids2017", "processed", "family.json")
HOST_NET = "192.168.10."
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
HOST_DET_SEEDS, TRAIN_CAP, CLASS_FLOOR = (0, 1), 400_000, 10_000


# ------------------------------------------------------------------ A: real-map traces
def rates():
    out = {}
    for wname in ("W0", "W2", "W3"):                          # W1 shares W0's links
        acc = {c: {"changes": 0, "returns": 0, "hours": 0.0} for c in ("veh", "ped", "static")}
        for seed in MAP_SEEDS:
            cfg = load_v2(None, {"seed": seed})
            node, ho, outage, vclass, _ = R8.world_links(seed, wname, cfg, "unchained")
            T = node.shape[0]
            for u in range(node.shape[1]):
                t = np.flatnonzero(ho[:, u])
                a = acc[str(vclass[u])]
                a["changes"] += len(t)
                a["hours"] += T / 3600
                if len(t) >= 2:
                    back = node[t[1:], u] == node[t[:-1] - 1, u]
                    a["returns"] += int((back & (np.diff(t) <= 30)).sum())
            if wname == "W0":
                acc.setdefault("outage", []).append(float(outage.mean()))
        o = acc.pop("outage", None)
        out[wname] = {c: {"per_h": a["changes"] / a["hours"], "return_30s": a["returns"] / max(a["changes"], 1)}
                      for c, a in acc.items()}
        if o:
            out[wname]["outage_share"] = float(np.mean(o))
        print("rates10", wname, out[wname], flush=True)
    json.dump(out, open(os.path.join(OUT, "r10_rates.json"), "w", encoding="utf-8"), indent=1)


def main_map(only=None):
    """R8.main (unchained) on the real-map traces, restricted to the scenario and budget of the plan."""
    from src.utils.common import env_info
    R8.DET_OFFSET = MAP_DET_OFFSET
    r7, r8 = R8.configs()
    out_dir = os.path.join(OUT, "r10_map")
    os.makedirs(out_dir, exist_ok=True)
    b = BUDGET_A
    for wname in WORLDS_A:
        if only and wname not in only:
            continue
        for d in DATASETS:
            out_p = os.path.join(out_dir, f"unchained_{wname}_{d}.json.gz")
            if os.path.exists(out_p):
                continue
            dl = {"llrh": r7[wname][d][b]["delta"], "llrd": r8["llrd"]["chosen"][wname][d][b]["delta"]}
            res, t0 = [], time.time()
            for seed in MAP_SEEDS:
                cfg = load_v2(None, {"seed": seed})
                for rep in range(2):
                    tr, refs, att, vclass, n_nodes, _ = R8.run(wname, d, "test", seed, rep, SCEN_A, cfg, "unchained", dl["llrh"])
                    honest = np.ones(tr["seen"].shape[1], bool)
                    honest[list(att)] = False
                    att_sw = {u: a["start"] // V.WINDOW_S for u, a in att.items()}
                    Wn = tr["seen"].shape[0]
                    hrs_dev = (Wn - R8.W_BURN) / R8.WIN_PER_H
                    above = refs["llrd"][0] >= R8.CEIL
                    unmon = float(above[R8.W_BURN:, honest].mean())
                    for stat in R8.STATS:
                        for pl in PLACE:
                            al, first = R8.alarms(tr, refs[stat], dl[stat], pl, n_nodes, [R8.H_THEORY[b]], att_sw, seed)
                            A = []
                            for u, a in att.items():
                                sw, f = att_sw[u], int(first[0, u])
                                okc = f < sw + R8.FOLLOW
                                A.append({"u": u, "vclass": a["vclass"], "contained": bool(okc),
                                          "delay_w": (f - sw + 1) if okc else None,
                                          "unmon": bool(above[min(sw, Wn - 1), u])})
                            res.append({"proto": "unchained", "world": wname, "dataset": d, "seed": seed, "rep": rep,
                                        "scenario": SCEN_A, "budget": b, "stat": stat, "placement": pl,
                                        "delta": dl[stat], "h": R8.H_THEORY[b], "unmonitored": unmon,
                                        "honest": R8.honest_record(al[0], honest, vclass, hrs_dev), "att": A})
                print("map10", wname, d, seed, f"{time.time() - t0:.0f}s", flush=True)
            R8.dump(res, out_p)
            json.dump({"deltas": dl, "det_offset": MAP_DET_OFFSET, **env_info()},
                      open(out_p.replace(".json.gz", ".env.json"), "w", encoding="utf-8"), indent=1, default=str)


# ------------------------------------------------------------------ matched-rate replay on subsets (B, C3)
def run_curves(jobs, out_dir, dataset):
    """R9's replay_curves for several subsets of device-sessions at once (the links of a trace are computed once).
    jobs: list of dicts with name, det, tag, ds, n, f, prof. Per job: events, contained and hours on the threshold grid
    per statistic and placement, in total and per trace seed -> out_dir/<name>_det<det>.json.gz"""
    from src.utils.common import env_info
    os.makedirs(out_dir, exist_ok=True)
    jobs = [j for j in jobs if not os.path.exists(os.path.join(out_dir, f"{j['name']}_det{j['det']}.json.gz"))]
    if not jobs:
        return
    r7, r8 = R8.configs()
    ch = R9.chosen()
    dl = {"llrh": r7["W1"][dataset]["b01"]["delta"], "llrd": r8["llrd"]["chosen"]["W1"][dataset]["b01"]["delta"]}
    dl["llrr"] = dl["llrd"]
    grid = V.GRID["llr"]
    acc = [{} for _ in jobs]
    t0 = time.time()
    for seed in FRESH_SEEDS:
        cfg = load_v2(None, {"seed": seed})
        node, ho, outage, vclass, n_nodes = R8.world_links(seed, "W1", cfg, "unchained")
        U, W = node.shape[1], node.shape[0] // V.WINDOW_S
        for ji, j in enumerate(jobs):
            for rep in range(2):
                rng = np.random.default_rng([seed, j["det"], rep, 1010, j["tag"]])
                seq_n, seq_f, dev_type, ues = R9.replay_sequences(j["ds"], j["n"], j["f"], U, W, rng)
                att = V.attackers(list(V.SCENARIOS).index("persistent_0.05"), seed, vclass, j["prof"])
                tr = R.simulate_replay(node, ho, outage, seq_n, seq_f, att, V.WINDOW_S, rng)
                honest = np.ones(U, bool)
                honest[list(att)] = False
                att_sw = {u: a["start"] // V.WINDOW_S for u, a in att.items()}
                hrs = float(honest.sum() * (tr["seen"].shape[0] - R8.W_BURN) / R8.WIN_PER_H)
                ph, vh = R.hourly_type_reference(tr, dev_type)
                pd_, vd = R.device_reference(tr, ph, vh, R8.LAG, R8.HIST, R8.M_PSEUDO, z=R8.Z_UCB, ceiling=R8.CEIL)
                refs = {"llrh": (ph, vh), "llrd": (pd_, vd)}
                rho = R9.llrr_rho(tr, dev_type, ch["source"])
                for stat in R9.STATS:
                    for pl in PLACE:
                        if stat == "llrr":
                            al, first = R9.llrr_alarms(tr, refs["llrd"], rho, ch["kappa"], dl[stat], pl, n_nodes, grid, att_sw, seed)
                        else:
                            al, first = R8.alarms(tr, refs[stat], dl[stat], pl, n_nodes, grid, att_sw, seed)
                        ev = al[:, honest].sum(1).astype(float)
                        ct = sum((first[:, u] < sw + R8.FOLLOW).astype(float) for u, sw in att_sw.items())
                        a = acc[ji].setdefault(f"{stat}/{pl}", {"ev": np.zeros(len(grid)), "cont": np.zeros(len(grid)),
                                                                "hours": 0.0, "n_att": 0, "by_seed": {}})
                        bs = a["by_seed"].setdefault(str(seed), {"ev": np.zeros(len(grid)), "cont": np.zeros(len(grid)),
                                                                 "hours": 0.0, "n_att": 0})
                        for x in (a, bs):
                            x["ev"] += ev
                            x["cont"] += ct
                            x["hours"] += hrs
                            x["n_att"] += len(att_sw)
        print("curves10", os.path.basename(out_dir), seed, f"{time.time() - t0:.0f}s", flush=True)

    def lists(o):
        return {k: lists(v) for k, v in o.items()} if isinstance(o, dict) else o.tolist() if isinstance(o, np.ndarray) else o
    for j, a in zip(jobs, acc):
        out_p = os.path.join(out_dir, f"{j['name']}_det{j['det']}.json.gz")
        R8.dump({"grid": grid.tolist(), "deltas": dl, "n_device_sessions": len(j["ds"]),
                 "devices": sorted(set(d["src"] for d in j["ds"])),
                 "observed_hours": float(sum(d["hours"] for d in j["ds"])), "rules": lists(a)}, out_p)
        json.dump({"post_hoc": True, "llrr": ch, **env_info()},
                  open(out_p.replace(".json.gz", ".env.json"), "w", encoding="utf-8"), indent=1, default=str)


def subset(ds, n, f, idx):
    return [ds[i] for i in idx], [n[i] for i in idx], [f[i] for i in idx]


def split():
    """B: the ten capture sessions of each base station are split into two halves (seeded permutation)."""
    info = json.load(open(os.path.join(R9.DATA_DIR, "sessions.json"), encoding="utf-8"))
    half = {}
    for bs in R9.BS:
        names = sorted(s["session"] for s in info["sessions"] if s["bs"] == bs)
        perm = np.random.default_rng([SPLIT_SEED, R9.BS.index(bs)]).permutation(len(names))
        for k, i in enumerate(perm):
            half[(bs, names[i])] = "H1" if k < len(names) // 2 else "H2"
    json.dump({f"{k[0]}/{k[1]}": v for k, v in half.items()},
              open(os.path.join(OUT, "r10_split_sessions.json"), "w", encoding="utf-8"), indent=1)
    jobs = []
    for det in R9.DET_SEEDS:
        ds, n, f, prof = R9.load_flags(det)
        for hi, h in enumerate(("H1", "H2")):
            idx = [i for i, d in enumerate(ds) if half[(d["bs"], d["session"])] == h]
            sds, sn, sf = subset(ds, n, f, idx)
            jobs.append({"name": h, "det": det, "tag": hi, "ds": sds, "n": sn, "f": sf, "prof": prof})
    run_curves(jobs, os.path.join(OUT, "r10_split"), "5g_nidd")


# ------------------------------------------------------------------ C: CIC-IDS2017 hosts
def day_seconds(ts, rng):
    """Seconds of the day from 'd/m/yyyy h:mm[:ss]' on a 12-hour clock without AM/PM (capture hours 08:00-18:00).
    Minute-resolution stamps get a uniformly random second (plan R10, part C)."""
    parts = ts.astype(str).str.strip().str.split(" ", expand=True)[1].str.split(":", expand=True)
    hh, mm = parts[0].astype(int).to_numpy(), parts[1].astype(int).to_numpy()
    hh = np.where(hh < 8, hh + 12, hh)
    ss = parts[2].astype(int).to_numpy() if parts.shape[1] > 2 else rng.integers(0, 60, size=len(hh))
    return hh * 3600 + mm * 60 + ss, parts.shape[1] > 2


def stratified_cap(y, cap, floor, rng):
    """Indices of at most about cap rows: every class keeps min(n_c, max(floor, share of the rest))."""
    cls, cnt = np.unique(y, return_counts=True)
    if len(y) <= cap:
        return np.arange(len(y))
    lo, hi = 0.0, 1.0
    for _ in range(60):
        frac = (lo + hi) / 2
        tot = sum(min(c, max(floor, int(round(c * frac)))) for c in cnt)
        lo, hi = (frac, hi) if tot < cap else (lo, frac)
    keep = []
    for c, k in zip(cls, cnt):
        m = min(k, max(floor, int(round(k * lo))))
        keep.append(rng.choice(np.flatnonzero(y == c), m, replace=False))
    return np.sort(np.concatenate(keep))


def host_prep():
    """One day at a time (memory): check against the official files, features, training rows, device-days."""
    from src.data.preprocess import CICIDS_FAMILY, _cicids_fine
    meta = json.load(open(HOST_META, encoding="utf-8"))
    classes, feats = meta["classes"], meta["features"]
    official = {os.path.basename(f["path"]): f["rows"] for f in meta["files"]}
    data_dir = os.path.join(HOSTS, "data")
    os.makedirs(data_dir, exist_ok=True)
    z = zipfile.ZipFile(HOST_ZIP)
    names = sorted(n for n in z.namelist() if n.endswith(".csv"))
    need = set(feats) | {"Source IP", "Timestamp", "Label"}
    ds_info, excluded, check, files_info = [], [], [], []
    for di, day in enumerate(DAYS):
        Xtr, ytr, Xatt, yatt, host_rows, attackers = [], [], [], [], {}, set()
        for name in [n for n in names if os.path.basename(n).lower().startswith(day.lower())]:
            base = os.path.basename(name)
            df = pd.read_csv(z.open(name), encoding="latin-1", low_memory=False, usecols=lambda c: c.strip() in need)
            df.columns = [c.strip() for c in df.columns]
            df = df.dropna(subset=["Label"]).reset_index(drop=True)
            fine = df["Label"].map(_cicids_fine)
            fam = fine.map(CICIDS_FAMILY).fillna("?").to_numpy()
            benign = fam == "Benign"
            dur = pd.to_numeric(df["Flow Duration"], errors="coerce")
            check.append({"file": base, "rows": int(len(df)), "official_rows": official.get(base),
                          "benign": int(benign.sum()), "flow_duration_sum": float(dur.sum())})
            rng = np.random.default_rng([10, names.index(name)])
            t, has_sec = day_seconds(df["Timestamp"], rng)
            X = R9.transform(df, meta)
            known = np.isin(fam, classes)                  # Infiltration and Heartbleed are not detector classes
            y = np.array([classes.index(x) if k else -1 for x, k in zip(fam, known)], dtype=np.int64)
            Xs, ys = R9.dedup(X[known], y[known])
            Xtr.append(Xs)
            ytr.append(ys)
            att_rows = np.flatnonzero(known & ~benign)
            if len(att_rows) > R9.ATT_SAMPLE:
                att_rows = np.sort(rng.choice(att_rows, R9.ATT_SAMPLE, replace=False))
            Xatt.append(X[att_rows])
            yatt.append(y[att_rows])
            src = df["Source IP"].astype(str).str.strip().to_numpy()
            attackers |= set(src[~benign])
            for s in sorted(set(src[benign])):
                if s.startswith(HOST_NET):
                    rows = np.flatnonzero(benign & (src == s))
                    host_rows.setdefault(s, []).append((t[rows], X[rows]))
            files_info.append({"file": base, "day": day, "rows": int(len(df)), "seconds_in_timestamp": bool(has_sec),
                               "labels": {k: int(v) for k, v in fine.value_counts().items()}})
            print("hostprep", base, len(df), "seconds" if has_sec else "minutes", flush=True)
            del df, X
        Xh, h_ds, h_w = [], [], []
        for s, parts in sorted(host_rows.items()):
            t = np.concatenate([p[0] for p in parts])
            span = max(t.max() - t.min(), R9.WINDOW_S)
            rec = {"bs": day, "session": day, "day": di, "src": s, "flows": int(len(t)), "rate": float(len(t) / span)}
            why = ("fewer than 30 benign flows" if len(t) < R9.MIN_FLOWS else
                   "sends attack-labelled flows that day" if s in attackers else
                   "more than 50 flows/s" if rec["rate"] > R9.MAX_RATE else None)
            if why:
                excluded.append({**rec, "reason": why})
                continue
            w = ((t - t.min()) // R9.WINDOW_S).astype(int)
            rec.update(id=len(ds_info), windows=int(w.max() + 1), hours=float((w.max() + 1) * R9.WINDOW_S / 3600))
            ds_info.append(rec)
            Xh.append(np.concatenate([p[1] for p in parts]))
            h_ds.append(np.full(len(t), rec["id"]))
            h_w.append(w)
        Xd, yd = R9.dedup(np.concatenate(Xtr), np.concatenate(ytr))
        np.savez_compressed(os.path.join(data_dir, f"{day}.npz"), X_tr=Xd, y_tr=yd, X_ue=np.concatenate(Xh),
                            ue_ds=np.concatenate(h_ds), ue_w=np.concatenate(h_w),
                            X_att=np.concatenate(Xatt), y_att=np.concatenate(yatt))
        print("hostprep", day, "train rows", len(yd), "host flows", sum(len(x) for x in Xh),
              "device-days", sum(1 for d in ds_info if d["session"] == day), flush=True)
        del Xtr, ytr, Xatt, yatt, host_rows, Xh, Xd
    ok = all(c["rows"] == c["official_rows"] for c in check)
    json.dump({"classes": classes, "device_sessions": ds_info, "sessions": files_info, "excluded": excluded,
               "check_against_official": {"all_row_counts_match": ok, "files": check},
               "archive": {"file": os.path.basename(HOST_ZIP), "bytes": os.path.getsize(HOST_ZIP),
                           "source": "https://huggingface.co/datasets/bencorn/CICIDS2017 (mirror of the CIC archive)"}},
              open(os.path.join(data_dir, "sessions.json"), "w", encoding="utf-8"), indent=1)
    print("hostprep: row counts match the official files:", ok, "| device-days", len(ds_info), "| excluded", len(excluded))


def host_flags():
    import torch
    from src.evaluation.metrics import predict_proba
    from src.utils.common import env_info
    data_dir, flag_dir = os.path.join(HOSTS, "data"), os.path.join(HOSTS, "flags")
    os.makedirs(flag_dir, exist_ok=True)
    info = json.load(open(os.path.join(data_dir, "sessions.json"), encoding="utf-8"))
    classes, ds = info["classes"], info["device_sessions"]
    K = len(classes)
    for seed in HOST_DET_SEEDS:
        out_p = os.path.join(flag_dir, f"seed{seed}.npz")
        if os.path.exists(out_p):
            continue
        n_ds = [np.zeros(d["windows"], np.int32) for d in ds]
        f_ds = [np.zeros(d["windows"], np.int32) for d in ds]
        det, att_n, summary = np.zeros((len(DAYS), K)), np.zeros((len(DAYS), K)), {}
        for fi, day in enumerate(DAYS):
            t0 = time.time()
            Xs, ys = [], []
            for other in DAYS:
                if other != day:
                    zz = np.load(os.path.join(data_dir, f"{other}.npz"))
                    Xs.append(zz["X_tr"])
                    ys.append(zz["y_tr"])
            X, y = R9.dedup(np.concatenate(Xs), np.concatenate(ys))
            del Xs, ys
            keep = stratified_cap(y, TRAIN_CAP, CLASS_FLOOR, np.random.default_rng([seed, fi, 77]))
            X, y = X[keep], y[keep]
            model, summ = R9.train_detector(X, y, seed, K)
            d = np.load(os.path.join(data_dir, f"{day}.npz"))
            flag_h = (1.0 - predict_proba(model, d["X_ue"], torch.device("cpu"))[:, 0]) >= 0.5
            for i in np.unique(d["ue_ds"]):
                m = d["ue_ds"] == i
                n_ds[i] = np.bincount(d["ue_w"][m], minlength=ds[i]["windows"]).astype(np.int32)
                f_ds[i] = np.bincount(d["ue_w"][m], weights=flag_h[m], minlength=ds[i]["windows"]).astype(np.int32)
            if len(d["y_att"]):
                flag_att = (1.0 - predict_proba(model, d["X_att"], torch.device("cpu"))[:, 0]) >= 0.5
                for k in range(1, K):
                    m = d["y_att"] == k
                    det[fi, k], att_n[fi, k] = flag_att[m].sum(), m.sum()
            summary[day] = {**summ, "train_classes": {classes[int(c)]: int(k) for c, k in zip(*np.unique(y, return_counts=True))},
                            "benign_flag_share": float(flag_h.mean()), "seconds": round(time.time() - t0)}
            print("hostflags", seed, day, "holdout macro-F1", round(summ["holdout_macro_f1"], 4), "epoch", summ["best_epoch"],
                  "benign flagged", round(float(flag_h.mean()), 4), f"{time.time() - t0:.0f}s", flush=True)
        fam = [c for c in classes if c != "Benign"]
        rate = det[:, 1:].sum(0) / np.maximum(att_n[:, 1:].sum(0), 1)
        np.savez_compressed(out_p, n=np.concatenate(n_ds), f=np.concatenate(f_ds),
                            off=np.cumsum([0] + [d["windows"] for d in ds]), fam=np.array(fam), fam_rate=rate,
                            fam_rate_fold=det[:, 1:] / np.maximum(att_n[:, 1:], 1), fam_n=att_n[:, 1:])
        json.dump({"seed": seed, "folds": summary, **env_info()},
                  open(out_p.replace(".npz", ".env.json"), "w", encoding="utf-8"), indent=1, default=str)


def use_hosts():
    """Point R9's description and replay at the host flags (same file formats)."""
    R9.DATA_DIR, R9.FLAG_DIR = os.path.join(HOSTS, "data"), os.path.join(HOSTS, "flags")
    R9.OUT, R9.DET_SEEDS, R9.REPLAY_DATASET = HOSTS, HOST_DET_SEEDS, "cicids2017"


def host_days(seen=False):
    """C3: matched-rate replay restricted to the device-days of each day.
    seen=True is a sensitivity analysis that is not in the plan. It was added after the cross-fitted detection rates
    were seen (before any containment result of C): every attack family of CIC-IDS2017 occurs on one day only, so a
    detector cross-fitted by day never saw the family it scores, and five of six families are flagged in under 5% of
    flows. This variant keeps the real benign flags and gives the attackers the detection rates of the manuscript's
    CIC-IDS2017 test detectors (mean over test seeds 42-51), i.e. attacks the detector was trained on."""
    use_hosts()
    jobs = []
    fam_seen = np.mean([V.profile("test", "cicids2017", s)["fam_rate"] for s in range(42, 52)], axis=0)
    for det in HOST_DET_SEEDS:
        ds, n, f, prof = R9.load_flags(det)
        if seen:
            prof = {**prof, "fam_rate": fam_seen}
        for di, day in enumerate(DAYS):
            idx = [i for i, d in enumerate(ds) if d["session"] == day]
            sds, sn, sf = subset(ds, n, f, idx)
            jobs.append({"name": day, "det": det, "tag": 20 + di, "ds": sds, "n": sn, "f": sf, "prof": prof})
    run_curves(jobs, os.path.join(OUT, "r10_days_seen" if seen else "r10_days"), "cicids2017")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rates", "main", "split", "hostprep", "hostflags", "hostdescribe", "hostreplay", "hostdays",
                                    "hostdays_seen"])
    ap.add_argument("--worlds", nargs="*")
    a = ap.parse_args()
    if a.cmd == "rates":
        rates()
    elif a.cmd == "main":
        main_map(a.worlds)
    elif a.cmd == "split":
        split()
    elif a.cmd == "hostprep":
        host_prep()
    elif a.cmd == "hostflags":
        host_flags()
    elif a.cmd in ("hostdays", "hostdays_seen"):
        host_days(seen=a.cmd.endswith("_seen"))
    else:
        use_hosts()
        {"hostdescribe": R9.describe, "hostreplay": R9.replay}[a.cmd]()
