#!/usr/bin/env python3
"""R9: real per-device benign flags and a dispersion-robust statistic the analysis plan.

prep      5G-NIDD per-session flows -> the study's feature transform; per base station: deduplicated training rows,
          UE benign flows of the included device-sessions, sampled attack flows -> results/v2/r9_data/
flags     detectors cross-fitted by base station (A: train BS2, score BS1; B: train BS1, score BS2), seeds 0-2 ->
          window flows and flags per device-session, family detection rates -> results/v2/r9_flags/seed<k>.npz
describe  A: flag rate, intra-window correlation and autocorrelation per UE -> results/v2/r9_describe.json
cal       B: llrr candidates on synthetic validation (traces 100-102, chained) -> configs/v2/r9.yaml
sweep     C: llrr on R8's synthetic flag variants (traces 52-61, unchained, W1 links) -> results/v2/r9_sweep/
replay    D: real-flag replay (traces 52-61, unchained, W1 links) -> results/v2/r9_replay/

python scripts/v2_r9.py prep | flags | describe | cal | sweep | replay
"""
import argparse
import copy
import gzip
import json
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import v2_r7 as R7  # noqa: E402
import v2_r8 as R8  # noqa: E402
import v2_robust as V  # noqa: E402
from src.v2 import robust as R  # noqa: E402
from src.v2.config import load_v2  # noqa: E402
from src.v2.plan import CAL_SEEDS, DATASETS  # noqa: E402

RAW = os.path.join(REPO, "data", "5g_nidd", "raw")
META = os.path.join(REPO, "data", "5g_nidd", "processed", "family.json")
OUT = os.path.join(REPO, "results", "v2")
DATA_DIR, FLAG_DIR = os.path.join(OUT, "r9_data"), os.path.join(OUT, "r9_flags")
BS = ("BS1", "BS2")
FOLDS = {"BS1": "BS2", "BS2": "BS1"}               # scored base station -> training base station
UE_PREFIX, MIN_FLOWS, MAX_RATE, ATT_SAMPLE = "10.155.", 30, 50.0, 20000
SESSION_TYPE = {"TCPConnect": "TCPConnectScan"}    # files without an Attack Type column
DET_SEEDS, EPOCHS, LR, BATCH = (0, 1, 2), 10, 0.003, 256
WINDOW_S = V.WINDOW_S
SOURCES, KAPPAS = ("none", "device", "type", "max"), (float("inf"), 2.0, 1.0, 0.5)
CAL_SETTINGS = ([("W1", "indep", None)] +
                [("W1", n, v["flag"]) for n, v in R8.VARIANTS if "flag" in v] + [("W4", "indep", None)])
STATS = ("llrh", "llrd", "llrr")
PLACE = ("carry", "memory", "reset")
REPLAY_DATASET = "5g_nidd"


def dump(obj, path):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, default=float)


def load_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ prep
def transform(df, meta):
    """The study's fitted feature transform (one-hot vocabulary, median imputation, standard scaling)."""
    from src.data.preprocess import _canon_text
    feats = meta["features"]
    onehot = {c: _canon_text(df[c]).fillna("nan").to_numpy() for c in meta["one_hot_columns"]}
    out = np.empty((len(df), len(feats)), np.float64)
    for j, f in enumerate(feats):
        base = f.split("=", 1)[0]
        if "=" in f and base in onehot:
            out[:, j] = onehot[base] == f.split("=", 1)[1]
            continue
        col = df[f]
        if not pd.api.types.is_numeric_dtype(col):
            col = pd.to_numeric(col.astype(str).str.strip(), errors="coerce")
        x = col.to_numpy(np.float64, copy=True)
        x[np.isinf(x)] = np.nan
        out[:, j] = np.where(np.isnan(x), meta["imputation_median"][f], x)
    mu, sc = np.array(meta["scaler"]["mean"]), np.array(meta["scaler"]["scale"])
    return ((out - mu) / sc).astype(np.float32)


def session_seconds(start):
    """StartTime is minutes:seconds; unwrap hour roll-overs along the file order (period 3600 s)."""
    parts = start.astype(str).str.strip().str.split(":", expand=True)
    assert parts.shape[1] == 2, "StartTime is not mm:ss"
    sec = parts[0].astype(float).to_numpy() * 60 + parts[1].astype(float).to_numpy()
    d = np.diff(sec, prepend=sec[0])
    d -= 3600 * np.round(d / 3600)
    t = sec[0] + np.cumsum(d)
    return t - t.min()


def dedup(X, y):
    """Exact duplicates removed; feature-identical rows with conflicting labels dropped."""
    h = pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy()
    df = pd.DataFrame({"h": h, "y": y})
    n_lab = df.groupby("h")["y"].nunique()
    keep = ~df.duplicated("h") & ~df["h"].isin(n_lab.index[n_lab > 1]).to_numpy()
    return X[keep.to_numpy()], y[keep.to_numpy()]


def prep():
    from src.data.preprocess import NIDD_FAMILY
    meta = json.load(open(META, encoding="utf-8"))
    classes = meta["classes"]
    raw_cols = [f for f in meta["features"] if "=" not in f] + list(meta["one_hot_columns"])
    os.makedirs(DATA_DIR, exist_ok=True)
    ds_info, sess_info, excluded = [], [], []
    for bi, bs in enumerate(BS):
        z = zipfile.ZipFile(os.path.join(RAW, f"{bs}_each_attack_csv.zip"))
        Xtr, ytr, Xue, ue_ds, ue_w, Xatt, yatt, att_sess = [], [], [], [], [], [], [], []
        for si, name in enumerate(sorted(n for n in z.namelist() if n.endswith(".csv"))):
            sess = os.path.basename(name)[:-4]
            cols = pd.read_csv(z.open(name), nrows=0).columns.tolist()
            use = raw_cols + ["StartTime", "SrcAddr", "Label"] + (["Attack Type"] if "Attack Type" in cols else [])
            df = pd.read_csv(z.open(name), usecols=use, low_memory=False)
            benign = (df["Label"].astype(str).str.strip() == "Benign").to_numpy()
            if "Attack Type" in df:
                fine = df["Attack Type"].astype(str).str.strip().to_numpy()
            else:
                fine = np.where(benign, "Benign", SESSION_TYPE[sess.rstrip("0123456789")])
            fam = np.array([NIDD_FAMILY[x] for x in fine])
            assert ((fam == "Benign") == benign).all(), f"label mismatch in {sess}"
            y = np.array([classes.index(x) for x in fam], dtype=np.int64)
            X = transform(df, meta)
            t = session_seconds(df["StartTime"])
            src = df["SrcAddr"].astype(str).str.strip().to_numpy()
            Xs, ys = dedup(X, y)
            Xtr.append(Xs)
            ytr.append(ys)
            att_rows = np.flatnonzero(~benign)
            rng = np.random.default_rng([0, bi, si])
            if len(att_rows) > ATT_SAMPLE:
                att_rows = np.sort(rng.choice(att_rows, ATT_SAMPLE, replace=False))
            Xatt.append(X[att_rows])
            yatt.append(y[att_rows])
            att_sess.append(np.full(len(att_rows), si))
            attackers_here = set(src[~benign])
            sess_info.append({"bs": bs, "session": sess, "rows": len(df), "benign": int(benign.sum()),
                              "attack_types": sorted(set(fine[~benign])), "minutes": float(t.max() / 60)})
            for s in sorted(set(src[benign])):
                if not s.startswith(UE_PREFIX):
                    continue
                rows = np.flatnonzero(benign & (src == s))
                span = max(t[rows].max() - t[rows].min(), WINDOW_S)
                rec = {"bs": bs, "session": sess, "src": s, "flows": len(rows), "rate": len(rows) / span}
                why = ("fewer than 30 benign flows" if len(rows) < MIN_FLOWS else
                       "attacker in this session" if s in attackers_here else
                       "more than 50 flows/s (probable label error)" if rec["rate"] > MAX_RATE else None)
                if why:
                    excluded.append({**rec, "reason": why})
                    continue
                w = (t[rows] // WINDOW_S).astype(int)
                w -= w.min()
                rec.update(id=len(ds_info), windows=int(w.max() + 1), hours=float((w.max() + 1) * WINDOW_S / 3600))
                ds_info.append(rec)
                Xue.append(X[rows])
                ue_ds.append(np.full(len(rows), rec["id"]))
                ue_w.append(w)
            print("prep9", bs, sess, len(df), flush=True)
        Xtr, ytr = dedup(np.concatenate(Xtr), np.concatenate(ytr))
        np.savez_compressed(os.path.join(DATA_DIR, f"{bs}.npz"), X_tr=Xtr, y_tr=ytr, X_ue=np.concatenate(Xue),
                            ue_ds=np.concatenate(ue_ds), ue_w=np.concatenate(ue_w), X_att=np.concatenate(Xatt),
                            y_att=np.concatenate(yatt), att_sess=np.concatenate(att_sess))
        print("prep9", bs, "train rows", len(ytr), "UE flows", sum(len(x) for x in Xue), flush=True)
    json.dump({"classes": classes, "device_sessions": ds_info, "sessions": sess_info, "excluded": excluded},
              open(os.path.join(DATA_DIR, "sessions.json"), "w", encoding="utf-8"), indent=1)


# ------------------------------------------------------------------ flags
def train_detector(X, y, seed, n_classes):
    import torch
    import torch.nn.functional as F
    from src.evaluation.metrics import evaluate
    from src.models.fib import build_model
    from src.utils.common import set_seed
    set_seed(seed)
    torch.set_num_threads(4)
    idx = np.random.default_rng([seed, 9]).permutation(len(y))
    ho, tr = idx[: len(y) // 10], idx[len(y) // 10:]
    model = build_model("cnn_bilstm", X.shape[1], n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xt, yt = torch.from_numpy(X[tr]), torch.from_numpy(y[tr])
    best, log = (-1.0, None, -1), []
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), BATCH):
            b = perm[i: i + BATCH]
            opt.zero_grad()
            F.cross_entropy(model(Xt[b]), yt[b]).backward()
            opt.step()
        m = evaluate(model, X[ho], y[ho], torch.device("cpu"), n_classes, benign=0)
        log.append({"epoch": ep, "macro_f1": float(m["macro_f1"]), "far": float(m["far"])})
        if m["macro_f1"] > best[0]:
            best = (float(m["macro_f1"]), copy.deepcopy(model.state_dict()), ep)
    model.load_state_dict(best[1])
    return model, {"best_epoch": best[2], "holdout_macro_f1": best[0], "epochs": log, "train_rows": int(len(tr))}


def flags():
    import torch
    from src.evaluation.metrics import predict_proba
    from src.utils.common import env_info
    info = json.load(open(os.path.join(DATA_DIR, "sessions.json"), encoding="utf-8"))
    classes = info["classes"]
    K = len(classes)
    ds = info["device_sessions"]
    os.makedirs(FLAG_DIR, exist_ok=True)
    data = {bs: dict(np.load(os.path.join(DATA_DIR, f"{bs}.npz"))) for bs in BS}
    for seed in DET_SEEDS:
        out_p = os.path.join(FLAG_DIR, f"seed{seed}.npz")
        if os.path.exists(out_p):
            continue
        n_ds = [np.zeros(d["windows"], np.int32) for d in ds]
        f_ds = [np.zeros(d["windows"], np.int32) for d in ds]
        det = np.zeros((2, K))
        att_n = np.zeros((2, K))
        summary = {}
        for fi, (bs, train_bs) in enumerate(FOLDS.items()):
            t0 = time.time()
            model, summ = train_detector(data[train_bs]["X_tr"], data[train_bs]["y_tr"], seed, K)
            d = data[bs]
            flag_ue = (1.0 - predict_proba(model, d["X_ue"], torch.device("cpu"))[:, 0]) >= 0.5
            for i in np.unique(d["ue_ds"]):
                m = d["ue_ds"] == i
                n_ds[i] = np.bincount(d["ue_w"][m], minlength=ds[i]["windows"]).astype(np.int32)
                f_ds[i] = np.bincount(d["ue_w"][m], weights=flag_ue[m], minlength=ds[i]["windows"]).astype(np.int32)
            flag_att = (1.0 - predict_proba(model, d["X_att"], torch.device("cpu"))[:, 0]) >= 0.5
            for k in range(1, K):
                m = d["y_att"] == k
                det[fi, k], att_n[fi, k] = flag_att[m].sum(), m.sum()
            summary[f"{bs}<-{train_bs}"] = {**summ, "seconds": round(time.time() - t0)}
            print("flags9", seed, bs, "trained on", train_bs, "holdout macro-F1", round(summ["holdout_macro_f1"], 4),
                  "epoch", summ["best_epoch"], f"{time.time() - t0:.0f}s", flush=True)
        fam = [c for c in classes if c != "Benign"]
        rate = det[:, 1:].sum(0) / np.maximum(att_n[:, 1:].sum(0), 1)
        np.savez_compressed(out_p, n=np.concatenate(n_ds), f=np.concatenate(f_ds),
                            off=np.cumsum([0] + [d["windows"] for d in ds]), fam=np.array(fam), fam_rate=rate,
                            fam_rate_fold=det[:, 1:] / np.maximum(att_n[:, 1:], 1), fam_n=att_n[:, 1:])
        json.dump({"seed": seed, "folds": summary, **env_info()},
                  open(out_p.replace(".npz", ".env.json"), "w", encoding="utf-8"), indent=1, default=str)


def load_flags(seed):
    info = json.load(open(os.path.join(DATA_DIR, "sessions.json"), encoding="utf-8"))
    z = np.load(os.path.join(FLAG_DIR, f"seed{seed}.npz"))
    off = z["off"]
    ds = info["device_sessions"]
    n = [z["n"][off[i]: off[i + 1]] for i in range(len(ds))]
    f = [z["f"][off[i]: off[i + 1]] for i in range(len(ds))]
    return ds, n, f, {"fam": z["fam"], "fam_rate": z["fam_rate"]}


# ------------------------------------------------------------------ A
def rho_moment(n, f):
    """Beta-binomial moment estimator over windows with at least two flows (as src/v2/robust.dispersion)."""
    m = n >= 2
    n, f = n[m].astype(float), f[m].astype(float)
    k, N, F = len(n), n.sum(), f.sum()
    if k < 10 or F == 0 or F == N:
        return np.nan, k
    p = F / N
    S = (f * f / n).sum() - F * F / N
    den = N - k - (n * n).sum() / N + 1
    return (float(np.clip((S / (p * (1 - p)) - (k - 1)) / den, 0, 0.5)) if den > 0 else np.nan), k


def acf(series_list, mean, lag):
    num = den = 0.0
    for s in series_list:
        x = s - mean
        ok = np.isfinite(x)
        den += np.nansum(x[ok] ** 2)
        if len(x) > lag:
            pair = ok[:-lag] & ok[lag:]
            num += (x[:-lag][pair] * x[lag:][pair]).sum()
    return num / den if den > 0 else np.nan


def describe():
    from scipy.stats import binom
    out = {}
    for seed in DET_SEEDS:
        ds, n, f, fam = load_flags(seed)
        rng = np.random.default_rng([seed, 31])
        per = {}
        for s in sorted(set(d["src"] for d in ds)):
            ids = [i for i, d in enumerate(ds) if d["src"] == s]
            nn, ff = np.concatenate([n[i] for i in ids]), np.concatenate([f[i] for i in ids])
            N, Fl = int(nn.sum()), int(ff.sum())
            p = Fl / N
            rho, k = rho_moment(nn, ff)
            boot = []
            if len(ids) > 1 and np.isfinite(rho):
                for _ in range(1000):
                    b = rng.choice(ids, len(ids), replace=True)
                    r, _k = rho_moment(np.concatenate([n[i] for i in b]), np.concatenate([f[i] for i in b]))
                    if np.isfinite(r):
                        boot.append(r)
            shares = [np.where(n[i] > 0, f[i] / np.maximum(n[i], 1), np.nan) for i in ids]
            act = nn > 0
            exc = float((ff[act] > binom.ppf(0.999, nn[act], p)).mean()) if 0 < p < 1 else 0.0
            per[s] = {"device_sessions": len(ids), "hours": float(sum(ds[i]["hours"] for i in ids)), "flows": N,
                      "flagged": Fl, "share": p, "flows_per_window": float(nn[act].mean()), "active_windows": int(act.sum()),
                      "rho": rho, "rho_k": int(k),
                      "rho_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if len(boot) >= 100 else None,
                      "acf1": float(acf(shares, p, 1)), "acf10": float(acf(shares, p, 10)), "excess_999": exc}
        rhos = np.array([v["rho"] for v in per.values() if np.isfinite(v["rho"])])
        allN = sum(v["flows"] for v in per.values())
        out[f"seed{seed}"] = {"per_ue": per, "fam": fam["fam"].tolist(), "fam_rate": fam["fam_rate"].tolist(),
                              "share_all": sum(v["flagged"] for v in per.values()) / allN,
                              "rho_median": float(np.median(rhos)) if len(rhos) else None,
                              "rho_min": float(rhos.min()) if len(rhos) else None,
                              "rho_max": float(rhos.max()) if len(rhos) else None, "n_rho": int(len(rhos)),
                              "n_ue": len(per)}
        print("describe9", seed, {k: v for k, v in out[f"seed{seed}"].items() if k != "per_ue"}, flush=True)
    json.dump(out, open(os.path.join(OUT, "r9_describe.json"), "w", encoding="utf-8"), indent=1, default=float)


# ------------------------------------------------------------------ llrr
def llrr_rho(tr, dev_type, source):
    if source == "none":
        return np.zeros(tr["seen"].shape)
    rho, k = R.dispersion(tr, R8.LAG, R8.HIST)
    if source == "device":
        return rho
    t = R.type_dispersion(rho, k, dev_type)
    return t if source == "type" else np.maximum(rho, t)


def llrr_alarms(tr, ref, rho, kappa, delta, placement, n_nodes, hs, att_sw, seed):
    p0, valid = ref
    inc = R.robust_increment(tr, p0, valid, delta, rho, kappa)
    return R.run_alarms("llr", delta, inc, tr["seen"] > 0, tr["node"], tr["ho"], placement, n_nodes, hs, R8.W_BURN,
                        att_sw, loss=R7.LOSS, rng=np.random.default_rng([seed, 55]))


def chosen():
    return yaml.safe_load(open(os.path.join(REPO, "configs", "v2", "r9.yaml"), encoding="utf-8"))["chosen"]


# ------------------------------------------------------------------ B
def cal():
    r7, r8 = R8.configs()
    rows = []
    for wname, vname, flag in CAL_SETTINGS:
        for d in DATASETS:
            acc = {(s, kp, b): {"ev": 0, "hrs": 0.0, "cont": 0, "n": 0}
                   for s in SOURCES for kp in KAPPAS for b in R8.BUDGETS}
            for seed in CAL_SEEDS:
                cfg = load_v2(None, {"seed": seed, "eval.split": "val"})
                for rep in range(3):
                    tr, refs, att, vclass, n_nodes, dev_type = R8.run(wname, d, "val", seed, rep, "persistent_0.05", cfg,
                                                                     "chained", R8.DELTAS[0], flag=flag)
                    honest = np.ones(tr["seen"].shape[1], bool)
                    honest[list(att)] = False
                    att_sw = {u: a["start"] // V.WINDOW_S for u, a in att.items()}
                    hrs = honest.sum() * (tr["seen"].shape[0] - R8.W_BURN) / R8.WIN_PER_H
                    rhos = {s: llrr_rho(tr, dev_type, s) for s in SOURCES}
                    for b in R8.BUDGETS:
                        dl = r8["llrd"]["chosen"][wname][d][b]["delta"]
                        for s in SOURCES:
                            for kp in KAPPAS:
                                al, first = llrr_alarms(tr, refs["llrd"], rhos[s], kp, dl, "carry", n_nodes,
                                                        [R8.H_THEORY[b]], att_sw, seed)
                                a = acc[(s, kp, b)]
                                a["ev"] += int(al[0, honest].sum())
                                a["hrs"] += hrs
                                a["cont"] += sum(int(first[0, u] < sw + R8.FOLLOW) for u, sw in att_sw.items())
                                a["n"] += len(att_sw)
                print("cal9", wname, vname, d, seed, flush=True)
            for (s, kp, b), a in acc.items():
                rows.append({"world": wname, "variant": vname, "dataset": d, "source": s, "kappa": kp, "budget": b,
                             "val_rate": a["ev"] / a["hrs"], "val_contained": a["cont"] / a["n"], "events": a["ev"],
                             "hours": a["hrs"]})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(REPO, "reports", "v2_r9_calibration.csv"), index=False)
    b01 = t[t.budget == "b01"]
    cand = []
    for (s, kp), g in b01.groupby(["source", "kappa"]):
        worst = float(g.val_rate.max())
        cont = float(g[(g.world == "W1") & (g.variant == "indep")].val_contained.mean())
        cand.append({"source": s, "kappa": float(kp), "feasible": bool(worst <= R8.BUDGETS["b01"]), "worst_rate": worst,
                     "contained_w1_indep": cont})
    ok = [c for c in cand if c["feasible"]]
    if ok:
        best = max(ok, key=lambda c: (round(c["contained_w1_indep"], 12), c["kappa"], -SOURCES.index(c["source"])))
    else:
        best = min(cand, key=lambda c: c["worst_rate"])
    out = {"provenance": "scripts/v2_r9.py cal (validation, traces 100-102 chained, 3 replicates; W1 with R8's flag "
                         "variants and W4 with independent flags; carry-over; persistent 5%)",
           "chosen": {"source": best["source"], "kappa": best["kappa"], "feasible": best["feasible"],
                      "worst_rate": best["worst_rate"], "contained_w1_indep": best["contained_w1_indep"]},
           "candidates": cand}
    yaml.safe_dump(json.loads(json.dumps(out)), open(os.path.join(REPO, "configs", "v2", "r9.yaml"), "w", encoding="utf-8"),
                   sort_keys=False)
    print("wrote configs/v2/r9.yaml", out["chosen"])


# ------------------------------------------------------------------ C
def sweep(only=None):
    from src.utils.common import env_info
    ch = chosen()
    out_dir = os.path.join(OUT, "r9_sweep")
    os.makedirs(out_dir, exist_ok=True)
    hs = [R8.H_THEORY["b01"], R8.H_THEORY["b1"]]
    for vname, v in R8.VARIANTS:
        if only and vname not in only:
            continue
        for d in DATASETS:
            out_p = os.path.join(out_dir, f"{vname}_{d}.json.gz")
            if os.path.exists(out_p):
                continue
            res, t0 = [], time.time()
            for seed in R8.TRACE_SEEDS:
                cfg = load_v2(None, {"seed": seed})
                for rep in range(2):
                    tr, refs, att, vclass, n_nodes, dev_type = R8.run("W1", d, "test", seed, rep, "persistent_0.05", cfg,
                                                                     "unchained", R8.DELTAS[0], sigma=v.get("sigma", 0.0),
                                                                     flag=v.get("flag"))
                    honest = np.ones(tr["seen"].shape[1], bool)
                    honest[list(att)] = False
                    att_sw = {u: a["start"] // V.WINDOW_S for u, a in att.items()}
                    hrs = float(honest.sum() * (tr["seen"].shape[0] - R8.W_BURN) / R8.WIN_PER_H)
                    unmon = float((refs["llrd"][0] >= R8.CEIL)[R8.W_BURN:, honest].mean())
                    rho = llrr_rho(tr, dev_type, ch["source"])
                    for dl in R8.DELTAS:
                        for pl in PLACE:
                            al, first = llrr_alarms(tr, refs["llrd"], rho, ch["kappa"], dl, pl, n_nodes, hs, att_sw, seed)
                            res.append({"variant": vname, "dataset": d, "seed": seed, "rep": rep, "stat": "llrr", "delta": dl,
                                        "placement": pl, "events": al[:, honest].sum(1).tolist(), "hours": hrs,
                                        "contained": [int(sum(R8.contained(first, att_sw, i).values())) for i in range(2)],
                                        "n_att": len(att_sw), "unmonitored": unmon})
                print("sweep9", vname, d, seed, f"{time.time() - t0:.0f}s", flush=True)
            dump(res, out_p)
            json.dump({"variant": v, "llrr": ch, **env_info()}, open(out_p.replace(".json.gz", ".env.json"), "w",
                                                                    encoding="utf-8"), indent=1, default=str)


# ------------------------------------------------------------------ D
def replay_sequences(ds, n, f, U, W, rng):
    """Block bootstrap of real UEs: device u gets UE r(u) (probability proportional to its hours) as its type, and its
    device-sessions concatenated in random order (re-permuted at every pass) from a random start."""
    ues = sorted(set(d["src"] for d in ds))
    ids = {s: [i for i, d in enumerate(ds) if d["src"] == s] for s in ues}
    hours = np.array([sum(ds[i]["hours"] for i in ids[s]) for s in ues])
    r = rng.choice(len(ues), size=U, p=hours / hours.sum())
    seq_n, seq_f = np.zeros((W, U), np.int64), np.zeros((W, U), np.int64)
    for u in range(U):
        parts_n, parts_f, length, off = [], [], 0, None
        while off is None or length < off + W:
            perm = rng.permutation(ids[ues[r[u]]])
            pn, pf = np.concatenate([n[i] for i in perm]), np.concatenate([f[i] for i in perm])
            if off is None:
                off = int(rng.integers(len(pn)))
            parts_n.append(pn)
            parts_f.append(pf)
            length += len(pn)
        seq_n[:, u] = np.concatenate(parts_n)[off: off + W]
        seq_f[:, u] = np.concatenate(parts_f)[off: off + W]
    return seq_n, seq_f, r, ues


def replay():
    from src.utils.common import env_info
    r7, r8 = R8.configs()
    ch = chosen()
    out_dir = os.path.join(OUT, "r9_replay")
    os.makedirs(out_dir, exist_ok=True)
    d = REPLAY_DATASET
    dl = {("llrh", b): r7["W1"][d][b]["delta"] for b in R8.BUDGETS}
    dl.update({(s, b): r8["llrd"]["chosen"]["W1"][d][b]["delta"] for s in ("llrd", "llrr") for b in R8.BUDGETS})
    for det in DET_SEEDS:
        out_p = os.path.join(out_dir, f"det{det}.json.gz")
        if os.path.exists(out_p):
            continue
        ds, n, f, prof = load_flags(det)
        res, t0 = [], time.time()
        for seed in R8.TRACE_SEEDS:
            cfg = load_v2(None, {"seed": seed})
            node, ho, outage, vclass, n_nodes = R8.world_links(seed, "W1", cfg, "unchained")
            U = node.shape[1]
            W = node.shape[0] // V.WINDOW_S
            for rep in range(2):
                rng = np.random.default_rng([seed, det, rep, 199])
                seq_n, seq_f, dev_type, ues = replay_sequences(ds, n, f, U, W, rng)
                att = V.attackers(list(V.SCENARIOS).index("persistent_0.05"), seed, vclass, prof)
                tr = R.simulate_replay(node, ho, outage, seq_n, seq_f, att, V.WINDOW_S, rng)
                honest = np.ones(U, bool)
                honest[list(att)] = False
                att_sw = {u: a["start"] // V.WINDOW_S for u, a in att.items()}
                hrs = float(honest.sum() * (tr["seen"].shape[0] - R8.W_BURN) / R8.WIN_PER_H)
                ph, vh = R.hourly_type_reference(tr, dev_type)
                pd_, vd = R.device_reference(tr, ph, vh, R8.LAG, R8.HIST, R8.M_PSEUDO, z=R8.Z_UCB, ceiling=R8.CEIL)
                refs = {"llrh": (ph, vh), "llrd": (pd_, vd)}
                rho = llrr_rho(tr, dev_type, ch["source"])
                unmon = float((pd_ >= R8.CEIL)[R8.W_BURN:, honest].mean())
                silent = float((tr["seen"][R8.W_BURN:, honest] == 0).mean())
                for b in R8.BUDGETS:
                    for stat in STATS:
                        for pl in PLACE:
                            if stat == "llrr":
                                al, first = llrr_alarms(tr, refs["llrd"], rho, ch["kappa"], dl[(stat, b)], pl, n_nodes,
                                                        [R8.H_THEORY[b]], att_sw, seed)
                            else:
                                al, first = R8.alarms(tr, refs[stat], dl[(stat, b)], pl, n_nodes, [R8.H_THEORY[b]], att_sw, seed)
                            c = R8.contained(first, att_sw)
                            res.append({"det": det, "seed": seed, "rep": rep, "budget": b, "stat": stat, "placement": pl,
                                        "delta": dl[(stat, b)], "events": int(al[0, honest].sum()), "hours": hrs,
                                        "contained": int(sum(c.values())), "n_att": len(c),
                                        "unmonitored": unmon if stat != "llrh" else 0.0, "silent": silent,
                                        "att_ue": [ues[int(dev_type[u])] for u in att]})
            print("replay9", det, seed, f"{time.time() - t0:.0f}s", flush=True)
        dump(res, out_p)
        json.dump({"deltas": {f"{k[0]}/{k[1]}": v for k, v in dl.items()}, "llrr": ch, "fam": prof["fam"].tolist(),
                   "fam_rate": prof["fam_rate"].tolist(), **env_info()},
                  open(out_p.replace(".json.gz", ".env.json"), "w", encoding="utf-8"), indent=1, default=str)


def replay_curves(att_share=0.05):
    """Post hoc (not in the plan; added after the first replay results): the replay of D with thresholds swept, so that
    placements and statistics can be compared at a matched false-alarm rate, as R8's section F. att_share other than
    0.05 changes only the attackers: honest devices draw the same flows and flags."""
    from src.utils.common import env_info
    r7, r8 = R8.configs()
    ch = chosen()
    out_dir = os.path.join(OUT, "r9_replay_curves" + ("" if att_share == 0.05 else f"_a{round(100 * att_share)}"))
    os.makedirs(out_dir, exist_ok=True)
    d = REPLAY_DATASET
    dl = {"llrh": r7["W1"][d]["b01"]["delta"], "llrd": r8["llrd"]["chosen"]["W1"][d]["b01"]["delta"]}
    dl["llrr"] = dl["llrd"]
    grid = V.GRID["llr"]
    for det in DET_SEEDS:
        out_p = os.path.join(out_dir, f"det{det}.json.gz")
        if os.path.exists(out_p):
            continue
        ds, n, f, prof = load_flags(det)
        acc = {}
        for seed in R8.TRACE_SEEDS:
            cfg = load_v2(None, {"seed": seed})
            node, ho, outage, vclass, n_nodes = R8.world_links(seed, "W1", cfg, "unchained")
            U = node.shape[1]
            W = node.shape[0] // V.WINDOW_S
            for rep in range(2):
                rng = np.random.default_rng([seed, det, rep, 199])        # the same draws as replay()
                seq_n, seq_f, dev_type, ues = replay_sequences(ds, n, f, U, W, rng)
                att = V.attackers(list(V.SCENARIOS).index("persistent_0.05"), seed, vclass, prof)
                tr = R.simulate_replay(node, ho, outage, seq_n, seq_f, att, V.WINDOW_S, rng, att_share=att_share)
                honest = np.ones(U, bool)
                honest[list(att)] = False
                att_sw = {u: a["start"] // V.WINDOW_S for u, a in att.items()}
                hrs = float(honest.sum() * (tr["seen"].shape[0] - R8.W_BURN) / R8.WIN_PER_H)
                ph, vh = R.hourly_type_reference(tr, dev_type)
                pd_, vd = R.device_reference(tr, ph, vh, R8.LAG, R8.HIST, R8.M_PSEUDO, z=R8.Z_UCB, ceiling=R8.CEIL)
                refs = {"llrh": (ph, vh), "llrd": (pd_, vd)}
                rho = llrr_rho(tr, dev_type, ch["source"])
                for stat in STATS:
                    for pl in PLACE:
                        if stat == "llrr":
                            al, first = llrr_alarms(tr, refs["llrd"], rho, ch["kappa"], dl[stat], pl, n_nodes, grid, att_sw, seed)
                        else:
                            al, first = R8.alarms(tr, refs[stat], dl[stat], pl, n_nodes, grid, att_sw, seed)
                        a = acc.setdefault(f"{stat}/{pl}", {"ev": np.zeros(len(grid)), "cont": np.zeros(len(grid)),
                                                            "hours": 0.0, "n_att": 0, "by_seed": {}})
                        ev = al[:, honest].sum(1)
                        ct = sum((first[:, u] < sw + R8.FOLLOW).astype(float) for u, sw in att_sw.items())
                        a["ev"] += ev
                        a["cont"] += ct
                        a["hours"] += hrs
                        a["n_att"] += len(att_sw)
                        bs = a["by_seed"].setdefault(str(seed), {"ev": np.zeros(len(grid)), "cont": np.zeros(len(grid)),
                                                                 "hours": 0.0, "n_att": 0})
                        bs["ev"] += ev
                        bs["cont"] += ct
                        bs["hours"] += hrs
                        bs["n_att"] += len(att_sw)
            print("replaycurves9", det, seed, flush=True)

        def lists(o):
            return {k: lists(v) for k, v in o.items()} if isinstance(o, dict) else o.tolist() if isinstance(o, np.ndarray) else o
        dump({"grid": grid.tolist(), "deltas": dl, "rules": lists(acc)}, out_p)
        json.dump({"post_hoc": True, "att_share": att_share, "llrr": ch, **env_info()},
                  open(out_p.replace(".json.gz", ".env.json"), "w", encoding="utf-8"), indent=1, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "flags", "describe", "cal", "sweep", "replay", "replaycurves"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--dets", nargs="*", type=int, help="detector seeds for replay / replaycurves (default all)")
    ap.add_argument("--share", type=float, default=0.05, help="attack-flow share for replaycurves (post hoc)")
    a = ap.parse_args()
    if a.dets:
        DET_SEEDS = tuple(a.dets)
    if a.cmd == "cal":
        os.environ["MERIT_FL_FORBID_TEST"] = "1"
        cal()
    elif a.cmd == "sweep":
        sweep(a.only)
    elif a.cmd == "replaycurves":
        replay_curves(a.share)
    else:
        {"prep": prep, "flags": flags, "describe": describe, "replay": replay}[a.cmd]()
