#!/usr/bin/env python3
"""Analysis of R9 the analysis plan. Output: reports/stats/V2_R9.md, v2_r9_stats.json.

A  real benign flags per UE: flagged share, intra-window correlation, autocorrelation (H9.1)
B  validation choice of llrr (configs/v2/r9.yaml)
C  llrr on R8's synthetic variants (fresh traces), against R8's llrh and llrd (H9.3 b, c; H9.4 synthetic; price)
D  real-flag replay: llrh, llrd, llrr (H9.2, H9.3 a, H9.4 real)
"""
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy import stats as st

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.evaluation.stats import bootstrap_ci, holm  # noqa: E402
from src.v2.plan import DATASETS  # noqa: E402

V2 = os.path.join(REPO, "results", "v2")
BUDGETS = {"b01": 0.001, "b1": 0.01}
SIGMA = [f"sigma_{s:g}" for s in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)]
FLAGV = ["betabin_0.01", "betabin_0.03", "betabin_0.1", "burst_3", "burst_10"]
md, js = ["# R9: real per-device benign flags and a dispersion-robust statistic (scripts/v2_r9_analyze.py)", "",
          "Plan: `ANALYSIS_PLAN_V2_R9.md`. Real flags: 5G-NIDD per-session flows, detectors cross-fitted by base station "
          "(seeds 0-2). Synthetic test: R8's traces 52-61.", ""], {}


def gz(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def poisson_ci(k, t):
    lo = st.chi2.ppf(0.025, 2 * k) / 2 / t if k > 0 else 0.0
    return lo, st.chi2.ppf(0.975, 2 * k + 2) / 2 / t


def configs():
    r7 = yaml.safe_load(open(os.path.join(REPO, "configs", "v2", "r7.yaml"), encoding="utf-8"))["chosen"]
    r8 = yaml.safe_load(open(os.path.join(REPO, "configs", "v2", "r8.yaml"), encoding="utf-8"))["llrd"]["chosen"]
    r9 = yaml.safe_load(open(os.path.join(REPO, "configs", "v2", "r9.yaml"), encoding="utf-8"))
    return r7, r8, r9


# ------------------------------------------------------------------ A
def section_describe():
    global md
    p = os.path.join(V2, "r9_describe.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p, encoding="utf-8"))
    info = json.load(open(os.path.join(V2, "r9_data", "sessions.json"), encoding="utf-8"))
    ds = info["device_sessions"]
    det = {}
    for s in d:
        env = json.load(open(os.path.join(V2, "r9_flags", f"{s}.env.json"), encoding="utf-8"))
        det[s] = {k: v["holdout_macro_f1"] for k, v in env["folds"].items()}
    ues = sorted(d["seed0"]["per_ue"])
    per = []
    for u in ues:
        vals = [d[s]["per_ue"][u] for s in d]
        rhos = [v["rho"] for v in vals if v["rho"] is not None and np.isfinite(v["rho"])]
        per.append({"ue": u, "hours": vals[0]["hours"], "device_sessions": vals[0]["device_sessions"],
                    "flows_per_window": vals[0]["flows_per_window"], "share": float(np.mean([v["share"] for v in vals])),
                    "rho": float(np.mean(rhos)) if rhos else np.nan, "n_seeds_rho": len(rhos),
                    "acf1": float(np.nanmean([v["acf1"] for v in vals])), "acf10": float(np.nanmean([v["acf10"] for v in vals])),
                    "excess_999": float(np.mean([v["excess_999"] for v in vals]))})
    t = pd.DataFrame(per)
    r = t.rho.dropna()
    out = {"n_ue": len(t), "n_device_sessions": len(ds), "device_hours": float(sum(x["hours"] for x in ds)),
           "excluded": {k: sum(1 for e in info["excluded"] if e["reason"] == k) for k in set(e["reason"] for e in info["excluded"])},
           "flows_per_window_median": float(t.flows_per_window.median()),
           "share_all_by_seed": {s: d[s]["share_all"] for s in d}, "share_ue_median": float(t.share.median()),
           "share_ue_min": float(t.share.min()), "share_ue_max": float(t.share.max()),
           "rho_n": int(len(r)), "rho_median": float(r.median()) if len(r) else None,
           "rho_min": float(r.min()) if len(r) else None, "rho_max": float(r.max()) if len(r) else None,
           "rho_share_ge_001": float((r >= 0.01).mean()) if len(r) else None,
           "rho_share_ge_003": float((r >= 0.03).mean()) if len(r) else None,
           "rho_share_ge_01": float((r >= 0.1).mean()) if len(r) else None,
           "acf1_median": float(t.acf1.median()), "acf10_median": float(t.acf10.median()),
           "excess_median": float(t.excess_999.median()), "excess_max": float(t.excess_999.max()),
           "fam": d["seed0"]["fam"], "fam_rate_by_seed": {s: d[s]["fam_rate"] for s in d},
           "detector_holdout_macro_f1": det, "per_ue": per}
    js["describe"] = out
    md.extend(["## A. Real benign flags (H9.1)", "",
               f"{out['n_device_sessions']} device-sessions of {out['n_ue']} UEs, {out['device_hours']:.1f} device-hours; "
               f"excluded: {out['excluded']}. Median flows per active window {out['flows_per_window_median']:.1f}.", "",
               f"Flagged share of benign UE flows per detector seed: "
               f"{', '.join(f'{100 * v:.2f}%' for v in out['share_all_by_seed'].values())}; per UE median "
               f"{100 * out['share_ue_median']:.2f}% (range {100 * out['share_ue_min']:.2f}-{100 * out['share_ue_max']:.2f}%).", "",
               f"Intra-window correlation rho (mean over seeds, {out['rho_n']} UEs with an estimate): median "
               f"{out['rho_median']}, range {out['rho_min']}-{out['rho_max']}; share of UEs with rho >= 0.01 / 0.03 / 0.1: "
               f"{out['rho_share_ge_001']}, {out['rho_share_ge_003']}, {out['rho_share_ge_01']}.", "",
               f"Lag-1 / lag-10 autocorrelation of the window share (median over UEs): {out['acf1_median']:.3f} / "
               f"{out['acf10_median']:.3f}. Windows above the binomial 99.9% quantile: median {100 * out['excess_median']:.2f}%, "
               f"max {100 * out['excess_max']:.2f}%.", "",
               f"Detector holdout macro-F1: {json.dumps(det)}. Family detection rates: "
               f"{json.dumps({s: [round(x, 3) for x in v] for s, v in out['fam_rate_by_seed'].items()})} ({out['fam']}).", "",
               "| UE | hours | sessions | flows/window | flagged share | rho | acf1 | acf10 | >99.9% |",
               "|---|---|---|---|---|---|---|---|---|"])
    md += [f"| {x['ue']} | {x['hours']:.2f} | {x['device_sessions']} | {x['flows_per_window']:.1f} | {100 * x['share']:.2f}% | "
           f"{x['rho']:.4f} | {x['acf1']:.3f} | {x['acf10']:.3f} | {100 * x['excess_999']:.2f}% |" for x in per]
    md.append("")


# ------------------------------------------------------------------ B
def section_cal(r9):
    global md
    js["cal"] = r9
    md.extend(["## B. Validation choice of llrr", "", f"Chosen: `{json.dumps(r9['chosen'])}`", "",
               "| source | kappa | feasible | worst %/h | contained (W1, independent) |", "|---|---|---|---|---|"])
    md += [f"| {c['source']} | {c['kappa']} | {'yes' if c['feasible'] else 'no'} | {100 * c['worst_rate']:.3f} | "
           f"{c['contained_w1_indep']:.3f} |" for c in sorted(r9["candidates"], key=lambda c: (c["source"], -c["kappa"]))]
    md.append("")


# ------------------------------------------------------------------ C
def load_sweep(name):
    rows = []
    p = os.path.join(V2, name)
    for f in sorted(os.listdir(p)) if os.path.isdir(p) else []:
        if f.endswith(".json.gz"):
            rows += gz(os.path.join(p, f))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df[["ev_b01", "ev_b1"]] = pd.DataFrame(df.events.tolist(), index=df.index)
    df[["c_b01", "c_b1"]] = pd.DataFrame(df.contained.tolist(), index=df.index)
    return df


def section_sweep(r7, r8):
    global md
    df = pd.concat([load_sweep("r8_sweep"), load_sweep("r9_sweep")], ignore_index=True)
    if df.empty or (df.stat == "llrr").sum() == 0:
        return
    chosen = {"llrh": {d: r7["W1"][d]["b01"]["delta"] for d in DATASETS},
              "llrd": {d: r8["W1"][d]["b01"]["delta"] for d in DATASETS},
              "llrr": {d: r8["W1"][d]["b01"]["delta"] for d in DATASETS}}
    sel = pd.concat([df[(df.stat == s) & (df.dataset == d) & (df.delta == chosen[s][d])] for s in chosen for d in DATASETS])
    curves = {}
    for stat in chosen:
        for v in SIGMA + FLAGV:
            g = sel[(sel.stat == stat) & (sel.variant == v) & (sel.placement == "carry")]
            if g.empty:
                continue
            rates = [float(g[g.dataset == d].ev_b01.sum() / g[g.dataset == d].hours.sum()) for d in DATASETS]
            cont = [float(g[g.dataset == d].c_b01.sum() / g[g.dataset == d].n_att.sum()) for d in DATASETS]
            curves[f"{stat}/{v}"] = {"rates": rates, "max_rate": max(rates), "within": all(r <= BUDGETS["b01"] for r in rates),
                                     "contained": float(np.mean(cont)), "contained_by_dataset": cont,
                                     "unmon": float(g.unmonitored.mean())}
    # H9.4 (synthetic): carry vs memory / reset, sigma = 0, seed level, Holm over 6
    rows = []
    g0 = sel[(sel.stat == "llrr") & (sel.variant == "sigma_0")]
    for d in DATASETS:
        g = g0[g0.dataset == d].assign(c=lambda x: x.c_b01 / x.n_att)
        s = g.groupby(["seed", "placement"]).c.mean().unstack()
        for other in ("memory", "reset"):
            dlt = (s["carry"] - s[other]).dropna()
            p = st.wilcoxon(dlt).pvalue if (dlt != 0).any() else 1.0
            lo, hi = bootstrap_ci(dlt.to_numpy())
            rows.append({"dataset": d, "vs": other, "carry": float(s["carry"].mean()), "other": float(s[other].mean()),
                         "mean_diff": float(dlt.mean()), "ci_lo": lo, "ci_hi": hi, "wins": int((dlt > 0).sum()),
                         "losses": int((dlt < 0).sum()), "p": float(p)})
    for r_, pa in zip(rows, holm([r["p"] for r in rows])):
        r_["p_holm"] = float(pa)
    beats = {d: all(any(r["dataset"] == d and r["vs"] == o and r["p_holm"] < 0.05 and r["mean_diff"] > 0 for r in rows)
                    for o in ("memory", "reset")) for d in DATASETS}
    b = {"all": all(curves.get(f"llrr/{v}", {}).get("within", False) for v in SIGMA + FLAGV),
         "flag_models": all(curves.get(f"llrr/{v}", {}).get("within", False) for v in ["sigma_0"] + FLAGV),
         "sigma": {v: curves.get(f"llrr/{v}", {}).get("within") for v in SIGMA}}
    price = {d: {"llrd": curves["llrd/sigma_0"]["contained_by_dataset"][i], "llrr": curves["llrr/sigma_0"]["contained_by_dataset"][i]}
             for i, d in enumerate(DATASETS)} if "llrd/sigma_0" in curves and "llrr/sigma_0" in curves else {}
    js["sweep"] = {"curves": curves, "H93b": b, "H93c": curves.get("llrr/sigma_0.3", {}).get("within"),
                   "H94_synthetic": {"rows": rows, "beats_both": beats, "n_beats_both": int(sum(beats.values())),
                                     "verdict": "robust" if sum(beats.values()) >= 2 else "not robust"},
                   "price": price}
    md.extend(["## C. Synthetic test (unchained, W1 links, fresh traces; carry, b01, validation delta)", "",
               f"H9.3(b) every variant within for llrr: **{b['all']}** (flag models and independent flags: {b['flag_models']}); "
               f"H9.3(c) sigma = 0.3: **{js['sweep']['H93c']}**.", "",
               "| variant | statistic | %/h per dataset | max %/h | within | contained | unmonitored |", "|---|---|---|---|---|---|---|"])
    for v in SIGMA + FLAGV:
        for stat in chosen:
            c = curves.get(f"{stat}/{v}")
            if c:
                md.append(f"| {v} | {stat} | {', '.join(f'{100 * r:.3f}' for r in c['rates'])} | {100 * c['max_rate']:.3f} | "
                          f"{'yes' if c['within'] else 'no'} | {c['contained']:.3f} | {c['unmon']:.3f} |")
    md.extend(["", f"**H9.4 synthetic (llrr, sigma = 0): {js['sweep']['H94_synthetic']['verdict']}** "
                   f"({js['sweep']['H94_synthetic']['n_beats_both']} of 3 datasets)", "",
               "| dataset | vs | carry | other | Δ [95% CI] | W / L | p (Holm) |", "|---|---|---|---|---|---|---|"])
    md += [f"| {x['dataset']} | {x['vs']} | {x['carry']:.3f} | {x['other']:.3f} | {x['mean_diff']:.3f} [{x['ci_lo']:.3f}, "
           f"{x['ci_hi']:.3f}] | {x['wins']} / {x['losses']} | {x['p_holm']:.4f} |" for x in rows]
    md.extend(["", f"Price of robustness (contained share, sigma = 0, carry): {json.dumps(price)}", ""])


# ------------------------------------------------------------------ D
def boot_rate(g, B=2000, seed=0):
    by = g.groupby("seed")[["events", "hours"]].sum()
    rng = np.random.default_rng(seed)
    ev, hr = by.events.to_numpy(), by.hours.to_numpy()
    idx = rng.integers(len(by), size=(B, len(by)))
    r = ev[idx].sum(1) / hr[idx].sum(1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def section_replay():
    global md
    rows = []
    for f in sorted(os.listdir(os.path.join(V2, "r9_replay"))) if os.path.isdir(os.path.join(V2, "r9_replay")) else []:
        if f.endswith(".json.gz"):
            rows += gz(os.path.join(V2, "r9_replay", f))
    if not rows:
        return
    df = pd.DataFrame(rows)
    out = {}
    for (b, stat, pl), g in df.groupby(["budget", "stat", "placement"]):
        ev, hr = int(g.events.sum()), float(g.hours.sum())
        lo, hi = poisson_ci(ev, hr)
        blo, bhi = boot_rate(g)
        out[f"{b}/{stat}/{pl}"] = {"events": ev, "hours": hr, "rate": ev / hr, "poisson_lo": lo, "poisson_hi": hi,
                                   "boot_lo": blo, "boot_hi": bhi, "within": ev / hr <= BUDGETS[b],
                                   "contained": float(g.contained.sum() / g.n_att.sum()), "unmon": float(g.unmonitored.mean()),
                                   "silent": float(g.silent.mean()), "runs": int(len(g))}
    tests = {}
    for stat in ("llrh", "llrd", "llrr"):
        g = df[(df.budget == "b01") & (df.stat == stat)].assign(c=lambda x: x.contained / x.n_att)
        s = g.groupby(["seed", "placement"]).c.mean().unstack()
        rr = []
        for other in ("memory", "reset"):
            dlt = (s["carry"] - s[other]).dropna()
            p = st.wilcoxon(dlt).pvalue if (dlt != 0).any() else 1.0
            lo, hi = bootstrap_ci(dlt.to_numpy())
            rr.append({"vs": other, "carry": float(s["carry"].mean()), "other": float(s[other].mean()), "mean_diff": float(dlt.mean()),
                       "ci_lo": lo, "ci_hi": hi, "wins": int((dlt > 0).sum()), "losses": int((dlt < 0).sum()), "p": float(p)})
        for r_, pa in zip(rr, holm([r["p"] for r in rr])):
            r_["p_holm"] = float(pa)
        tests[stat] = {"rows": rr, "beats_both": all(r["p_holm"] < 0.05 and r["mean_diff"] > 0 for r in rr)}
    js["replay"] = {"budget": out, "placement": tests,
                    "H92": {s: out[f"b01/{s}/carry"]["within"] for s in ("llrh", "llrd")},
                    "H93a": out["b01/llrr/carry"]["within"], "H94_real": tests["llrr"]["beats_both"]}
    md.extend(["## D. Real-flag replay (traces 52-61, unchained, W1 links; 3 detector seeds x 2 bootstrap replicates)", "",
               f"H9.2 (llrh, llrd within 0.1%/h with carry): {json.dumps(js['replay']['H92'])}; "
               f"H9.3(a) llrr within: **{js['replay']['H93a']}**; H9.4 real (llrr carry beats memory and reset): "
               f"**{js['replay']['H94_real']}**.", "",
               "| budget | statistic | placement | events | device-hours | %/h [Poisson 95%] [trace bootstrap 95%] | within | "
               "contained | unmonitored | silent |", "|---|---|---|---|---|---|---|---|---|---|"])
    for k, v in out.items():
        b, stat, pl = k.split("/")
        md.append(f"| {b} | {stat} | {pl} | {v['events']} | {v['hours']:.0f} | {100 * v['rate']:.4f} [{100 * v['poisson_lo']:.4f}, "
                  f"{100 * v['poisson_hi']:.4f}] [{100 * v['boot_lo']:.4f}, {100 * v['boot_hi']:.4f}] | {'yes' if v['within'] else 'no'} | "
                  f"{v['contained']:.3f} | {v['unmon']:.3f} | {v['silent']:.3f} |")
    md.append("")
    for stat, t in tests.items():
        md.extend([f"**{stat}: carry beats both (Holm over 2): {t['beats_both']}**", "",
                   "| vs | carry | other | Δ [95% CI] | W / L | p (Holm) |", "|---|---|---|---|---|---|"])
        md += [f"| {x['vs']} | {x['carry']:.3f} | {x['other']:.3f} | {x['mean_diff']:.3f} [{x['ci_lo']:.3f}, {x['ci_hi']:.3f}] | "
               f"{x['wins']} / {x['losses']} | {x['p_holm']:.4f} |" for x in t["rows"]]
        md.append("")


def section_replay_curves(name="r9_replay_curves", key="replay_matched", share="5%"):
    """Post hoc: real-flag replay at a matched false-alarm rate (thresholds swept on the replay, as R8's section F)."""
    global md
    p = os.path.join(V2, name)
    files = sorted(f for f in os.listdir(p) if f.endswith(".json.gz")) if os.path.isdir(p) else []
    if not files:
        return
    acc, grid = {}, None
    for f in files:
        z = gz(os.path.join(p, f))
        grid = np.array(z["grid"])
        for k, a in z["rules"].items():
            t = acc.setdefault(k, {"ev": 0, "cont": 0, "hours": 0.0, "n_att": 0, "by_seed": {}})
            t["ev"] = t["ev"] + np.array(a["ev"])
            t["cont"] = t["cont"] + np.array(a["cont"])
            t["hours"] += a["hours"]
            t["n_att"] += a["n_att"]
            for s, b in a["by_seed"].items():
                u = t["by_seed"].setdefault(s, {"ev": 0, "cont": 0, "hours": 0.0, "n_att": 0})
                u["ev"] = u["ev"] + np.array(b["ev"])
                u["cont"] = u["cont"] + np.array(b["cont"])
                u["hours"] += b["hours"]
                u["n_att"] += b["n_att"]
    out = {}
    for k, t in acc.items():
        rate = t["ev"] / t["hours"]
        ok = np.flatnonzero(rate <= BUDGETS["b01"])
        i = int(ok.min()) if len(ok) else None
        out[k] = {"h": float(grid[i]) if i is not None else None, "rate": float(rate[i]) if i is not None else None,
                  "contained": float(t["cont"][i] / t["n_att"]) if i is not None else None, "detectors": len(files),
                  "by_seed": {s: float(u["cont"][i] / u["n_att"]) for s, u in t["by_seed"].items()} if i is not None else {}}
    tests = {}
    for stat in ("llrh", "llrd", "llrr"):
        rr = []
        c = out.get(f"{stat}/carry", {})
        for other in ("memory", "reset"):
            o = out.get(f"{stat}/{other}", {})
            if not c.get("by_seed") or not o.get("by_seed"):
                continue
            seeds = sorted(c["by_seed"])
            dlt = np.array([c["by_seed"][s] - o["by_seed"][s] for s in seeds])
            pv = st.wilcoxon(dlt).pvalue if (dlt != 0).any() else 1.0
            lo, hi = bootstrap_ci(dlt)
            rr.append({"vs": other, "mean_diff": float(dlt.mean()), "ci_lo": lo, "ci_hi": hi, "wins": int((dlt > 0).sum()),
                       "losses": int((dlt < 0).sum()), "p": float(pv)})
        for r_, pa in zip(rr, holm([r["p"] for r in rr])):
            r_["p_holm"] = float(pa)
        tests[stat] = rr
    js[key] = {"rules": out, "placement": tests, "detectors": len(files)}
    md.extend([f"## E. Real-flag replay at a matched rate, {share} attack flows (post hoc, descriptive)", "",
               "Thresholds swept on the replay; each rule at the lowest threshold whose rate is at most 0.1%/h "
               f"(pooled over {len(files)} detector seeds, 10 traces, 2 replicates).", "",
               "| rule | h | %/h | contained |", "|---|---|---|---|"])
    for k, v in out.items():
        md.append(f"| {k} | {num_(v['h'])} | {num_(v['rate'], 100, 4)} | {num_(v['contained'], 1, 3)} |")
    md.append("")
    for stat, rr in tests.items():
        md += [f"**{stat}**: " + "; ".join(f"carry vs {r['vs']} Δ {r['mean_diff']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}], "
                                           f"W/L {r['wins']}/{r['losses']}, p (Holm) {r['p_holm']:.4f}" for r in rr), ""]


def num_(x, scale=1.0, d=2):
    return "--" if x is None else f"{scale * x:.{d}f}"


def main():
    r7, r8, r9 = configs()
    section_describe()
    section_cal(r9)
    section_sweep(r7, r8)
    section_replay()
    section_replay_curves()
    section_replay_curves("r9_replay_curves_a20", "replay_matched_a20", "20%")
    v = {}
    if "sweep" in js and "replay" in js:
        v = {"H92": js["replay"]["H92"], "H93a_real": js["replay"]["H93a"], "H93b_all_variants": js["sweep"]["H93b"]["all"],
             "H93c_sigma03": js["sweep"]["H93c"],
             "H93_confirmed": bool(js["replay"]["H93a"] and js["sweep"]["H93b"]["all"] and js["sweep"]["H93c"]),
             "H94_synthetic": js["sweep"]["H94_synthetic"]["verdict"], "H94_real": js["replay"]["H94_real"]}
    js["verdicts"] = v
    md.extend(["## Verdicts", "", "```", json.dumps(v, indent=1), "```", ""])
    open(os.path.join(REPO, "reports", "stats", "V2_R9.md"), "w", encoding="utf-8").write("\n".join(md))
    json.dump(js, open(os.path.join(REPO, "reports", "stats", "v2_r9_stats.json"), "w", encoding="utf-8"), indent=1, default=float)
    print("\n".join(md[-12:]))


if __name__ == "__main__":
    main()
