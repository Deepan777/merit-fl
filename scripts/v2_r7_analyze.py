#!/usr/bin/env python3
"""Analysis of R7 the analysis plan; post hoc). Output: reports/stats/V2_R7.md, v2_r7_stats.json.

H7.1  steady-state honest false-alarm rate on test with the theory threshold, carry-over, per world and dataset
H7.2  carry vs memory and vs reset, persistent 5%, seed level (n = 10), Holm within world over 6
H7.3  clustered compromise, adaptive attackers at the floor, lost context (descriptive)
      matched-rate operating curves (descriptive)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.evaluation.stats import bootstrap_ci, holm  # noqa: E402
from src.v2.plan import DATASETS  # noqa: E402

WORLDS = ("W0", "W1", "W2", "W3", "W4")
BUDGETS = {"b01": 0.001, "b1": 0.01}
md, js = ["# R7: self-calibrated evidence and attacks on the evidence (post-hoc; scripts/v2_r7_analyze.py)", ""], {}


def poisson_ci(k, t):
    lo = st.chi2.ppf(0.025, 2 * k) / 2 / t if k > 0 else 0.0
    return lo, st.chi2.ppf(0.975, 2 * k + 2) / 2 / t


def load():
    att, hon = [], []
    for w in WORLDS:
        for d in DATASETS:
            p = os.path.join(REPO, "results", "v2", "r7_test", f"{w}_{d}.json")
            if not os.path.exists(p):
                continue
            for r in json.load(open(p, encoding="utf-8")):
                key = {k: r[k] for k in ("world", "dataset", "seed", "rep", "scenario", "budget", "placement")}
                hon.append({**key, "events": r["honest_events"], "hours": r["honest_hours"],
                            **{f"ev_{c}": v[0] for c, v in r["honest_by_class"].items()},
                            **{f"hr_{c}": v[1] for c, v in r["honest_by_class"].items()}})
                for a in r["att"]:
                    att.append({**key, **a})
    return pd.DataFrame(att), pd.DataFrame(hon)


def main():
    global md
    a, h = load()
    # H7.1
    md.extend(["## H7.1: steady-state honest false-alarm rate on test, theory threshold h = ln(ARL0)", "",
               "| world | budget | dataset | placement | events | device-hours | %/h [95% CI] | within |", "|---|---|---|---|---|---|---|---|"])
    h71 = {}
    for w in WORLDS:
        for b, B in BUDGETS.items():
            for d in DATASETS:
                for pl in ("carry", "memory", "reset", "carry_loss"):
                    g = h[(h.world == w) & (h.budget == b) & (h.dataset == d) & (h.placement == pl)]
                    if g.empty:
                        continue
                    ev, hr = int(g.events.sum()), float(g.hours.sum())
                    lo, hi = poisson_ci(ev, hr)
                    h71[f"{w}/{b}/{d}/{pl}"] = {"events": ev, "hours": hr, "rate": ev / hr, "lo": lo, "hi": hi, "within": ev / hr <= B}
                    if pl == "carry":
                        md.append(f"| {w} | {b} | {d} | {pl} | {ev} | {hr:.0f} | {100 * ev / hr:.4f} [{100 * lo:.4f}, {100 * hi:.4f}] | "
                                  f"{'yes' if ev / hr <= B else 'no'} |")
    sup = all(h71[f"{w}/b01/{d}/carry"]["within"] for w in ("W0", "W1", "W2", "W3") for d in DATASETS if f"{w}/b01/{d}/carry" in h71)
    js["H71"] = {"rows": h71, "supported": bool(sup),
                 "W4_within": [h71[f"W4/b01/{d}/carry"]["within"] for d in DATASETS if f"W4/b01/{d}/carry" in h71]}
    md.extend(["", f"**H7.1 (W0-W3, main budget): {'supported' if sup else 'not supported'}**; W4 within: {js['H71']['W4_within']}", ""])
    # H7.2
    md.extend(["## H7.2: carry vs memory and vs reset at the theory threshold, persistent 5% (seed level, Holm within world over 6)", ""])
    h72 = {}
    for b in BUDGETS:
        for w in WORLDS:
            rows = []
            for d in DATASETS:
                g = a[(a.world == w) & (a.budget == b) & (a.dataset == d) & (a.scenario == "persistent_0.05")]
                if g.empty:
                    continue
                s = g.groupby(["seed", "placement"]).contained.mean().unstack()
                for other in ("memory", "reset"):
                    dlt = (s["carry"] - s[other]).dropna()
                    p = st.wilcoxon(dlt).pvalue if (dlt != 0).any() else 1.0
                    lo, hi = bootstrap_ci(dlt.to_numpy())
                    rows.append({"dataset": d, "vs": other, "carry": float(s["carry"].mean()), "other": float(s[other].mean()),
                                 "mean_diff": float(dlt.mean()), "ci_lo": lo, "ci_hi": hi, "wins": int((dlt > 0).sum()),
                                 "losses": int((dlt < 0).sum()), "p": float(p)})
            for r_, pa in zip(rows, holm([r["p"] for r in rows])):
                r_["p_holm"] = float(pa)
            per = [{"dataset": d, "beats_both": all(any(r["dataset"] == d and r["vs"] == o and r["p_holm"] < 0.05 and r["mean_diff"] > 0
                                                        for r in rows) for o in ("memory", "reset"))} for d in DATASETS]
            verdict = "robust" if sum(x["beats_both"] for x in per) >= 2 else "not robust"
            h72[f"{w}/{b}"] = {"rows": rows, "per_dataset": per, "verdict": verdict}
            md += [f"**{w}, {b}: {verdict}**", "", "| dataset | vs | carry | other | Δ [95% CI] | W / L | p (Holm) |",
                   "|---|---|---|---|---|---|---|"]
            md += [f"| {r['dataset']} | {r['vs']} | {r['carry']:.3f} | {r['other']:.3f} | {r['mean_diff']:.3f} [{r['ci_lo']:.3f}, "
                   f"{r['ci_hi']:.3f}] | {r['wins']} / {r['losses']} | {r['p_holm']:.4f} |" for r in rows]
            md.append("")
    js["H72"] = h72
    # summary table: contained share by scenario, placement, world (main budget, pooled datasets)
    sc = a[a.budget == "b01"].groupby(["world", "placement", "scenario"]).contained.mean().unstack()
    js["by_scenario"] = {f"{w}/{pl}": row.dropna().to_dict() for (w, pl), row in sc.iterrows()}
    md.extend(["## Contained share by scenario (main budget, pooled datasets)", "",
               "| world | placement | " + " | ".join(sc.columns) + " |", "|---|---|" + "---|" * len(sc.columns)])
    for (w, pl), row in sc.iterrows():
        md.append(f"| {w} | {pl} | " + " | ".join(f"{v:.3f}" for v in row.values) + " |")
    md.append("")
    # H7.3 adaptive: containment and floors
    ad = a[(a.budget == "b01") & a.scenario.str.startswith("adaptive")]
    js["adaptive"] = {f"{w}/{s_}": float(g[g.placement == "carry"].contained.mean())
                      for (w, s_), g in ad.groupby(["world", "scenario"])}
    fl = a[(a.budget == "b01") & (a.scenario == "adaptive_1_2") & (a.placement == "carry")]
    js["floor"] = {d: {"median_pct": float(100 * g.floor.median()), "share_above_1": float((g.floor >= 1).mean())}
                   for d, g in fl[fl.world == "W1"].groupby("dataset")}
    # lost context
    js["loss"] = {f"{w}": {pl: float(a[(a.world == w) & (a.budget == "b01") & (a.scenario == "persistent_0.05") &
                                       (a.placement == pl)].contained.mean()) for pl in ("carry", "carry_loss", "reset")} for w in WORLDS}
    # by class, W1
    pers = a[(a.budget == "b01") & a.scenario.isin(["persistent_0.02", "persistent_0.05", "persistent_0.1", "persistent_0.2"])]
    js["by_class"] = {f"{w}/{pl}": {c: float(pers[(pers.world == w) & (pers.placement == pl) & (pers.vclass == c)].contained.mean())
                                    for c in ("veh", "ped", "static")} for w in WORLDS for pl in ("carry", "memory", "reset")}
    # curves (matched rates)
    curve, far_grid = {}, np.geomspace(1e-4, 0.1, 30)
    matched = {}
    for w in WORLDS:
        for d in DATASETS:
            p = os.path.join(REPO, "results", "v2", "r7_curves", f"{w}_{d}.json")
            if not os.path.exists(p):
                continue
            J = json.load(open(p, encoding="utf-8"))
            for pl, A in J["rules"].items():
                ev, hrs = np.array(A["ev"], float), np.array(A["hours"], float)
                cont, n = np.array(A["cont"], float), np.array(A["n_att"], float)
                far = ev.sum(0) / hrs.sum()
                share = cont.sum(0) / n.sum()
                cv = []
                for T in far_grid:
                    ok = np.flatnonzero(far <= T)
                    cv.append(float(share[int(ok.min())]) if len(ok) else np.nan)
                curve[f"{w}/{d}/{pl}"] = cv
                ok = np.flatnonzero(far <= 0.001)
                matched[f"{w}/{d}/{pl}"] = float(share[int(ok.min())]) if len(ok) else None
    js["curve"], js["far_grid"], js["matched_b01"] = curve, far_grid.tolist(), matched
    js["verdicts"] = {k: v["verdict"] for k, v in h72.items()}
    open(os.path.join(REPO, "reports", "stats", "V2_R7.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump(js, open(os.path.join(REPO, "reports", "stats", "v2_r7_stats.json"), "w", encoding="utf-8"), indent=1, default=float)
    print("wrote V2_R7.md; H7.1", js["H71"]["supported"], "; H7.2", js["verdicts"])


if __name__ == "__main__":
    main()
