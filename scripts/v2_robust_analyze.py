#!/usr/bin/env python3
"""Analysis of the R6 robustness experiment (post-hoc; the analysis plan.
Input: results/v2/robust_test/<world>_<dataset>.json; configs/v2/robust.yaml.
Output: reports/stats/V2_ROBUST.md, reports/stats/v2_robust_stats.json.

R1  per world, dataset and statistic at the main budget: carry vs memory and carry vs reset on the
    contained share of persistent 5% attackers (unit: trace seed, replicates averaged); Wilcoxon, Holm
    within world over 18 tests; "robust" in a world if, for carry-over's best statistic (validation),
    carry beats both in >= 2 of 3 datasets.
R2  steady-state honest false-alarm rate on test, with Poisson 95% intervals, against the budget.
R3  contained share by class; containment against the attacker's longest stay at one edge node.
R4  contained share of every rule (all scenarios), per world.
"""
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

WORLDS = ("W0", "W1", "W2", "W3")
STATS = ("share", "llr", "ewma")
PL = ("carry", "memory", "reset")
MAIN = "b01"
md, js = ["# R6 robustness of C1 (post-hoc; scripts/v2_robust_analyze.py)", ""], {}


def load():
    att, hon = [], []
    for w in WORLDS:
        for d in DATASETS:
            p = os.path.join(REPO, "results", "v2", "robust_test", f"{w}_{d}.json")
            if not os.path.exists(p):
                continue
            for r in json.load(open(p, encoding="utf-8")):
                key = {k: r[k] for k in ("world", "dataset", "seed", "rep", "scenario", "budget", "rule")}
                hon.append({**key, "events": r["honest_events"], "hours": r["honest_hours"],
                            **{f"ev_{c}": v[0] for c, v in r["honest_by_class"].items()},
                            **{f"hr_{c}": v[1] for c, v in r["honest_by_class"].items()}})
                for a in r["att"]:
                    att.append({**key, **a})
    return pd.DataFrame(att), pd.DataFrame(hon)


def df_md(df):
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(out)


def poisson_ci(k, t):
    lo = st.chi2.ppf(0.025, 2 * k) / 2 / t if k > 0 else 0.0
    hi = st.chi2.ppf(0.975, 2 * k + 2) / 2 / t
    return lo, hi


TARGETS = {"t01": 0.001, "t1": 0.01}


def curves_analysis():
    """Amendment A2: placements compared at equal false-alarm rates matched on the test runs."""
    global md
    out = {"matched": {}, "R6": {}, "curve": {}}
    far_grid = np.geomspace(1e-4, 0.1, 30)
    for w in WORLDS:
        store = {}
        for d in DATASETS:
            p = os.path.join(REPO, "results", "v2", "robust_curves", f"{w}_{d}.json")
            if not os.path.exists(p):
                continue
            J = json.load(open(p, encoding="utf-8"))
            for key, A in J["rules"].items():
                ev, hrs = np.array(A["ev"], float), np.array(A["hours"], float)
                cont, n = np.array(A["cont"], float), np.array(A["n_att"], float)
                far = ev.sum(0) / hrs.sum()
                share = cont.sum(0) / n.sum()
                for t, T in TARGETS.items():
                    ok = np.flatnonzero(far <= T)
                    i = int(ok.min()) if len(ok) else len(far) - 1
                    store[(d, key, t)] = {"far": float(far[i]), "matched": bool(len(ok)), "per_seed": (cont[:, i] / n).tolist(),
                                          "contained": float(share[i])}
                    out["matched"][f"{w}/{d}/{key}/{t}"] = {k: v for k, v in store[(d, key, t)].items()}
                # curve: contained at matched thresholds over a grid of false-alarm rates
                cv = []
                for T in far_grid:
                    ok = np.flatnonzero(far <= T)
                    cv.append(float(share[int(ok.min())]) if len(ok) else np.nan)
                out["curve"][f"{w}/{d}/{key}"] = cv
        for t in TARGETS:
            rows = []
            for d in DATASETS:
                for st_ in STATS:
                    c = store.get((d, f"{st_}/carry", t))
                    if not c:
                        continue
                    for other in ("memory", "reset"):
                        o = store[(d, f"{st_}/{other}", t)]
                        dlt = np.array(c["per_seed"]) - np.array(o["per_seed"])
                        p = st.wilcoxon(dlt).pvalue if (dlt != 0).any() else 1.0
                        rows.append({"dataset": d, "stat": st_, "vs": other, "mean_diff": float(dlt.mean()),
                                     "wins": int((dlt > 0).sum()), "losses": int((dlt < 0).sum()), "p": float(p),
                                     "carry": c["contained"], "other": o["contained"]})
            for r_, pa in zip(rows, holm([r["p"] for r in rows])):
                r_["p_holm"] = float(pa)
            per_ds = []
            for d in DATASETS:
                cands = [s_ for s_ in STATS if (d, f"{s_}/carry", t) in store and store[(d, f"{s_}/carry", t)]["matched"]]
                if not cands:
                    continue
                best = max(cands, key=lambda s_: store[(d, f"{s_}/carry", t)]["contained"])
                ok = all(any(r["dataset"] == d and r["stat"] == best and r["vs"] == o_ and r["p_holm"] < 0.05 and r["mean_diff"] > 0
                             for r in rows) for o_ in ("memory", "reset"))
                per_ds.append({"dataset": d, "best_stat": best, "carry_beats_both": bool(ok)})
            verdict = "robust" if sum(x["carry_beats_both"] for x in per_ds) >= 2 else "not robust"
            out["R6"][f"{w}/{t}"] = {"rows": rows, "per_dataset": per_ds, "verdict": verdict}
            md.extend([f"**A2 {w}, equal test false-alarm rate {100 * TARGETS[t]:g}%/h** — verdict: **{verdict}** (" +
                       "; ".join(f"{x['dataset']}: best {x['best_stat']}, beats both {x['carry_beats_both']}" for x in per_ds) + ")", "",
                       "| dataset | statistic | vs | carry | other | mean Δ (seed) | W / L | p (Holm) |", "|---|---|---|---|---|---|---|---|"] +
                      [f"| {r['dataset']} | {r['stat']} | {r['vs']} | {r['carry']:.3f} | {r['other']:.3f} | {r['mean_diff']:.3f} | "
                       f"{r['wins']} / {r['losses']} | {r['p_holm']:.4f} |" for r in rows] + [""])
    out["far_grid"] = far_grid.tolist()
    return out


def main():
    global md
    a, h = load()
    conf = yaml.safe_load(open(os.path.join(REPO, "configs", "v2", "robust.yaml"), encoding="utf-8"))
    B = conf["budgets"]
    a["stat"], a["placement"] = a.rule.str.split("/").str[0], a.rule.str.split("/").str[1]
    h["stat"], h["placement"] = h.rule.str.split("/").str[0], h.rule.str.split("/").str[1]

    # ---------------- R4 / main table: contained share per world and rule (persistent 5%), hazard
    md.extend(["## Contained share of persistent 5% attackers within 60 min, and steady-state honest false-alarm "
               "rate on test (% per device-hour), at the main budget (0.1% per device-hour)", ""])
    tab = {}
    md += ["| world | dataset | rule | contained | false alarms %/h [95% CI] |", "|---|---|---|---|---|"]
    for w in WORLDS:
        for d in DATASETS:
            for st_ in STATS:
                for pl in PL:
                    r = f"{st_}/{pl}"
                    g = a[(a.world == w) & (a.dataset == d) & (a.budget == MAIN) & (a.rule == r) & (a.scenario == "persistent_0.05")]
                    hh = h[(h.world == w) & (h.dataset == d) & (h.budget == MAIN) & (h.rule == r)]
                    if g.empty:
                        continue
                    ev, hr = hh.events.sum(), hh.hours.sum()
                    lo, hi = poisson_ci(ev, hr)
                    tab[f"{w}/{d}/{r}"] = {"contained": float(g.contained.mean()), "hazard": float(ev / hr),
                                           "hz_lo": float(lo), "hz_hi": float(hi), "events": int(ev), "hours": float(hr)}
                    md.append(f"| {w} | {d} | {r} | {g.contained.mean():.3f} | {100 * ev / hr:.3f} [{100 * lo:.3f}, {100 * hi:.3f}] |")
    md.append("")
    js["table"] = tab

    # ---------------- R1
    md.extend(["## R1: carry vs memory and vs reset, persistent 5%, per world, dataset and statistic (seed level, n = 10; "
               "Holm within world over 18)", ""])
    r1 = {}
    for w in WORLDS:
        rows = []
        for d in DATASETS:
            for st_ in STATS:
                g = a[(a.world == w) & (a.dataset == d) & (a.budget == MAIN) & (a.stat == st_) & (a.scenario == "persistent_0.05")]
                if g.empty:
                    continue
                s = g.groupby(["seed", "placement"]).contained.mean().unstack()
                for other in ("memory", "reset"):
                    dlt = (s["carry"] - s[other]).dropna()
                    p = st.wilcoxon(dlt).pvalue if (dlt != 0).any() else 1.0
                    lo, hi = bootstrap_ci(dlt.to_numpy())
                    rows.append({"world": w, "dataset": d, "stat": st_, "vs": other, "n": int(len(dlt)),
                                 "mean_diff": float(dlt.mean()), "ci_lo": lo, "ci_hi": hi,
                                 "wins": int((dlt > 0).sum()), "losses": int((dlt < 0).sum()), "p": float(p)})
        for r_, pa in zip(rows, holm([r["p"] for r in rows])):
            r_["p_holm"] = float(pa)
        # verdict: carry-over's best statistic on validation, per dataset
        robust_ds = []
        for d in DATASETS:
            ch = conf["chosen"][w][d][MAIN]
            feasible = [s_ for s_ in STATS if ch[f"{s_}/carry"]["val_hazard"] <= B[MAIN]] or list(STATS)
            best = max(feasible, key=lambda s_: ch[f"{s_}/carry"]["val_contained"])
            ok = all(any(r["dataset"] == d and r["stat"] == best and r["vs"] == o and r["p_holm"] < 0.05 and r["mean_diff"] > 0
                         for r in rows) for o in ("memory", "reset"))
            robust_ds.append({"dataset": d, "best_stat": best, "carry_beats_both": bool(ok)})
        verdict = "robust" if sum(x["carry_beats_both"] for x in robust_ds) >= 2 else "not robust"
        r1[w] = {"rows": rows, "per_dataset": robust_ds, "verdict": verdict}
        md += [f"**{w}** — verdict: **{verdict}** (" + "; ".join(f"{x['dataset']}: best {x['best_stat']}, carry beats both "
               f"{x['carry_beats_both']}" for x in robust_ds) + ")", "",
               "| dataset | statistic | vs | mean Δ [95% CI] | W / L | p (Holm) |", "|---|---|---|---|---|---|"]
        md += [f"| {r['dataset']} | {r['stat']} | {r['vs']} | {r['mean_diff']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | "
               f"{r['wins']} / {r['losses']} | {r['p_holm']:.4f} |" for r in rows]
        md.append("")
    js["R1"] = r1

    # ---------------- R2: calibration transfer
    md.extend(["## R2: steady-state honest false-alarm rate on test vs budget (pooled over datasets per world)", ""])
    r2 = {}
    md += ["| world | budget | rule | events | device-hours | %/h [95% CI] | within budget |", "|---|---|---|---|---|---|---|"]
    for w in WORLDS:
        for b, bv in B.items():
            for st_ in STATS:
                for pl in PL:
                    r = f"{st_}/{pl}"
                    hh = h[(h.world == w) & (h.budget == b) & (h.rule == r)]
                    if hh.empty:
                        continue
                    ev, hr = hh.events.sum(), hh.hours.sum()
                    lo, hi = poisson_ci(ev, hr)
                    r2[f"{w}/{b}/{r}"] = {"events": int(ev), "hours": float(hr), "rate": float(ev / hr), "lo": lo, "hi": hi,
                                          "within": bool(lo <= bv)}
                    md.append(f"| {w} | {b} | {r} | {ev} | {hr:.0f} | {100 * ev / hr:.3f} [{100 * lo:.3f}, {100 * hi:.3f}] | "
                              f"{'yes' if lo <= bv else 'no'} |")
    md.append("")
    js["R2"] = r2

    # ---------------- R3: classes and stays
    md.extend(["## R3: contained share by class (persistent scenarios, main budget, pooled datasets)", ""])
    r3 = {}
    md += ["| world | rule | vehicles | pedestrians | static |", "|---|---|---|---|---|"]
    pers = a[a.scenario.str.startswith("persistent") & (a.budget == MAIN)]
    for w in WORLDS:
        for st_ in STATS:
            for pl in PL:
                g = pers[(pers.world == w) & (pers.rule == f"{st_}/{pl}")]
                if g.empty:
                    continue
                v = {c: float(g[g.vclass == c].contained.mean()) for c in ("veh", "ped", "static")}
                r3[f"{w}/{st_}/{pl}"] = v
                md.append(f"| {w} | {st_}/{pl} | {v['veh']:.3f} | {v['ped']:.3f} | {v['static']:.3f} |")
    md.append("")
    stays = []
    g = a[(a.scenario == "persistent_0.05") & (a.budget == MAIN)]
    bins = [0, 2, 4, 8, 16, 32, 121]
    g = g.assign(stay=pd.cut(g.residence_w, bins, right=False))
    for (w, r, b_), gg in g.groupby(["world", "rule", "stay"], observed=True):
        stays.append({"world": w, "rule": r, "stay_lo_w": int(b_.left), "stay_hi_w": int(b_.right),
                      "contained": float(gg.contained.mean()), "n": int(len(gg))})
    js["R3"] = {"by_class": r3, "by_stay": stays}

    # ---------------- all scenarios, main budget, pooled datasets
    md.extend(["## Contained share by scenario (main budget, pooled datasets)", ""])
    sc = a[a.budget == MAIN].groupby(["world", "rule", "scenario"]).contained.mean().unstack()
    md += [df_md(sc), ""]
    js["by_scenario"] = {f"{w}/{r}": row.dropna().to_dict() for (w, r), row in sc.iterrows()}
    # exposure with and without per-flow blocking, persistent 5%
    ex = a[(a.budget == MAIN) & (a.scenario == "persistent_0.05")]
    ex = ex.groupby(["world", "rule"]).apply(lambda x: pd.Series({
        "exposure": x.exposure.sum() / max(x.attack_total.sum(), 1),
        "exposure_blocking": x.exposure_blocking.sum() / max(x.attack_total.sum(), 1)}), include_groups=False)
    js["exposure"] = {f"{w}/{r}": row.to_dict() for (w, r), row in ex.iterrows()}
    md.extend(["## Exposure (share of attack flows not blocked within 60 min), persistent 5%, with and without per-flow "
               "blocking of flagged flows", "", df_md(ex), ""])
    js["verdicts"] = {w: r1[w]["verdict"] for w in WORLDS if w in r1}
    md.extend(["## Amendment A2: placements at equal false-alarm rates matched on test (persistent 5%)", ""])
    js["A2"] = curves_analysis()
    js["verdicts_A2"] = {k: v["verdict"] for k, v in js["A2"]["R6"].items()}
    open(os.path.join(REPO, "reports", "stats", "V2_ROBUST.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump(js, open(os.path.join(REPO, "reports", "stats", "v2_robust_stats.json"), "w", encoding="utf-8"),
              indent=1, default=float)
    print("wrote reports/stats/V2_ROBUST.md", js["verdicts"])


if __name__ == "__main__":
    main()
