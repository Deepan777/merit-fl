#!/usr/bin/env python3
"""Re-analyses of the pre-registered v2 data (post hoc);
no new runs. Post-hoc, reported as such. Output: reports/stats/V2_REANALYSIS.md, v2_reanalysis.json.

1. C1 primary tests at the level of independent mobility/attacker realisations: the three datasets share
   traces and attackers, so the unit is the trace seed (n = 10, datasets averaged); also for the
   detectors of residence-aware training with forwarding (R5).
2. Mobility dependence within classes: Spearman correlation of (carry - recalibrated reset) with the
   attacker's handover rate per class, with a seed-cluster bootstrap 95% CI (no naive p-values).
3. False containment per dataset and device class, all methods, including the stealthy scenario.
4. Time to containment: the pre-registered conditional comparison (from v2_stats.json) and an
   exploratory restricted mean time to containment within 30 min (not contained = 30), seed level.
5. C2: uplink to target with censored runs ranked worst (as the plan states), per-dataset tests with
   Holm over the three datasets, and equivalence checks (90% bootstrap CI within +-0.02).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.evaluation.stats import bootstrap_ci, holm  # noqa: E402
from src.v2.plan import DATASETS  # noqa: E402

PRIMARY = ("persistent_0.02", "persistent_0.05", "persistent_0.1", "persistent_0.2", "handover_timed", "on_off_0.2")
PERSIST = PRIMARY[:4]
md, js = ["# v2 re-analyses after the pre-specified analysis (post-hoc; scripts/v2_reanalysis.py)", ""], {}


def load(sub):
    rows, att = [], []
    for d in DATASETS:
        for r in json.load(open(os.path.join(REPO, "results", "v2", sub, f"{d}.json"), encoding="utf-8")):
            a, h = pd.DataFrame(r["att"]), pd.DataFrame(r["hon"])
            rows.append({"dataset": d, "seed": r["seed"], "scenario": r["scenario"], "method": r["method"],
                         "contained": a.contained.mean(), "false": h.contained.mean()})
            for x in r["att"]:
                att.append({"dataset": d, "seed": r["seed"], "scenario": r["scenario"], "method": r["method"], **x})
    return pd.DataFrame(rows), pd.DataFrame(att)


def seed_level(c, label):
    global md
    out = []
    for s in PRIMARY:
        g = c[c.scenario == s].groupby(["seed", "method"]).contained.mean().unstack()
        dlt = g["cusum_carry"] - g["cusum_reset_cal"]
        p = st.wilcoxon(dlt).pvalue if (dlt != 0).any() else 1.0
        lo, hi = bootstrap_ci(dlt.to_numpy())
        out.append({"scenario": s, "n": int(len(dlt)), "mean_diff": float(dlt.mean()), "ci_lo": lo, "ci_hi": hi,
                    "wins": int((dlt > 0).sum()), "losses": int((dlt < 0).sum()), "p": float(p)})
    for r, pa in zip(out, holm([r["p"] for r in out])):
        r["p_holm"] = float(pa)
    md.extend([f"**{label}: carry-over minus recalibrated reset, contained share, seed level (n = 10, datasets "
               f"averaged; Holm over 6)**", "", "| scenario | mean Δ [95% CI] | W / L | p | p (Holm) |", "|---|---|---|---|---|"])
    md += [f"| {r['scenario']} | {r['mean_diff']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | {r['wins']} / {r['losses']} | "
           f"{r['p']:.4f} | {r['p_holm']:.4f} |" for r in out]
    md.append("")
    return out


def within_class(a, rng_seed=0):
    g = a[a.scenario.isin(PERSIST) & a.method.isin(["cusum_carry", "cusum_reset_cal"])]
    piv = g.pivot_table(index=["dataset", "seed", "scenario", "u", "vclass", "ho_per_h"], columns="method",
                        values="contained", aggfunc="mean").reset_index()
    piv["diff"] = piv.cusum_carry.astype(float) - piv.cusum_reset_cal.astype(float)
    rng = np.random.default_rng(rng_seed)
    seeds = piv.seed.unique()
    out = {}
    md.extend(["**Mobility dependence: Spearman correlation of (carry − recalibrated reset) with the attacker's "
               "handover rate, by class (seed-cluster bootstrap 95% CI, 2,000 resamples)**", "",
               "| class | n attacker-runs | ρ [95% CI] |", "|---|---|---|"])
    for name, m in (("vehicles", piv.vclass == "veh"), ("pedestrians", piv.vclass == "ped"),
                    ("mobile (vehicles and pedestrians)", piv.vclass != "static"), ("all", np.ones(len(piv), bool))):
        sub = piv[m]
        rho = st.spearmanr(sub.ho_per_h, sub["diff"]).statistic
        bs = []
        for _ in range(2000):
            pick = rng.choice(seeds, size=len(seeds), replace=True)
            b = pd.concat([sub[sub.seed == s] for s in pick])
            if b["diff"].nunique() > 1 and b.ho_per_h.nunique() > 1:
                bs.append(st.spearmanr(b.ho_per_h, b["diff"]).statistic)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[name] = {"rho": float(rho), "ci_lo": float(lo), "ci_hi": float(hi), "n": int(len(sub))}
        md.append(f"| {name} | {len(sub)} | {rho:.2f} [{lo:.2f}, {hi:.2f}] |")
    md.append("")
    return out


def false_by_class(c_all, a_dummy, sub):
    """False containment of honest devices per dataset, method and class (needs device classes)."""
    from src.v2.config import load_v2
    from src.v2.hfl import load_trace
    vc = {}
    rows = []
    for d in DATASETS:
        for r in json.load(open(os.path.join(REPO, "results", "v2", sub, f"{d}.json"), encoding="utf-8")):
            s = r["seed"]
            if s not in vc:
                vc[s] = load_trace(load_v2(None, {"seed": s}), s)[1]
            for x in r["hon"]:
                rows.append({"dataset": d, "scenario": r["scenario"], "method": r["method"], "vclass": str(vc[s][x["u"]]),
                             "contained": x["contained"]})
    t = pd.DataFrame(rows)
    out = {}
    md.extend(["**False containment of honest devices (% within the 1-h run), per dataset and class (primary "
               "scenarios; last column: stealthy scenario, all classes)**", "",
               "| dataset | method | all | vehicles | pedestrians | static | stealthy |", "|---|---|---|---|---|---|---|"])
    for d in DATASETS:
        for m in ("cusum_carry", "cusum_reset_cal", "cusum_reset", "trust_carry", "trust_cold"):
            g = t[(t.dataset == d) & (t.method == m)]
            p = g[g.scenario.isin(PRIMARY)]
            vals = [100 * p.contained.mean()] + [100 * p[p.vclass == c].contained.mean() for c in ("veh", "ped", "static")]
            vals.append(100 * g[g.scenario == "stealthy_0.005"].contained.mean())
            out[f"{d}/{m}"] = vals
            md.append(f"| {d} | {m} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    md.append("")
    return out


def rmst(a, horizon_min=30):
    g = a[a.scenario.isin(PRIMARY) & a.method.isin(["cusum_carry", "cusum_reset_cal"])].copy()
    g["t"] = np.where(g.contained, np.minimum(g.delay_s / 60, horizon_min), horizon_min)
    piv = g.groupby(["seed", "scenario", "method"]).t.mean().unstack()
    out = []
    md.extend([f"**Exploratory: restricted mean time to containment within {horizon_min} min (attackers not contained "
               f"within {horizon_min} min count as {horizon_min}); carry-over minus recalibrated reset, seed level (n = 10)**",
               "", "| scenario | carry (min) | reset (min) | mean Δ [95% CI] | p |", "|---|---|---|---|---|"])
    for s in PRIMARY:
        p = piv.xs(s, level="scenario")
        dlt = p.cusum_carry - p.cusum_reset_cal
        lo, hi = bootstrap_ci(dlt.to_numpy())
        pv = st.wilcoxon(dlt).pvalue if (dlt != 0).any() else 1.0
        out.append({"scenario": s, "carry": float(p.cusum_carry.mean()), "reset": float(p.cusum_reset_cal.mean()),
                    "mean_diff": float(dlt.mean()), "ci_lo": lo, "ci_hi": hi, "p": float(pv)})
        md.append(f"| {s} | {p.cusum_carry.mean():.1f} | {p.cusum_reset_cal.mean():.1f} | {dlt.mean():.1f} [{lo:.1f}, {hi:.1f}] | {pv:.4f} |")
    md.append("")
    return out


def c2():
    import v2_analyze as va
    df = va.load_hfl()
    t = va.to_target(df[df.group == "test_hfl"].copy())
    out = {}
    # uplink to target, censored runs ranked worst (plan)
    big = 1e9
    t["utt_rank"] = t.uplink_to_target.replace(np.inf, big)
    a = t[t.label == "residence+fw"].set_index(["dataset", "seed"]).utt_rank
    b = t[t.label == "random"].set_index(["dataset", "seed"]).utt_rank
    j = a.to_frame("a").join(b.to_frame("b"))
    dif = j.b - j.a
    p = st.wilcoxon(dif).pvalue
    out["uplink_ranked_worst"] = {"n": int(len(dif)), "wins": int((dif > 0).sum()), "losses": int((dif < 0).sum()),
                                  "ties": int((dif == 0).sum()), "p": float(p)}
    md.extend(["**C2: uplink to target with censored runs ranked worst (as pre-registered)**", "",
               f"n = {len(dif)}, residence+fw better / worse / tied: {(dif > 0).sum()} / {(dif < 0).sum()} / {(dif == 0).sum()}, "
               f"Wilcoxon p = {p:.3f}.", ""])
    # per dataset with Holm over 3
    rows = []
    for d in DATASETS:
        g = t[t.dataset == d]
        a = g[g.label == "residence+fw"].set_index("seed").f1
        b = g[g.label == "random"].set_index("seed").f1
        dd = (a - b).dropna()
        rows.append({"dataset": d, "mean_diff": float(dd.mean()), "p": float(st.wilcoxon(dd).pvalue)})
    for r, pa in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = float(pa)
    out["per_dataset_holm"] = rows
    md.extend(["**C2 per dataset, residence+fw − random macro-F1, Holm over 3 datasets**", "",
               "| dataset | mean Δ | p | p (Holm) |", "|---|---|---|---|"] +
              [f"| {r['dataset']} | {r['mean_diff']:.3f} | {r['p']:.4f} | {r['p_holm']:.4f} |" for r in rows] + [""])
    # equivalence: 90% bootstrap CI of the paired difference within +-0.02
    eq = []
    for other in ("random+fw", "speed", "speed+fw", "residence"):
        a = t[t.label == "residence+fw"].set_index(["dataset", "seed"]).f1
        b = t[t.label == other].set_index(["dataset", "seed"]).f1
        dd = (a - b).dropna().to_numpy()
        rng = np.random.default_rng(0)
        bs = [rng.choice(dd, len(dd)).mean() for _ in range(10000)]
        lo, hi = np.percentile(bs, [5, 95])
        eq.append({"vs": other, "mean_diff": float(dd.mean()), "ci90_lo": float(lo), "ci90_hi": float(hi),
                   "equivalent": bool(lo > -0.02 and hi < 0.02)})
    out["equivalence"] = eq
    md.extend(["**C2 equivalence check (margin ±0.02 macro-F1; 90% bootstrap CI of the paired difference, n = 30)**", "",
               "| residence+fw vs | mean Δ | 90% CI | equivalent |", "|---|---|---|---|"] +
              [f"| {r['vs']} | {r['mean_diff']:.3f} | [{r['ci90_lo']:.3f}, {r['ci90_hi']:.3f}] | {'yes' if r['equivalent'] else 'no'} |"
               for r in eq] + [""])
    return out


def main():
    c, a = load("containment_test")
    md.extend(["## C1: unit of analysis", "", "The three datasets share the mobility traces and the attacker devices of a "
               "seed, so the independent replicates are the 10 trace seeds.", ""])
    js["seed_level"] = seed_level(c, "Pre-registered detectors (random participation)")
    c5, a5 = load("containment_test_residence_fw")
    js["seed_level_resfw"] = seed_level(c5, "R5: detectors of residence-aware training with forwarding")
    js["resfw_summary"] = {s: {m: float(c5[(c5.scenario == s) & (c5.method == m)].contained.mean())
                               for m in ("cusum_carry", "cusum_reset_cal")} for s in PRIMARY}
    js["resfw_false"] = {m: float(c5[c5.scenario.isin(PRIMARY) & (c5.method == m)]["false"].mean())
                         for m in ("cusum_carry", "cusum_reset_cal")}
    md.extend(["## C1: mobility dependence", ""])
    js["within_class"] = within_class(a)
    md.extend(["## C1: false containment", ""])
    js["false_by_class"] = false_by_class(c, a, "containment_test")
    md.extend(["## C1: time to containment", ""])
    pre = json.load(open(os.path.join(REPO, "reports", "stats", "v2_stats.json"), encoding="utf-8"))["V1"]["secondary"]["delay"]
    md.extend(["Pre-registered (conditional on containment; positive = carry-over faster; Holm over 6): " +
               "; ".join(f"{r['vs'].split(', ')[1]}: {r['mean_diff'] / 60:+.1f} min (n = {r['n']}, Holm p = {r['p_holm']:.4f})"
                         for r in pre) + ".", ""])
    js["delay_preregistered"] = pre
    js["rmst30"] = rmst(a)
    md.extend(["## C2", ""])
    js["c2"] = c2()
    open(os.path.join(REPO, "reports", "stats", "V2_REANALYSIS.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump(js, open(os.path.join(REPO, "reports", "stats", "v2_reanalysis.json"), "w", encoding="utf-8"), indent=1, default=float)
    print("wrote reports/stats/V2_REANALYSIS.md")


if __name__ == "__main__":
    main()
