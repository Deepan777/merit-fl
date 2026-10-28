#!/usr/bin/env python3
"""Analysis of R8 the analysis plan. Output: reports/stats/V2_R8.md, v2_r8_stats.json.

A/B   H8.1-H8.3: theory threshold within budget (llrh, carry) and carry vs memory/reset (seed level, Holm within world
      over 6), on fresh traces, chained and unchained; the same for llrd (H8.4 in W1 and W4, unchained)
      contained share by scenario; population mix; unmonitored shares
C     ablation (reference x threshold)
D/E   within-type spread sweep (sigma*), overdispersion and bursts
F     matched-rate comparators
      edge-change rates
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
WORLDS = ("W0", "W1", "W2", "W3", "W4")
BUDGETS = {"b01": 0.001, "b1": 0.01}
PROTOS = ("chained", "unchained")
MIX = {"veh": 0.15, "ped": 0.15, "static": 0.70}
md, js = ["# R8: confirmation on fresh traces and limits of the self-calibrated statistic (scripts/v2_r8_analyze.py)", "",
          "Plan: `ANALYSIS_PLAN_V2_R8.md` (amendment A1 before any test run). Fresh traces 52-61; detectors of test seeds 42-51.", ""], {}


def gz(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def poisson_ci(k, t):
    lo = st.chi2.ppf(0.025, 2 * k) / 2 / t if k > 0 else 0.0
    return lo, st.chi2.ppf(0.975, 2 * k + 2) / 2 / t


def load_main():
    att, hon = [], []
    for p in PROTOS:
        for w in WORLDS:
            for d in DATASETS:
                f = os.path.join(V2, "r8_main", f"{p}_{w}_{d}.json.gz")
                if not os.path.exists(f):
                    continue
                for r in gz(f):
                    key = {k: r[k] for k in ("proto", "world", "dataset", "seed", "rep", "scenario", "budget", "stat", "placement")}
                    hon.append({**key, "events": r["honest"]["events"], "hours": r["honest"]["hours"], "unmon": r["unmonitored"],
                                **{f"ev_{c}": v[0] for c, v in r["honest"]["by_class"].items()},
                                **{f"hr_{c}": v[1] for c, v in r["honest"]["by_class"].items()}})
                    for a in r["att"]:
                        att.append({**key, **a})
    return pd.DataFrame(att), pd.DataFrame(hon)


def budget_rows(h, proto, stat, worlds):
    out = {}
    for w in worlds:
        for b, B in BUDGETS.items():
            for d in DATASETS:
                for pl in ("carry", "memory", "reset", "carry_loss"):
                    g = h[(h.proto == proto) & (h.stat == stat) & (h.world == w) & (h.budget == b) & (h.dataset == d) & (h.placement == pl)]
                    if g.empty:
                        continue
                    ev, hr = int(g.events.sum()), float(g.hours.sum())
                    lo, hi = poisson_ci(ev, hr)
                    out[f"{w}/{b}/{d}/{pl}"] = {"events": ev, "hours": hr, "rate": ev / hr, "lo": lo, "hi": hi, "within": ev / hr <= B,
                                                "unmon": float(g.unmon.mean())}
    return out


def placement_tests(a, proto, stat, b="b01", scen="persistent_0.05"):
    res = {}
    for w in WORLDS:
        rows = []
        for d in DATASETS:
            g = a[(a.proto == proto) & (a.stat == stat) & (a.world == w) & (a.budget == b) & (a.dataset == d) & (a.scenario == scen)]
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
        if not rows:
            continue
        for r_, pa in zip(rows, holm([r["p"] for r in rows])):
            r_["p_holm"] = float(pa)
        per = [{"dataset": d, "beats_both": all(any(r["dataset"] == d and r["vs"] == o and r["p_holm"] < 0.05 and r["mean_diff"] > 0
                                                    for r in rows) for o in ("memory", "reset"))} for d in DATASETS]
        res[w] = {"rows": rows, "per_dataset": per, "n_beats_both": sum(x["beats_both"] for x in per),
                  "verdict": "robust" if sum(x["beats_both"] for x in per) >= 2 else "not robust",
                  "wins": sum(r["wins"] for r in rows), "losses": sum(r["losses"] for r in rows)}
    return res


def section_main(a, h):
    global md
    js["main"] = {}
    for proto in PROTOS:
        if h.empty or (h.proto == proto).sum() == 0:
            continue
        J = js["main"][proto] = {}
        for stat in ("llrh", "llrd"):
            br = budget_rows(h, proto, stat, WORLDS)
            pt = placement_tests(a, proto, stat)
            pt1 = placement_tests(a, proto, stat, b="b1")
            within_w03 = all(br[f"{w}/b01/{d}/carry"]["within"] for w in ("W0", "W1", "W2", "W3") for d in DATASETS
                             if f"{w}/b01/{d}/carry" in br)
            J[stat] = {"budget": br, "placement": pt, "placement_b1": pt1, "within_W0_W3": bool(within_w03),
                       "within_by_world": {w: all(br[f"{w}/b01/{d}/carry"]["within"] for d in DATASETS if f"{w}/b01/{d}/carry" in br)
                                           for w in WORLDS if any(f"{w}/b01/{d}/carry" in br for d in DATASETS)},
                       "max_rate_W0_W3": max([br[k]["rate"] for k in br if k.split("/")[0] in ("W0", "W1", "W2", "W3") and
                                              k.endswith("/carry") and "/b01/" in k] or [np.nan]),
                       "rate_W4": [br[f"W4/b01/{d}/carry"]["rate"] for d in DATASETS if f"W4/b01/{d}/carry" in br]}
            md.extend([f"## {proto}, {stat}: steady-state honest false-alarm rate (carry, b01) and placement tests", "",
                       "| world | dataset | events | device-hours | %/h [95% CI] | within | unmonitored |", "|---|---|---|---|---|---|---|"])
            for k, v in br.items():
                w, b, d, pl = k.split("/")
                if b == "b01" and pl == "carry":
                    um = f"{v['unmon']:.3f}" if stat == "llrd" else "—"
                    md.append(f"| {w} | {d} | {v['events']} | {v['hours']:.0f} | {100 * v['rate']:.4f} [{100 * v['lo']:.4f}, {100 * v['hi']:.4f}] | "
                              f"{'yes' if v['within'] else 'no'} | {um} |")
            md.append("")
            for w, r in pt.items():
                md += [f"**{w}: {r['verdict']}** ({r['n_beats_both']} of 3 datasets; wins {r['wins']}, losses {r['losses']})", "",
                       "| dataset | vs | carry | other | Δ [95% CI] | W / L | p (Holm) |", "|---|---|---|---|---|---|---|"]
                md += [f"| {x['dataset']} | {x['vs']} | {x['carry']:.3f} | {x['other']:.3f} | {x['mean_diff']:.3f} [{x['ci_lo']:.3f}, "
                       f"{x['ci_hi']:.3f}] | {x['wins']} / {x['losses']} | {x['p_holm']:.4f} |" for x in r["rows"]]
                md.append("")
            sc = a[(a.proto == proto) & (a.stat == stat) & (a.budget == "b01")].groupby(["world", "placement", "scenario"]).contained.mean().unstack()
            J[stat]["by_scenario"] = {f"{w}/{pl}": row.dropna().to_dict() for (w, pl), row in sc.iterrows()}
            pers = a[(a.proto == proto) & (a.stat == stat) & (a.budget == "b01") &
                     a.scenario.isin(["persistent_0.02", "persistent_0.05", "persistent_0.1", "persistent_0.2"])]
            byc = {}
            for w in WORLDS:
                for pl in ("carry", "memory", "reset"):
                    g = pers[(pers.world == w) & (pers.placement == pl)]
                    if g.empty:
                        continue
                    c = {k: float(g[g.vclass == k].contained.mean()) for k in MIX}
                    byc[f"{w}/{pl}"] = {**c, "iot_mix": float(sum(MIX[k] * c[k] for k in MIX))}
            J[stat]["by_class"] = byc
            if stat == "llrd":
                u = a[(a.proto == proto) & (a.stat == "llrd") & (a.budget == "b01") & (a.placement == "carry")]
                J[stat]["att_unmonitored"] = {w: float(g.unmon.mean()) for w, g in u.groupby("world")}
            fl = a[(a.proto == proto) & (a.stat == stat) & (a.budget == "b01") & (a.placement == "carry") &
                   (a.scenario == "adaptive_1_2") & (a.world == "W1")]
            if not fl.empty and "floor" in fl:
                J[stat]["floor_pct_by_dataset"] = {d: float(100 * g.floor.median()) for d, g in fl.groupby("dataset")}
    # verdicts
    v = {}
    for proto in js["main"]:
        J = js["main"][proto]
        v[proto] = {"H_budget_llrh": J["llrh"]["within_W0_W3"],
                    "H_place_llrh": {w: J["llrh"]["placement"][w]["verdict"] for w in J["llrh"]["placement"]},
                    "replicated": bool(J["llrh"]["within_W0_W3"] and all(J["llrh"]["placement"].get(w, {}).get("verdict") == "robust"
                                                                         for w in ("W0", "W1", "W3"))),
                    "H84_llrd_within_W1_W4": {w: J["llrd"]["within_by_world"].get(w) for w in ("W1", "W4")},
                    "llrd_place": {w: J["llrd"]["placement"][w]["verdict"] for w in J["llrd"]["placement"]}}
    js["verdicts"] = v
    md.extend(["## Verdicts", "", "```", json.dumps(v, indent=1), "```", ""])


def section_ablation():
    global md
    rows = []
    for d in DATASETS:
        f = os.path.join(V2, "r8_ablation", f"W1_{d}.json.gz")
        if os.path.exists(f):
            rows += gz(f)
    if not rows:
        return
    df = pd.DataFrame(rows)
    out = {}
    md.extend(["## C. Ablation (chained, W1, b01, carry, persistent 5%)", "",
               "| reference | threshold | dataset | h | %/h | within | contained |", "|---|---|---|---|---|---|---|"])
    for (ref, th, d), g in df.groupby(["reference", "threshold", "dataset"]):
        rate, cont = g.events.sum() / g.hours.sum(), g.contained.sum() / g.n_att.sum()
        out[f"{ref}/{th}/{d}"] = {"h": float(g.h.iloc[0]), "rate": float(rate), "within": bool(rate <= 0.001), "contained": float(cont)}
        md.append(f"| {ref} | {th} | {d} | {g.h.iloc[0]:.2f} | {100 * rate:.4f} | {'yes' if rate <= 0.001 else 'no'} | {cont:.3f} |")
    js["ablation"] = out
    md.append("")


def section_sweep():
    global md
    r7 = yaml.safe_load(open(os.path.join(REPO, "configs", "v2", "r7.yaml"), encoding="utf-8"))["chosen"]
    r8 = yaml.safe_load(open(os.path.join(REPO, "configs", "v2", "r8.yaml"), encoding="utf-8"))["llrd"]["chosen"]
    rows = []
    for f in sorted(os.listdir(os.path.join(V2, "r8_sweep"))) if os.path.isdir(os.path.join(V2, "r8_sweep")) else []:
        if f.endswith(".json.gz"):
            rows += gz(os.path.join(V2, "r8_sweep", f))
    if not rows:
        return
    df = pd.DataFrame(rows)
    df[["ev_b01", "ev_b1"]] = pd.DataFrame(df.events.tolist(), index=df.index)
    df[["c_b01", "c_b1"]] = pd.DataFrame(df.contained.tolist(), index=df.index)
    agg = {}
    for (v, stat, dl, pl, d), g in df.groupby(["variant", "stat", "delta", "placement", "dataset"]):
        agg[f"{v}/{stat}/{dl:g}/{pl}/{d}"] = {"rate_b01": float(g.ev_b01.sum() / g.hours.sum()), "rate_b1": float(g.ev_b1.sum() / g.hours.sum()),
                                              "cont_b01": float(g.c_b01.sum() / g.n_att.sum()), "cont_b1": float(g.c_b1.sum() / g.n_att.sum()),
                                              "unmon": float(g.unmonitored.mean())}
    js["sweep"] = agg
    chosen = {"llrh": {d: r7["W1"][d]["b01"]["delta"] for d in DATASETS}, "llrd": {d: r8["W1"][d]["b01"]["delta"] for d in DATASETS}}

    def at_val(v, stat, key):
        return [agg.get(f"{v}/{stat}/{chosen[stat][d]:g}/carry/{d}", {}).get(key, np.nan) for d in DATASETS]

    summ = {}
    for stat in ("llrh", "llrd"):
        for fam, vs in (("sigma", [f"sigma_{s:g}" for s in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)]),
                        ("betabin", ["sigma_0", "betabin_0.01", "betabin_0.03", "betabin_0.1"]),
                        ("burst", ["sigma_0", "burst_3", "burst_10"])):
            curve = []
            for v in vs:
                r = at_val(v, stat, "rate_b01")
                c = at_val(v, stat, "cont_b01")
                curve.append({"variant": v, "rates": r, "max_rate": float(np.nanmax(r)) if not np.all(np.isnan(r)) else np.nan,
                              "within": bool(np.all(np.array(r) <= 0.001)), "contained": float(np.nanmean(c)),
                              "unmon": float(np.nanmean(at_val(v, stat, "unmon")))})
            star = None
            for x in curve:
                if x["within"]:
                    star = x["variant"]
                else:
                    break
            summ[f"{stat}/{fam}"] = {"curve": curve, "largest_within": star}
    js["sweep_summary"] = summ
    md.extend(["## D/E. Limits (unchained, W1 links; validation delta; carry; b01)", ""])
    for k, s in summ.items():
        md.append(f"**{k}: largest variant within budget on all datasets: {s['largest_within']}**")
        md.append("")
        md.append("| variant | %/h per dataset | max %/h | within | contained | unmonitored |")
        md.append("|---|---|---|---|---|---|")
        md += [f"| {x['variant']} | {', '.join(f'{100 * r:.3f}' for r in x['rates'])} | {100 * x['max_rate']:.3f} | "
               f"{'yes' if x['within'] else 'no'} | {x['contained']:.3f} | {x['unmon']:.3f} |" for x in s["curve"]]
        md.append("")


def section_curves():
    global md
    out, far = {}, 0.001
    for w in ("W0", "W1"):
        for d in DATASETS:
            f = os.path.join(V2, "r8_curves", f"{w}_{d}.json.gz")
            if not os.path.exists(f):
                continue
            J = gz(f)
            for name, A in J["rules"].items():
                ev, hrs = np.array(A["ev"], float).sum(0), float(np.sum(A["hours"]))
                cont, n = np.array(A["cont"], float).sum(0), float(np.sum(A["n_att"]))
                rate = ev / hrs
                ok = np.flatnonzero(rate <= far)
                out[f"{w}/{d}/{name}"] = float(cont[int(ok.min())] / n) if len(ok) else None
    if not out:
        return
    fams = {"MERIT-FL llrh carry": ["llrh/carry"], "MERIT-FL llrd carry": ["llrd/carry"], "llrh reset": ["llrh/reset"],
            "TRW-SPRT per node (reset)": ["sprt/reset"], "TRW-SPRT carried": ["sprt/carry"],
            "Beta reputation cold": ["beta0.9/reset", "beta0.98/reset"], "Beta reputation carried": ["beta0.9/carry", "beta0.98/carry"],
            "EWMA reset": ["ewma0.02/reset", "ewma0.05/reset", "ewma0.1/reset"], "EWMA carried": ["ewma0.02/carry", "ewma0.05/carry", "ewma0.1/carry"]}
    best = {}
    for w in ("W0", "W1"):
        for fam, names in fams.items():
            per_d = []
            for d in DATASETS:
                vals = [out.get(f"{w}/{d}/{n}") for n in names]
                vals = [x for x in vals if x is not None]
                per_d.append(max(vals) if vals else None)
            # best parameter chosen on the mean over datasets (favours the comparator)
            cand = []
            for n in names:
                v = [out.get(f"{w}/{d}/{n}") for d in DATASETS]
                if all(x is not None for x in v):
                    cand.append((float(np.mean(v)), n, v))
            best[f"{w}/{fam}"] = {"param": max(cand)[1] if cand else None, "per_dataset": max(cand)[2] if cand else per_d,
                                  "mean": max(cand)[0] if cand else None}
    js["curves"], js["curves_best"] = out, best
    md.extend(["## F. Comparators at a matched rate of 0.1% per device-hour (unchained, persistent 5%; best parameter on test)", "",
               "| world | rule | parameter | 5G-NIDD | CICIoT2023 | CICIDS2017 | mean |", "|---|---|---|---|---|---|---|"])
    for k, v in best.items():
        w, fam = k.split("/", 1)
        cells = " | ".join("—" if x is None else f"{x:.3f}" for x in v["per_dataset"])
        mean = "—" if v["mean"] is None else f"{v['mean']:.3f}"
        md.append(f"| {w} | {fam} | {v['param']} | {cells} | {mean} |")
    md.append("")


def section_rates():
    global md
    f = os.path.join(V2, "r8_rates.json")
    if not os.path.exists(f):
        return
    R = json.load(open(f, encoding="utf-8"))
    js["rates"] = R
    md.extend(["## Edge changes per device-hour and returns within 30 s", "", "| protocol/world | class | per hour | return within 30 s |", "|---|---|---|---|"])
    for k, v in R.items():
        for c, x in v.items():
            md.append(f"| {k} | {c} | {x['per_h']:.1f} | {x['return_30s']:.3f} |")
    md.append("")


def main():
    a, h = load_main()
    if not h.empty:
        section_main(a, h)
    section_ablation()
    section_sweep()
    section_curves()
    section_rates()
    open(os.path.join(REPO, "reports", "stats", "V2_R8.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump(js, open(os.path.join(REPO, "reports", "stats", "v2_r8_stats.json"), "w", encoding="utf-8"), indent=1, default=float)
    print("wrote V2_R8.md;", json.dumps(js.get("verdicts", {})))


if __name__ == "__main__":
    main()
