#!/usr/bin/env python3
"""v2 statistics, implementing the analysis plan (pre-registered).
Inputs: experiments/v2_registry.csv (test_hfl, test_round runs; metrics.json per run) and
results/v2/containment_test/<dataset>.json. Outputs: reports/stats/V2_STATS.md, v2_stats.json.

python scripts/v2_analyze.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.evaluation.stats import bootstrap_ci, compare_family, holm  # noqa: E402
from src.v2.plan import DATASETS  # noqa: E402

NI_F1, NI_FALSE, NI_TOKEN = 0.02, 0.005, 0.05
PRIMARY_SCEN = ("persistent_0.02", "persistent_0.05", "persistent_0.1", "persistent_0.2", "handover_timed", "on_off_0.2")
md, js = [], {}


def f(v, d=3):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def table(rows, title, d=3, ni=None):
    md.extend([f"**{title}**", "", "| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) |" + (" non-inferior |" if ni else ""),
               "|---|---|---|---|---|---|---|---|" + ("---|" if ni else "")])
    for r in rows:
        line = (f"| {r['vs']} | {r['n']} | {f(r['mean_diff'], d)} [{f(r['diff_ci_lo'], d)}, {f(r['diff_ci_hi'], d)}] | "
                f"{f(r['hl'], d)} | {r['wins']} / {r['losses']} / {r['ties']} | {f(r['rank_biserial'], 2)} | "
                f"{f(r['p'], 4)} | {f(r['p_holm'], 4)} |")
        if ni:
            line += f" {'yes' if r['diff_ci_lo'] > -ni else 'no'} |"
        md.append(line)
    md.append("")


def clean(rows):
    return [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()} for r in rows]


# ------------------------------------------------------------------ C2 (hierarchical FL)
def load_hfl():
    r = pd.read_csv(os.path.join(REPO, "experiments", "v2_registry.csv"))
    r = r[r.group.isin(["test_hfl", "test_round"]) & (r.status == "done")]
    rows = []
    for x in r.itertuples():
        m = json.load(open(os.path.join(REPO, x.run_dir, "metrics.json"), encoding="utf-8"))
        s = m["stats"]
        rows.append({"group": x.group, "dataset": x.dataset, "seed": int(x.seed), "label": x.arm,
                     "setting": int(x.run_id.split("/set")[1].split("/")[0]), "round_s": m["round_s"],
                     "f1": m["macro_f1_last3"], "f1_final": m["final"]["macro_f1"],
                     "interrupted": s["interrupted"] / max(s["selected"], 1),
                     "lost": (s["lost_handover"] + s["lost_outage"]) / max(s["selected"], 1),
                     "energy": m["energy_J"], "uplink": m["uplink_MB"], "curve": m["curve"], "rounds": m["rounds"]})
    return pd.DataFrame(rows)


def to_target(df):
    out = []
    for d in DATASETS:
        g = df[(df.dataset == d)]
        target = 0.9 * g[g.label == "random"].f1_final.median()
        for x in g.itertuples():
            hit = [c for c in x.curve if c["macro_f1"] >= target]
            rr = hit[0]["round"] if hit else x.rounds + 1                     # censored: ranked worst
            up = hit[0].get("uplink_MB", np.nan) if hit else np.inf
            out.append((x.Index, rr, up))
    for i, rr, up in out:
        df.loc[i, "rounds_to_target"] = rr
        df.loc[i, "uplink_to_target"] = up
    return df


def v3(df):
    global md
    md.extend(["## V3: mobility-aware participation (C2)", ""])
    t = df[df.group == "test_hfl"].copy()
    t = to_target(t)
    js["V3_summary"] = {d: {a: {**{k: float(g[k].mean()) for k in ("f1", "interrupted", "lost", "energy", "uplink",
                                                                     "rounds_to_target")},
                                "f1_sd": float(g.f1.std()), "n": int(len(g)),
                                "uplink_to_target": float(g.uplink_to_target.replace(np.inf, np.nan).mean()),
                                "reached": float(np.isfinite(g.uplink_to_target).mean())}
                            for a, g in t[t.dataset == d].groupby("label")} for d in DATASETS}
    js["V3_pooled"] = {a: {"interrupted": float(g.interrupted.mean()), "lost": float(g.lost.mean()),
                           "energy": float(g.energy.mean()), "uplink": float(g.uplink.mean()),
                           "rounds_to_target": float(g.rounds_to_target.mean()),
                           "reached": float(np.isfinite(g.uplink_to_target).mean())}
                       for a, g in t.groupby("label")}
    js["V3_round"] = {f"{a}@{int(rs)}": {"f1": float(g.f1.mean()), "interrupted": float(g.interrupted.mean()), "n": int(len(g))}
                      for (a, rs), g in df[df.seed.isin(range(42, 47))].groupby(["label", "round_s"])}
    md += ["| dataset | arm | macro-F1 (last 3) | interrupted | lost | energy (J) | uplink (MB) | rounds to target |",
           "|---|---|---|---|---|---|---|---|"]
    for d in DATASETS:
        for a, g in t[t.dataset == d].groupby("label"):
            md.append(f"| {d} | {a} | {g.f1.mean():.3f} ± {g.f1.std():.3f} | {g.interrupted.mean():.3f} | {g.lost.mean():.3f} | "
                      f"{g.energy.mean():.0f} | {g.uplink.mean():.0f} | {g.rounds_to_target.mean():.1f} |")
    md.append("")
    prim = compare_family(t, "f1", "residence+fw", ["random"])
    table(prim, "Primary: residence+fw − random, macro-F1 (pooled, n = 30; non-inferiority margin 0.02)", ni=NI_F1)
    fam = compare_family(t, "f1", "residence+fw", ["random", "random+fw", "speed", "speed+fw", "residence"])
    table(fam, "Secondary: residence+fw − other arms, macro-F1 (Holm over 5)", ni=NI_F1)
    inter = compare_family(t, "interrupted", "residence+fw", ["random"], higher_is_better=False)
    table(inter, "Interrupted share of selected updates (positive = residence+fw lower)")
    sel_only = compare_family(t, "interrupted", "residence", ["random"], higher_is_better=False)
    table(sel_only, "Interrupted share, residence (no forwarding) vs random (positive = residence lower)")
    rtt = compare_family(t, "rounds_to_target", "residence+fw", ["random"], higher_is_better=False)
    table(rtt, "Rounds to 0.9 x random's median final macro-F1 (positive = residence+fw faster)", d=1)
    t_up = t[np.isfinite(t.uplink_to_target)]
    utt = compare_family(t, "uplink_to_target", "residence+fw", ["random"], higher_is_better=False)
    table(utt, "Uplink MB to target (positive = residence+fw less; censored runs ranked worst as +inf)", d=1)
    fw = []
    for sel in ("random", "speed", "residence"):
        r_ = compare_family(t, "f1", f"{sel}+fw", [sel])[0]
        fw.append({**r_, "vs": f"{sel}+fw vs {sel}"})
    for r_, pa in zip(fw, holm([r["p"] for r in fw])):
        r_["p_holm"] = float(pa)
    table(fw, "Forwarding on − off within each selection rule, macro-F1 (Holm over 3)")
    energy = compare_family(t, "energy", "residence+fw", ["random"], higher_is_better=False)
    table(energy, "Device training energy (positive = residence+fw less)", d=0)
    per_ds = {}
    for d in DATASETS:
        rows = compare_family(t[t.dataset == d], "f1", "residence+fw", ["random"])
        per_ds[d] = clean(rows)
        table(rows, f"Per dataset {d}: residence+fw − random, macro-F1 (n = 10)", ni=NI_F1)
    rd = df[df.group == "test_round"]
    rows_r = []
    for i, rs in enumerate(sorted(rd.round_s.unique())):
        r_ = compare_family(rd[rd.round_s == rs], "f1", "residence+fw", ["random"])[0]
        ri = compare_family(rd[rd.round_s == rs], "interrupted", "residence+fw", ["random"], higher_is_better=False)[0]
        rows_r += [{**r_, "vs": f"random @ {rs} s (macro-F1)"}, {**ri, "vs": f"random @ {rs} s (interrupted)"}]
    for r_, pa in zip(rows_r, holm([r["p"] for r in rows_r])):
        r_["p_holm"] = float(pa)
    table(rows_r, "Round length: residence+fw − random (n = 15 per length)")
    ni_ok = prim[0]["diff_ci_lo"] > -NI_F1
    better_int = inter[0]["p_holm"] < 0.05 and inter[0]["rank_biserial"] > 0
    better_tt = (rtt[0]["p_holm"] < 0.05 and rtt[0]["rank_biserial"] > 0) or (utt[0]["p_holm"] < 0.05 and utt[0]["rank_biserial"] > 0)
    verdict = ("supported" if ni_ok and better_int and better_tt else "partially supported" if ni_ok and better_int
               else "not supported")
    md.extend([f"**Rule outcome (plan): {verdict}** (non-inferior {ni_ok}; fewer interruptions {better_int}; "
               f"faster or cheaper to target {better_tt}).", ""])
    js["V3"] = {"primary": clean(prim), "family": clean(fam), "interrupted": clean(inter), "interrupted_sel_only": clean(sel_only),
                "rounds_to_target": clean(rtt), "uplink_to_target": clean(utt), "forwarding": clean(fw),
                "energy": clean(energy), "per_dataset": per_ds, "round_length": clean(rows_r), "verdict": verdict}


# ------------------------------------------------------------------ C1 (containment)
def load_containment():
    rows, att_rows = [], []
    for d in DATASETS:
        p = os.path.join(REPO, "results", "v2", "containment_test", f"{d}.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p, encoding="utf-8")):
            a, h = pd.DataFrame(r["att"]), pd.DataFrame(r["hon"])
            rows.append({"dataset": r["dataset"], "seed": r["seed"], "scenario": r["scenario"], "label": r["method"],
                         "contained": a.contained.mean(), "delay": a.delay_s.mean(),
                         "exposure": a.exposure.sum() / max(a.attack_total.sum(), 1), "false": h.contained.mean(),
                         "blocked_benign": h.blocked_benign.sum(), "p_fa": r["p_fa"], "p_det": r["p_det"]})
            for x in r["att"]:
                att_rows.append({"dataset": r["dataset"], "seed": r["seed"], "scenario": r["scenario"], "method": r["method"],
                                 "u": x["u"], "vclass": x["vclass"], "contained": x["contained"],
                                 "ho_per_h": x.get("ho_per_h", np.nan)})
    return pd.DataFrame(rows), pd.DataFrame(att_rows)


def v1(c, a):
    global md
    md.extend(["## V1: carry-over of sequential evidence (C1)", ""])
    summ = c.groupby(["scenario", "label"])[["contained", "delay", "exposure", "false"]].mean().reset_index()
    js["V1_summary"] = summ.to_dict("records")
    md += ["| scenario | method | contained | delay (s) | exposure | false containment |", "|---|---|---|---|---|---|"]
    for x in summ.itertuples():
        md.append(f"| {x.scenario} | {x.label} | {x.contained:.3f} | {f(x.delay, 0)} | {x.exposure:.3f} | {x.false:.4f} |")
    md.append("")
    def family(metric, other, hib=True):
        rows = []
        for s in PRIMARY_SCEN:
            g = c[c.scenario == s]
            r_ = compare_family(g, metric, "cusum_carry", [other], higher_is_better=hib)[0]
            rows.append({**r_, "vs": f"{other}, {s}"})
        for r_, pa in zip(rows, holm([r["p"] for r in rows])):
            r_["p_holm"] = float(pa)
        return rows

    prim = family("contained", "cusum_reset_cal")
    table(prim, "Primary: cusum_carry − cusum_reset_cal (equal false-containment budget), contained share "
                "(pooled n = 30 per scenario; Holm over 6)")
    sec = {"same_threshold": clean(family("contained", "cusum_reset"))}
    table(sec["same_threshold"], "Secondary: cusum_carry − cusum_reset (same k, h), contained share (Holm over 6)")
    for metric in ("delay", "exposure"):
        rows = family(metric, "cusum_reset_cal", hib=False)
        sec[metric] = clean(rows)
        table(rows, f"Secondary: {metric}, cusum_carry vs cusum_reset_cal (positive = carry better)", d=3)
    v1rows = []
    for s in PRIMARY_SCEN:
        g = c[c.scenario == s]
        for o in ("trust_carry", "trust_cold"):
            r_ = compare_family(g, "contained", "cusum_carry", [o])[0]
            v1rows.append({**r_, "vs": f"{o}, {s}"})
    for r_, pa in zip(v1rows, holm([r["p"] for r in v1rows])):
        r_["p_holm"] = float(pa)
    table(v1rows, "Secondary: cusum_carry − v1 smoothed trust, contained share (Holm over 12)")
    fc = c[c.scenario.isin(PRIMARY_SCEN)].groupby(["dataset", "seed", "label"])["false"].mean().reset_index()
    fa = compare_family(fc, "false", "cusum_carry", ["cusum_reset_cal"], higher_is_better=False)
    table(fa, "False containment of honest devices (positive = carry lower; non-inferiority margin 0.005)", d=4, ni=NI_FALSE)
    mob = {}
    md += ["Contained share by velocity class (persistent scenarios):", "",
           "| class | carry | reset (recalibrated) | reset (same threshold) |", "|---|---|---|---|"]
    for vc in ("veh", "ped", "static"):
        g = a[a.scenario.isin(PRIMARY_SCEN[:4]) & (a.vclass == vc)]
        cr = g[g.method == "cusum_carry"].contained.mean()
        rc = g[g.method == "cusum_reset_cal"].contained.mean()
        rs = g[g.method == "cusum_reset"].contained.mean()
        mob[vc] = {"carry": float(cr), "reset_cal": float(rc), "reset": float(rs)}
        md.append(f"| {vc} | {cr:.3f} | {rc:.3f} | {rs:.3f} |")
    md.append("")
    g = a[a.scenario.isin(PRIMARY_SCEN[:4])]
    piv = g.pivot_table(index=["dataset", "seed", "scenario", "u", "vclass", "ho_per_h"], columns="method", values="contained",
                        aggfunc="mean").reset_index()
    if {"cusum_carry", "cusum_reset_cal"} <= set(piv.columns) and piv.ho_per_h.notna().any():
        rho, pval = st.spearmanr(piv.ho_per_h, piv.cusum_carry.astype(float) - piv.cusum_reset_cal.astype(float))
        mob["spearman"] = {"rho": float(rho), "p": float(pval), "n": int(len(piv))}
        md.append(f"Spearman correlation of (carry − recalibrated reset) containment with the attacker's handover rate: "
                  f"rho = {rho:.3f} (p = {pval:.2g}, n = {len(piv)} attacker-runs).")
        md.append("")
    wins = sum(r["p_holm"] < 0.05 and r["rank_biserial"] > 0 for r in prim)
    ni = fa[0]["diff_ci_lo"] > -NI_FALSE
    verdict = ("supported" if wins >= 4 and ni else "partially supported" if wins >= 1 else "not supported")
    md.extend([f"**Rule outcome (plan): {verdict}** ({wins}/6 primary scenarios significant; false containment "
               f"non-inferior {ni}).", ""])
    per_ds = {}
    for d in DATASETS:
        rows = []
        for s in PRIMARY_SCEN:
            g = c[(c.scenario == s) & (c.dataset == d)]
            r_ = compare_family(g, "contained", "cusum_carry", ["cusum_reset_cal"])[0]
            rows.append({**r_, "vs": f"reset_cal, {s}"})
        for r_, pa in zip(rows, holm([r["p"] for r in rows])):
            r_["p_holm"] = float(pa)
        per_ds[d] = clean(rows)
        table(rows, f"Per dataset {d}: cusum_carry − cusum_reset_cal, contained share (n = 10; Holm over 6)")
    js["V1"] = {"primary": clean(prim), "secondary": sec, "vs_v1_trust": clean(v1rows), "false": clean(fa),
                "mobility": mob, "per_dataset": per_ds, "verdict": verdict}


def v2tok(c):
    md.extend(["## V2: token attacks", ""])
    rows = []
    for tok in ("omit", "forge"):
        g = c[c.scenario.isin(["persistent_0.05", f"persistent_0.05_{tok}"]) & (c.label == "cusum_carry")].copy()
        g["label"] = np.where(g.scenario == "persistent_0.05", "authentic", tok)
        r_ = compare_family(g, "contained", tok, ["authentic"])[0]
        rows.append({**r_, "vs": f"{tok} vs authentic (THCP v2)"})
    for r_, pa in zip(rows, holm([r["p"] for r in rows])):
        r_["p_holm"] = float(pa)
    table(rows, "Contained share with a withheld or forged token minus with the authentic token (non-inferiority margin 0.05)",
          ni=NI_TOKEN)
    g1 = c[c.scenario.isin(["persistent_0.05", "persistent_0.05_omit"]) & (c.label == "trust_carry")]
    desc = g1.groupby("scenario").contained.mean().to_dict()
    md.extend([f"v1 smoothed trust with THCP v1, contained share: authentic {desc.get('persistent_0.05', np.nan):.3f}, "
               f"withheld token {desc.get('persistent_0.05_omit', np.nan):.3f}.", ""])
    ni = all(r["diff_ci_lo"] > -NI_TOKEN for r in rows)
    js["V2"] = {"rows": clean(rows), "v1_trust": {k: float(v) for k, v in desc.items()},
                "verdict": "supported" if ni else "not supported"}
    md.extend([f"**Rule outcome (plan): {js['V2']['verdict']}**.", ""])


def v4(c):
    global md
    md.extend(["## V4: boundaries", ""])
    s = c[c.scenario == "stealthy_0.005"].groupby("label").contained.mean()
    marg = c.groupby("dataset")[["p_fa", "p_det"]].mean()
    md += ["Stealthy attackers (rate 0.005), contained share: " + ", ".join(f"{k} {v:.3f}" for k, v in s.items()), "",
           "| dataset | detector p_fa | p_det | margin | carry contained (persistent 0.05) |", "|---|---|---|---|---|"]
    for d in marg.index:
        cc = c[(c.dataset == d) & (c.scenario == "persistent_0.05") & (c.label == "cusum_carry")].contained.mean()
        md.append(f"| {d} | {marg.loc[d, 'p_fa']:.3f} | {marg.loc[d, 'p_det']:.3f} | {marg.loc[d, 'p_det'] - marg.loc[d, 'p_fa']:.3f} | {cc:.3f} |")
    md.append("")
    js["V4"] = {"stealthy": {k: float(v) for k, v in s.items()},
                "margin": {d: {"p_fa": float(marg.loc[d, "p_fa"]), "p_det": float(marg.loc[d, "p_det"])} for d in marg.index}}


def main():
    md.extend(["# v2 statistics (scripts/v2_analyze.py; plan: the analysis plan", ""])
    df = load_hfl()
    if len(df):
        v3(df)
    c, a = load_containment()
    if len(c):
        v1(c, a)
        v2tok(c)
        v4(c)
    open(os.path.join(REPO, "reports", "stats", "V2_STATS.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump(js, open(os.path.join(REPO, "reports", "stats", "v2_stats.json"), "w", encoding="utf-8"), indent=1, default=float)
    print("wrote reports/stats/V2_STATS.md")


if __name__ == "__main__":
    main()
