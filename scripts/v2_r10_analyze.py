#!/usr/bin/env python3
"""Analysis of R10 (reports/stats/ANALYSIS_PLAN_V2_R10.md) -> reports/stats/V2_R10.md, v2_r10_stats.json.

A  real-map traces: false containment, contained shares, placement verdicts (R8's criteria), edge-change rates
B  5G-NIDD: threshold calibrated on one half of the capture sessions, applied to the other half
C  CIC-IDS2017 hosts: description (C1), derived threshold (C2), day-to-day calibration (C3)

python scripts/v2_r10_analyze.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import v2_r8_analyze as A8  # noqa: E402

V2 = os.path.join(REPO, "results", "v2")
OUT = os.path.join(REPO, "reports", "stats")
WORLDS = ("W0", "W1", "W2", "W3")
WNAME = {"W0": "homogeneous", "W1": "heterogeneous types", "W2": "edge node per four cells", "W3": "stricter handover"}
DATASETS = A8.DATASETS if hasattr(A8, "DATASETS") else ("5g_nidd", "ciciot2023", "cicids2017")
BUDGETS = {"b01": 0.001, "b1": 0.01}
STATS3, PLACE = ("llrh", "llrd", "llrr"), ("carry", "memory", "reset")
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
js, md = {}, ["# R10: real-map mobility, calibration validity and a second real-traffic source", "",
              "Plan: `ANALYSIS_PLAN_V2_R10.md` (committed before any R10 code, data or run). All of R10 is post hoc with "
              "respect to the pre-specified evaluation; nothing was tuned.", ""]


# ------------------------------------------------------------------ A
def section_map():
    d_ = os.path.join(V2, "r10_map")
    if not os.path.isdir(d_):
        return
    att, hon = [], []
    for w in WORLDS:
        for d in DATASETS:
            f = os.path.join(d_, f"unchained_{w}_{d}.json.gz")
            if not os.path.exists(f):
                continue
            for r in A8.gz(f):
                key = {k: r[k] for k in ("proto", "world", "dataset", "seed", "rep", "scenario", "budget", "stat", "placement")}
                hon.append({**key, "events": r["honest"]["events"], "hours": r["honest"]["hours"], "unmon": r["unmonitored"]})
                att += [{**key, **a} for a in r["att"]]
    if not att:
        return
    a, h = pd.DataFrame(att), pd.DataFrame(hon)
    out = {}
    for stat in ("llrh", "llrd"):
        bud = A8.budget_rows(h, "unchained", stat, WORLDS)
        plc = A8.placement_tests(a, "unchained", stat)
        rows = {}
        for w in WORLDS:
            fr = [bud[f"{w}/b01/{d}/carry"]["rate"] for d in DATASETS if f"{w}/b01/{d}/carry" in bud]
            if not fr:
                continue
            g = a[(a.stat == stat) & (a.world == w)]
            cont = {pl: float(g[g.placement == pl].groupby("dataset").contained.mean().mean()) for pl in PLACE}
            veh = {pl: float(g[(g.placement == pl) & (g.vclass == "veh")].groupby("dataset").contained.mean().mean()) for pl in PLACE}
            rows[w] = {"false_max": float(max(fr)), "false_by_dataset": {d: bud[f"{w}/b01/{d}/carry"]["rate"] for d in DATASETS
                                                                     if f"{w}/b01/{d}/carry" in bud},
                       "within": bool(max(fr) <= BUDGETS["b01"]), "contained": cont, "contained_vehicles": veh,
                       "verdict": plc.get(w, {}).get("verdict"), "n_beats_both": plc.get(w, {}).get("n_beats_both"),
                       "wins": plc.get(w, {}).get("wins"), "losses": plc.get(w, {}).get("losses"),
                       "unmonitored": float(h[(h.stat == stat) & (h.world == w)].unmon.mean()),
                       "tests": plc.get(w, {}).get("rows")}
        out[stat] = rows
    rates = json.load(open(os.path.join(V2, "r10_rates.json"))) if os.path.exists(os.path.join(V2, "r10_rates.json")) else {}
    grid = json.load(open(os.path.join(V2, "r8_rates.json"))) if os.path.exists(os.path.join(V2, "r8_rates.json")) else {}
    js["map"] = {"stats": out, "rates": rates, "rates_grid": {k.split("/")[1]: v for k, v in grid.items() if k.startswith("unchained/")},
                 "net": json.load(open(os.path.join(REPO, "data", "osm", "net_info.json")))}
    md.extend(["## A. Real street network (Luxembourg City centre, traces 62-71, unchained protocol)", "",
               f"Network: {js['map']['net']['road_km']} km of roads, {js['map']['net']['edges_kept']} edges. Edge changes per "
               "device-hour (real map / synthetic grid): " +
               "; ".join(f"{w} vehicles {rates[w]['veh']['per_h']:.0f} / {js['map']['rates_grid'].get(w, {}).get('veh', {}).get('per_h', float('nan')):.0f}, "
                         f"pedestrians {rates[w]['ped']['per_h']:.0f} / {js['map']['rates_grid'].get(w, {}).get('ped', {}).get('per_h', float('nan')):.0f}"
                         for w in rates) + ".", ""])
    for stat, rows in out.items():
        md.extend([f"**{stat}** (h = ln ARL0, budget 0.1%/h)", "",
                   "| world | false %/h (max) | within | carry | memory | reset | verdict | datasets | W / L |", "|---|---|---|---|---|---|---|---|---|"])
        for w, r in rows.items():
            md.append(f"| {WNAME[w]} | {100 * r['false_max']:.4f} | {'yes' if r['within'] else 'no'} | {r['contained']['carry']:.3f} | "
                      f"{r['contained']['memory']:.3f} | {r['contained']['reset']:.3f} | {r['verdict']} | {r['n_beats_both']} of 3 | "
                      f"{r['wins']} / {r['losses']} |")
        md.append("")


# ------------------------------------------------------------------ matched-rate transfer (B, C3)
def load_curves(folder, name):
    acc, grid, meta = {}, None, {"detectors": 0, "observed_hours": None, "devices": None}
    p = os.path.join(V2, folder)
    for f in sorted(os.listdir(p)) if os.path.isdir(p) else []:
        if f.startswith(name + "_det") and f.endswith(".json.gz"):
            z = A8.gz(os.path.join(p, f))
            grid = np.array(z["grid"])
            meta["detectors"] += 1
            meta["observed_hours"], meta["devices"] = z["observed_hours"], z["devices"]
            for k, a in z["rules"].items():
                t = acc.setdefault(k, {"ev": 0, "cont": 0, "hours": 0.0, "n_att": 0})
                t["ev"] = t["ev"] + np.array(a["ev"])
                t["cont"] = t["cont"] + np.array(a["cont"])
                t["hours"] += a["hours"]
                t["n_att"] += a["n_att"]
    return acc, grid, meta


def transfer(folder, cal, test):
    """Threshold: the lowest grid value at which carry-over's rate on the calibration replay is at most the budget;
    applied to all placements on the test replay."""
    ca, grid, cm = load_curves(folder, cal)
    te, _, tm = load_curves(folder, test)
    if not ca or not te:
        return None
    out = {"cal": cal, "test": test, "cal_observed_hours": cm["observed_hours"], "test_observed_hours": tm["observed_hours"],
           "cal_devices": len(cm["devices"]), "test_devices": len(tm["devices"]), "detectors": cm["detectors"], "rules": {}}
    for stat in STATS3:
        for b, B in BUDGETS.items():
            rate = ca[f"{stat}/carry"]["ev"] / ca[f"{stat}/carry"]["hours"]
            ok = np.flatnonzero(rate <= B)
            if not len(ok):
                out["rules"][f"{stat}/{b}"] = None
                continue
            i = int(ok.min())
            t = te[f"{stat}/carry"]
            ev, hr = float(t["ev"][i]), float(t["hours"])
            lo, hi = A8.poisson_ci(int(round(ev)), hr)
            own = np.flatnonzero(t["ev"] / hr <= B)                       # the test half's own calibrated threshold
            out["rules"][f"{stat}/{b}"] = {
                "h": float(grid[i]), "cal_rate": float(rate[i]), "test_rate": ev / hr, "test_lo": float(lo), "test_hi": float(hi),
                "ratio": ev / hr / B, "events": ev, "hours": hr, "h_test_own": float(grid[int(own.min())]) if len(own) else None,
                "contained": {pl: float(te[f"{stat}/{pl}"]["cont"][i] / te[f"{stat}/{pl}"]["n_att"]) for pl in PLACE}}
    return out


def transfer_table(rows, title):
    md.extend([title, "", "| calibrated on | applied to | statistic | budget | h | test %/h [Poisson 95%] | ratio to budget | "
               "carry | memory | reset |", "|---|---|---|---|---|---|---|---|---|---|"])
    for r in rows:
        for k, v in r["rules"].items():
            stat, b = k.split("/")
            if v is None:
                md.append(f"| {r['cal']} | {r['test']} | {stat} | {b} | -- | no threshold on the grid meets the budget | | | | |")
                continue
            md.append(f"| {r['cal']} | {r['test']} | {stat} | {100 * BUDGETS[b]:g}%/h | {v['h']:.1f} | {100 * v['test_rate']:.4f} "
                      f"[{100 * v['test_lo']:.4f}, {100 * v['test_hi']:.4f}] | {v['ratio']:.2f} | {v['contained']['carry']:.3f} | "
                      f"{v['contained']['memory']:.3f} | {v['contained']['reset']:.3f} |")
    md.append("")


def classify(rows):
    """Counts of calibration-test pairs within the budget, within twice the budget, and above, per statistic and budget."""
    out = {}
    for stat in STATS3:
        for b in BUDGETS:
            v = [r["rules"].get(f"{stat}/{b}") for r in rows]
            v = [x for x in v if x]
            out[f"{stat}/{b}"] = {"pairs": len(v), "within": sum(x["ratio"] <= 1 for x in v),
                                  "within_twice": sum(x["ratio"] <= 2 for x in v),
                                  "worst_ratio": max((x["ratio"] for x in v), default=None),
                                  "median_ratio": float(np.median([x["ratio"] for x in v])) if v else None}
    return out


def section_split():
    rows = [r for r in (transfer("r10_split", "H1", "H2"), transfer("r10_split", "H2", "H1")) if r]
    if not rows:
        return
    js["split"] = {"pairs": rows, "summary": classify(rows)}
    transfer_table(rows, "## B. 5G-NIDD: threshold calibrated on one half of the capture sessions, applied to the other half")
    md.extend(["Observed device-hours: " + ", ".join(f"{r['cal']} {r['cal_observed_hours']:.1f} h ({r['cal_devices']} devices)" for r in rows)
               + ". The halves are small; the replay reuses them, so the Poisson intervals understate the uncertainty.", ""])


# ------------------------------------------------------------------ C
def section_hosts():
    global md
    hp = os.path.join(V2, "r10_hosts")
    info_p = os.path.join(hp, "data", "sessions.json")
    if not os.path.exists(info_p):
        return
    info = json.load(open(info_p, encoding="utf-8"))
    ds = info["device_sessions"]
    H = {"device_days": len(ds), "hosts": len(set(d["src"] for d in ds)), "hours": float(sum(d["hours"] for d in ds)),
         "per_day": {day: sum(1 for d in ds if d["session"] == day) for day in DAYS},
         "excluded": len(info["excluded"]), "excluded_reasons": pd.Series([e["reason"] for e in info["excluded"]]).value_counts().to_dict(),
         "rows_match_official": info["check_against_official"]["all_row_counts_match"],
         "flows": int(sum(d["flows"] for d in ds)),
         "minute_resolution_files": sum(1 for s in info["sessions"] if not s["seconds_in_timestamp"]), "files": len(info["sessions"])}
    md.extend(["## C. CIC-IDS2017 hosts (five capture days)", "",
               f"Row counts match the official files: **{H['rows_match_official']}**. {H['device_days']} device-days of {H['hosts']} hosts, "
               f"{H['hours']:.0f} device-hours, {H['flows']:,} benign flows; device-days per day {H['per_day']}; excluded {H['excluded']} "
               f"({H['excluded_reasons']}). {H['minute_resolution_files']} of {H['files']} files have minute-resolution timestamps.", ""])
    # detectors
    dets = {}
    for f in sorted(os.listdir(os.path.join(hp, "flags"))) if os.path.isdir(os.path.join(hp, "flags")) else []:
        if f.endswith(".env.json"):
            e = json.load(open(os.path.join(hp, "flags", f), encoding="utf-8"))
            dets[f"seed{e['seed']}"] = {day: {"holdout_macro_f1": v["holdout_macro_f1"], "benign_flag_share": v["benign_flag_share"]}
                                        for day, v in e["folds"].items()}
    H["detectors"] = dets
    if dets:
        f1 = [v["holdout_macro_f1"] for s in dets.values() for v in s.values()]
        fs = [v["benign_flag_share"] for s in dets.values() for v in s.values()]
        H["holdout_f1_min"], H["holdout_f1_max"] = float(min(f1)), float(max(f1))
        H["benign_flag_min"], H["benign_flag_max"] = float(min(fs)), float(max(fs))
        md.extend([f"Cross-fitted detectors ({len(f1)}): held-out macro-F1 {min(f1):.3f}-{max(f1):.3f}; share of host benign flows "
                   f"flagged per scored day {100 * min(fs):.2f}%-{100 * max(fs):.2f}%.", ""])
    # C1
    dp = os.path.join(hp, "r9_describe.json")
    if os.path.exists(dp):
        D = json.load(open(dp, encoding="utf-8"))
        per = {}
        for s, v in D.items():
            for host, x in v["per_ue"].items():
                per.setdefault(host, []).append(x)
        hosts = {h: {"share": float(np.mean([x["share"] for x in v])), "rho": float(np.nanmean([x["rho"] for x in v])),
                     "acf1": float(np.nanmean([x["acf1"] for x in v])), "excess_999": float(np.mean([x["excess_999"] for x in v])),
                     "hours": v[0]["hours"], "flows_per_window": v[0]["flows_per_window"], "device_days": v[0]["device_sessions"]}
                 for h, v in per.items()}
        rho = np.array([x["rho"] for x in hosts.values()])
        fam = {s: dict(zip(v["fam"], v["fam_rate"])) for s, v in D.items()}
        H["describe"] = {"hosts": hosts, "rho_median": float(np.nanmedian(rho)), "rho_min": float(np.nanmin(rho)),
                         "rho_max": float(np.nanmax(rho)), "share_rho_ge_001": float(np.mean(rho >= 0.01)),
                         "share_all": float(np.mean([v["share_all"] for v in D.values()])),
                         "flows_per_window_median": float(np.median([x["flows_per_window"] for x in hosts.values()])),
                         "fam_rate": {k: float(np.mean([fam[s][k] for s in fam])) for k in next(iter(fam.values()))}}
        md.extend(["### C1. Benign flags per host (mean over detector seeds)", "",
                   f"Median intra-window correlation {H['describe']['rho_median']:.3f} (range {H['describe']['rho_min']:.3f}-"
                   f"{H['describe']['rho_max']:.3f}); {100 * H['describe']['share_rho_ge_001']:.0f}% of hosts at or above 0.01. "
                   f"Cross-fitted detection rates of the attack families (each family occurs on one day only, so the scoring "
                   f"detector never saw it): {json.dumps({k: round(v, 3) for k, v in H['describe']['fam_rate'].items()})}.", "",
                   "| host | device-days | hours | flows/window | flagged share | rho | acf(1) | windows above binomial 99.9% |",
                   "|---|---|---|---|---|---|---|---|"])
        md += [f"| {h} | {x['device_days']} | {x['hours']:.1f} | {x['flows_per_window']:.1f} | {x['share']:.4f} | {x['rho']:.3f} | "
               f"{x['acf1']:.3f} | {x['excess_999']:.4f} |" for h, x in sorted(hosts.items())]
        md.append("")
    # C2
    rp = os.path.join(hp, "r9_replay")
    rows = []
    for f in sorted(os.listdir(rp)) if os.path.isdir(rp) else []:
        if f.endswith(".json.gz"):
            rows += A8.gz(os.path.join(rp, f))
    if rows:
        df = pd.DataFrame(rows)
        out = {}
        for (b, stat, pl), g in df.groupby(["budget", "stat", "placement"]):
            ev, hr = int(g.events.sum()), float(g.hours.sum())
            lo, hi = A8.poisson_ci(ev, hr)
            out[f"{b}/{stat}/{pl}"] = {"events": ev, "hours": hr, "rate": ev / hr, "lo": float(lo), "hi": float(hi),
                                       "within": bool(ev / hr <= BUDGETS[b]), "contained": float(g.contained.sum() / g.n_att.sum()),
                                       "unmon": float(g.unmonitored.mean())}
        H["replay"] = out
        md.extend(["### C2. Derived threshold h = ln ARL0 on the replayed host flags", "",
                   "| budget | statistic | placement | %/h [Poisson 95%] | within | contained | unmonitored |", "|---|---|---|---|---|---|---|"])
        md += [f"| {k.split('/')[0]} | {k.split('/')[1]} | {k.split('/')[2]} | {100 * v['rate']:.4f} [{100 * v['lo']:.4f}, {100 * v['hi']:.4f}] | "
               f"{'yes' if v['within'] else 'no'} | {v['contained']:.3f} | {v['unmon']:.3f} |" for k, v in out.items()]
        md.append("")
    # C3
    pairs = [("Monday", d) for d in DAYS[1:]] + [(DAYS[i], DAYS[i + 1]) for i in range(1, 4)]
    rows = [r for r in (transfer("r10_days", c, t) for c, t in pairs) if r]
    if rows:
        H["days"] = {"pairs": rows, "summary": classify(rows),
                     "summary_monday": classify([r for r in rows if r["cal"] == "Monday"])}
        transfer_table(rows, "### C3. Day-to-day calibration: threshold calibrated on one day, applied to a later day")
        md.extend(["Pairs within the budget / within twice the budget / all, per statistic and budget: " +
                   "; ".join(f"{k}: {v['within']} / {v['within_twice']} / {v['pairs']} (worst ratio {v['worst_ratio']:.1f})"
                             for k, v in H["days"]["summary"].items() if v["pairs"]) + ".", ""])
    rows_s = [r for r in (transfer("r10_days_seen", c, t) for c, t in pairs) if r]
    if rows_s:
        H["days_seen"] = {"pairs": rows_s, "summary": classify(rows_s)}
        transfer_table(rows_s, "### C3 sensitivity (not in the plan): attackers at the detection rates of seen attacks "
                               "(the manuscript's CIC-IDS2017 test detectors); benign flags unchanged")
    js["hosts"] = H


def main():
    section_map()
    section_split()
    section_hosts()
    os.makedirs(OUT, exist_ok=True)
    json.dump(js, open(os.path.join(OUT, "v2_r10_stats.json"), "w", encoding="utf-8"), indent=1, default=float)
    open(os.path.join(OUT, "V2_R10.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("wrote V2_R10.md; sections:", list(js))


if __name__ == "__main__":
    main()
