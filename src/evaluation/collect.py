"""One tidy row per Phase 4 run, read from experiments/registry.csv and each
run's metrics.json (results/ only; nothing is recomputed from data)."""
import csv
import json
import os

import numpy as np
import pandas as pd

HIGH_RISK = 0.5          # lower edge of the highest Risk-Place risk bin


def _get(d, *path, default=np.nan):
    for p in path:
        if not isinstance(d, dict) or d.get(p) is None:
            return default
        d = d[p]
    return d


def _placement(p, prefix):
    hi = [b for b in p.get("by_risk", []) if b["risk_range"][0] == HIGH_RISK]
    return {f"{prefix}budget_sat": _get(p, "budget_satisfaction"),
            f"{prefix}budget_sat_high_risk": hi[0]["budget_satisfaction"] if hi and hi[0]["n"] else np.nan,
            f"{prefix}n_high_risk": hi[0]["n"] if hi else 0,
            f"{prefix}energy_J": _get(p, "energy_J"),
            f"{prefix}latency_ms": _get(p, "latency_ms_mean"),
            f"{prefix}latency_p95_ms": _get(p, "latency_ms_p95"),
            f"{prefix}unreachable": _get(p, "unreachable")}


def run_row(r, m):
    """Flatten one run (registry row r, metrics m)."""
    final, h2, sp = m["final"], m["h2"], m["slow_path"]
    rare = m.get("rare_classes") or []
    pcf = final.get("per_class_f1") or []
    row = {
        "run_id": r["run_id"], "group": r["group"], "dataset": r["dataset"], "method": r["method"],
        "ablation": r["ablation"], "seed": int(r["seed"]), "setting": int(r["run_id"].split("/set")[1].split("/")[0]),
        "overrides": r["overrides"], "label": r["method"] + ("+" + r["ablation"] if r["ablation"] else ""),
        # detection (holdout test split, argmax decision)
        "macro_f1": final["macro_f1"], "acc": final["acc"], "far": final["far"],
        "rare_f1": float(np.mean([pcf[k] for k in rare])) if rare and pcf else np.nan,
        # stream (per-flow decisions with the method's own operating point and trust)
        "stream_macro_f1": _get(m, "stream", "macro_f1"), "stream_detection": _get(m, "stream", "attack_detection"),
        "stream_far": _get(m, "stream", "far"), "j_sec": _get(m, "security", "j_sec"),
        "residual_exposure": _get(m, "security", "residual_exposure"),
        # handover windows (H2)
        "inside_detection": _get(h2, "inside", "attack_detection"),
        "outside_detection": _get(h2, "outside", "attack_detection"),
        "gap_detection": _get(h2, "gap", "attack_detection"),
        "inside_far": _get(h2, "inside", "far"), "outside_far": _get(h2, "outside", "far"),
        "detection_dip": _get(h2, "detection_dip"), "macro_f1_dip": _get(h2, "macro_f1_dip"),
        "far_rise": _get(h2, "far_rise"), "window_flows_ok": bool(h2.get("window_flows_ok")),
        "n_inside": _get(h2, "inside", "n"),
        "injected_inside_detection": _get(h2, "injected_inside", "attack_detection"),
        "injected_outside_detection": _get(h2, "injected_outside", "attack_detection"),
        # trust (H5)
        "trust_stability": _get(m, "trust", "stability_per_device_round"),
        "quarantine_device_rounds": _get(m, "trust", "quarantine_device_rounds"),
        **{f"handover_{k}": _get(m, "trust", "handover_outcomes", k, default=0)
           for k in ("carried", "expired", "missing", "tampered", "revoked")},
        # communication, energy, cost (H3, scalability)
        "uplink_MB_total": _get(m, "comm", "uplink_bytes_total") / 2**20,
        "uplink_MB_per_round": _get(m, "comm", "uplink_MB_per_round_mean"),
        "train_J": _get(m, "energy_sim", "train_J"), "dropped_clients": _get(m, "energy_sim", "dropped_clients"),
        "dead_devices": _get(m, "energy_sim", "dead_devices"),
        "wall_clock_s": _get(m, "cost_measured", "wall_clock_s"), "round_s": _get(m, "cost_measured", "round_s_mean"),
        "peak_vram_MB": _get(m, "cost_measured", "peak_vram_MB"), "params": _get(m, "cost_measured", "params"),
        "size_MB": _get(m, "cost_measured", "size_MB"), "flops_per_sample": _get(m, "cost_measured", "flops_per_sample"),
        "latency_gpu_ms": _get(m, "cost_measured", "latency", "cuda_bs1", "per_sample_ms"),
        "latency_cpu_ms": _get(m, "cost_measured", "latency", "cpu_bs1", "per_sample_ms"),
        "explain_stability": _get(m, "explain", "stability"),
        # slow path (H7)
        "slow_backend": sp["backend"], "llm_calls": sp["llm_calls"], "slow_invocations": sp["invocations"],
        "reevaluated": sp["reevaluated_flows"], "reeval_changed": sp["reeval_changed_decisions"],
        "flows_total": sp["flows_total"], "llm_calls_per_flow": sp["llm_calls_per_flow"],
        # placement (H6): the method's own, and static edge on the same trace (MERIT-FL)
        **_placement(m["placement"], "place_"),
        **_placement(m["placement"].get("counterfactual_static_edge") or {}, "static_"),
        # learning curve (H3): holdout macro-F1 every log_every rounds, cumulative uplink at each checkpoint
        "curve_rounds": json.dumps([c["round"] for c in m["curve"]]),
        "curve_f1": json.dumps([c["macro_f1"] for c in m["curve"]]),
        "curve_uplink_MB": json.dumps([float(np.sum(m["history"]["uplink_bytes"][:c["round"]])) / 2**20
                                       for c in m["curve"]]),
    }
    a3 = m.get("a3_revoked_device")
    if a3:
        after = [h for h in a3["handovers"] if h["round"] > a3["revoke_round"]]
        row.update({"a3_max_tau_after": a3["max_tau_after_revocation"],
                    "a3_selected_after": a3["selected_after_revocation"],
                    "a3_handovers_after": len(after),
                    "a3_handovers_after_denied": sum(h["outcome"] == "revoked" and h["tau_after"] == 0.0 for h in after),
                    "a3_detection_after": _get(a3, "flows_after_revocation", "attack_detection")})
    return row


def load_runs(repo, registry="experiments/registry.csv"):
    rows = list(csv.DictReader(open(os.path.join(repo, registry), newline="", encoding="utf-8")))
    out = []
    for r in rows:
        if r["status"] != "done":
            continue
        with open(os.path.join(repo, r["run_dir"], "metrics.json"), encoding="utf-8") as fh:
            out.append(run_row(r, json.load(fh)))
    return pd.DataFrame(out)
