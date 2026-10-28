"""C2: mobility-aware hierarchical federated training (the design notes, Section 3).

Per round of fl.round_s seconds:
1. every edge node selects up to fl.clients_per_edge of the devices associated with it at the
   start of the round that are not in outage and can afford the round (energy gate; exhausted
   devices are never eligible), by fl.selection:
     random     uniformly
     speed      v1 mobility stability exp(-4 h - v/20) (penalises fast devices)
     residence  predicted probability of staying in the cell for the round (src/v2/network.py)
     oracle     devices that will in fact stay and stay connected (upper bound, not deployable)
2. selected devices train locally from their edge node's model on their buffers;
3. at the end of the round an update is delivered to the edge node the device started at if it
   is connected and did not hand over; if it handed over it is lost, or, with fl.forwarding, the
   old edge node forwards it to the new one, where it is aggregated with fl.staleness_weight;
   a device in outage at the end of the round cannot upload;
4. edge nodes aggregate the updates they received (weights by data size, clipped to fl.clip_norm);
5. every fl.cloud_every rounds the cloud averages the edge models (weighted by the data each
   aggregated since the last synchronisation) and broadcasts the result.
Holdout macro-F1 of the cloud model is evaluated every eval.every_rounds rounds (at
synchronisation); the reported accuracy is the mean of the last three checkpoints.
"""
import copy
import json
import os
import time

import numpy as np
import torch

from ..evaluation.metrics import evaluate
from ..federated.core import local_train, tw_agg
from ..models.fib import build_model, count_flops
from ..utils.common import env_info, make_run_dir, set_seed
from .network import compute_links, edge_grid, p_stay, time_to_exit
from .traffic import Buffers, V2Data


def load_trace(cfg, seed):
    z = np.load(os.path.join(cfg["net"]["trace_dir"], f"seed{seed}.npz"))
    pos = z["pos"].astype(np.float64)[: cfg["net"]["horizon_s"], : cfg["net"]["n_devices"]]
    return pos, z["vclass"][: cfg["net"]["n_devices"]], float(z["side"])


def mobility_stability(handover, speed, t, window=50):
    h = handover[max(0, t - window): t].mean(0) if t > 0 else np.zeros(handover.shape[1])
    return np.exp(-4.0 * h - speed[t] / 20.0)


def _average(models, weights):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    sd = [m.state_dict() for m in models]
    out = {}
    for k, v in sd[0].items():
        out[k] = sum(float(wi) * s[k].float() for wi, s in zip(w, sd)).to(v.dtype) if v.dtype.is_floating_point else v
    return out


def run_hfl(cfg, log=print):
    seed = cfg["seed"]
    set_seed(seed)
    rng = np.random.default_rng([seed, 11])
    device = torch.device("cuda") if cfg["device"] == "cuda" else torch.device("cpu")
    torch.set_num_threads(cfg["threads"])
    run_dir = make_run_dir(cfg["output_dir"], f"hfl-{cfg['fl']['selection']}-{'fw' if cfg['fl']['forwarding'] else 'nofw'}-s{seed}")
    t_start = time.time()
    data = V2Data(cfg, seed)
    pos, vclass, side = load_trace(cfg, seed)
    edges = edge_grid(side, cfg["net"]["edges_per_side"])
    links = compute_links(pos, edges, seed)
    U, E = pos.shape[1], len(edges)
    fl, en = cfg["fl"], cfg["energy"]
    Tr = fl["round_s"]
    R = cfg["net"]["horizon_s"] // Tr
    buffers = Buffers(data, U, E, fl["window"], fl["per_round"], cfg["data"], rng)
    buffers.observe(links["serving"][0], n=fl["window"])          # initial buffers
    model = build_model(cfg["model"]["name"], data.n_features, data.K).to(device)
    edge_models = [copy.deepcopy(model) for _ in range(E)]
    since_sync = np.zeros(E)
    flops = count_flops(model, data.n_features)
    e_epoch = 3 * flops * fl["window"] * en["j_per_flop"]
    model_bits = 32 * sum(p.numel() for p in model.parameters())
    battery = np.where(np.isin(vclass, en["mains"]), np.inf,
                       rng.uniform(*en["battery_epochs"], size=U) * e_epoch)
    stats = {k: 0 for k in ("selected", "delivered", "interrupted", "forwarded", "lost_handover",
                            "lost_outage", "dead_skipped")}
    energy_j, uplink_bits, curve = 0.0, 0.0, []
    for r in range(R):
        t0, t1 = r * Tr, min((r + 1) * Tr, cfg["net"]["horizon_s"]) - 1
        serving0 = links["serving"][t0].astype(int)
        out0 = links["outage"][t0]
        ho_round = links["handover"][t0 + 1: t1 + 1].any(0)
        conn_end = ~links["outage"][t1]
        need = fl["local_epochs"] * e_epoch + en["tx_power_w"] * model_bits / np.maximum(
            10e6 * np.log2(1 + 10 ** (links["snr_db"][t0] / 10)), 1e3)
        score = None
        if fl["selection"] == "speed":
            score = mobility_stability(links["handover"], links["speed"], t0)
        elif fl["selection"] == "residence":
            tte = time_to_exit(pos[t0], links["vel"][t0].astype(np.float64), serving0, edges)
            score = p_stay(tte, Tr, fl["p_stay_scale_s"])
        elif fl["selection"] == "oracle":
            score = (~ho_round & conn_end).astype(float)
        received = [[] for _ in range(E)]                      # (delta, weight) per edge
        for e in range(E):
            cand = np.flatnonzero((serving0 == e) & ~out0 & (battery >= need))
            stats["dead_skipped"] += int(((serving0 == e) & ~out0 & (battery < need)).sum())
            if len(cand) == 0:
                continue
            order = rng.permutation(cand)
            if score is not None:                              # stable sort keeps random tie-breaking
                order = order[np.argsort(-score[order], kind="stable")]
            for u in order[: fl["clients_per_edge"]]:
                X, y = buffers.data_of(u)
                if len(y) == 0:
                    continue
                delta, n, _ = local_train(edge_models[e], X, y, device, fl["local_epochs"], fl["lr"], fl["batch"])
                battery[u] -= need[u]
                energy_j += float(need[u])
                stats["selected"] += 1
                if not conn_end[u]:
                    stats["lost_outage"] += 1
                    continue
                uplink_bits += model_bits
                if not ho_round[u]:
                    received[e].append((delta, n))
                    stats["delivered"] += 1
                elif fl["forwarding"]:
                    e2 = int(links["serving"][t1, u])
                    received[e2].append((delta, n * fl["staleness_weight"]))
                    stats["interrupted"] += 1
                    stats["forwarded"] += 1
                else:
                    stats["interrupted"] += 1
                    stats["lost_handover"] += 1
        for e in range(E):
            if received[e]:
                tw_agg(edge_models[e], [d for d, _ in received[e]], [w for _, w in received[e]], fl["clip_norm"])
                since_sync[e] += sum(w for _, w in received[e])
        if (r + 1) % fl["cloud_every"] == 0 and since_sync.sum() > 0:
            active = np.flatnonzero(since_sync > 0)
            model.load_state_dict(_average([edge_models[e] for e in active], since_sync[active]))
            for m in edge_models:
                m.load_state_dict(model.state_dict())
            since_sync[:] = 0
            if (r + 1) % cfg["eval"]["every_rounds"] == 0:
                m_ = evaluate(model, data.X_hold, data.y_hold, device, data.K, data.benign)
                curve.append({"round": r + 1, "macro_f1": float(m_["macro_f1"]), "acc": float(m_["acc"]),
                              "uplink_MB": uplink_bits / 8e6, "energy_J": energy_j,
                              "delivered": stats["delivered"] + stats["forwarded"]})
                log(f"r{r + 1}: macro-F1 {m_['macro_f1']:.4f} | delivered {stats['delivered']} "
                    f"interrupted {stats['interrupted']} lost(outage) {stats['lost_outage']}")
        buffers.observe(links["serving"][t1])
    final = evaluate(model, data.X_hold, data.y_hold, device, data.K, data.benign)
    out = {"seed": seed, "dataset": cfg["data"]["name"], "selection": fl["selection"], "forwarding": fl["forwarding"],
           "round_s": Tr, "rounds": R, "curve": curve,
           "macro_f1_last3": float(np.mean([c["macro_f1"] for c in curve[-3:]])) if curve else None,
           "final": {k: (float(v) if np.isscalar(v) else v) for k, v in final.items() if k in ("macro_f1", "acc", "far")},
           "stats": stats, "energy_J": energy_j, "uplink_MB": uplink_bits / 8e6,
           "handovers_per_device_hour": float(links["handover"].sum() / U / (cfg["net"]["horizon_s"] / 3600)),
           "outage_share": float(links["outage"].mean()), "wall_s": time.time() - t_start,
           "peak_vram_MB": float(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0.0}
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    json.dump(out, open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8"), indent=1, default=float)
    json.dump({"config": cfg}, open(os.path.join(run_dir, "config_resolved.json"), "w", encoding="utf-8"), indent=1)
    json.dump(env_info(), open(os.path.join(run_dir, "env.json"), "w", encoding="utf-8"), indent=1)
    out["run_dir"] = run_dir
    return out
