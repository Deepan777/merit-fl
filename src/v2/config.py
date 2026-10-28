"""Configuration of v2 runs: defaults below, overridden by configs/v2/<dataset>.yaml (frozen after
calibration on validation) and by run overrides (dotted keys)."""
import copy

import yaml

DEFAULTS = {
    "seed": 42,
    "output_dir": "results/v2",
    "device": "cpu",          # cpu | cuda; CPU by default: the laptop's free memory cannot hold a CUDA context
    "threads": 4,
    "data": {"name": "5g_nidd", "root": "data", "taxonomy": "family", "cell_profile": "prior_tilt",
             "cell_spread": 1.5, "cell_concentration": 0.3, "dirichlet_alpha": None},
    "model": {"name": "cnn_bilstm"},
    "eval": {"split": "test", "every_rounds": 10},
    "net": {"n_devices": 150, "edges_per_side": 4, "horizon_s": 3600, "trace_dir": "data/v2_traces"},
    "fl": {"round_s": 30, "clients_per_edge": 1, "cloud_every": 2, "window": 300, "per_round": 60,
           "local_epochs": 1, "lr": 0.001, "batch": 256, "clip_norm": 10.0, "staleness_weight": 0.5,
           "selection": "random", "forwarding": False, "p_stay_scale_s": 10.0},
    "energy": {"battery_epochs": [20, 200], "mains": ["veh"], "j_per_flop": 2.0e-9, "tx_power_w": 0.2},
}


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def set_dotted(cfg, key, value):
    *parents, leaf = key.split(".")
    node = cfg
    for p in parents:
        node = node[p]
    if leaf not in node:
        raise KeyError(f"unknown v2 config key {key}")
    node[leaf] = value


def load_v2(path=None, overrides=None):
    cfg = deep_merge(DEFAULTS, yaml.safe_load(open(path, encoding="utf-8")) if path else {})
    for k, v in (overrides or {}).items():
        set_dotted(cfg, k, v)
    return cfg
