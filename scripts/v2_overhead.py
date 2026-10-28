#!/usr/bin/env python3
"""Overheads of MERIT-FL v2 at an edge node : CPU time to score flows with a test detector
(4 threads, batches of 4,096), time of one evidence update for 1,000 devices, and the sizes of a token and
of a context record. Output: reports/stats/v2_overhead.json.

python scripts/v2_overhead.py
"""
import hashlib
import hmac
import json
import os
import struct
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.models.fib import build_model  # noqa: E402


def main():
    torch.set_num_threads(4)
    out = {"threads": 4}
    for n_feat, name in ((67, "5g_nidd"), (37, "ciciot2023"), (60, "cicids2017")):
        model = build_model("cnn_bilstm", n_feat, 8).eval()
        X = torch.randn(4096, n_feat)
        with torch.no_grad():
            model(X)
            t0 = time.perf_counter()
            reps = 10
            for _ in range(reps):
                model(X)
            dt = (time.perf_counter() - t0) / (reps * 4096)
        out[f"us_per_flow_{name}"] = dt * 1e6
        out[f"flows_per_s_{name}"] = 1 / dt
    S = np.zeros(1000)
    x = np.random.default_rng(0).normal(0, 0.05, 1000)
    t0 = time.perf_counter()
    for _ in range(1000):
        S = np.maximum(0.0, S + x - 0.005)
    out["us_cusum_update_1000_devices"] = (time.perf_counter() - t0) / 1000 * 1e6
    key = os.urandom(32)
    payload = struct.pack("<QdIQ", 12345, 0.42, 7, 1_700_000_000)          # device id, S, edge node, time
    tag = hmac.new(key, payload, hashlib.sha256).digest()
    out["token_bytes"] = len(payload) + len(tag)
    out["context_record_bytes"] = len(struct.pack("<Qd", 12345, 0.42))
    t0 = time.perf_counter()
    for _ in range(10000):
        hmac.new(key, payload, hashlib.sha256).digest()
    out["us_per_hmac"] = (time.perf_counter() - t0) / 10000 * 1e6
    json.dump(out, open(os.path.join(REPO, "reports", "stats", "v2_overhead.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
