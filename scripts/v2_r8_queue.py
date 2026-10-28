#!/usr/bin/env python3
"""Run the R8 test jobs (scripts/v2_r8.py) with at most LANES at a time, waiting for free commit memory before each
start. Jobs whose outputs exist are skipped (v2_r8.py also skips finished world/dataset files, so a rerun resumes).
Log: results/v2_r8_queue.log; status: results/v2_r8_queue_status.txt.

python scripts/v2_r8_queue.py
"""
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from memutil import free_commit_gb  # noqa: E402

DATASETS = ("5g_nidd", "ciciot2023", "cicids2017")
V2 = os.path.join(REPO, "results", "v2")
SWEEP_GROUPS = [["sigma_0", "sigma_0.05", "sigma_0.1"], ["sigma_0.15", "sigma_0.2", "sigma_0.3"],
                ["betabin_0.01", "betabin_0.03", "betabin_0.1"], ["burst_3", "burst_10"]]


def _all(sub, names):
    return lambda: all(os.path.exists(os.path.join(V2, sub, f"{n}_{d}.env.json")) for n in names for d in DATASETS)


JOBS = ([(["main", "--proto", "chained", "--worlds", w], _all("r8_main", [f"chained_{w}"])) for w in ("W0", "W1", "W2", "W3")] +
        [(["main", "--proto", "unchained", "--worlds", w], _all("r8_main", [f"unchained_{w}"])) for w in ("W0", "W1", "W2", "W3")] +
        [(["main", "--proto", p, "--worlds", "W4"], _all("r8_main", [f"{p}_W4"])) for p in ("chained", "unchained")] +
        [(["sweep", "--worlds", *g], _all("r8_sweep", g)) for g in SWEEP_GROUPS] +
        [(["curves", "--worlds", w], _all("r8_curves", [w])) for w in ("W0", "W1")] +
        [(["ablation"], _all("r8_ablation", ["W1"])),
         (["rates"], lambda: os.path.exists(os.path.join(V2, "r8_rates.json")))])
LANES, MIN_FREE_GB = 3, 1.5


def main():
    status = os.path.join(REPO, "results", "v2_r8_queue_status.txt")
    todo = [j for j in JOBS if not j[1]()]
    log = open(os.path.join(REPO, "results", "v2_r8_queue.log"), "a", buffering=1)
    print(time.strftime("%H:%M:%S"), "todo", [" ".join(a) for a, _ in todo], file=log, flush=True)
    running = []
    while todo or running:
        running = [(a, p) for a, p in running if p.poll() is None]
        while todo and len(running) < LANES and free_commit_gb() >= MIN_FREE_GB:
            args, _ = todo.pop(0)
            name = "_".join(a.lstrip("-") for a in args)[:60]
            out = open(os.path.join(REPO, "results", f"v2_r8_{name}.log"), "a")
            p = subprocess.Popen([sys.executable, "-W", "ignore", "-u", os.path.join(REPO, "scripts", "v2_r8.py"), *args],
                                 cwd=REPO, stdout=out, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            running.append((args, p))
            print(time.strftime("%H:%M:%S"), "started", " ".join(args), file=log, flush=True)
            time.sleep(30)
        open(status, "w").write(time.strftime("%Y-%m-%d %H:%M:%S") +
                                f": running {[' '.join(a) for a, _ in running]}, todo {len(todo)}\n")
        time.sleep(30)
    left = [" ".join(a) for a, ok in JOBS if not ok()]
    open(status, "w").write(time.strftime("%Y-%m-%d %H:%M:%S") + f": finished; incomplete {left}\n")
    print(time.strftime("%H:%M:%S"), "finished; incomplete", left, file=log, flush=True)


if __name__ == "__main__":
    main()
