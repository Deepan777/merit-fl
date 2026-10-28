#!/usr/bin/env python3
"""Auto-resume of the v2 test runs (started by the scheduled task of scripts/v2_autorun.ps1).

Each start: if a v2 runner is already alive (any process running scripts/v2_run.py run, or another
instance of this script), exit; if every run of the groups is done, exit; otherwise run the runner of
the given checkout in this process (scripts/v2_run.py: resumes from experiments/v2_registry.csv, so
runs that were interrupted by sleep, a restart or a closed session are simply run again). Runs
without a window under pythonw; output goes to results/v2_autorun_<time>.log of that checkout.

pythonw scripts/v2_autorun.py --repo <main checkout> [--workers 2] [--min-free-gb 1.0]
"""
import argparse
import csv
import os
import sys
import time

import psutil

GROUPS = ("test_hfl", "test_round")


def runner_alive():
    # this process and its ancestors: a venv's python/pythonw.exe is a launcher that runs the base
    # interpreter as a child, so the launcher's command line also names this script
    me = os.getpid()
    mine = {me} | {p.pid for p in psutil.Process(me).parents()}
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except (psutil.Error, TypeError):
            continue
        if p.info["pid"] in mine or not (p.info["name"] or "").lower().startswith("python"):
            continue
        if ("v2_run.py" in cmd and " run" in cmd) or "v2_autorun.py" in cmd:
            return p.info["pid"]
    return None


def remaining(repo):
    with open(os.path.join(repo, "experiments", "v2_registry.csv"), newline="", encoding="utf-8") as fh:
        return sum(r["group"] in GROUPS and r["status"] != "done" for r in csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-free-gb", type=float, default=1.0)
    a = ap.parse_args()
    repo = os.path.abspath(a.repo)
    status = os.path.join(repo, "results", "v2_autorun_status.txt")
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    pid = runner_alive()
    if pid:
        open(status, "w", encoding="utf-8").write(f"{stamp}: runner alive (pid {pid}), nothing to do\n")
        return
    left = remaining(repo)
    if left == 0:
        open(status, "w", encoding="utf-8").write(f"{stamp}: all v2 test runs done\n")
        return
    log = os.path.join(repo, "results", f"v2_autorun_{time.strftime('%Y%m%d-%H%M%S')}.log")
    fh = open(log, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = fh                       # pythonw has no console
    open(status, "w", encoding="utf-8").write(f"{stamp}: started runner for {left} remaining runs, log {log}\n")
    os.chdir(repo)
    sys.path.insert(0, os.path.join(repo, "scripts"))
    sys.argv = [os.path.join(repo, "scripts", "v2_run.py")]
    import v2_run                                     # the checkout's own runner (same code as the runs)
    v2_run.run(list(GROUPS), a.workers, None, a.min_free_gb, "random")
    open(status, "a", encoding="utf-8").write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: runner finished\n")


if __name__ == "__main__":
    main()
