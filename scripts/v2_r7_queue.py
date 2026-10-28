#!/usr/bin/env python3
"""Run the remaining R7 jobs (scripts/v2_r7.py test/curves per world) with at most LANES at a time, from a
scheduled task so that they survive a closed session (scripts/v2_r7_queue.ps1). Jobs whose outputs exist are
skipped by v2_r7.py itself; if another instance of this queue is alive, exit. Waits for free memory before
starting a job. Log: results/v2_r7_queue.log; status: results/v2_r7_queue_status.txt.

pythonw scripts/v2_r7_queue.py
"""
import os
import subprocess
import sys
import time

import psutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ("5g_nidd", "ciciot2023", "cicids2017")
JOBS = [("test", w) for w in ("W0", "W1", "W2", "W3", "W4")] + [("curves", w) for w in ("W0", "W1", "W2", "W3", "W4")]
LANES, MIN_FREE_GB = 2, 1.2


def done(cmd, w):
    sub = "r7_test" if cmd == "test" else "r7_curves"
    return all(os.path.exists(os.path.join(REPO, "results", "v2", sub, f"{w}_{d}.env.json")) for d in DATASETS)


def other_queue_alive():
    me = os.getpid()
    mine = {me} | {p.pid for p in psutil.Process(me).parents()}
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except (psutil.Error, TypeError):
            continue
        if p.info["pid"] not in mine and (p.info["name"] or "").lower().startswith("python") and \
                ("v2_r7_queue.py" in cmd or "v2_r7.py" in cmd):
            return True
    return False


def free_gb():
    """Free commit memory (RAM + page file), as the experiment runner measures it."""
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from memutil import free_commit_gb
    return free_commit_gb()


def main():
    status = os.path.join(REPO, "results", "v2_r7_queue_status.txt")
    if other_queue_alive():
        open(status, "w").write(time.strftime("%Y-%m-%d %H:%M:%S") + ": another R7 job is alive\n")
        return
    todo = [j for j in JOBS if not done(*j)]
    log = open(os.path.join(REPO, "results", "v2_r7_queue.log"), "a", buffering=1)
    sys.stdout = sys.stderr = log
    print(time.strftime("%H:%M:%S"), "todo", todo, flush=True)
    py = sys.executable
    running = []
    while todo or running:
        running = [(j, p) for j, p in running if p.poll() is None]
        while todo and len(running) < LANES and free_gb() >= MIN_FREE_GB:
            cmd, w = todo.pop(0)
            out = open(os.path.join(REPO, "results", f"v2_r7_{cmd}_{w}.log"), "a")
            p = subprocess.Popen([py, "-W", "ignore", "-u", os.path.join(REPO, "scripts", "v2_r7.py"), cmd, "--worlds", w],
                                 cwd=REPO, stdout=out, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            running.append(((cmd, w), p))
            print(time.strftime("%H:%M:%S"), "started", cmd, w, flush=True)
            time.sleep(20)
        open(status, "w").write(time.strftime("%Y-%m-%d %H:%M:%S") + f": running {[j for j, _ in running]}, todo {todo}\n")
        time.sleep(30)
    left = [j for j in JOBS if not done(*j)]
    open(status, "w").write(time.strftime("%Y-%m-%d %H:%M:%S") + f": finished; incomplete {left}\n")
    print(time.strftime("%H:%M:%S"), "finished; incomplete", left, flush=True)


if __name__ == "__main__":
    main()
