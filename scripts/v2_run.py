#!/usr/bin/env python3
"""Run v2 hierarchical-FL experiments from experiments/v2_registry.csv (src/v2/plan.py), resumable,
on CPU worker processes. A run's config: src/v2/config.py defaults <- configs/v2/<dataset>.yaml
(frozen after calibration; absent during calibration) <- the row's overrides. Calibration groups set
eval.split = val and the runner then refuses any test read (MERIT_FL_FORBID_TEST). A worker starts
only while --min-free-gb of commit memory is free.

python scripts/v2_run.py plan
python scripts/v2_run.py run --groups cal_hfl [--workers 2] [--limit N]
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
REG = os.path.join(REPO, "experiments", "v2_registry.csv")
FIELDS = ["run_id", "group", "dataset", "arm", "seed", "overrides", "status", "run_dir", "wall_s",
          "macro_f1_last3", "finished_at", "error"]


def read():
    return list(csv.DictReader(open(REG, newline="", encoding="utf-8"))) if os.path.exists(REG) else []


def write(rows):
    tmp = REG + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)
    os.replace(tmp, REG)


def plan():
    from src.v2.plan import plan_rows
    old = {r["run_id"]: r for r in read()}
    rows = []
    for r in plan_rows():
        o = old.get(r["run_id"])
        rows.append(o if o and o["status"] == "done" else {**r, "status": "planned"})
    write(rows)
    print(f"{len(rows)} rows; done {sum(r['status'] == 'done' for r in rows)}")


def job(row):
    import traceback
    if json.loads(row["overrides"]).get("eval.split") == "val":
        os.environ["MERIT_FL_FORBID_TEST"] = "1"
    from src.v2.config import load_v2
    from src.v2.hfl import run_hfl
    t0 = time.time()
    try:
        frozen = os.path.join(REPO, "configs", "v2", f"{row['dataset']}.yaml")
        ov = {"seed": int(row["seed"]), "data.name": row["dataset"],
              "output_dir": os.path.join("results", "v2", row["group"], row["dataset"]), **json.loads(row["overrides"])}
        cfg = load_v2(frozen if os.path.exists(frozen) and row["group"] != "cal_hfl" else None, ov)
        out = run_hfl(cfg, log=lambda *a: None)
        return {**row, "status": "done", "run_dir": out["run_dir"], "wall_s": round(time.time() - t0, 1),
                "macro_f1_last3": out["macro_f1_last3"], "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "error": ""}
    except Exception:
        return {**row, "status": "failed", "wall_s": round(time.time() - t0, 1), "error": traceback.format_exc()[-1500:]}


def run(groups, workers, limit, min_free, first_arm=None):
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from memutil import free_commit_gb
    rows = read()
    todo = [r for r in rows if r["status"] in ("planned", "failed") and (not groups or r["group"] in groups)]
    if first_arm:                                     # e.g. the detectors the containment experiment needs
        todo = sorted(todo, key=lambda r: (r["group"] != "test_hfl" or r["arm"] != first_arm))
    todo = todo[:limit] if limit else todo
    print(f"{len(todo)} runs to execute", flush=True)
    idx = {r["run_id"]: i for i, r in enumerate(rows)}
    done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
        pending, queue = set(), list(todo)
        while queue or pending:
            while queue and len(pending) < workers and free_commit_gb() >= min_free:
                pending.add(ex.submit(job, queue.pop(0)))
                time.sleep(5)
            if not pending:
                time.sleep(30)
                continue
            fin, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in fin:
                res = f.result()
                rows[idx[res["run_id"]]] = res
                write(rows)
                done += 1
                print(f"[{done}/{len(todo)}] {res['run_id']}: {res['status']} ({res['wall_s']} s) "
                      f"F1={res.get('macro_f1_last3', '')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run"])
    ap.add_argument("--groups", nargs="*")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--min-free-gb", type=float, default=1.5)
    ap.add_argument("--first-arm", default=None, help="run this test_hfl arm first")
    a = ap.parse_args()
    plan() if a.cmd == "plan" else run(a.groups, a.workers, a.limit, a.min_free_gb, a.first_arm)
