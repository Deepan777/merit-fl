#!/usr/bin/env python3
"""v2 mobility traces with SUMO (the design notes, Section 4).

Per seed: a 2 x 2 km city grid (9 x 9 junctions, 250 m blocks, traffic lights, sidewalks and
crossings), vehicles and pedestrians on long random routes (so that they stay in the network for
the whole horizon), and static devices at random positions. SUMO runs with 1-s steps and writes
floating-car data; the positions of every device at every second are saved to
data/v2_traces/seed<S>.npz (pos: T x U x 2 in metres, vclass: U, alive: T x U).

SUMO 1.27.1 is installed outside the repository (the project path is too long for SUMO's nested
files under the Windows path limit): set SUMO_HOME or use the default D:\\sumo-1.27.1\\sumo.

python scripts/v2_traces.py --seeds 100 101 102 42 43 ... [--devices 150] [--horizon 3600]
R8 (6-h traces, routes scaled to the horizon): python scripts/v2_traces.py --seeds 52 ... 61 --horizon 21600
"""
import argparse
import gzip
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "data", "v2_traces")
SUMO_HOME = os.environ.get("SUMO_HOME", r"D:\sumo-1.27.1\sumo")
GRID_N, BLOCK = 9, 250.0                       # 9 x 9 junctions, 250 m apart: 2 km x 2 km
SHARES = {"veh": 0.5, "ped": 0.3, "static": 0.2}


def tool(*args, cwd):
    env = {**os.environ, "SUMO_HOME": SUMO_HOME,
           "PYTHONPATH": os.path.join(SUMO_HOME, "tools") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    r = subprocess.run(list(args), cwd=cwd, env=env, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{args[0]} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r


def make_trace(seed, n_dev, horizon, keep_work=False):
    work = os.path.join(OUT, f"work_s{seed}")
    os.makedirs(work, exist_ok=True)
    b = lambda name: os.path.join(SUMO_HOME, "bin", name)
    rt = os.path.join(SUMO_HOME, "tools", "randomTrips.py")
    n_veh, n_ped = int(round(SHARES["veh"] * n_dev)), int(round(SHARES["ped"] * n_dev))
    n_static = n_dev - n_veh - n_ped
    tool(b("netgenerate.exe"), "--grid", "--grid.number", str(GRID_N), "--grid.length", str(BLOCK),
         "--tls.guess", "true", "--sidewalks.guess", "true", "--crossings.guess", "true",
         "--default.speed", "13.89", "-o", "net.net.xml", "--seed", str(seed), cwd=work)
    # long routes: about 200 intermediate edges per hour for vehicles (13.9 m/s, 250 m edges), 40 for pedestrians
    scale = max(1, int(round(horizon / 3600)))
    tool(sys.executable, rt, "-n", "net.net.xml", "-o", "veh.trips.xml", "-r", "veh.rou.xml", "--seed", str(seed),
         "-b", "0", "-e", str(n_veh), "-p", "1", "--intermediate", str(200 * scale), "--prefix", "veh",
         "--validate", "--fringe-factor", "1", cwd=work)
    tool(sys.executable, rt, "-n", "net.net.xml", "-o", "ped.trips.xml", "-r", "ped.rou.xml", "--seed", str(seed + 1000),
         "-b", "0", "-e", str(n_ped), "-p", "1", "--intermediate", str(40 * scale), "--pedestrians", "--prefix", "ped", cwd=work)
    tool(b("sumo.exe"), "-n", "net.net.xml", "-r", "veh.rou.xml,ped.rou.xml", "--step-length", "1",
         "--end", str(horizon), "--seed", str(seed), "--fcd-output", "fcd.xml.gz", "--no-step-log", "true",
         "--time-to-teleport", "300", "--ignore-route-errors", "true", cwd=work)
    ids = [f"veh{i}" for i in range(n_veh)] + [f"ped{i}" for i in range(n_ped)]
    col = {d: i for i, d in enumerate(ids)}
    pos = np.full((horizon, n_dev, 2), np.nan, dtype=np.float32)
    for _, el in ET.iterparse(gzip.open(os.path.join(work, "fcd.xml.gz")), events=("end",)):
        if el.tag == "timestep":
            t = int(float(el.get("time")))
            if t < horizon:
                for o in el:
                    j = col.get(o.get("id"))
                    if j is not None:
                        pos[t, j] = (float(o.get("x")), float(o.get("y")))
            el.clear()
    alive = ~np.isnan(pos[..., 0])
    for j in range(n_veh + n_ped):                     # before departure / after arrival: hold position
        idx = np.flatnonzero(alive[:, j])
        if len(idx) == 0:
            continue
        pos[: idx[0], j] = pos[idx[0], j]
        last = idx[0]
        for t in range(idx[0], horizon):
            if alive[t, j]:
                last = t
            else:
                pos[t, j] = pos[last, j]
    rng = np.random.default_rng(seed + 2000)
    side = (GRID_N - 1) * BLOCK
    pos[:, n_veh + n_ped:] = rng.uniform(0, side, size=(n_static, 2)).astype(np.float32)
    vclass = np.array(["veh"] * n_veh + ["ped"] * n_ped + ["static"] * n_static)
    out = os.path.join(OUT, f"seed{seed}.npz")
    np.savez_compressed(out, pos=pos, vclass=vclass, alive=alive, side=side, horizon=horizon)
    if not keep_work:
        shutil.rmtree(work)
    moving = alive[:, : n_veh + n_ped].mean(0)
    return out, float(moving.mean()), float(np.nanmean(np.linalg.norm(np.diff(pos[:, :n_veh], axis=0), axis=2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--devices", type=int, default=150)
    ap.add_argument("--horizon", type=int, default=3600)
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for s in a.seeds:
        out, alive, speed = make_trace(s, a.devices, a.horizon, a.keep_work)
        print(f"seed {s}: {os.path.relpath(out, REPO)}  moving devices in network {alive:.2f} of the time, "
              f"mean vehicle speed {speed:.1f} m/s", flush=True)


if __name__ == "__main__":
    main()
