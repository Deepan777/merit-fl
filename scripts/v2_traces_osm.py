#!/usr/bin/env python3
"""R10-A: mobility traces on a real street network (reports/stats/ANALYSIS_PLAN_V2_R10.md, part A).

map     download the OpenStreetMap extract of Luxembourg City centre (about 2 km x 2 km) once from the main
        OpenStreetMap API, in tiles -> data/osm/luxembourg_centre_tile<ij>.osm.xml.gz and SOURCE.json (bounding box,
        time, SHA-256)
net     netconvert: roads open to passenger cars inside the bounding box, guessed sidewalks, crossings and traffic
        lights, restricted to the largest strongly connected component -> data/osm/luxembourg_centre.net.xml
traces  per seed: vehicles and pedestrians on long random trips (trips generated in excess, the first valid ones kept),
        static devices uniform in the 2 km square, SUMO with 1-s steps -> data/v2_traces/seed<S>.npz in the format of
        scripts/v2_traces.py (pos: T x U x 2 in metres, vclass, alive, side, horizon)

Map data (c) OpenStreetMap contributors, Open Database Licence.

python scripts/v2_traces_osm.py map | net | traces --seeds 62 ... 71 [--horizon 21600]
"""
import argparse
import datetime
import gzip
import hashlib
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import v2_traces as T  # noqa: E402

OSM = os.path.join(REPO, "data", "osm")
NAME = "luxembourg_centre"
BBOX = (6.1157, 49.6023, 6.1435, 49.6203)             # west, south, east, north (plan R10)
SIDE = 2000.0
TILES = 3
NET_FILE = os.path.join(OSM, f"{NAME}.net.xml")
EXCESS = 2                                            # trips generated per kept device
VEH_VIA, PED_VIA = 100, 40                            # intermediate way points per hour of horizon


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tile_files():
    return [os.path.join(OSM, f"{NAME}_tile{i}{j}.osm.xml.gz") for i in range(TILES) for j in range(TILES)]


def get_map():
    """The Overpass servers named in the plan did not answer on 2026-11-10 (time-outs), so the extract comes from the
    main OpenStreetMap API (the same data), in TILES x TILES tiles because that API limits the nodes per request."""
    import urllib.request
    os.makedirs(OSM, exist_ok=True)
    w, s, e, n = BBOX
    files = tile_files()
    if not all(os.path.exists(f) for f in files):
        for i in range(TILES):
            for j in range(TILES):
                out = os.path.join(OSM, f"{NAME}_tile{i}{j}.osm.xml.gz")
                if os.path.exists(out):
                    continue
                bb = (w + (e - w) * i / TILES, s + (n - s) * j / TILES, w + (e - w) * (i + 1) / TILES, s + (n - s) * (j + 1) / TILES)
                url = "https://api.openstreetmap.org/api/0.6/map?bbox=" + ",".join(f"{v:.6f}" for v in bb)
                req = urllib.request.Request(url, headers={"User-Agent": "merit-fl-research/1.0"})
                data = urllib.request.urlopen(req, timeout=180).read()
                with gzip.open(out, "wb") as f:
                    f.write(data)
                print("tile", i, j, len(data), "bytes", flush=True)
        json.dump({"source": "OpenStreetMap main API (api.openstreetmap.org/api/0.6/map), %d x %d tiles" % (TILES, TILES),
                   "bbox_west_south_east_north": BBOX,
                   "downloaded": datetime.datetime.now().isoformat(timespec="seconds"),
                   "files": {os.path.basename(f): {"bytes": os.path.getsize(f), "sha256": sha256(f)} for f in files},
                   "licence": "Open Database Licence (ODbL), (c) OpenStreetMap contributors"},
                  open(os.path.join(OSM, "SOURCE.json"), "w", encoding="utf-8"), indent=1)
    print("map:", [(os.path.basename(f), os.path.getsize(f)) for f in files])


def strongly_connected(net_path):
    """Edge ids of the largest strongly connected component of the car network (junction graph, Kosaraju)."""
    fwd, rev, edges = {}, {}, []
    for _, el in ET.iterparse(net_path, events=("end",)):
        if el.tag == "edge" and el.get("function") is None:
            cars = any("passenger" in (ln.get("allow") or "passenger") and "passenger" not in (ln.get("disallow") or "")
                       for ln in el.findall("lane"))
            if cars:
                a, b = el.get("from"), el.get("to")
                edges.append((el.get("id"), a, b))
                fwd.setdefault(a, []).append(b)
                rev.setdefault(b, []).append(a)
                fwd.setdefault(b, [])
                rev.setdefault(a, [])
            el.clear()
    order, seen = [], set()
    for s in fwd:                                               # iterative depth-first search, finishing order
        if s in seen:
            continue
        stack = [(s, iter(fwd[s]))]
        seen.add(s)
        while stack:
            v, it = stack[-1]
            for w in it:
                if w not in seen:
                    seen.add(w)
                    stack.append((w, iter(fwd[w])))
                    break
            else:
                order.append(v)
                stack.pop()
    comp, best = {}, []
    for s in reversed(order):
        if s in comp:
            continue
        members, stack = [], [s]
        comp[s] = s
        while stack:
            v = stack.pop()
            members.append(v)
            for w in rev[v]:
                if w not in comp:
                    comp[w] = s
                    stack.append(w)
        if len(members) > len(best):
            best = members
    keep = set(best)
    return [e for e, a, b in edges if a in keep and b in keep], len(edges)


def build_net():
    if os.path.exists(NET_FILE):
        print("net: exists")
        return
    b = lambda name: os.path.join(T.SUMO_HOME, "bin", name)  # noqa: E731
    raw = os.path.join(OSM, "net0.net.xml")
    T.tool(b("netconvert.exe"), "--osm-files", ",".join(os.path.basename(f) for f in tile_files()),
           "--keep-edges.in-geo-boundary", ",".join(map(str, BBOX)), "--keep-edges.by-vclass", "passenger",
           "--geometry.remove", "true", "--roundabouts.guess", "true", "--ramps.guess", "true", "--junctions.join", "true",
           "--tls.guess-signals", "true", "--tls.discard-simple", "true", "--tls.join", "true",
           "--sidewalks.guess", "true", "--crossings.guess", "true", "--remove-edges.isolated", "true",
           "--no-warnings", "true", "-o", os.path.basename(raw), cwd=OSM)
    keep, n_all = strongly_connected(raw)
    open(os.path.join(OSM, "keep_edges.txt"), "w", encoding="utf-8").write("\n".join(keep) + "\n")
    T.tool(b("netconvert.exe"), "--sumo-net-file", os.path.basename(raw), "--keep-edges.input-file", "keep_edges.txt",
           "--remove-edges.isolated", "true", "--crossings.guess", "true", "--no-warnings", "true",
           "-o", os.path.basename(NET_FILE), cwd=OSM)
    os.remove(raw)
    x0, y0, x1, y1 = boundary()
    length = sum(float(ln.get("length")) for _, e in ET.iterparse(NET_FILE, events=("end",)) if e.tag == "edge"
                 and e.get("function") is None for ln in e.findall("lane")[:1])
    info = {"edges_car": n_all, "edges_kept": len(keep), "extent_m": [round(x1 - x0), round(y1 - y0)],
            "road_km": round(length / 1000, 1), "sha256": sha256(NET_FILE)}
    json.dump(info, open(os.path.join(OSM, "net_info.json"), "w", encoding="utf-8"), indent=1)
    print("net:", info)


def boundary():
    for _, el in ET.iterparse(NET_FILE, events=("start",)):
        if el.tag == "location":
            return tuple(float(v) for v in el.get("convBoundary").split(","))
    raise RuntimeError("no location element")


def first_n(path, tag, n, prefix):
    """Keep the first n routed vehicles or persons of a route file and number them prefix0 ... prefix(n-1)."""
    tree = ET.parse(path)
    root = tree.getroot()
    kept = 0
    for el in list(root):
        if el.tag == tag:
            if kept < n:
                el.set("id", f"{prefix}{kept}")
                kept += 1
            else:
                root.remove(el)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return kept


def make_trace(seed, n_dev, horizon, keep_work=False):
    work = os.path.join(T.OUT, f"work_s{seed}")
    os.makedirs(work, exist_ok=True)
    shutil.copy(NET_FILE, os.path.join(work, "net.net.xml"))
    b = lambda name: os.path.join(T.SUMO_HOME, "bin", name)  # noqa: E731
    rt = os.path.join(T.SUMO_HOME, "tools", "randomTrips.py")
    n_veh, n_ped = int(round(T.SHARES["veh"] * n_dev)), int(round(T.SHARES["ped"] * n_dev))
    n_static = n_dev - n_veh - n_ped
    scale = max(1, int(round(horizon / 3600)))
    T.tool(sys.executable, rt, "-n", "net.net.xml", "-o", "veh.trips.xml", "-r", "veh.rou.xml", "--seed", str(seed),
           "-b", "0", "-e", str(EXCESS * n_veh), "-p", "1", "--intermediate", str(VEH_VIA * scale), "--prefix", "v",
           "--validate", "--fringe-factor", "1", cwd=work)
    T.tool(sys.executable, rt, "-n", "net.net.xml", "-o", "ped.trips.xml", "-r", "ped.rou.xml", "--seed", str(seed + 1000),
           "-b", "0", "-e", str(EXCESS * n_ped), "-p", "1", "--intermediate", str(PED_VIA * scale), "--pedestrians",
           "--prefix", "p", cwd=work)
    kv = first_n(os.path.join(work, "veh.rou.xml"), "vehicle", n_veh, "veh")
    kp = first_n(os.path.join(work, "ped.rou.xml"), "person", n_ped, "ped")
    if kv < n_veh or kp < n_ped:
        raise RuntimeError(f"seed {seed}: only {kv} vehicles and {kp} pedestrians could be routed")
    T.tool(b("sumo.exe"), "-n", "net.net.xml", "-r", "veh.rou.xml,ped.rou.xml", "--step-length", "1",
           "--end", str(horizon), "--seed", str(seed), "--fcd-output", "fcd.xml.gz", "--no-step-log", "true",
           "--time-to-teleport", "300", "--ignore-route-errors", "true", "--no-warnings", "true", cwd=work)
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
            raise RuntimeError(f"seed {seed}: device {ids[j]} never appeared")
        pos[: idx[0], j] = pos[idx[0], j]
        last = idx[0]
        for t in range(idx[0], horizon):
            if alive[t, j]:
                last = t
            else:
                pos[t, j] = pos[last, j]
    x0, y0, x1, y1 = boundary()                         # centre the network on the 2 km square of the cell grid
    pos[:, : n_veh + n_ped, 0] += SIDE / 2 - (x0 + x1) / 2
    pos[:, : n_veh + n_ped, 1] += SIDE / 2 - (y0 + y1) / 2
    rng = np.random.default_rng(seed + 2000)
    pos[:, n_veh + n_ped:] = rng.uniform(0, SIDE, size=(n_static, 2)).astype(np.float32)
    vclass = np.array(["veh"] * n_veh + ["ped"] * n_ped + ["static"] * n_static)
    out = os.path.join(T.OUT, f"seed{seed}.npz")
    if os.path.exists(out):
        raise RuntimeError(f"{out} exists: traces are never overwritten")
    np.savez_compressed(out, pos=pos, vclass=vclass, alive=alive, side=SIDE, horizon=horizon, map=NAME)
    if not keep_work:
        shutil.rmtree(work)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=2)
    info = {"seed": seed, "moving_in_network": float(alive[:, : n_veh + n_ped].mean()),
            "veh_in_network": float(alive[:, :n_veh].mean()), "ped_in_network": float(alive[:, n_veh:n_veh + n_ped].mean()),
            "veh_speed_mps": float(step[:, :n_veh].mean()), "ped_speed_mps": float(step[:, n_veh:n_veh + n_ped].mean()),
            "outside_square": float(((pos < 0) | (pos > SIDE)).any(axis=2).mean())}
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["map", "net", "traces"])
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--devices", type=int, default=150)
    ap.add_argument("--horizon", type=int, default=21600)
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()
    if a.cmd == "map":
        get_map()
    elif a.cmd == "net":
        build_net()
    else:
        os.makedirs(T.OUT, exist_ok=True)
        log = os.path.join(OSM, "traces_summary.jsonl")
        for s in a.seeds:
            if os.path.exists(os.path.join(T.OUT, f"seed{s}.npz")):
                print(f"seed {s}: exists", flush=True)
                continue
            out, info = make_trace(s, a.devices, a.horizon, a.keep_work)
            open(log, "a", encoding="utf-8").write(json.dumps(info) + "\n")
            print(f"seed {s}: {os.path.relpath(out, REPO)} {json.dumps(info)}", flush=True)


if __name__ == "__main__":
    main()
