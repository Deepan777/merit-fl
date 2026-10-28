"""Real-dataset preprocessing for MERIT-FL.

    python scripts/preprocess.py --dataset cicids2017 [--root data] [--taxonomy family]

Raw sources: <root>/<name>/raw/ by default, or any files / folders / zip
archives given with --raw (zip members are read without extraction). Steps,
each logged with row/column counts for data/DATASET_CARDS.md:
   1  read every raw CSV (numeric datasets in chunks, hashed in float64 for
      exact duplicate detection, stored as float32); strip column names
   2  raw label -> canonical fine label -> taxonomy class (unknown labels abort)
   3  drop identifier / timestamp / payload / label-copy columns (reasons in SPECS)
   4  numeric coercion (inf -> NaN); low-cardinality text one-hot encoded,
      high-cardinality text dropped
   5  drop exact duplicate rows, then feature-duplicates with conflicting labels
   6  drop classes with fewer than `min_class_count` rows
   7  stratified cap to `max_rows` (classes below `class_floor` kept whole)
   8  stratified train/val/test split with a fixed `split_seed` — before any fitting
   9  fitted on train only: median imputation, removal of constant and duplicate
      columns, StandardScaler; applied to all splits
  10  save processed/<taxonomy>.npz + .json, and processed/audit_<taxonomy>.npz
      (dropped identifier columns of train/val rows, for the leakage audit)
"""
import glob
import hashlib
import json
import os
import re
import time
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ID = "network identifier (address/port/flow id): names testbed hosts, not behaviour"
TIME = "timestamp or capture-order index: encodes when attacks were scheduled in the capture"
PAYLOAD = "packet payload or free text: not a flow statistic; would let the model read attack strings"
LABEL = "copy of the label"
SEQ = "raw sequence/transaction number or checksum: pseudo-random per packet or session"
PORT = ("service port: names the targeted service, and in a testbed the port-to-attack "
        "mapping is fixed by the lab set-up rather than by attack behaviour, so it would not "
        "transfer to other networks; dropped conservatively (per-dataset evidence in the "
        "leakage audit)")
CAPTURE_TIME = ("capture-time artifact: values sit in narrow, non-overlapping bands per attack "
                "family (~8.3e7 vs ~1.66e8), i.e. they identify the recording session rather "
                "than flow behaviour (see leakage audit)")


def _norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


# ----------------------------- CICIDS2017 -----------------------------
def _cicids_fine(raw):
    s = _norm(raw)
    low = s.lower()
    if low == "benign":
        return "Benign"
    if low.startswith("web attack"):
        for key, name in (("brute", "Brute Force"), ("xss", "XSS"), ("sql", "Sql Injection")):
            if key in low:
                return f"Web Attack - {name}"
    return s


CICIDS_FAMILY = {
    "Benign": "Benign", "DoS Hulk": "DoS", "DoS GoldenEye": "DoS", "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS", "DDoS": "DDoS", "PortScan": "PortScan",
    "FTP-Patator": "BruteForce", "SSH-Patator": "BruteForce", "Bot": "Bot",
    "Web Attack - Brute Force": "WebAttack", "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack", "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
}

# ----------------------------- Edge-IIoTset -----------------------------
EDGE_FAMILY = {
    "Benign": "Benign", "DDoS_UDP": "DDoS", "DDoS_ICMP": "DDoS", "DDoS_TCP": "DDoS",
    "DDoS_HTTP": "DDoS", "Port_Scanning": "InfoGathering", "Fingerprinting": "InfoGathering",
    "Vulnerability_scanner": "InfoGathering", "MITM": "MITM", "XSS": "Injection",
    "SQL_injection": "Injection", "Uploading": "Injection", "Backdoor": "Malware",
    "Password": "Malware", "Ransomware": "Malware",
}

# ----------------------------- CICIoT2023 -----------------------------
CICIOT_OTHER = {
    "VulnerabilityScan": "Recon", "MITM-ArpSpoofing": "Spoofing", "DNS_Spoofing": "Spoofing",
    "DictionaryBruteForce": "BruteForce", "SqlInjection": "Web", "XSS": "Web",
    "CommandInjection": "Web", "Uploading_Attack": "Web", "Backdoor_Malware": "Web",
    "BrowserHijacking": "Web",
}


def _ciciot_family(fine):
    if fine == "Benign":
        return "Benign"
    for prefix, fam in (("DDoS-", "DDoS"), ("DoS-", "DoS"), ("Mirai-", "Mirai"), ("Recon-", "Recon")):
        if fine.startswith(prefix):
            return fam
    return CICIOT_OTHER.get(fine)


SPECS = {
    "cicids2017": {
        "numeric": True,
        "title": "CIC-IDS2017",
        "source_url": "https://www.unb.ca/cic/datasets/ids-2017.html",
        "files": "MachineLearningCSV.zip -> MachineLearningCVE/*.csv (8 files)",
        "reference": "Sharafaldin, Lashkari, Ghorbani, 'Toward Generating a New Intrusion "
                     "Detection Dataset and Intrusion Traffic Characterization', ICISSP 2018 "
                     "(as given on the dataset page; DOI to be verified in Phase 7)",
        "label": "Label",
        "fine": _cicids_fine,
        "family": CICIDS_FAMILY.get,
        "drop": {"Flow ID": ID, "Source IP": ID, "Source Port": ID, "Destination IP": ID,
                 "Timestamp": TIME, "Destination Port": PORT},
    },
    "edge_iiotset": {
        "numeric": False,
        "title": "Edge-IIoTset",
        "source_url": "https://www.kaggle.com/datasets/mohamedamineferrag/"
                      "edgeiiotset-cyber-security-dataset-of-iot-iiot",
        "files": "Selected dataset for ML and DL / DNN-EdgeIIoT-dataset.csv",
        "reference": "Ferrag et al., 'Edge-IIoTset: A New Comprehensive Realistic Cyber Security "
                     "Dataset of IoT and IIoT Applications for Centralized and Federated Learning', "
                     "IEEE Access 2022; IEEE DataPort DOI 10.21227/mbc1-1h68 (both to be verified "
                     "by DOI in Phase 7). Kaggle labels the dataset CC BY-NC-SA 4.0",
        "label": "Attack_type",
        # Text fields are restricted to their protocol vocabulary; anything else
        # (attack strings that tools write into the HTTP request line, malformed
        # MQTT fields) becomes "<other>", so the model sees "malformed" but never
        # the attack text itself.
        "vocab": {"http.request.version": ["HTTP/1.0", "HTTP/1.1", "0"],
                  "http.request.method": ["GET", "HEAD", "POST", "PUT", "DELETE", "CONNECT",
                                          "OPTIONS", "TRACE", "PATCH", "PROPFIND", "SEARCH", "0"],
                  "mqtt.conack.flags": ["0", "0x00000000", "0x00000001"],
                  "mqtt.protoname": ["MQTT", "0"],
                  "mqtt.topic": ["Temperature_and_Humidity", "0"]},
        "fine": lambda raw: "Benign" if _norm(raw).lower() == "normal" else _norm(raw),
        "family": EDGE_FAMILY.get,
        "drop": {"frame.time": TIME, "ip.src_host": ID, "ip.dst_host": ID,
                 "arp.src.proto_ipv4": ID, "arp.dst.proto_ipv4": ID, "tcp.srcport": ID,
                 "tcp.dstport": PORT, "udp.port": PORT, "udp.stream": TIME,
                 "icmp.transmit_timestamp": TIME, "tcp.ack_raw": SEQ, "tcp.checksum": SEQ,
                 "icmp.checksum": SEQ, "mbtcp.trans_id": SEQ, "http.file_data": PAYLOAD,
                 "http.request.full_uri": PAYLOAD, "http.request.uri.query": PAYLOAD,
                 "tcp.payload": PAYLOAD, "tcp.options": PAYLOAD, "mqtt.msg": PAYLOAD,
                 "dns.qry.name": PAYLOAD, "Attack_label": LABEL},
    },
    "ciciot2023": {
        "numeric": True,
        "title": "CICIoT2023",
        "source_url": "https://www.unb.ca/cic/datasets/iotdataset-2023.html",
        "files": "CSV/*.csv (merged feature files)",
        "reference": "Neto et al., 'CICIoT2023: A real-time dataset and benchmark for "
                     "large-scale attacks in IoT environment', Sensors 2023 (as given on the "
                     "dataset page; DOI to be verified in Phase 7)",
        "label": "label",
        "fine": lambda raw: "Benign" if _norm(raw) == "BenignTraffic" else _norm(raw),
        "family": _ciciot_family,
        "drop": {"IAT": CAPTURE_TIME},
    },
}

# 5G-NIDD (v2, mobile-native): Argus flow features captured at two base stations of the University
# of Oulu 5G test network; one attacker machine per base station. Families group the 8 attack types.
NIDD_FAMILY = {"Benign": "Benign", "UDPFlood": "Flood", "SYNFlood": "Flood", "ICMPFlood": "Flood",
               "HTTPFlood": "HTTP-DoS", "SlowrateDoS": "HTTP-DoS",
               "SYNScan": "Scan", "TCPConnectScan": "Scan", "UDPScan": "Scan"}
HOST = ("host fingerprint (IP time-to-live / hop count): in a testbed with one attacker machine it "
        "names the host, not the behaviour")
SPECS["5g_nidd"] = {
    "numeric": False,
    "chunked_text": True,                                 # 275 MB of mixed CSV: two-pass chunked read
    "chunk": 100_000,
    "title": "5G-NIDD",
    "source_url": "https://etsin.fairdata.fi/dataset/9d13ef28-2ca7-44b0-9950-225359afac65",
    "files": "Combined.zip -> Combined.csv (both base stations, all attack scenarios)",
    "members": ["Combined.zip::Combined.csv"],           # the per-attack files duplicate these flows
    "reference": "Samarakoon et al., '5G-NIDD: A Comprehensive Network Intrusion Detection Dataset "
                 "Generated over 5G Wireless Network', arXiv:2212.01298; data DOI "
                 "10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361 (CC BY 4.0)",
    "label": "Attack Type",
    "fine": _norm,
    "family": NIDD_FAMILY.get,
    "drop": {"Unnamed: 0": TIME, "Seq": SEQ, "Offset": TIME, "SrcTCPBase": SEQ, "DstTCPBase": SEQ,
             "sVid": ID, "dVid": ID, "sTtl": HOST, "dTtl": HOST, "sHops": HOST, "dHops": HOST,
             "Label": LABEL, "Attack Tool": LABEL},
}

# Robustness variant (Phase 4): Edge-IIoTset without the dominant TCP
# sequence fields flagged in the leakage audit.
SPECS["edge_iiotset_noseq"] = {
    **SPECS["edge_iiotset"], "title": "Edge-IIoTset (without tcp.seq / tcp.ack)",
    "drop": {**SPECS["edge_iiotset"]["drop"],
             "tcp.seq": "robustness variant: dominant connection-progress field removed",
             "tcp.ack": "robustness variant: dominant connection-progress field removed"},
}

DEFAULT_PARAMS = {"taxonomy": "family", "min_class_count": 100, "max_rows": 600_000,
                  "class_floor": 10_000, "val_size": 0.1, "test_size": 0.2,
                  "split_seed": 0, "max_categories": 32, "max_files": None}


CHUNK = 500_000


def _sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _zip_members(path):
    with zipfile.ZipFile(path) as zf:
        return [f"{path}::{i.filename}" for i in zf.infolist() if i.filename.lower().endswith(".csv")]


def raw_sources(root, name, raw=None):
    """CSV sources: files, folders (recursive), zip archives (their CSV
    members, read without extraction) and single members written
    'archive.zip::path/in/archive.csv'. Default: <root>/<name>/raw."""
    members = SPECS.get(name, {}).get("members")
    if not raw and members:                                  # only the listed archive members
        raw = [os.path.join(root, name, "raw", m) for m in members]
    paths = list(raw) if raw else [os.path.join(root, name, "raw")]
    out = []
    for p in paths:
        if "::" in p:                                    # one member of a zip archive
            archive, member = p.split("::", 1)
            if f"{archive}::{member}" not in _zip_members(archive):
                raise FileNotFoundError(p)
            out.append(p)
        elif os.path.isdir(p):
            for f in sorted(glob.glob(os.path.join(p, "**", "*"), recursive=True)):
                if f.lower().endswith(".csv"):
                    out.append(f)
                elif f.lower().endswith(".zip"):
                    out += _zip_members(f)
        elif p.lower().endswith(".zip") and os.path.isfile(p):
            out += _zip_members(p)
        elif os.path.isfile(p):
            out.append(p)
        else:
            raise FileNotFoundError(p)
    if not out:
        raise FileNotFoundError(f"no CSV files in {paths}")
    return out


def _chunks(src, chunksize):
    kw = dict(low_memory=False, encoding="utf-8", encoding_errors="replace")
    if "::" in src:
        archive, member = src.split("::", 1)
        with zipfile.ZipFile(archive) as zf, zf.open(member) as f:
            if chunksize:
                yield from pd.read_csv(f, chunksize=chunksize, **kw)
            else:
                yield pd.read_csv(f, **kw)
    elif chunksize:
        yield from pd.read_csv(src, chunksize=chunksize, **kw)
    else:
        yield pd.read_csv(src, **kw)


def _provenance(src, cache):
    """Size and SHA-256 of a source; for a zip member, the SHA-256 of the
    archive plus the CRC-32 of the member."""
    if "::" in src:
        archive, member = src.split("::", 1)
        if archive not in cache:
            cache[archive] = _sha256(archive)
        with zipfile.ZipFile(archive) as zf:
            info = zf.getinfo(member)
        return {"path": src, "bytes": info.file_size, "sha256": cache[archive],
                "sha256_of": "archive", "crc32": f"{info.CRC:08x}"}
    return {"path": src, "bytes": os.path.getsize(src), "sha256": _sha256(src), "sha256_of": "file"}


def _numeric_like(col, thresh=0.99):
    s = col.dropna()
    if s.empty or pd.api.types.is_numeric_dtype(s):
        return True
    return pd.to_numeric(s.astype(str).str.strip(), errors="coerce").notna().mean() >= thresh


def _codes(name, col):
    """Numeric view of a dropped column for the leakage audit: numbers as is,
    time columns as epoch seconds, anything else as category codes."""
    num = pd.to_numeric(col, errors="coerce")
    if num.notna().mean() >= 0.99:
        return num.to_numpy(np.float64)
    if "time" in name.lower():
        t = pd.to_datetime(col, errors="coerce", format="mixed")
        if t.notna().mean() >= 0.99:
            return (t - pd.Timestamp(0)).dt.total_seconds().to_numpy(np.float64)
    return pd.factorize(col.astype(str))[0].astype(np.float64)


def _unique_names(cols):
    """Stripped header names, made unique ('x', 'x' -> 'x', 'x.dup1')."""
    seen, out = {}, []
    for c in cols:
        c = _norm(c)
        out.append(c if c not in seen else f"{c}.dup{seen[c]}")
        seen[c] = seen.get(c, 0) + 1
    return out


def _canon_text(col):
    """Strip text; write number-like strings canonically ('0.0' -> '0')."""
    s_ = col.astype(str).str.strip()
    num = pd.to_numeric(s_, errors="coerce")
    return s_.where(num.isna(), num.map(lambda v: f"{v:g}"))


def _to_float64(df, stats):
    """Coerce to float64 (counting unparsable values); inf -> NaN."""
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            before = int(df[c].notna().sum())
            df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce")
            stats["unparsable_values"] += before - int(df[c].notna().sum())
    a = df.to_numpy(np.float64, copy=True)
    inf = np.isinf(a)
    stats["inf_values"] += int(inf.sum())
    a[inf] = np.nan
    nan = np.isnan(a)
    stats["nan_values"] += int(nan.sum())
    stats["rows_with_nan_or_inf"] += int(nan.any(axis=1).sum())
    return pd.DataFrame(a, columns=df.columns)


def _row_hash(X):
    return pd.util.hash_pandas_object(X, index=False).to_numpy()


def _load_chunked_text(sources, spec, p, log, meta):
    """Mixed dataset read in chunks with the same result as reading it whole: pass 1 finds the
    text columns and their values on all rows (one-hot if at most max_categories values, else
    dropped); pass 2 encodes every chunk with that fixed vocabulary. For files too large to hold
    as text in memory (5G-NIDD)."""
    lab, drop = spec["label"], spec["drop"]
    text_vals, cols_ref = {}, None
    for src in sources:                                          # pass 1: vocabulary
        for df in _chunks(src, spec.get("chunk", CHUNK)):
            df.columns = _unique_names(df.columns)
            cols_ref = cols_ref or list(df.columns)
            for c in df.columns:
                if c == lab or c in drop or (c not in text_vals and _numeric_like(df[c])):
                    continue
                text_vals.setdefault(c, set()).update(_canon_text(df[c]).fillna("nan").unique().tolist())
    onehot = {c: sorted(v) for c, v in text_vals.items() if len(v) <= p["max_categories"]}
    text_drop = {c: len(v) for c, v in text_vals.items() if len(v) > p["max_categories"]}
    stats = {k: 0 for k in ("rows_without_label", "unparsable_values", "inf_values",
                            "nan_values", "rows_with_nan_or_inf")}
    Xs, hs, labels, audit, files, cache, feats = [], [], [], {}, [], {}, None
    for src in sources:                                          # pass 2: encode chunk by chunk
        rows = 0
        for df in _chunks(src, spec.get("chunk", CHUNK)):
            df.columns = _unique_names(df.columns)
            if list(df.columns) != cols_ref:
                raise ValueError(f"column mismatch in {src}")
            rows += len(df)
            miss = df[lab].isna() | (df[lab].astype(str).str.strip() == "")
            stats["rows_without_label"] += int(miss.sum())
            df = df[~miss].reset_index(drop=True)
            labels.append(df.pop(lab).astype(str).str.strip())
            for c in drop:
                if c in df.columns:
                    audit.setdefault(c, []).append(df[c])
            df = df.drop(columns=[c for c in drop if c in df.columns] + list(text_drop))
            num = {}
            for c, vocab in onehot.items():
                vals = _canon_text(df.pop(c)).fillna("nan")
                for v in vocab:
                    num[f"{c}={v}"] = (vals == v).astype(np.float64)
            X = _to_float64(pd.concat([df, pd.DataFrame(num)], axis=1), stats)
            feats = list(X.columns)
            hs.append(_row_hash(X))
            Xs.append(X.to_numpy(np.float32))
        files.append({**_provenance(src, cache), "rows": rows})
        log(f"[{meta['dataset']}] read {src}: {rows:,} rows")
    meta["files"] = files
    meta["dropped_columns"] = {c: r for c, r in drop.items() if c in cols_ref}
    meta["absent_drop_columns"] = sorted(set(drop) - set(cols_ref))
    meta["one_hot_columns"] = onehot
    meta["vocabulary_mapping"] = {}
    meta["dropped_text_columns"] = {c: f"high-cardinality text ({n} values)" for c, n in text_drop.items()}
    meta.update(stats)
    X = np.concatenate(Xs)
    raw_lab = pd.concat(labels, ignore_index=True)
    audit = {c: _codes(c, pd.concat(v, ignore_index=True)) for c, v in audit.items()}
    n_raw = sum(f["rows"] for f in files)
    return X, feats, np.concatenate(hs), raw_lab, audit, n_raw, len(cols_ref) - 1


def _load(sources, spec, p, log, meta):
    """Read every source. Numeric datasets are read in chunks: each chunk is
    hashed in float64 (exact duplicate detection) and stored as float32.
    Mixed datasets (text columns) are read whole, so that text handling
    (one-hot or drop) is decided on all rows."""
    lab, drop = spec["label"], spec["drop"]
    if not spec["numeric"] and spec.get("chunked_text"):
        return _load_chunked_text(sources, spec, p, log, meta)
    stats = {k: 0 for k in ("rows_without_label", "unparsable_values", "inf_values",
                            "nan_values", "rows_with_nan_or_inf")}
    Xs, hs, labels, frames, audit, files, cache = [], [], [], [], {}, [], {}
    cols_ref = feats = None
    for src in sources:
        rows = 0
        for df in _chunks(src, CHUNK if spec["numeric"] else None):
            df.columns = _unique_names(df.columns)
            if cols_ref is None:
                cols_ref = list(df.columns)
            elif list(df.columns) != cols_ref:
                raise ValueError(f"column mismatch in {src}")
            rows += len(df)
            miss = df[lab].isna() | (df[lab].astype(str).str.strip() == "")
            stats["rows_without_label"] += int(miss.sum())
            df = df[~miss].reset_index(drop=True)
            labels.append(df.pop(lab).astype(str).str.strip())
            for c in drop:
                if c in df.columns:
                    audit.setdefault(c, []).append(df[c])
            df = df.drop(columns=[c for c in drop if c in df.columns])
            if spec["numeric"]:
                X = _to_float64(df, stats)
                feats = list(X.columns)
                hs.append(_row_hash(X))
                Xs.append(X.to_numpy(np.float32))
            else:
                frames.append(df)
        files.append({**_provenance(src, cache), "rows": rows})
        log(f"[{meta['dataset']}] read {src}: {rows:,} rows")
    meta["files"] = files
    meta["dropped_columns"] = {c: r for c, r in drop.items() if c in cols_ref}
    meta["absent_drop_columns"] = sorted(set(drop) - set(cols_ref))
    onehot, text_drop, vocab_log = {}, {}, {}
    if not spec["numeric"]:
        df = pd.concat(frames, ignore_index=True)
        del frames
        num = {}
        for c in list(df.columns):
            if _numeric_like(df[c]):
                continue
            vals = _canon_text(df.pop(c))
            allowed = spec.get("vocab", {}).get(c)
            if allowed is not None:
                outside = ~vals.isin(allowed)
                vocab_log[c] = {"allowed": allowed, "rows_mapped_to_other": int(outside.sum()),
                                "distinct_values_mapped": int(vals[outside].nunique())}
                vals = vals.where(~outside, "<other>")
            if vals.nunique() <= p["max_categories"]:
                onehot[c] = sorted(vals.unique().tolist())
                for v in onehot[c]:
                    num[f"{c}={v}"] = (vals == v).astype(np.float64)
            else:
                text_drop[c] = int(vals.nunique())
        X = _to_float64(pd.concat([df, pd.DataFrame(num)], axis=1), stats)
        feats = list(X.columns)
        hs.append(_row_hash(X))
        Xs.append(X.to_numpy(np.float32))
        del df, X
    meta["one_hot_columns"] = onehot
    meta["vocabulary_mapping"] = vocab_log
    meta["dropped_text_columns"] = {c: f"high-cardinality text ({n} values)" for c, n in text_drop.items()}
    meta.update(stats)
    X = np.concatenate(Xs)
    del Xs
    raw_lab = pd.concat(labels, ignore_index=True)
    audit = {c: _codes(c, pd.concat(v, ignore_index=True)) for c, v in audit.items()}
    n_raw = sum(f["rows"] for f in files)
    return X, feats, np.concatenate(hs), raw_lab, audit, n_raw, len(cols_ref) - 1


def cap_indices(y, max_rows, class_floor, rng):
    """Stratified cap: classes with <= class_floor rows kept whole, the others
    sampled with one common fraction so their relative frequencies hold."""
    if len(y) <= max_rows:
        return np.arange(len(y))
    classes, counts = np.unique(y, return_counts=True)
    small = counts <= class_floor
    budget = max_rows - counts[small].sum()
    frac = budget / counts[~small].sum()
    keep = []
    for k, n, sm in zip(classes, counts, small):
        idx = np.flatnonzero(y == k)
        keep.append(idx if sm else rng.choice(idx, size=max(1, int(round(n * frac))), replace=False))
    return np.sort(np.concatenate(keep))


def preprocess(name, root="data", log=print, raw=None, note=None, **params):
    p = {**DEFAULT_PARAMS, **params}
    spec = SPECS[name]
    t0 = time.time()
    steps = []
    meta = {"dataset": name, "title": spec["title"], "source_url": spec["source_url"],
            "expected_files": spec["files"], "reference": spec["reference"],
            "params": p, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "source_note": note}

    def step(s, rows, cols, note=""):
        steps.append({"step": s, "rows": int(rows), "cols": int(cols), "note": note})
        log(f"[{name}] {s}: {rows:,} rows x {cols} cols {note}")

    # 1-4 read, drop identifier/time/payload columns, numeric coercion
    sources = raw_sources(root, name, raw)
    if p["max_files"] and len(sources) > p["max_files"]:
        pick = np.random.default_rng(p["split_seed"]).choice(len(sources), p["max_files"], replace=False)
        sources = [sources[i] for i in sorted(pick)]
    X, feats, hx, raw_lab, audit_cols, n_raw, n_cols_raw = _load(sources, spec, p, log, meta)
    step("raw", n_raw, n_cols_raw)
    step("drop identifier/time/payload columns", len(X), n_cols_raw - len(meta["dropped_columns"]),
         f"({len(meta['dropped_columns'])} dropped, {meta['rows_without_label']:,} unlabelled rows)")
    step("numeric features", len(X), X.shape[1],
         f"(one-hot {len(meta['one_hot_columns'])}, text dropped {len(meta['dropped_text_columns'])}, "
         f"inf {meta['inf_values']:,})")

    # 2 labels: raw -> fine -> taxonomy class (unknown labels abort)
    meta["raw_label_counts"] = {k: int(v) for k, v in raw_lab.value_counts().items()}
    fine_of = {r: spec["fine"](r) for r in raw_lab.unique()}
    fine = raw_lab.map(fine_of)
    if p["taxonomy"] == "family":
        cls_of = {f_: spec["family"](f_) for f_ in fine.unique()}
        unknown = sorted(f_ for f_, c in cls_of.items() if c is None)
        if unknown:
            raise ValueError(f"labels without a taxonomy mapping: {unknown}")
    else:
        cls_of = {f_: f_ for f_ in fine.unique()}
    meta["mapping"] = {f_: cls_of[f_] for f_ in sorted(cls_of)}
    target = fine.map(cls_of)

    # 5 duplicates: exact rows, then identical features with conflicting labels
    hy = pd.util.hash_pandas_object(target, index=False).to_numpy()
    exact = pd.Series(hx ^ (hy * np.uint64(0x9E3779B97F4A7C15))).duplicated().to_numpy()
    keep = ~exact
    fl = pd.DataFrame({"h": hx[keep], "y": target.to_numpy()[keep]})
    n_lab = fl.groupby("h")["y"].nunique()
    conflict = np.zeros(len(X), bool)
    conflict[np.flatnonzero(keep)] = fl["h"].isin(n_lab.index[n_lab > 1]).to_numpy()
    keep &= ~conflict
    meta["duplicates_exact"] = int(exact.sum())
    meta["duplicates_conflicting_rows"] = int(conflict.sum())
    meta["exact_duplicates_by_class"] = {k: int(v) for k, v in target[exact].value_counts().items()}
    X, target, fine = X[keep], target[keep].reset_index(drop=True), fine[keep].reset_index(drop=True)
    audit_cols = {c: v[keep] for c, v in audit_cols.items()}
    step("deduplicate", len(X), X.shape[1],
         f"(exact {meta['duplicates_exact']:,}, conflicting {meta['duplicates_conflicting_rows']:,})")
    meta["fine_counts_after_dedup"] = {k: int(v) for k, v in fine.value_counts().items()}
    meta["class_counts_after_dedup"] = {k: int(v) for k, v in target.value_counts().items()}

    # 6 drop classes too small to split and evaluate
    counts = target.value_counts()
    small = counts[counts < p["min_class_count"]]
    meta["dropped_classes"] = {k: int(v) for k, v in small.items()}
    ok = ~target.isin(small.index).to_numpy()
    X, target = X[ok], target[ok].reset_index(drop=True)
    audit_cols = {c: v[ok] for c, v in audit_cols.items()}
    step("drop classes < min_class_count", len(X), X.shape[1], f"({sorted(small.index.tolist())})")

    # 7 stratified cap
    classes = sorted(target.unique(), key=lambda c: (c != "Benign", c))   # Benign first
    y = target.map({c: i for i, c in enumerate(classes)}).to_numpy(np.int64)
    idx = cap_indices(y, p["max_rows"], p["class_floor"], np.random.default_rng(p["split_seed"]))
    X, y = X[idx], y[idx]
    audit_cols = {c: v[idx] for c, v in audit_cols.items()}
    step("stratified cap", len(X), X.shape[1], f"(max_rows {p['max_rows']:,}, floor {p['class_floor']:,})")

    # 8 split before any fitting
    held = p["val_size"] + p["test_size"]
    tr, rest = train_test_split(np.arange(len(y)), test_size=held, stratify=y,
                                random_state=p["split_seed"])
    va, te = train_test_split(rest, test_size=p["test_size"] / held, stratify=y[rest],
                              random_state=p["split_seed"])

    # 9 fitted on train only: imputation, constant/duplicate columns, scaling
    X = X.astype(np.float64)
    with np.errstate(all="ignore"):
        med = np.nanmedian(X[tr], axis=0)
    med = np.where(np.isnan(med), 0.0, med)                  # all-NaN in train -> 0
    X = np.where(np.isnan(X), med, X)
    Xtr = X[tr]
    const = [j for j in range(X.shape[1]) if np.ptp(Xtr[:, j]) == 0]
    groups, dup = {}, {}
    for j in range(X.shape[1]):
        if j not in const:
            groups.setdefault(hashlib.sha1(Xtr[:, j].tobytes()).hexdigest(), []).append(j)
    for js in groups.values():
        for j in js[1:]:
            if np.array_equal(Xtr[:, j], Xtr[:, js[0]]):
                dup[j] = js[0]
    kept = [j for j in range(X.shape[1]) if j not in const and j not in dup]
    meta["train_constant_columns"] = [feats[j] for j in const]
    meta["train_duplicate_columns"] = {feats[j]: feats[o] for j, o in dup.items()}
    scaler = StandardScaler().fit(Xtr[:, kept])
    Z = scaler.transform(X[:, kept]).astype(np.float32)
    step("train-fitted cleaning + scaling", len(Z), Z.shape[1],
         f"(constant {len(const)}, duplicate cols {len(dup)})")
    meta["features"] = [feats[j] for j in kept]
    meta["classes"] = classes
    meta["imputation_median"] = {feats[j]: float(med[j]) for j in kept}
    meta["scaler"] = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    meta["class_counts"] = {
        s: {classes[k]: int(n) for k, n in zip(*np.unique(y[i], return_counts=True))}
        for s, i in (("cap", np.arange(len(y))), ("train", tr), ("val", va), ("test", te))}
    meta["steps"] = steps
    meta["seconds"] = round(time.time() - t0, 1)

    # 10 save
    out = os.path.join(root, name, "processed")
    os.makedirs(out, exist_ok=True)
    tax = p["taxonomy"]
    np.savez_compressed(os.path.join(out, f"{tax}.npz"),
                        X_train=Z[tr], y_train=y[tr], X_val=Z[va], y_val=y[va],
                        X_test=Z[te], y_test=y[te])
    if audit_cols:
        names = sorted(audit_cols)
        A = np.stack([audit_cols[c] for c in names], axis=1)
        np.savez_compressed(os.path.join(out, f"audit_{tax}.npz"), names=np.array(names),
                            X_train=A[tr], y_train=y[tr], X_val=A[va], y_val=y[va])
    with open(os.path.join(out, f"{tax}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"[{name}] saved {out}/{tax}.npz ({meta['seconds']} s)")
    return meta


FORBID_TEST_ENV = "MERIT_FL_FORBID_TEST"     # set by tuning: any test-split read raises


def assert_test_allowed(splits):
    if "test" in splits and os.environ.get(FORBID_TEST_ENV):
        raise RuntimeError("test split read attempted while tuning (validation only)")


def load_processed(root, name, taxonomy="family", splits=("train", "val", "test")):
    """Cached splits; only the requested splits are read from disk."""
    assert_test_allowed(splits)
    base = os.path.join(root, name, "processed", taxonomy)
    if not os.path.exists(base + ".npz"):
        raise FileNotFoundError(f"{base}.npz missing: run scripts/preprocess.py --dataset {name}")
    with open(base + ".json", encoding="utf-8") as f:
        meta = json.load(f)
    with np.load(base + ".npz") as z:
        data = {s: (z[f"X_{s}"], z[f"y_{s}"]) for s in splits}
    return data, meta
