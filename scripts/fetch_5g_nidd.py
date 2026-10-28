#!/usr/bin/env python3
"""Download the flow-level files of 5G-NIDD (University of Oulu; CC BY 4.0;
DOI 10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361) from the Fairdata (Etsin) mirror into
data/5g_nidd/raw/, verifying each file's SHA-256 against the Metax record. The packet captures
(BS1.zip, BS2.zip, ..., several GB) are not needed and are not downloaded.

Provenance is written to data/5g_nidd/raw/SOURCE.json.

python scripts/fetch_5g_nidd.py
"""
import hashlib
import json
import os
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "data", "5g_nidd", "raw")
DATASET = "9d13ef28-2ca7-44b0-9950-225359afac65"
DOI = "10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361"
FILES = ("/README.pdf", "/Combined.zip", "/BS1_each_attack_csv.zip", "/BS2_each_attack_csv.zip")
UA = {"User-Agent": "merit-fl-data/1.0 (research dataset download)"}


def get(url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = {f["file_path"]: f for f in json.loads(get(f"https://metax.fairdata.fi/rest/v2/datasets/{DATASET}/files"))}
    record = {"dataset": DATASET, "doi": DOI, "license": "CC BY 4.0", "source": "https://etsin.fairdata.fi/dataset/" + DATASET,
              "downloaded": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": []}
    for path in FILES:
        m = meta[path]
        dest = os.path.join(OUT, path.lstrip("/"))
        if not (os.path.exists(dest) and hashlib.sha256(open(dest, "rb").read()).hexdigest() == m["checksum"]["value"]):
            body = json.dumps({"cr_id": DATASET, "file": path}).encode()
            url = json.loads(get("https://etsin.fairdata.fi/api/download/authorize", body,
                                 {"Content-Type": "application/json"}))["url"]
            blob = get(url)
            digest = hashlib.sha256(blob).hexdigest()
            if digest != m["checksum"]["value"] or len(blob) != m["byte_size"]:
                raise SystemExit(f"{path}: checksum or size mismatch ({digest}, {len(blob)} bytes)")
            open(dest, "wb").write(blob)
        record["files"].append({"path": path, "bytes": m["byte_size"], "sha256": m["checksum"]["value"]})
        print(f"ok {path} ({m['byte_size'] / 1e6:.1f} MB, sha256 verified)", flush=True)
    json.dump(record, open(os.path.join(OUT, "SOURCE.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
