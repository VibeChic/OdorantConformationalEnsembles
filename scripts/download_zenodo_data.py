#!/usr/bin/env python3
"""Download the QH-Odor-1821 bundle from Zenodo and verify checksums.

    python scripts/download_zenodo_data.py --record 1234567 --out data

`--record` accepts a numeric id or a DOI suffix (the part after zenodo.).
Metadata and file links come from the public API, so no token is required.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests

API = "https://zenodo.org/api"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--api", default=API)
    ap.add_argument("--include", nargs="*", default=None,
                    help="only download keys containing one of these substrings "
                         "(e.g. --include cohorts probes metrics)")
    args = ap.parse_args()

    rec = str(args.record).rstrip("/").split(".")[-1]
    r = requests.get(f"{args.api}/records/{rec}", timeout=60)
    r.raise_for_status()
    meta = r.json()
    files = meta.get("files", [])
    if args.include:
        files = [f for f in files if any(s in f.get("key", "") for s in args.include)]
    total = sum(f.get("size", 0) for f in files)
    print(f"record {rec} (id {meta.get('id')}): {len(files)} file(s), "
          f"{total/1e6:.1f} MB, {meta.get('metadata', {}).get('title', '')[:70]}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        key = f.get("key") or f["links"]["self"].rsplit("/", 1)[-1]
        url = f["links"]["self"]
        name = key.replace("__", "/")          # restore the logical tree
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and f.get("checksum", "").startswith("md5:"):
            if md5(dest) == f["checksum"].split(":", 1)[1]:
                print(f"  skip (verified) {name}", flush=True)
                continue
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(1 << 22):
                    fh.write(chunk)
            tmp.replace(dest)
        ck = f.get("checksum", "")
        if ck.startswith("md5:"):
            ok = md5(dest) == ck.split(":", 1)[1]
            print(f"  {'ok ' if ok else 'BAD'} {name}", flush=True)
            if not ok:
                return 1
    print(f"done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
