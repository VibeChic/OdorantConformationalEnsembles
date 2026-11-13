#!/usr/bin/env python3
"""Create a Zenodo deposition, upload the QH-Odor-1821 bundle, optionally publish.

Environment
-----------
ZENODO_TOKEN      personal access token (deposit:write)
ZENODO_API_URL    default https://zenodo.org/api

    python scripts/upload_zenodo_data.py --dir dist/zenodo_data_QH-Odor-1821
    python scripts/upload_zenodo_data.py --dir ... --publish

Writes release_metadata.json (deposition id, data_doi, uploaded files).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm

META = {
    "upload_type": "dataset",
    "title": "QH-Odor-1821: complete 300 K GFN2-xTB conformational ensembles and multi-tier olfactory benchmark data",
    "description": (
        "Complete 300 K GFN2-xTB (--ohess) conformational ensembles "
        "(106,526 Phase II conformers at C_max=200 for 1,821 flexible odorants and "
        "55,895 Phase I conformers at C_max=20 for 4,809 odorants), ALPB implicit "
        "solvation tables (16,836 conformers), 60-d USRCAT pharmacophore entropy "
        "tables, topological and 3D dispersion observables, 21-arm neural "
        "representation benchmark metrics, and figure source data."
    ),
    "license": "cc-by-4.0",
    "keywords": ["conformational ensemble", "odour breadth", "molecular representation",
                 "polar-surface fluctuation", "rotatable-bond topology",
                 "pre-registered benchmark", "GFN2-xTB"],
    "prereserve_doi": True,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--api", default=os.environ.get("ZENODO_API_URL", "https://zenodo.org/api"))
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--meta-out", default="release_metadata.json")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("ZENODO_TOKEN is not set", file=sys.stderr)
        return 2
    h = {"Authorization": f"Bearer {token}"}

    root = Path(args.dir)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        print(f"no files under {root}", file=sys.stderr)
        return 2

    r = requests.post(f"{args.api}/deposit/depositions", json={}, headers=h, timeout=60)
    r.raise_for_status()
    dep = r.json()
    dep_id, bucket = dep["id"], dep["links"]["bucket"]
    doi = dep["metadata"]["prereserve_doi"]["doi"]
    print(f"deposition {dep_id}, reserved DOI {doi}")

    r = requests.put(f"{args.api}/deposit/depositions/{dep_id}", json={"metadata": META},
                     headers=h, timeout=60)
    r.raise_for_status()

    uploaded = []
    for p in tqdm(files, unit="file"):
        rel = str(p.relative_to(root))
        with open(p, "rb") as fh:
            resp = requests.put(f"{bucket}/{rel}", data=fh, headers=h, timeout=600)
        resp.raise_for_status()
        info = resp.json()
        uploaded.append({"file": rel, "bytes": p.stat().st_size,
                         "checksum": info.get("checksum"), "id": info.get("id")})
    print(f"uploaded {len(uploaded)} file(s)")

    if args.publish:
        r = requests.post(f"{args.api}/deposit/depositions/{dep_id}/actions/publish",
                          headers=h, timeout=120)
        r.raise_for_status()
        print("published")

    payload = {"data_doi": doi, "deposition_id": dep_id, "published": bool(args.publish),
               "files": uploaded}
    out = Path(args.meta_out)
    if out.exists():
        old = json.loads(out.read_text())
        old.update(payload)
        payload = old
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
