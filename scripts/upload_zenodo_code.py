#!/usr/bin/env python3
"""Archive a GitHub release tarball/zip as a Zenodo software deposition.

    git archive --format=zip --prefix=OdorantConformationalEnsembles-1.0.0/ \
        -o dist/OdorantConformationalEnsembles-v1.0.0.zip v1.0.0
    python scripts/upload_zenodo_code.py --zip dist/OdorantConformationalEnsembles-v1.0.0.zip \
        --repo https://github.com/VibeChic/OdorantConformationalEnsembles \
        --data-doi 10.5281/zenodo.XXXXXXX [--publish]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

TITLE = ("Code for: Odorant conformational ensembles encode topology, molecular size "
         "and polar-surface coherence, not shape")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data-doi", default=None)
    ap.add_argument("--api", default=os.environ.get("ZENODO_API_URL", "https://zenodo.org/api"))
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--meta-out", default="release_metadata.json")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("ZENODO_TOKEN is not set", file=sys.stderr)
        return 2
    h = {"Authorization": f"Bearer {token}"}

    related = [{"relation": "isSupplementTo", "identifier": f"{args.repo}/tree/v1.0.0",
                "scheme": "url", "resource_type": "software"}]
    if args.data_doi:
        related.append({"relation": "isSupplementTo",
                        "identifier": f"https://doi.org/{args.data_doi}",
                        "scheme": "doi", "resource_type": "dataset"})

    meta = {"upload_type": "software", "title": TITLE, "license": "MIT",
            "version": "1.0.0", "access_right": "open",
            "creators": [{"name": "The OdorantConformationalEnsembles authors"}],
            "related_identifiers": related}

    r = requests.post(f"{args.api}/deposit/depositions", json={}, headers=h, timeout=60)
    r.raise_for_status()
    dep = r.json()
    dep_id, bucket = dep["id"], dep["links"]["bucket"]
    doi = dep["metadata"]["prereserve_doi"]["doi"]
    print(f"deposition {dep_id}, reserved DOI {doi}")

    requests.put(f"{args.api}/deposit/depositions/{dep_id}", json={"metadata": meta},
                 headers=h, timeout=60).raise_for_status()

    z = Path(args.zip)
    with open(z, "rb") as fh:
        resp = requests.put(f"{bucket}/{z.name}", data=fh, headers=h, timeout=600)
    resp.raise_for_status()
    print("uploaded", z.name, resp.json().get("checksum"))

    if args.publish:
        requests.post(f"{args.api}/deposit/depositions/{dep_id}/actions/publish",
                      headers=h, timeout=120).raise_for_status()
        print("published")

    out = Path(args.meta_out)
    payload = json.loads(out.read_text()) if out.exists() else {}
    payload.update({"code_doi": doi, "code_deposition_id": dep_id,
                    "code_published": bool(args.publish),
                    "repo": args.repo, "code_file": z.name})
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
