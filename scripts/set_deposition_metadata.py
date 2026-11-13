#!/usr/bin/env python3
"""Set creators / related identifiers on an existing Zenodo draft.

    python scripts/set_deposition_metadata.py --deposition-id 24187223 \
        --creators creators.json

creators.json:
[
  {"name": "Surname, Given", "affiliation": "Institute", "orcid": "0000-0002-...."}
]
A record cannot be published without at least one creator.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposition-id", type=int, required=True)
    ap.add_argument("--creators", required=True)
    ap.add_argument("--api", default=os.environ.get("ZENODO_API_URL", "https://zenodo.org/api"))
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("ZENODO_TOKEN is not set", file=sys.stderr)
        return 2
    h = {"Authorization": f"Bearer {token}"}

    creators = json.loads(Path(args.creators).read_text())
    if not creators:
        print("creators list is empty", file=sys.stderr)
        return 2
    for i, c in enumerate(creators, 1):
        if not c.get("name"):
            print(f"creator {i} has no name", file=sys.stderr)
            return 2
        if c.get("orcid") and not c["orcid"].startswith("0000-"):
            print(f"creator {i} ORCID looks malformed: {c['orcid']}", file=sys.stderr)

    dep = requests.get(f"{args.api}/deposit/depositions/{args.deposition_id}", headers=h, timeout=60).json()
    meta = dep["metadata"]
    payload = {k: meta.get(k) for k in
               ["title", "upload_type", "description", "license", "keywords",
                "related_identifiers", "language", "access_right", "version", "notes"]}
    payload = {k: v for k, v in payload.items() if v is not None}
    payload["creators"] = creators

    r = requests.put(f"{args.api}/deposit/depositions/{args.deposition_id}",
                     json={"metadata": payload}, headers=h, timeout=60)
    print("update:", r.status_code)
    if r.status_code >= 400:
        print(r.text[:600])
        return 1
    print("creators:", [c["name"] for c in r.json()["metadata"]["creators"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
