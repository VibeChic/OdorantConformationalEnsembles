#!/usr/bin/env python3
"""T0 -- freeze the v3.0 baseline before any new run touches the tree.

Records, in one json:

  * sha256 of the five inputs the audit depends on (handoff doc section 3.4)
  * the audit exit code and its full stdout (so a later FAIL can be diffed)
  * a code_sha256 for every source file that can change a number, plus a
    single tree hash (there is no git repository here)
  * the per-arm seed counts actually present on disk

Usage:
  python p2/freeze_baseline.py            # writes results/_freeze_<date>.json
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(PROJ)

INPUTS = {
    "curated_csv": os.path.join(REPO, "data", "curated_GS_LF_merged_4983.csv"),
    "manifest": os.path.join(REPO, "data", "3d", "run1", "run1_gnn_manifest.csv"),
    "stereo_map": os.path.join(PROJ, "cache", "stereo_map.csv"),
    "exp2_scalars": os.path.join(PROJ, "cache", "exp2_scalars.csv"),
    "summary_10seed": os.path.join(PROJ, "results", "summary_10seed.json"),
}

CODE_GLOBS = ["p2", "src/data", "src/models", "src/utils"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_files():
    out = []
    for g in CODE_GLOBS:
        root = os.path.join(REPO if not g.startswith("p2") else PROJ, g)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for fn in filenames:
                if fn.endswith(".py"):
                    out.append(os.path.join(dirpath, fn))
    return sorted(out)


def seed_counts():
    counts = {}
    res = os.path.join(PROJ, "results")
    for dirpath, dirnames, filenames in os.walk(res):
        if "metrics.json" in filenames:
            rel = os.path.relpath(dirpath, res)
            arm = os.path.dirname(rel)
            counts[arm] = counts.get(arm, 0) + 1
    return dict(sorted(counts.items()))


def main():
    files = code_files()
    code = {os.path.relpath(p, REPO): sha256(p) for p in files
            if "paper2/p2" in p or "/src/" in p}
    tree = hashlib.sha256(
        "".join(f"{k}:{v}\n" for k, v in sorted(code.items())).encode()).hexdigest()

    proc = subprocess.run([sys.executable, os.path.join(PROJ, "p2", "audit.py")],
                          capture_output=True, text=True, cwd=PROJ)

    out = {
        "frozen_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "manuscript": "Paper2_manuscript_v3_flexibility.md",
        "inputs": {k: sha256(v) for k, v in INPUTS.items()},
        "inputs_short": {k: sha256(v)[:16] for k, v in INPUTS.items()},
        "audit": {"exit_code": proc.returncode,
                  "verdict": "PASS" if proc.returncode == 0 else "FAIL",
                  "stdout_sha256": hashlib.sha256(proc.stdout.encode()).hexdigest(),
                  "stdout_tail": proc.stdout.strip().splitlines()[-1]},
        "code_tree_sha256": tree,
        "code_sha256": code,
        "runs_per_arm": seed_counts(),
    }
    p = os.path.join(PROJ, "results", "_freeze_2026-10-26.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("audit", "inputs_short", "code_tree_sha256")},
                     indent=1))
    print(f"[frozen] {p}")
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
