#!/usr/bin/env python3
"""Fail if a candidate release contains the proprietary 138-descriptor matrix
or commercial catalogue text.

Usage
-----
    python scripts/check_proprietary_leakage.py --root dist/zenodo_data_QH-Odor-1821
    python scripts/check_proprietary_leakage.py --root cache        # should FAIL (self-test)

Exit code 0 only if no file trips a rule.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BANNED_TEXT = ["goodscents", "good scents", "leffingwell", "the good scents company"]
N_LABELS = 138


def scan_table(path: Path):
    reasons = []
    try:
        import pandas as pd
    except Exception:
        return ["pandas unavailable"]
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, nrows=2000)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            return []
    except Exception as exc:
        return [f"unreadable: {type(exc).__name__}"]
    if df.shape[1] == N_LABELS:
        reasons.append(f"{N_LABELS} columns (full descriptor matrix)")
    binary_like = 0
    for col in df.columns[:400]:
        s = df[col]
        if str(s.dtype) in ("bool", "int64", "int32", "float64"):
            vals = set(s.dropna().unique()[:5])
            if vals and vals <= {0, 1, 0.0, 1.0}:
                binary_like += 1
    if binary_like >= N_LABELS * 0.9:
        reasons.append(f"{binary_like} binary-like columns")
    return reasons


def scan_text(path: Path):
    reasons = []
    try:
        blob = path.read_text(errors="ignore").lower()[:2_000_000]
    except Exception:
        return reasons
    for token in BANNED_TEXT:
        if token in blob:
            reasons.append(f"banned string: {token}")
    return reasons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"root not found: {root}")
        return 2

    findings = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in {".csv", ".parquet"}:
            reasons = scan_table(p)
        elif p.suffix.lower() in {".json", ".txt", ".md", ".tsv"}:
            reasons = scan_text(p)
        else:
            continue
        if reasons:
            findings.append({"file": str(p), "reasons": reasons})

    for f in findings:
        print(f"[FAIL] {f['file']}: {'; '.join(f['reasons'])}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"root": str(root), "findings": findings}, indent=2))

    print(f"\n{len(findings)} file(s) tripped a rule under {root}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
