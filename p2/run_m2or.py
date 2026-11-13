#!/usr/bin/env python3
"""Driver for the M2OR grid (arms A/B/C x seeds x split), resumable.

  python p2/run_m2or.py --split receptor --workers 4 --cells 6
  python p2/run_m2or.py --split ligand --arms A B --cells 4

A cell = one (split, arm, seed) run (A/C ~1.2 min, B ~2.5 min alone).
"""
import argparse
import os
import subprocess
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pending(split, arms):
    todo = []
    for arm in arms:
        for s in range(10):
            d = os.path.join(PROJ, "results", "s3_tiers", "m2or", split, arm, f"seed{s}")
            if not os.path.exists(os.path.join(d, "metrics.json")):
                todo.append((arm, s, d))
    return todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="receptor", choices=["receptor", "ligand"])
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cells", type=int, default=6)
    ap.add_argument("--minutes", type=float, default=9.0)
    args = ap.parse_args()

    todo = pending(args.split, args.arms)
    print(f"[m2or] {len(todo)} pending for {args.split}", flush=True)
    t0 = time.time()
    running, done, launched = [], 0, 0
    while running or (todo and launched < args.cells
                      and time.time() - t0 < args.minutes * 60):
        while todo and len(running) < args.workers and launched < args.cells \
                and time.time() - t0 < args.minutes * 60:
            arm, s, d = todo.pop(0)
            os.makedirs(d, exist_ok=True)
            log = open(os.path.join(d, "stdout.log"), "w")
            p = subprocess.Popen([sys.executable, os.path.join(PROJ, "p2", "train_m2or.py"),
                                  "--arm", arm, "--split", args.split, "--seed", str(s)],
                                 cwd=PROJ, stdout=log, stderr=subprocess.STDOUT)
            running.append((p, arm, s, log))
            launched += 1
        time.sleep(2)
        for item in list(running):
            p, arm, s, log = item
            if p.poll() is not None:
                log.close()
                running.remove(item)
                if p.returncode == 0:
                    done += 1
                    print(f"[ok] {args.split} {arm} seed{s} ({time.time()-t0:.0f}s)", flush=True)
                else:
                    print(f"[FAIL] {args.split} {arm} seed{s}", flush=True)
    print(f"[m2or] {done} done, {len(pending(args.split, args.arms))} left in {args.split}")


if __name__ == "__main__":
    main()
