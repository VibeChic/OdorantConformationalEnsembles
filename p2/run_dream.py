#!/usr/bin/env python3
"""Driver for the DREAM grid (4 arms x 10 seeds), resumable and cell-capped.

  python p2/run_dream.py --workers 4 --cells 8
  python p2/run_dream.py --only en6 --cells 4

A cell = one (arm, seed) run covering all 5 folds (~2.6 min alone).
"""
import argparse
import os
import subprocess
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ["r1", "en1", "en6", "morgan"]


def pending(only=""):
    todo = []
    for arm in ARMS:
        if only and arm != only:
            continue
        for s in range(10):
            out = os.path.join(PROJ, "results", "s3_tiers", "dream", arm, f"seed{s}")
            if not os.path.exists(os.path.join(out, "metrics.json")):
                todo.append((arm, s, out))
    return todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cells", type=int, default=8)
    ap.add_argument("--only", default="")
    ap.add_argument("--minutes", type=float, default=9.0)
    args = ap.parse_args()

    todo = pending(args.only)
    print(f"[dream] {len(todo)} pending", flush=True)
    t0 = time.time()
    running, done, launched = [], 0, 0
    while running or (todo and launched < args.cells and time.time() - t0 < args.minutes * 60):
        while todo and len(running) < args.workers and launched < args.cells \
                and time.time() - t0 < args.minutes * 60:
            arm, s, out = todo.pop(0)
            os.makedirs(out, exist_ok=True)
            log = open(os.path.join(out, "stdout.log"), "w")
            p = subprocess.Popen([sys.executable, os.path.join(PROJ, "p2", "train_dream.py"),
                                  "--arm", arm, "--seed", str(s)],
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
                    print(f"[ok] {arm} seed{s} ({time.time()-t0:.0f}s)", flush=True)
                else:
                    print(f"[FAIL] {arm} seed{s}", flush=True)
    print(f"[dream] {done} done, {len(pending(args.only))} left overall")


if __name__ == "__main__":
    main()
