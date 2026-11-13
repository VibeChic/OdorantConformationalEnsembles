#!/usr/bin/env python3
"""Run the T4/T5 grid with N workers, skipping cells that already have metrics.json.

The GPU (11 GB) comfortably holds 4 concurrent runs of the 1.2 M-parameter MIL
model; measured throughput is ~2x at 2 workers, so 4 is the practical default.

Usage
  python p2/run_grid.py --task all --workers 4 --minutes 9
  python p2/run_grid.py --task gauss --workers 3
"""
import argparse
import os
import subprocess
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SCRIPT = os.path.join(PROJ, "p2", "train_r3_variants.py")

TEMP_TAUS = ["10", "100", "600", "1500", "inf"]
GAUSS_SIGMAS = [("005", "0.05"), ("015", "0.15"), ("030", "0.30"), ("rmsd", "rmsd")]


def cells(task):
    out = []
    if task in ("temp", "all"):
        for t in TEMP_TAUS:
            for s in range(10):
                out.append((["--mode", "temp", "--tau", t],
                            f"results/s1_mil/temp_{t}/seed{s}"))
    if task in ("gauss", "all"):
        for tag, sig in GAUSS_SIGMAS:
            for s in range(10):
                out.append((["--mode", "gauss", "--sigma", sig],
                            f"results/s1_mil/gauss_{tag}/seed{s}"))
    return out


def pending(task):
    todo = []
    for args, out in cells(task):
        s = int(out.rsplit("seed", 1)[1])
        if not os.path.exists(os.path.join(PROJ, out, "metrics.json")):
            todo.append((args + ["--seed", str(s), "--out", out], out))
    return todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["temp", "gauss", "all"])
    ap.add_argument("--only", default="", help="substring filter on the output path")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--minutes", type=float, default=9.0)
    ap.add_argument("--cells", type=int, default=0, help="launch at most N cells (0 = all)")
    args = ap.parse_args()

    todo = pending(args.task)
    if args.only:
        todo = [t for t in todo if args.only in t[1]]
    print(f"[grid] {len(todo)} cells pending, {args.workers} workers, "
          f"budget {args.minutes} min", flush=True)
    t0 = time.time()
    running, done, failed, launched = [], 0, [], 0
    while todo or running:
        # budget only stops *launching* new cells; in-flight ones are always
        # collected so a call never leaves orphaned processes behind
        while (todo and len(running) < args.workers
               and (time.time() - t0) < args.minutes * 60
               and (args.cells == 0 or launched < args.cells)):
            cmd, out = todo.pop(0)
            os.makedirs(os.path.join(PROJ, out), exist_ok=True)
            log = open(os.path.join(PROJ, out, "stdout.log"), "w")
            launched += 1
            p = subprocess.Popen([PY, SCRIPT] + cmd, cwd=PROJ,
                                 stdout=log, stderr=subprocess.STDOUT)
            running.append((p, out, log))
        time.sleep(2)
        for item in list(running):
            p, out, log = item
            if p.poll() is not None:
                log.close()
                running.remove(item)
                if p.returncode == 0:
                    done += 1
                    print(f"[ok] {out}  ({time.time()-t0:.0f}s)", flush=True)
                else:
                    failed.append(out)
                    print(f"[FAIL {p.returncode}] {out}", flush=True)
        if not running:
            hit_cap = args.cells and launched >= args.cells
            hit_time = (time.time() - t0) >= args.minutes * 60
            if hit_cap or hit_time or not todo:
                break
        left = len(todo) + len(running)
        print(f"  … {done} done, {left} left, {time.time()-t0:.0f}s", flush=True)
    print(f"[grid] finished {done}, failed {len(failed)}, "
          f"remaining {len(pending(args.task))}")


if __name__ == "__main__":
    main()
