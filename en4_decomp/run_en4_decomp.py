#!/usr/bin/env python3
"""Run the EN4 size-vs-shape decomposition (2 arms x 10 seeds).

Each seed runs in its own subprocess with the arm injected into p2.train_s2.ARMS at
runtime, so the original training script is untouched. Outputs go to
  <this dir>/results/<arm>/seed<N>/
and are resumable: finished seeds are skipped.

Run from anywhere:
  python run_en4_decomp.py
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "paper2"))
ARMS = {
    "en4_size": ["Rg_mean", "Rg_sd"],
    "en4_shape": ["NPR1_mean", "NPR2_mean", "NPR1_sd", "NPR2_sd"],
}
CHILD = r"""
import sys, json
sys.path.insert(0, {proj!r})
import p2.train_s2 as t
t.ARMS[{arm!r}] = {cols!r}
sys.argv = ["train_s2", "--arm", {arm!r}, "--seed", str({seed!r}), "--out", {out!r}]
t.main()
"""

ap = argparse.ArgumentParser()
ap.add_argument("--max-runs", type=int, default=100)
ap.add_argument("--parallel", type=int, default=1)
args = ap.parse_args()

os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
todo = []
for arm, cols in ARMS.items():
    for seed in range(10):
        out = os.path.join(HERE, "results", arm, f"seed{seed}")
        if os.path.exists(os.path.join(out, "metrics.json")):
            continue
        todo.append((arm, cols, seed, out))
print(f"{len(todo)} runs pending", flush=True)
todo = todo[: args.max_runs]

t0 = time.time()
running = []
for i, (arm, cols, seed, out) in enumerate(todo, 1):
    os.makedirs(out, exist_ok=True)
    code = CHILD.format(proj=PROJ, arm=arm, cols=cols, seed=seed, out=out)
    log = os.path.join(HERE, "logs", f"{arm}_seed{seed}.log")
    fh = open(log, "w")
    pr = subprocess.Popen([sys.executable, "-c", code], stdout=fh, stderr=subprocess.STDOUT,
                          cwd=PROJ)
    running.append((pr, fh, arm, seed))
    while len(running) >= args.parallel:
        pr0, fh0, a0, s0 = running[0]
        rc = pr0.wait()
        fh0.close()
        running.pop(0)
        el = (time.time() - t0) / 60
        print(f"[done] {'ok ' if rc == 0 else 'FAIL'} {a0} seed{s0}  ({el:.1f} min elapsed)",
              flush=True)
for pr0, fh0, a0, s0 in running:
    rc = pr0.wait()
    fh0.close()
    print(f"[done] {'ok ' if rc == 0 else 'FAIL'} {a0} seed{s0}  ({(time.time()-t0)/60:.1f} min)",
          flush=True)

# compact summary of what was produced
rows = []
for arm in ARMS:
    for seed in range(10):
        f = os.path.join(HERE, "results", arm, f"seed{seed}", "metrics.json")
        if os.path.exists(f):
            m = json.load(open(f))
            rows.append(dict(arm=arm, seed=seed, aucpr=m["test_macro_aucpr"],
                             n_params=m["n_params"], best_epoch=m["best_epoch"]))
json.dump(rows, open(os.path.join(HERE, "runs.json"), "w"), indent=1)
print(f"done: {len(rows)}/20 metrics.json", flush=True)
