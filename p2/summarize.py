#!/usr/bin/env python3
"""Summarise a result tree and run seed-level paired sign-flip tests.

  python p2/summarize.py --root results/s1_mil --baseline results/r1/gin \
      --title "Rung 3" --out reports/10_rung3.md

Every arm must have a metrics.json per seed. Arms with fewer seeds than the
baseline are compared on the intersection of seeds, and the report says so.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p2.stats import sign_flip_test  # noqa: E402


def collect(root):
    arms = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "seed*", "metrics.json"))):
        d = json.load(open(f))
        arm = os.path.basename(os.path.dirname(os.path.dirname(f)))
        seed = int(os.path.basename(os.path.dirname(f)).replace("seed", ""))
        arms.setdefault(arm, {})[seed] = {
            "aucpr": d["test_macro_aucpr"], "micro": d["test_micro_auroc"],
            "epochs": len(d.get("history", [])), "params": d.get("n_params"),
            "minutes": d.get("train_min")}
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--title", default="summary")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arms = collect(args.root)
    base = collect(args.baseline) if args.baseline else {}
    base_arm = list(base)[0] if base else None

    lines = [f"# {args.title}", "",
             "由 `p2/summarize.py` 生成；数字直接读 `metrics.json`，未手工录入。", "",
             "| arm | n seeds | macro-AUCPR | micro-AUROC | epochs | params | min/run |",
             "|---|---|---|---|---|---|---|"]
    order = sorted(arms, key=lambda a: -np.mean([v["aucpr"] for v in arms[a].values()]))
    for a in order:
        v = arms[a]
        au = [x["aucpr"] for x in v.values()]
        mi = [x["micro"] for x in v.values()]
        ep = [x["epochs"] for x in v.values()]
        pa = next((x["params"] for x in v.values() if x["params"]), None)
        mn = [x["minutes"] for x in v.values() if x["minutes"]]
        lines.append(f"| {a} | {len(au)} | {np.mean(au):.4f} ± {np.std(au, ddof=1):.4f} | "
                     f"{np.mean(mi):.4f} ± {np.std(mi, ddof=1):.4f} | "
                     f"{np.mean(ep):.0f} | {pa:,} | {np.mean(mn):.1f} |"
                     if pa and mn else
                     f"| {a} | {len(au)} | {np.mean(au):.4f} ± {np.std(au, ddof=1):.4f} | "
                     f"{np.mean(mi):.4f} ± {np.std(mi, ddof=1):.4f} | {np.mean(ep):.0f} | — | — |")

    if base_arm:
        b = base[base_arm]
        lines += ["", f"## Paired vs `{base_arm}` (common seeds only)", "",
                  "| arm | common seeds | mean Δ | SD | sign-flip p | verdict |",
                  "|---|---|---|---|---|---|"]
        for a in order:
            common = sorted(set(arms[a]) & set(b))
            if not common:
                continue
            d = [arms[a][s]["aucpr"] - b[s]["aucpr"] for s in common]
            t = sign_flip_test(d)
            verdict = ("A > B" if t["mean"] > 0 and t["p"] < 0.05 else
                       "B > A" if t["mean"] < 0 and t["p"] < 0.05 else "not significant")
            lines.append(f"| {a} | {common} | {t['mean']:+.4f} | {t['sd']:.4f} | "
                         f"{t['p']:.3f} | {verdict} |")
        lines += ["", "> 判定口径（预注册）：双侧符号翻转置换，α = 0.05，单位是 seed。"
                  "分子级置信区间另算，不能与这里混用。"]

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
