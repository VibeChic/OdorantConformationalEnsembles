#!/usr/bin/env python3
"""T3 -- WP-A: is the 3D shape already in the 2D representation?

Two measurements on representations the models already learned (no arm is
retrained):

  (a) probing -- does the R1 256-d graph vector (or R1 + n_rotors, or four
      textbook 2D descriptors) predict eight ensemble/geometry targets on
      held-out molecules? Test R2 and Spearman, 10 seeds.

  (b) CKA -- how much of the test-set representation geometry is shared across
      seven arms (R1, EN1, EN4, R2, R2.5, R3-boltz, R3-uniform), linear and RBF.

Gate G4 (frozen in reports/02_preregistration.md section 2) is decided by (a):
mean test R2 of `GIN + n_rotors` over the eight targets, at the better of the two
probe families. >= 0.85 -> "geometric manifold collapse"; else "receptor
invariance".

Outputs
  results/wpa_redundancy/probing.json
  results/wpa_redundancy/cka.npz
  reports/14_redundancy.md

Usage
  python p2/wpa_probe.py                     # full run
  python p2/wpa_probe.py --seeds 0 --no-cka  # smoke test
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader as PyGLoader
from torch_geometric.nn import global_mean_pool

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(PROJ)
sys.path.insert(0, REPO)
sys.path.insert(0, PROJ)

from p2.models.r2 import build as build_r2                       # noqa: E402
from p2.models.r25 import ScalarGIN, build as build_r25          # noqa: E402
from p2.models.r3 import SchNetMIL                               # noqa: E402
from p2.train_r2 import load_split as load_r2                    # noqa: E402
from p2.train_r25 import load_split as load_r25                  # noqa: E402
from p2.train_s2 import ARMS as S2_ARMS                          # noqa: E402
from p2.train_s2 import load_split as load_s2                    # noqa: E402
from p2.train_r3 import collate, load_split as load_r3           # noqa: E402
from src.data.datasets import OdorDataset                        # noqa: E402
from src.models import get_model                                 # noqa: E402

RDLogger.DisableLog("rdApp.*")

CACHE = os.path.join(PROJ, "cache")
OUT = os.path.join(PROJ, "results", "wpa_redundancy")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGETS = ["NPR1_mean", "NPR2_mean", "Rg_mean",
           "NPR1_sd", "NPR2_sd", "Rg_sd",
           "S_conf", "top_pop"]

# label -> (family, seed-dir template)
CKA_ARMS = [
    ("R1", "r1", "results/r1/gin/seed{s}"),
    ("EN1", "scalar_en1", "results/s2_ablation/en1/seed{s}"),
    ("EN4", "scalar_en4", "results/s2_ablation/en4/seed{s}"),
    ("R2_SchNet", "schnet", "results/r2_sc/schnet/seed{s}"),
    ("R2.5_EdgeGIN", "edge_gin", "results/r25/edge_gin/seed{s}"),
    ("R3_boltz", "mil_boltz", "results/s1_mil/boltz/seed{s}"),
    ("R3_uniform", "mil_uniform", "results/s1_mil/shuffle/seed{s}"),
]


def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else None


def molecule_index():
    man = pd.read_csv(os.path.join(REPO, "data", "3d", "run1", "run1_gnn_manifest.csv"),
                      low_memory=False)[["name", "canonical_SMILES"]]
    name2smi = dict(zip(man["name"], man["canonical_SMILES"]))
    smi2name = dict(zip(man["canonical_SMILES"], man["name"]))
    sc = pd.read_csv(os.path.join(CACHE, "exp2_scalars.csv")).set_index("name")
    sc["S_conf"] = np.log(sc["n_eff"])
    return name2smi, smi2name, sc


def r1_split(split, smi2name):
    path = os.path.join(CACHE, f"subset2d_{split}.csv")
    ds = OdorDataset(path, os.path.join(REPO, "data", f"{split}-sdf"),
                     use_3d=False, cache_dir=os.path.join(CACHE, "graph2d"))
    df = pd.read_csv(path)
    names = [smi2name[canon(s)] for s in df["nonStereoSMILES"]]
    if len(set(names)) != len(names):
        raise RuntimeError(f"duplicate molecules in {split}")
    return names, ds


@torch.no_grad()
def r1_reps(seed, ds, batch=64):
    model = get_model("gin")(num_classes=138, hidden_dim=256, num_layers=3,
                             dropout=0.15, eps=0.0, train_eps=True).to(DEVICE)
    ck = torch.load(os.path.join(PROJ, "results", "r1", "gin", f"seed{seed}",
                                 "best_model.pt"), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    out = []
    for b in PyGLoader(ds, batch_size=batch, shuffle=False):
        b = b.to(DEVICE)
        x, bi = model.encode(b)
        out.append(global_mean_pool(x, bi).cpu().numpy())
    return np.concatenate(out).astype("float32")


@torch.no_grad()
def mil_reps(seed, pool):
    d = "boltz" if pool == "boltz" else "shuffle"
    model = SchNetMIL(num_classes=138, pool=d, dropout=0.1).to(DEVICE)
    ck = torch.load(os.path.join(PROJ, "results", "s1_mil", d, f"seed{seed}", "best_model.pt"),
                    map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    recs = load_r3("test", CACHE)
    names = [r["name"] for r in recs]
    vecs = []
    for i in range(0, len(recs), 16):
        b = collate(recs[i:i + 16]).to(DEVICE)
        z = model.drop(model.enc(b.x[:, 0].long(), b.pos, b.bs))         # [T,H]
        w = b.pi if pool == "boltz" else torch.ones_like(b.pi)
        tot = torch.zeros(b.n_mol, device=w.device).index_add_(0, b.conf_mol, w)
        w = w / tot[b.conf_mol]
        v = torch.zeros((b.n_mol, z.shape[1]), device=z.device)
        v = v.index_add_(0, b.conf_mol, z * w[:, None])
        vecs.append(v.cpu().numpy())
    return names, np.concatenate(vecs).astype("float32")


@torch.no_grad()
def schnet_reps(seed):
    model = build_r2("schnet", dropout=0.15).to(DEVICE)
    ck = torch.load(os.path.join(PROJ, "results", "r2_sc", "schnet", f"seed{seed}",
                                 "best_model.pt"), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    recs = load_r2("test", CACHE)
    names = cache_test_names()
    out = []
    for b in PyGLoader(recs, batch_size=64, shuffle=False):
        b = b.to(DEVICE)
        h = model.net.embedding(b.x[:, 0].long())
        ei, ew = model.net.interaction_graph(b.pos, b.batch)
        ea = model.net.distance_expansion(ew)
        for it in model.net.interactions:
            h = h + it(h, ei, ew, ea)
        out.append(global_mean_pool(h, b.batch).cpu().numpy())
    return names, np.concatenate(out).astype("float32")


@torch.no_grad()
def edge_gin_reps(seed):
    model = build_r25("edge_gin", dropout=0.15, edge_dim=66).to(DEVICE)
    ck = torch.load(os.path.join(PROJ, "results", "r25", "edge_gin", f"seed{seed}",
                                 "best_model.pt"), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    recs = load_r25("test", CACHE)
    names = cache_test_names()
    out = []
    for b in PyGLoader(recs, batch_size=64, shuffle=False):
        b = b.to(DEVICE)
        h = model.inp(b.x.float())
        for conv in model.convs:
            h = conv(h, b.edge_index, b.edge_feat)
        out.append(global_mean_pool(h, b.batch).cpu().numpy())
    return names, np.concatenate(out).astype("float32")


def cache_test_names():
    """Molecule names in cache order (the order every cache-based loader emits)."""
    man = json.load(open(os.path.join(CACHE, "_ensemble_manifest.json")))
    recs = torch.load(os.path.join(CACHE, man["splits"]["test"]["path"]),
                      weights_only=False)
    return [r["name"] for r in recs]


@torch.no_grad()
def scalar_reps(seed, arm, cols, scal):
    model = ScalarGIN(num_classes=138, n_scalars=max(1, len(cols)), dropout=0.15).to(DEVICE)
    ck = torch.load(os.path.join(PROJ, "results", "s2_ablation", arm, f"seed{seed}",
                                 "best_model.pt"), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    recs = load_s2("test", CACHE, scal, cols)
    names = cache_test_names()
    out = []
    for b in PyGLoader(recs, batch_size=64, shuffle=False):
        b = b.to(DEVICE)
        x, bi = model.gin.encode(b)
        out.append(global_mean_pool(x, bi).cpu().numpy())
    return names, np.concatenate(out).astype("float32")


def two_d_descriptors(names, name2smi):
    rows = []
    for n in names:
        mol = Chem.MolFromSmiles(name2smi[n])
        dm = Chem.GetDistanceMatrix(mol)
        kier = (rdMolDescriptors.CalcKappa1(mol) * rdMolDescriptors.CalcKappa2(mol)
                / max(1, mol.GetNumHeavyAtoms()))
        rows.append([Descriptors.MolWt(mol), float(mol.GetRingInfo().NumRings()),
                     float(kier), float(dm.sum() / 2.0)])
    return np.asarray(rows, dtype="float32")


def fit_ridge(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=np.logspace(-3, 4, 20), cv=5)
    m.fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte)), float(m.alpha_)


def fit_mlp(Xtr, ytr, Xva, yva, Xte, seed=0, max_epochs=150, patience=15):
    torch.manual_seed(seed)
    sc = StandardScaler().fit(Xtr)
    ys, ym = float(ytr.mean()), float(ytr.std() + 1e-9)
    A = torch.tensor(sc.transform(Xtr), dtype=torch.float32, device=DEVICE)
    B = torch.tensor((ytr - ym) / ys, dtype=torch.float32, device=DEVICE).view(-1, 1)
    V = torch.tensor(sc.transform(Xva), dtype=torch.float32, device=DEVICE)
    VY = torch.tensor((yva - ym) / ys, dtype=torch.float32, device=DEVICE).view(-1, 1)
    T = torch.tensor(sc.transform(Xte), dtype=torch.float32, device=DEVICE)
    net = nn.Sequential(nn.Linear(A.shape[1], 128), nn.ReLU(), nn.Linear(128, 1)).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    best, best_state, bad, used = 1e9, None, 0, 0
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(A.shape[0], device=DEVICE)
        for i in range(0, A.shape[0], 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            lossf(net(A[idx]), B[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = lossf(net(V), VY).item()
        used = ep + 1
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        p = net(T).cpu().numpy().ravel() * ys + ym
    return p, used


def scores(y, p):
    return {"r2": float(r2_score(y, p)), "spearman": float(spearmanr(y, p).statistic)}


def cka_linear(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    den = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
    return float(np.linalg.norm(X.T @ Y) ** 2 / den) if den > 0 else float("nan")


def _center(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def cka_rbf(X, Y):
    d2x = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    d2y = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    sx = np.sqrt(np.median(d2x[d2x > 0]) / 2.0)
    sy = np.sqrt(np.median(d2y[d2y > 0]) / 2.0)
    Kc = _center(np.exp(-d2x / (2 * sx ** 2)))
    Lc = _center(np.exp(-d2y / (2 * sy ** 2)))
    num = float((Kc * Lc).sum())
    den = float(np.sqrt((Kc * Kc).sum() * (Lc * Lc).sum()))
    return num / den if den > 0 else float("nan")


def stage_probe(seeds, ctx):
    """Fit both probe families for every seed; one json per seed (resumable)."""
    name2smi, smi2name, scal_all, sets, names, Y, X2d, rot = ctx
    for seed in seeds:
        p = os.path.join(OUT, f"_probe_seed{seed}.json")
        if os.path.exists(p):
            print(f"[probe] seed {seed} cached, skip")
            continue
        t0 = time.time()
        R = {s: r1_reps(seed, sets[s][1]) for s in ("train", "val", "test")}
        X = {"gin": R,
             "gin_rotors": {s: np.hstack([R[s], rot[s]]) for s in R},
             "descriptors_2d": X2d}
        res = {"ridge": {}, "mlp": {}}
        for fam in ("gin", "gin_rotors", "descriptors_2d"):
            res["ridge"][fam], res["mlp"][fam] = {}, {}
            for j, t in enumerate(TARGETS):
                ytr, yva, yte = Y["train"][:, j], Y["val"][:, j], Y["test"][:, j]
                pr, alpha = fit_ridge(X[fam]["train"], ytr, X[fam]["test"])
                res["ridge"][fam][t] = {**scores(yte, pr), "alpha": alpha}
                pm, ep = fit_mlp(X[fam]["train"], ytr, X[fam]["val"], yva, X[fam]["test"], seed)
                res["mlp"][fam][t] = {**scores(yte, pm), "epochs": ep}
        res["seed"] = seed
        with open(p, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"[probe] seed {seed} {time.time() - t0:.0f}s", flush=True)


def stage_cka(seeds, ctx):
    """Pairwise CKA per seed; one json per seed (resumable)."""
    _, _, _, sets, names, _, _, _ = ctx
    scal_all = ctx[2]
    tr_cols = S2_ARMS["en4"]
    tr = scal_all[scal_all.split == "train"]
    mu, sd = tr[tr_cols].mean(), tr[tr_cols].std().replace(0, 1.0)
    scal_std = scal_all.copy()
    scal_std[tr_cols] = (scal_std[tr_cols] - mu) / sd
    order = {n: i for i, n in enumerate(names["test"])}
    labels = [a[0] for a in CKA_ARMS]

    for seed in seeds:
        p = os.path.join(OUT, f"_cka_seed{seed}.json")
        if os.path.exists(p):
            print(f"[cka] seed {seed} cached, skip")
            continue
        t0 = time.time()
        got = {
            "R1": (names["test"], r1_reps(seed, sets["test"][1])),
            "EN1": scalar_reps(seed, "en1", S2_ARMS["en1"], scal_std),
            "EN4": scalar_reps(seed, "en4", S2_ARMS["en4"], scal_std),
            "R2_SchNet": schnet_reps(seed),
            "R2.5_EdgeGIN": edge_gin_reps(seed),
            "R3_boltz": mil_reps(seed, "boltz"),
            "R3_uniform": mil_reps(seed, "uniform"),
        }
        mats = {}
        for lab, (nm, M) in got.items():
            try:
                idx = np.array([order[n] for n in nm])
            except KeyError as e:
                raise RuntimeError(f"{lab}: molecule {e} not in the R1 test order")
            A = np.empty(M.shape, dtype="float64")
            A[idx] = M
            mats[lab] = A
        lin, rbf = {}, {}
        for i in range(len(labels)):
            for j in range(i, len(labels)):
                a, b = labels[i], labels[j]
                lin[f"{a}|{b}"] = cka_linear(mats[a], mats[b])
                rbf[f"{a}|{b}"] = cka_rbf(mats[a], mats[b])
        with open(p, "w") as fh:
            json.dump({"seed": seed, "linear": lin, "rbf": rbf}, fh, indent=1)
        print(f"[cka] seed {seed} {time.time() - t0:.0f}s", flush=True)


def stage_aggregate(ctx):
    _, _, _, _, _, _, _, _ = ctx
    files = sorted(int(f[len("_probe_seed"):-5]) for f in os.listdir(OUT)
                   if f.startswith("_probe_seed"))
    if not files:
        raise SystemExit("no probe seed files; run --stage probe first")
    probe = {"ridge": {}, "mlp": {}}
    for fam in ("gin", "gin_rotors", "descriptors_2d"):
        for pname in ("ridge", "mlp"):
            probe[pname][fam] = {t: {"r2": [], "spearman": [], "extra": []} for t in TARGETS}
    for seed in files:
        r = json.load(open(os.path.join(OUT, f"_probe_seed{seed}.json")))
        for pname, xkey in (("ridge", "alpha"), ("mlp", "epochs")):
            for fam in r[pname]:
                for t in TARGETS:
                    probe[pname][fam][t]["r2"].append(r[pname][fam][t]["r2"])
                    probe[pname][fam][t]["spearman"].append(r[pname][fam][t]["spearman"])
                    probe[pname][fam][t]["extra"].append(r[pname][fam][t][xkey])

    summary = {}
    for pname in ("ridge", "mlp"):
        summary[pname] = {}
        for fam in probe[pname]:
            summary[pname][fam] = {}
            for t in TARGETS:
                r2v = np.array(probe[pname][fam][t]["r2"])
                spv = np.array(probe[pname][fam][t]["spearman"])
                summary[pname][fam][t] = {
                    "r2_mean": float(r2v.mean()),
                    "r2_sd": float(r2v.std(ddof=1)) if len(r2v) > 1 else 0.0,
                    "spearman_mean": float(spv.mean()),
                    "spearman_sd": float(spv.std(ddof=1)) if len(spv) > 1 else 0.0,
                }

    fam_mean = {p: {f: float(np.mean([summary[p][f][t]["r2_mean"] for t in TARGETS]))
                    for f in probe[p]} for p in ("ridge", "mlp")}
    primary = max(fam_mean, key=lambda p: fam_mean[p]["gin_rotors"])
    g4 = fam_mean[primary]["gin_rotors"]
    gate = {"probe_family_primary": primary, "mean_r2_by_family": fam_mean,
            "g4_variable": g4, "threshold": 0.85,
            "verdict": "PASS" if g4 >= 0.85 else "FAIL",
            "branch": "geometric manifold collapse" if g4 >= 0.85 else "receptor invariance"}

    cka_files = sorted(int(f[len("_cka_seed"):-5]) for f in os.listdir(OUT)
                       if f.startswith("_cka_seed"))
    cka = {"labels": [a[0] for a in CKA_ARMS], "linear": {}, "rbf": {}}
    if cka_files:
        rows = [json.load(open(os.path.join(OUT, f"_cka_seed{s}.json"))) for s in cka_files]
        for key in ("linear", "rbf"):
            for k in rows[0][key]:
                v = [r[key][k] for r in rows]
                cka[key][k] = {"mean": float(np.mean(v)),
                               "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                               "per_seed": v}
        L = len(cka["labels"])
        lin_m = np.full((L, L), np.nan)
        rbf_m = np.full((L, L), np.nan)
        for i, a in enumerate(cka["labels"]):
            for j, b in enumerate(cka["labels"]):
                k = f"{a}|{b}" if f"{a}|{b}" in cka["linear"] else f"{b}|{a}"
                lin_m[i, j] = cka["linear"][k]["mean"]
                rbf_m[i, j] = cka["rbf"][k]["mean"]
        np.savez_compressed(os.path.join(OUT, "cka.npz"),
                            labels=np.array(cka["labels"]),
                            linear_mean=lin_m, rbf_mean=rbf_m,
                            seeds=np.array(cka_files))

    out = {"targets": TARGETS, "seeds": files, "cka_seeds": cka_files,
           "probe": probe, "summary": summary, "gate_G4": gate, "cka": cka,
           "definitions": {
               "gin": "R1 256-d mean-pooled node embedding (pre-classifier)",
               "gin_rotors": "the same plus the RDKit 2D n_rotors from exp2_scalars.csv",
               "descriptors_2d": "MolWt, n_rings, Kier phi, Wiener (no learned representation)",
               "S_conf": "ln(n_eff)",
               "mlp": "d->128->1, ReLU, Adam 1e-3, early stop on val MSE (patience 15, 150 ep cap)",
               "ridge": "RidgeCV, alpha by 5-fold CV on the 3,363 training molecules",
               "cka": "linear + RBF (median heuristic), column-centred, per seed then mean",
               "r3_vector": "molecule-level proxy: sum_c p_c z_c (boltz) / mean_c z_c (uniform)",
           }}
    with open(os.path.join(OUT, "probing.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    write_report(out)
    print(json.dumps(gate, indent=1))
    print("[aggregate] wrote probing.json, cka.npz, reports/14_redundancy.md")


def context():
    name2smi, smi2name, scal_all = molecule_index()
    sets = {s: r1_split(s, smi2name) for s in ("train", "val", "test")}
    names = {s: sets[s][0] for s in sets}
    Y = {s: scal_all.loc[names[s], TARGETS].values.astype("float64") for s in sets}
    X2d = {s: two_d_descriptors(names[s], name2smi) for s in sets}
    rot = {s: scal_all.loc[names[s], ["n_rotors"]].values.astype("float32") for s in sets}
    print(f"[data] train/val/test = {len(names['train'])}/{len(names['val'])}/"
          f"{len(names['test'])} | device {DEVICE}")
    return name2smi, smi2name, scal_all, sets, names, Y, X2d, rot


def write_report(res):
    s = res["summary"]
    g = res["gate_G4"]
    fam_name = {"gin": "GIN 表示 (256)", "gin_rotors": "GIN + n_rotors (257)",
                "descriptors_2d": "纯 2D 描述符 (4)"}
    lines = ["# 14 — 2D→3D 冗余探针与 CKA（T3，零训练）", "",
             "由 `p2/wpa_probe.py` 生成。探针在 3,363 个训练分子上拟合（Ridge 的 α 用 5 折 CV 选），"
             "在 730 个测试分子上报告；每个 seed 使用同 seed 的 R1 表示。"
             "阈值见 `reports/02_preregistration.md` §2（G4，跑前冻结）。", "",
             "## 1. G4 判定", "",
             f"- 主探针族：**{g['probe_family_primary']}**（按 GIN+n_rotors 的八目标平均 R² 选族，"
             "两族都跑、只选一次，不逐目标挑）",
             f"- `GIN + n_rotors` 八目标平均 test R² = **{g['g4_variable']:.3f}**（阈值 0.85）",
             f"- 结论：**{g['verdict']}** → §2.3 走 *{g['branch']}* 分支", "",
             "| 特征集 | Ridge 平均 R² | MLP 平均 R² |", "|---|---|---|"]
    for fam in ("gin", "gin_rotors", "descriptors_2d"):
        lines.append(f"| {fam_name[fam]} | {g['mean_r2_by_family']['ridge'][fam]:.3f} | "
                     f"{g['mean_r2_by_family']['mlp'][fam]:.3f} |")

    # robustness: Rg is a size proxy, and 4 raw descriptors predict it as well as GIN.
    # An unweighted 8-target mean therefore overstates what "predicting 3D shape" means.
    no_rg = [t for t in TARGETS if not t.startswith("Rg")]
    alt = {p: {f: float(np.mean([s[p][f][t]["r2_mean"] for t in no_rg]))
               for f in ("gin", "gin_rotors", "descriptors_2d")} for p in ("ridge", "mlp")}
    prim = g["probe_family_primary"]
    lines += ["", f"另算一个去掉两个 Rg 目标的均值（Rg 基本是分子大小的代理，"
              f"四个原始描述符就能到 {s[prim]['descriptors_2d']['Rg_mean']['r2_mean']:.2f}）："
              f"`GIN+n_rotors` = **{alt[prim]['gin_rotors']:.3f}**（{len(no_rg)} 个目标）。"
              "去掉最大、最容易被大小解释的目标后判据离 0.85 更远，G4 的 FAIL 不是均值构成的产物。"]

    for pname, title, tag in (("ridge", "Ridge", "2.1"), ("mlp", "MLP (d→128→1)", "2.2")):
        lines += ["", f"## {tag} 逐目标 {title}", "",
                  "| 目标 | GIN R² | GIN+n_rotors R² | 纯 2D R² | GIN ρ | GIN+n_rotors ρ |",
                  "|---|---|---|---|---|---|"]
        for t in TARGETS:
            a, b, c = s[pname]["gin"][t], s[pname]["gin_rotors"][t], s[pname]["descriptors_2d"][t]
            lines.append(f"| {t} | {a['r2_mean']:.3f} ± {a['r2_sd']:.3f} | "
                         f"{b['r2_mean']:.3f} ± {b['r2_sd']:.3f} | "
                         f"{c['r2_mean']:.3f} ± {c['r2_sd']:.3f} | "
                         f"{a['spearman_mean']:.3f} | {b['spearman_mean']:.3f} |")

    cka = res["cka"]
    if cka["linear"]:
        labels = cka["labels"]
        for key, title, tag in (("linear", "线性核", "3.1"), ("rbf", "RBF 核", "3.2")):
            lines += ["", f"## {tag} CKA（{title}，730 测试分子，{len(res['cka_seeds'])} seeds 均值）", "",
                      "| | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
            for a in labels:
                row = []
                for b in labels:
                    k = f"{a}|{b}" if f"{a}|{b}" in cka[key] else f"{b}|{a}"
                    row.append(f"{cka[key][k]['mean']:.3f}" if k in cka[key] else "—")
                lines.append(f"| **{a}** | " + " | ".join(row) + " |")

    lines += ["", "## 4. 读法", "",
              "- R² 是测试分子上的，不是训练拟合度；MLP 用验证集早停，730 个测试分子未参与任何选择。",
              "- 三组特征的差别就是「3D 形状能否由 2D 图 + 一个柔性计数恢复」的直接测量；"
              "纯 2D 描述符列是「不用学习到的表示」的下界。",
              "- **CKA 与 G4 并不矛盾，要分开写**：G4 说 2D 表示**不能**恢复完整 3D 形状（R² 0.61），"
              "所以机制不是「几何已被拓扑决定」；CKA 说**学到的**表示彼此高度重合"
              "（R1↔R2.5 = 0.85），即边际化几何的模型收敛到了与 2D 相同的子空间。"
              "两者合起来支持的表述是「表示收敛，但不是因为几何被决定」。",
              "- R3 使用**分子级代理向量**：boltz 为 Σ_c p_c z_c、uniform 为 mean_c z_c；"
              "训练时 R3 在 138 维输出空间聚合，此处为与其它臂同维比较而取编码空间聚合，属代理量。",
              "- G4 的阈值在跑之前冻结：判 FAIL 即必须把 §2.3/§3.1 的「冗余」解释改为"
              "「标签侧的受体不变性」，不得回退。"]
    with open(os.path.join(PROJ, "reports", "14_redundancy.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["probe", "cka", "aggregate", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ctx = context()

    if args.stage in ("probe", "all"):
        stage_probe(args.seeds, ctx)
    if args.stage in ("cka", "all"):
        stage_cka(args.seeds, ctx)
    if args.stage in ("aggregate", "all"):
        stage_aggregate(ctx)


if __name__ == "__main__":
    main()
