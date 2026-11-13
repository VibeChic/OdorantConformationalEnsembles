"""Common utilities for Paper 3.

All paths are resolved relative to the repository root (gnn_comparison01).
Read-only inputs: paper2/cache, data/3d/run1, data/external, data/mmp.
Writable outputs: paper3/cache, paper3/results, paper3/reports, paper3/logs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

ROOT = Path(__file__).resolve().parents[2]
PAPER2_CACHE = ROOT / "paper2" / "cache"
DATA = ROOT / "data"
PAPER3 = ROOT / "paper3"
CACHE = PAPER3 / "cache"
RESULTS = PAPER3 / "results"
REPORTS = PAPER3 / "reports"
LOGS = PAPER3 / "logs"
SDF_DIR = DATA / "3d" / "run1" / "structures"
PT = Chem.GetPeriodicTable()


def load_cache() -> tuple[list[dict], dict[str, str]]:
    """Load the 4,809-molecule ensemble cache and return records + split map."""
    man = json.loads((PAPER2_CACHE / "_ensemble_manifest.json").read_text())
    records: list[dict] = []
    split_of: dict[str, str] = {}
    for split in ("train", "val", "test"):
        recs = torch.load(PAPER2_CACHE / man["splits"][split]["path"], weights_only=False)
        for r in recs:
            name = str(r["name"])
            records.append(r)
            split_of[name] = split
    return records, split_of


def load_run1_table() -> pd.DataFrame:
    """Per-molecule metadata from the 3D run (4,811 rows incl. 2 failures)."""
    df = pd.read_csv(DATA / "3d" / "run1" / "run1_molecules.csv", low_memory=False)
    df = df[df["status"] == "ok"].copy()
    df["name"] = df["name"].astype(str)
    return df.set_index("name", drop=False)


def load_sdf(name: str) -> list[Chem.Mol]:
    path = SDF_DIR / f"{name}.sdf"
    if not path.exists():
        raise FileNotFoundError(path)
    mols = [m for m in Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True) if m is not None]
    if not mols:
        raise ValueError(f"no readable conformers in {path}")
    return mols


def heavy_atom_coords(mol: Chem.Mol, conf_id: int = 0) -> np.ndarray:
    conf = mol.GetConformer(conf_id)
    idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    return np.asarray([list(conf.GetAtomPosition(i)) for i in idx], dtype=float)


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray, allow_reflection: bool = False) -> float:
    """RMSD after Kabsch alignment of Q onto P.

    P, Q: (N, 3) arrays with the same atom order.
    If allow_reflection is False, improper rotations are forbidden (det(R)=+1),
    which is required for chiral molecules.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    if allow_reflection:
        R = Vt.T @ U.T
    else:
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
    # R maps P onto Q in row-vector form (Qc ~ Pc @ R); applying R.T instead
    # rotates the wrong way and inflates every RMSD.
    Qa = Qc @ R
    return float(np.sqrt(np.mean(np.sum((Pc - Qa) ** 2, axis=1))))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_var(values: np.ndarray, weights: np.ndarray) -> float:
    mu = weighted_mean(values, weights)
    return float(np.sum(weights * (values - mu) ** 2) / np.sum(weights))


def rg_mass_weighted(pos: np.ndarray, masses: np.ndarray) -> float:
    com = np.sum(pos * masses[:, None], axis=0) / np.sum(masses)
    r = pos - com
    return float(np.sqrt(np.sum(masses * np.sum(r * r, axis=1)) / np.sum(masses)))


def npr_mass_weighted(pos: np.ndarray, masses: np.ndarray) -> tuple[float, float, float]:
    """Return (I1/I3, I2/I3, I1) for the mass-weighted inertia tensor."""
    com = np.sum(pos * masses[:, None], axis=0) / np.sum(masses)
    r = pos - com
    I = np.zeros((3, 3), dtype=float)
    for mi, ri in zip(masses, r):
        I += mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))
    eig = np.linalg.eigvalsh(I)  # ascending
    I1, I2, I3 = (float(eig[0]), float(eig[1]), float(eig[2]))
    if I3 <= 0:
        return float("nan"), float("nan"), float("nan")
    return I1 / I3, I2 / I3, I1


def polar_sasa(mol: Chem.Mol, conf_id: int, probe_radius: float = 1.4) -> float:
    """Polar SASA (N/O/S/P atoms) using RDKit's FreeSASA Shrake-Rupley implementation."""
    from rdkit.Chem import rdFreeSASA

    pt = Chem.GetPeriodicTable()
    radii = [pt.GetRvdw(a.GetAtomicNum()) for a in mol.GetAtoms()]
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        a.SetIntProp("SASAClass", rdFreeSASA.Polar if z in (7, 8, 15, 16) else rdFreeSASA.APolar)
        a.SetProp("SASAClassName", "Polar" if z in (7, 8, 15, 16) else "APolar")
    opts = rdFreeSASA.SASAOpts()
    opts.algorithm = rdFreeSASA.ShrakeRupley
    opts.probeRadius = float(probe_radius)
    q = rdFreeSASA.MakeFreeSasaPolarAtomQuery()
    return float(rdFreeSASA.CalcSASA(mol, radii, int(conf_id), q, opts))


def rdkit_descriptors(smiles: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"mw": float("nan"), "clogp": float("nan"), "n_ha": float("nan"), "tpsa": float("nan")}
    return {
        "mw": float(Descriptors.MolWt(mol)),
        "clogp": float(Descriptors.MolLogP(mol)),
        "n_ha": float(rdMolDescriptors.CalcNumHBA(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
    }


def ensure_dirs() -> None:
    for p in (CACHE, RESULTS, REPORTS, LOGS):
        p.mkdir(parents=True, exist_ok=True)
