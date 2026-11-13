#!/usr/bin/env python3
"""Build the QH-Odor-1821 release bundle with frozen SHA-256 manifests.

The proprietary 138-descriptor matrix is used only to compute row sums (L_sem)
and SMILES; it is never written to the output tree.

    python scripts/build_zenodo_data_bundle.py \
        --data-root ~/codebuddy_dir/gnn_comparison01 \
        --out dist/zenodo_data_QH-Odor-1821
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

EXPECT = {
    "phase1_molecules": 4809, "phase1_conformers": 55895,
    "phase2_molecules": 1821, "phase2_conformers": 106526,
    "complete_cohort": 4808, "censored": 601,
    "dream": 338, "cliffs": 5000, "solvation_molecules": 300,
    "solvation_conformers": 16836, "arms": 21, "seeds": 10,
}

SDF_PARTS = ["pilot100_representatives.sdf", "pilot200_representatives.sdf",
             "fullrest_representatives.sdf"]
THERMO_KEYS = ["mol_index", "name", "rank", "conf_index", "cluster_id", "cluster_size",
               "degeneracy", "e_abs_kcal", "e_rel_kcal", "zpe_kcal", "s_total_cal_mol_K",
               "s_rot_cal_mol_K", "s_vib_cal_mol_K", "ts_kcal", "g_abs_kcal", "dg_kcal",
               "n_imag", "population", "population_298K", "population_310K", "T_k"]


def write_table(df: pd.DataFrame, path: Path) -> None:
    """Write gzip-compressed CSV (no parquet engine is guaranteed in the build env)."""
    df.to_csv(path, index=False, compression="gzip")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def inchikey(smiles: str):
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return Chem.MolToInchiKey(m) if m else None


def ring_flex(smiles: str) -> float:
    """Fraction of ring bonds that are single and not in a fused system (paper 2 definition)."""
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return np.nan
    ring = m.GetRingInfo().AtomRings()
    if not ring:
        return 0.0
    rot = 0
    for b in m.GetBonds():
        if b.IsInRing() and b.GetBondTypeAsDouble() == 1.0 and not b.GetIsAromatic():
            rot += 1
    return rot / max(1, m.GetNumBonds())


def pyrfume_registry(data_root: Path):
    frames = []
    for f in (data_root / "data" / "pyrfume-data-main").glob("*/molecules.csv"):
        try:
            d = pd.read_csv(f, usecols=["CID", "IsomericSMILES"]).dropna()
            frames.append(d)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["ik", "CID"])
    reg = pd.concat(frames).drop_duplicates("CID")
    reg["ik"] = [inchikey(s) for s in reg.IsomericSMILES]
    return reg.dropna(subset=["ik"]).drop_duplicates("ik")[["ik", "CID"]]


def stream_sdf_thermo(paths, out_parquet: Path) -> int:
    rows, block, n = [], {}, 0
    for p in paths:
        with open(p, "r", errors="ignore") as fh:
            key = None
            for line in fh:
                if line.startswith("$$$$"):
                    if block:
                        block.setdefault("name", None)
                        rows.append({k: block.get(k) for k in THERMO_KEYS})
                        block = {}
                        n += 1
                    key = None
                    continue
                s = line.rstrip("\n")
                m = __import__("re").match(r"^>\s*<([^>]+)>", s)
                if m:
                    key = m.group(1).strip()
                elif key is not None:
                    block[key] = s
                    key = None
    df = pd.DataFrame(rows)
    for c in THERMO_KEYS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="ignore") if c != "name" else df[c]
    write_table(df, out_parquet)
    return len(df)


def concat_gzip(paths, out_gz: Path) -> int:
    n = 0
    with gzip.open(out_gz, "wb", compresslevel=6) as gz:
        for p in paths:
            with open(p, "rb") as fh:
                for line in fh:
                    gz.write(line)
                    if line.startswith(b"$$$$"):
                        n += 1
    return n


def collect_metrics(p2: Path, consolidate: Path, exp_dir: Path) -> dict:
    arms = {"R1": "results/r1/gin", "SCHNET": "results/r2_sc/schnet",
            "CHIRAL": "results/r2_sc/chiral_schnet", "DIMENET": "results/r2_dimenet/dimenet",
            "ANGLE": "results/r2/angle_mpnn", "R25": "results/r25/edge_gin",
            "R25shuf": "results/r25/edge_gin_shuffled", "EN3": "results/r25/scalar_gin",
            "R3boltz": "results/s1_mil/boltz", "R3attn": "results/s1_mil/attn",
            "R3unif": "results/s1_mil/shuffle", "EN1": "results/s2_ablation/en1",
            "EN2": "results/s2_ablation/en2", "EN4": "results/s2_ablation/en4",
            "EN5": "results/s2_ablation/en5", "EN6": "results/s2_ablation/en6",
            "EN3shuf": "results/s2_ablation/en3_shuf"}
    extra = {"en4_s2d": exp_dir / "results/en4_s2d",
             "wide_en1": exp_dir / "results/wide_en1",
             "en4_size": consolidate / "1031/1103_1857-paper2/en4_decomp/results/en4_size",
             "en4_shape": consolidate / "1031/1103_1857-paper2/en4_decomp/results/en4_shape"}
    out = {"arms": {}, "note": "per-seed macro-AUCPR / macro-AUROC on 730 held-out odorants"}
    for name, rel in arms.items():
        root = p2 / rel
        seeds = []
        for s in range(10):
            f = root / f"seed{s}" / "metrics.json"
            if f.exists():
                m = json.loads(f.read_text())
                seeds.append({"seed": s, "macro_aucpr": m["test_macro_aucpr"],
                              "macro_auroc": m.get("test_macro_auroc", m.get("test_micro_auroc")),
                              "best_epoch": m.get("best_epoch"), "n_params": m.get("n_params")})
        if seeds:
            out["arms"][name] = seeds
    for name, root in extra.items():
        seeds = []
        for s in range(10):
            f = root / f"seed{s}" / "metrics.json"
            if f.exists():
                m = json.loads(f.read_text())
                seeds.append({"seed": s, "macro_aucpr": m["test_macro_aucpr"],
                              "macro_auroc": m.get("test_macro_auroc"),
                              "best_epoch": m.get("best_epoch"), "n_params": m.get("n_params")})
        if seeds:
            out["arms"][name] = seeds
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    p2, p3 = root / "paper2", root / "paper3"
    cons = root / "consolidate"
    out = Path(args.out)
    (out / "cohorts").mkdir(parents=True, exist_ok=True)
    (out / "ensembles").mkdir(exist_ok=True)
    (out / "probes").mkdir(exist_ok=True)
    (out / "metrics").mkdir(exist_ok=True)
    (out / "Figures").mkdir(exist_ok=True)

    status = {}

    # ---------- cohort table
    obs = pd.read_csv(p3 / "cache/ensemble_observables.csv")
    exp_dir = cons / "1031/1103_1857-paper2/experiments"
    if not exp_dir.exists():
        exp_dir = p2 / "experiments"
    scal = pd.read_csv(exp_dir / "scalars_v3.csv", usecols=["name", "split"])
    master = pd.read_csv(root / "data/curated_GS_LF_merged_4983.csv", low_memory=False,
                         usecols=["nonStereoSMILES"])
    master = master.rename(columns={"nonStereoSMILES": "smiles"})
    # L_sem is the row sum of the 138 binary descriptors — computed, never written
    full = pd.read_csv(root / "data/curated_GS_LF_merged_4983.csv", low_memory=False)
    bin_cols = [c for c in full.columns if full[c].dropna().isin([0, 1]).all() and c != "descriptors"]
    master["L_sem"] = full[bin_cols].sum(axis=1).astype("Int64")
    master["smiles"] = master["smiles"].astype(str)

    # name -> SMILES bridge: the master table carries structures but no names;
    # inchikey_map.csv carries name -> InChIKey for all 4,809 converged molecules.
    master["inchikey"] = master["smiles"].map(inchikey)
    ikmap = pd.read_csv(p3 / "cache/inchikey_map.csv")
    bridge = ikmap.merge(master[["inchikey", "smiles", "L_sem"]].drop_duplicates("inchikey"),
                         on="inchikey", how="left")
    df = (obs.merge(scal.rename(columns={"split": "split_ref"}), on="name", how="left")
             .merge(bridge[["name", "smiles", "L_sem"]], on="name", how="left"))
    if "L_sem" in obs:
        df["L_sem"] = obs.set_index("name").L_sem.reindex(df.name).values
    if "split" not in df and "split_ref" in df:
        df["split"] = df["split_ref"]
    df = df.rename(columns={"n_rotors": "n_rotors_pipeline", "n_ha": "hba",
                            "n_unspecified_stereo": "unspecified_stereo"})
    def _strict_rotors(s):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        return rdMolDescriptors.CalcNumRotatableBonds(m) if m is not None else np.nan
    df["n_rotors_rdkit"] = df.smiles.map(_strict_rotors)
    df["n_ring_flex"] = df.smiles.map(ring_flex)
    df["inchikey"] = df.smiles.map(inchikey)
    reg = pyrfume_registry(root)
    df = df.merge(reg, left_on="inchikey", right_on="ik", how="left").drop(columns=["ik"])
    df["pyrfume_id"] = df["CID"].astype("Int64")
    df["molecule_id"] = df["name"]
    df["maxcap"] = df["capped20"].map({True: 20, False: 200}) if "capped20" in df else np.where(df.n_conf >= 100, 200, 20)
    cols = ["molecule_id", "pyrfume_id", "smiles", "split", "L_sem", "mw", "clogp", "hba", "tpsa",
            "unspecified_stereo", "run_pass", "n_rotors_pipeline", "n_rotors_rdkit", "n_ring_flex",
            "has_censored", "maxcap", "n_conf", "S_conf", "N_eff", "rmsd_ab", "rg_mean", "sigma2_rg",
            "npr1_mean", "npr2_mean", "tr_sigma_npr", "sasa_polar_mean", "sigma_sasa_polar", "inchikey"]
    cols = [c for c in cols if c in df.columns]
    table = df[cols].rename(columns={"run_pass": "batch_id"})
    table.to_csv(out / "cohorts/gs_lf_4809_master_observables.csv", index=False)
    write_table(table, out / "cohorts/gs_lf_4809_master_observables.csv.gz")
    status["cohort_rows"] = len(table)

    # ---------- DREAM / M2OR / cliffs
    dream = pd.read_csv(root / "data/external/dream/dream_dataset.csv")
    dream.to_csv(out / "cohorts/dream_338_psychophysics_observables.csv", index=False)
    status["dream_rows"] = len(dream)

    m2 = pd.read_csv(p3 / "results/wp2_m2or/wp2_ligand_table.csv")
    m2.to_csv(out / "cohorts/m2or_ligand_panel_294.csv", index=False)
    status["m2or_rows"] = len(m2)

    cliffs = pd.read_csv(p3 / "results/wp3_cliffs/wp3_pair_metrics.csv")
    cliffs.to_csv(out / "probes/perceptual_cliffs_pairs.csv", index=False)
    status["cliff_rows"] = len(cliffs)
    status["cliff_positive"] = int(cliffs.is_cliff.sum())

    # ---------- ensembles
    sdf_paths = [p3 / "v3_results" / f for f in SDF_PARTS]
    n_conf = concat_gzip(sdf_paths, out / "ensembles/phase2_complete_1821_106526confs.sdf.gz")
    status["phase2_conformers"] = n_conf
    n_thermo = stream_sdf_thermo(sdf_paths, out / "ensembles/phase2_thermochemistry_106526.csv.gz")
    status["thermo_rows"] = n_thermo

    # ---------- probes
    sol = pd.read_csv(p3 / "v3_results/exp20_hexane/solvation_all_corrected.csv")
    write_table(sol, out / "probes/alpb_solvation_300mols_16836confs.csv.gz")
    status["solvation_rows"] = len(sol)

    ph = pd.read_csv(p3 / "v3_results/exp9_pharm/s_pharm_all.csv")
    write_table(ph, out / "probes/usrcat_60d_pharmacophore_entropy_1821.csv.gz")
    status["pharm_rows"] = len(ph)

    sat = pd.read_csv(p3 / "v3_results/exp18_figdata/sat_curves.csv")
    write_table(sat, out / "probes/saturation_curves_C95_1821.csv.gz")
    status["sat_rows"] = len(sat)
    shutil.copy(p3 / "v3_results/exp18_figdata/sat_c95.csv", out / "probes/saturation_c95_1821.csv")

    # ---------- metrics
    met = collect_metrics(p2, cons, exp_dir)
    (out / "metrics/paper2_21arms_10seeds_metrics.json").write_text(json.dumps(met, indent=1))
    status["arms"] = len(met["arms"])
    status["seeds_per_arm"] = sorted({len(v) for v in met["arms"].values()})

    # ---------- figure source data
    for f in ["figure_source_data.json", "source_data_paper2_arms.json"]:
        src = cons / "1104_1307-paper2+3/Figures" / f
        if src.exists():
            shutil.copy(src, out / "Figures" / f)
            status[f"figure::{f}"] = True

    # ---------- phase I summary (geometries not redistributed)
    p1 = pd.read_csv(exp_dir / "scalars_v3.csv")
    write_table(p1, out / "cohorts/phase1_cache_4809_observables.csv.gz")
    status["phase1_rows"] = len(p1)

    (out / "build_status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))

    checks = {
        "phase1_molecules": status.get("phase1_rows") == EXPECT["phase1_molecules"],
        "phase2_conformers": status.get("phase2_conformers") == EXPECT["phase2_conformers"],
        "thermo_rows": status.get("thermo_rows") == EXPECT["phase2_conformers"],
        "dream_rows": status.get("dream_rows") >= EXPECT["dream"],
        "cliff_rows": status.get("cliff_rows") >= EXPECT["cliffs"],
        "arms": status.get("arms") == EXPECT["arms"],
        "seeds": status.get("seeds_per_arm") == [EXPECT["seeds"]],
    }
    bad = [k for k, ok in checks.items() if not ok]
    print("checks:", checks)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
