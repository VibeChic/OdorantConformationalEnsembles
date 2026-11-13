#!/usr/bin/env python3
"""M2OR stage 0 -- resolve every panel receptor to a sequence.

The panel's 436 "receptors" are not all UniProt accessions: some are mutant
labels (`hOR10A2_H43R_H207R_K258T`), some are indels, some are aliases. This
script produces one table that the dual tower can consume:

    entry, kind, parent_accession, sequence, n_subs, status

and a manifest with sha256. Nothing is trained here; `--esm` additionally caches
ESM-2 650M embeddings (weights already on disk -- no download).

Outputs
  cache/m2or_receptor_sequences.csv
  cache/m2or_receptor_manifest.json
  cache/esm2_m2or_receptors.npz          (only with --esm)

Usage
  python p2/m2or_stage0.py
  python p2/m2or_stage0.py --esm
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import requests

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(PROJ)
CACHE = os.path.join(PROJ, "cache")
ESM_DIR = os.path.expanduser("~/codebuddy_dir/models/esm2_t33_650M_UR50D")

UNIPROT = "https://rest.uniprot.org/uniprotkb"
ACC_RE = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")
MUT_RE = re.compile(r"^h?(OR[0-9]+[A-Z]{1,2}[0-9]+)((?:_[A-Z][0-9]+[A-Z*])*)$")
SUB_RE = re.compile(r"_([A-Z])([0-9]+)([A-Z*])")


def fasta_to_dict(text):
    out, cur, buf = {}, None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if cur:
                out[cur] = "".join(buf)
            parts = line[1:].split("|")
            cur = parts[1] if len(parts) > 2 else line[1:].split()[0]
            buf = []
        elif line.strip():
            buf.append(line.strip())
    if cur:
        out[cur] = "".join(buf)
    return out


def fetch_accessions(accs, chunk=40):
    got = {}
    for i in range(0, len(accs), chunk):
        r = requests.get(f"{UNIPROT}/accessions",
                         params={"accessions": ",".join(accs[i:i + chunk]),
                                 "format": "fasta"}, timeout=90)
        if r.status_code == 200:
            got.update(fasta_to_dict(r.text))
        time.sleep(0.25)
    return got


def fetch_one(acc):
    r = requests.get(f"{UNIPROT}/{acc}.fasta", timeout=60)
    return fasta_to_dict(r.text) if r.status_code == 200 else {}


def gene_to_accession(genes, chunk=18):
    """Map gene symbol -> UniProt accession (human, reviewed first, then longest)."""
    rows = []
    for i in range(0, len(genes), chunk):
        q = " OR ".join(f"gene_exact:{g}" for g in genes[i:i + chunk])
        r = requests.get(f"{UNIPROT}/search",
                         params={"query": f"({q}) AND organism_id:9606",
                                 "fields": "accession,gene_primary,reviewed,length",
                                 "format": "tsv", "size": 500}, timeout=90)
        if r.status_code == 200:
            lines = r.text.strip().splitlines()
            for ln in lines[1:]:
                f = ln.split("\t")
                if len(f) >= 4:
                    rows.append({"gene": f[1], "acc": f[0],
                                 "reviewed": f[2] == "reviewed", "len": int(f[3])})
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    df = df.sort_values(["reviewed", "len"], ascending=[False, False])
    return df.drop_duplicates("gene").set_index("gene")["acc"].to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm", action="store_true", help="also cache ESM-2 pooled embeddings")
    ap.add_argument("--tokens", action="store_true", help="also cache ESM-2 residue tokens")
    args = ap.parse_args()

    mat = pd.read_csv(os.path.join(REPO, "data", "external", "m2or",
                                   "m2or_receptor_ligand.csv"), low_memory=False)
    entries = list(dict.fromkeys(mat.receptor.astype(str)))
    print(f"[panel] {len(entries)} receptor entries")

    wt = [e for e in entries if ACC_RE.match(e)]
    mut = [e for e in entries if not ACC_RE.match(e)]
    print(f"[panel] accession-like {len(wt)} | mutant/alias {len(mut)}")

    seqs = fetch_accessions(wt)
    missing = [a for a in wt if a not in seqs]
    print(f"[fasta] {len(seqs)}/{len(wt)} on first pass; retrying {len(missing)} one by one")
    for a in missing:
        seqs.update(fetch_one(a))
        time.sleep(0.2)
    still = [a for a in wt if a not in seqs]
    print(f"[fasta] recovered {len(wt) - len(still)}/{len(wt)}; unresolved {still}")

    # --- mutant labels ------------------------------------------------
    parsed, unparsed = {}, []
    for e in mut:
        m = MUT_RE.match(e)
        if not m:
            unparsed.append(e)
            continue
        gene, suffix = m.group(1), m.group(2)
        subs = [(w, int(p), t) for w, p, t in SUB_RE.findall(suffix)]
        if any(t == "*" for _, _, t in subs):
            unparsed.append(e)          # premature stop: not a usable receptor
            continue
        parsed[e] = (gene, subs)
    print(f"[mutants] parsed {len(parsed)} | unsupported {len(unparsed)}")

    genes = sorted({g for g, _ in parsed.values()})
    g2a = gene_to_accession(genes)
    print(f"[genes] mapped {len(g2a)}/{len(genes)}")
    parent_seqs = fetch_accessions(list({a for a in g2a.values() if a not in seqs}))
    seqs.update(parent_seqs)

    rows, problems, offsets = [], [], []
    for e in wt:
        if e in seqs:
            rows.append({"entry": e, "kind": "wildtype", "parent_accession": e,
                         "sequence": seqs[e], "n_subs": 0, "status": "ok"})
        else:
            rows.append({"entry": e, "kind": "wildtype", "parent_accession": "",
                         "sequence": "", "n_subs": 0, "status": "no_sequence"})

    for e in mut:
        if e in unparsed:
            rows.append({"entry": e, "kind": "mutant", "parent_accession": "",
                         "sequence": "", "n_subs": 0, "status": "unsupported_label"})
            continue
        gene, subs = parsed[e]
        acc = g2a.get(gene)
        if acc is None or acc not in seqs:
            rows.append({"entry": e, "kind": "mutant", "parent_accession": acc or "",
                         "sequence": "", "n_subs": len(subs),
                         "status": "no_parent_sequence"})
            continue
        parent = seqs[acc]
        best = None
        for off in (0, 1, -1, 2, -2):
            if all(1 <= pos + off <= len(parent) and parent[pos + off - 1] == wt_aa
                   for wt_aa, pos, _ in subs):
                best = off
                break
        if best is None:
            # allele difference between the M2OR naming reference and the UniProt
            # canonical: apply at the stated position when in range, but flag it.
            if all(1 <= p <= len(parent) for _, p, _ in subs):
                s = list(parent)
                for _, pos, to_aa in subs:
                    s[pos - 1] = to_aa
                rows.append({"entry": e, "kind": "mutant", "parent_accession": acc,
                             "sequence": "".join(s), "n_subs": len(subs),
                             "status": "allele_mismatch"})
                problems.append((e, acc, [f"{w}{p}{t}" for w, p, t in subs]))
            else:
                rows.append({"entry": e, "kind": "mutant", "parent_accession": acc,
                             "sequence": "", "n_subs": len(subs),
                             "status": "substitution_mismatch"})
                problems.append((e, acc, [f"{w}{p}{t}" for w, p, t in subs]))
            continue
        s = list(parent)
        for wt_aa, pos, to_aa in subs:
            s[pos + best - 1] = to_aa
        rows.append({"entry": e, "kind": "mutant", "parent_accession": acc,
                     "sequence": "".join(s), "n_subs": len(subs),
                     "status": "ok" if best == 0 else f"ok_offset{best:+d}"})
        if best != 0:
            offsets.append((e, best))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CACHE, "m2or_receptor_sequences.csv"), index=False)

    counts = df.status.value_counts().to_dict()
    lens = df.loc[df.status.str.startswith("ok"), "sequence"].str.len()
    manifest = {
        "panel_entries": len(entries),
        "wildtype_entries": len(wt),
        "mutant_entries": len(mut),
        "status_counts": counts,
        "sequence_len": {"min": int(lens.min()), "median": float(lens.median()),
                         "max": int(lens.max())} if len(lens) else {},
        "substitution_mismatches": problems[:20],
        "n_substitution_mismatch": len(problems),
        "offset_applied": offsets,
        "unsupported_labels": unparsed,
        "csv_sha256": hashlib.sha256(
            open(os.path.join(CACHE, "m2or_receptor_sequences.csv"), "rb").read()).hexdigest(),
    }
    with open(os.path.join(CACHE, "m2or_receptor_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps({k: manifest[k] for k in
                      ("status_counts", "sequence_len", "n_substitution_mismatch")}, indent=1))

    if args.esm:
        cache_esm(df)
    if args.tokens:
        cache_tokens(df)


def cache_esm(df):
    import torch
    from transformers import AutoModel, AutoTokenizer

    ok = df[df.sequence.notna() & (df.sequence != "")].reset_index(drop=True)
    print(f"[esm] embedding {len(ok)} sequences from {ESM_DIR}")
    tok = AutoTokenizer.from_pretrained(ESM_DIR)
    model = AutoModel.from_pretrained(ESM_DIR, torch_dtype=torch.float16).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)

    embs = []
    with torch.no_grad():
        for i in range(0, len(ok), 8):
            batch = ok.sequence[i:i + 8].tolist()
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=1024).to(dev)
            out = model(**enc).last_hidden_state.float()             # [B,T,H]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            embs.append(((out * mask).sum(1) / mask.sum(1)).cpu().numpy().astype("float16"))
            if i % 80 == 0:
                print(f"[esm] {i}/{len(ok)}", flush=True)
    E = np.concatenate(embs)
    np.savez_compressed(os.path.join(CACHE, "esm2_m2or_receptors.npz"),
                        entry=np.array(ok.entry), emb=E)
    print(f"[esm] {E.shape} -> cache/esm2_m2or_receptors.npz")


def cache_tokens(df):
    """Residue-level ESM-2 tokens (special tokens removed), concatenated + offsets.

    n = 423 receptors x ~314 residues x 1280 d x fp16 ~ 340 MB on disk; the model
    gathers from `offsets` so padding is never stored.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    ok = df[df.sequence.notna() & (df.sequence != "")].reset_index(drop=True)
    print(f"[tokens] embedding {len(ok)} sequences from {ESM_DIR}")
    tok = AutoTokenizer.from_pretrained(ESM_DIR)
    model = AutoModel.from_pretrained(ESM_DIR, torch_dtype=torch.float16).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)

    chunks, lengths = [], []
    with torch.no_grad():
        for i in range(0, len(ok), 8):
            batch = ok.sequence[i:i + 8].tolist()
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=1024).to(dev)
            h = model(**enc).last_hidden_state                     # [B,T,H]
            for j, L in enumerate(enc["attention_mask"].sum(1).tolist()):
                # drop <cls> at 0 and <eos> at L-1 -> residues only
                chunks.append(h[j, 1:L - 1].to(torch.float16).cpu().numpy())
                lengths.append(L - 2)
            if i % 120 == 0:
                print(f"[tokens] {i}/{len(ok)}", flush=True)
    T = np.concatenate(chunks)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    np.savez(os.path.join(CACHE, "esm2_m2or_tokens.npz"),
             entry=np.array(ok.entry), tokens=T, offsets=offsets, lengths=np.array(lengths))
    print(f"[tokens] {T.shape} ({T.nbytes/1e6:.0f} MB) -> cache/esm2_m2or_tokens.npz")


if __name__ == "__main__":
    main()
