"""Per-conformer geometry features for the A2 appendix (WP5)."""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p3.common import CACHE, PT, ensure_dirs, load_cache, load_sdf, npr_mass_weighted, polar_sasa, rg_mass_weighted
from p3.observables import match_conformers

def main():
    ensure_dirs()
    recs, split = load_cache()
    rows=[]; t0=time.time()
    for i,r in enumerate(recs):
        name=str(r["name"]); pos=np.asarray(r["pos"].numpy(),float); pi=np.asarray(r["pi"].numpy(),float)
        z=np.asarray(r["x"].numpy()[:,0],int); m=np.asarray([PT.GetAtomicWeight(int(zz)) for zz in z],float)
        dE=np.asarray(r["dE"].numpy(),float)
        try:
            mols=load_sdf(name); mapping,_=match_conformers(pos,mols)
        except Exception:
            mols=[]; mapping=[]
        for c in range(pos.shape[0]):
            n1,n2,_=npr_mass_weighted(pos[c],m)
            rg=rg_mass_weighted(pos[c],m)
            sasa=polar_sasa(mols[mapping[c]],0) if mapping else float("nan")
            rows.append({"name":name,"conf_idx":c,"npr1":n1,"npr2":n2,"rg":rg,"sasa_polar":sasa,"dE":dE[c],"pi":pi[c],"split":split[name]})
        if (i+1)%500==0: print(f"{i+1}/{len(recs)} {time.time()-t0:.0f}s",flush=True)
    df=pd.DataFrame(rows); df.to_csv(CACHE/"conformer_features.csv",index=False)
    print(df.shape, df.isna().sum().to_dict())
if __name__=="__main__": main()
