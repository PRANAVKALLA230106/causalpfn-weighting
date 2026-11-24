"""IHDP benchmark: H1 estimators on Hill surfaces A and B (simulated) and the 10 official
NPCI files, plus H2 linear/nonlinear attribution for CausalPFN-S and the GP S-learner.

python -I run_ihdp.py --surfaces 60 --reps 4 --out results/ihdp
"""
import argparse, json, os, sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, torch
import cfm.decomp as dc
from cfm.decomp import load_pretrained, tau_both, decompose_ate
from cfm.ihdp import load_ihdp, npci_file, IHDPSurface
from cfm.baselines import BASELINES
from cfm.gp_torch import gp_s_learner_torch, gp_implied_weights
from cfm.experiments import heldout_fit, _attrib

ap = argparse.ArgumentParser()
ap.add_argument("--surfaces", type=int, default=60); ap.add_argument("--reps", type=int, default=4)
ap.add_argument("--h2_surfaces", type=int, default=20); ap.add_argument("--out", default="results/ihdp")
ap.add_argument("--ckpt", default=None); ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args(); torch.set_num_threads(a.threads); os.makedirs(a.out, exist_ok=True)
model = load_pretrained(a.ckpt); X, t, _ = load_ihdp()
EST = ["DiM", "S-OLS", "T-OLS", "IPW", "Hajek", "AIPW2", "kNN"]
path = os.path.join(a.out, "h1_ihdp.csv")
done = set(map(tuple, pd.read_csv(path)[["dgp_id", "rep"]].values)) if os.path.exists(path) else set()
t0 = time.time()

def row_for(gid, family, surface, rep, d):
    Xd, td, y = d["X"], d["t"], d["y"]
    S, RA = tau_both(model, Xd, td, y); HO, fact = heldout_fit(model, Xd, td, y, seed=rep)
    mu_f = np.where(td == 1, d["mu1"], d["mu0"]); r2 = lambda p_: 1 - ((y - p_) ** 2).mean() / y.var()
    return dict(dgp_id=gid, family=family, surface=surface, rep=rep, psi=d["psi"], oracle=np.nan, strength=np.nan,
                **{k: BASELINES[k](Xd, td, y) for k in EST}, **{"CausalPFN-S": S, "CausalPFN-RA": RA, "CausalPFN-HO": HO},
                **{"GP-S": gp_s_learner_torch(Xd, td, y, device="cpu")}, fact_r2_pfn=r2(fact), fact_r2_oracle=r2(mu_f))

jobs = [(f"ihdp_npci-{k}", "ihdp_npci", k, 0, None) for k in range(1, 11)]
jobs += [(f"ihdp_{s}-{j}", f"ihdp_{s}", j, r, s) for j in range(a.surfaces) for s in ("A", "B") for r in range(a.reps)]
for gid, fam, surf, rep, setting in jobs:
    if (gid, rep) in done: continue
    d = npci_file(surf) if setting is None else IHDPSurface(setting, surf, X, t).sample(rep)
    pd.DataFrame([row_for(gid, fam, surf, rep, d)]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    if rep == 0: print(f"[ihdp h1] {gid} {time.time()-t0:.0f}s", flush=True)

# ---- H2 attribution on a subset (rep 0)
p2 = os.path.join(a.out, "h2_ihdp.jsonl")
done2 = {json.loads(l)["dgp_id"] for l in open(p2)} if os.path.exists(p2) else set()
dc.CHECKPOINT_LAYERS = True
for s in ("A", "B"):
    for j in range(a.h2_surfaces):
        gid = f"ihdp_{s}-{j}"
        if gid in done2: continue
        d = IHDPSurface(s, j, X, t).sample(0); Xd, td, y = d["X"], d["t"], d["y"]
        D = decompose_ate(model, Xd, td, y, "S"); gtau, gw, gres = gp_implied_weights(Xd, td, y, device="cpu")
        dim_w = np.where(td == 1, 1 / td.sum(), -1 / (1 - td).sum())
        rec = dict(dgp_id=gid, setting=s, psi=d["psi"], dim=D.dim, tau_pfn=D.tau, tau_gp=gtau, euler=[D.euler_resid, gres],
                   dim_attrib=_attrib(dim_w, d), pfn_attrib=_attrib(D.w, d), gp_attrib=_attrib(gw, d),
                   labeldep=float(np.linalg.norm(D.w_labeldep) / (np.linalg.norm(D.w_labelfree) + np.linalg.norm(D.w_labeldep))))
        with open(p2, "a") as fh: fh.write(json.dumps(rec, default=float) + "\n")
        print(f"[ihdp h2] {gid} kappa_pfn={(D.tau-D.dim)/(d['psi']-D.dim):.2f} {time.time()-t0:.0f}s", flush=True)
print("done", flush=True)
