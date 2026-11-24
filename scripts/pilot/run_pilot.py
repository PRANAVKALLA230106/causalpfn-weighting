"""Pilot: all estimators on a DGP suite, cached per (dgp, rep).

python -I run_pilot.py --n_dgp 20 --reps 20 [--ckpt causalpfn_v0.pt] --out results/pilot
"""
import argparse, os, sys, time, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from cfm.dgp import make_suite, make_ric_hard, make_ric_hard_clip, RIC_HARD_GRID
from cfm.baselines import BASELINES, oracle_fluctuation
from cfm.decomp import load_pretrained, random_init_model, tau_only, tau_both
from cfm.kappa import per_dgp, family_summary

ap = argparse.ArgumentParser()
ap.add_argument("--n_dgp", type=int, default=20)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--ckpt", default=None)
ap.add_argument("--suite", default="base", choices=["base", "ric_hard", "ric_hard_clip"])
ap.add_argument("--skip", default="", help="comma list of baselines to skip")
ap.add_argument("--out", default="results/pilot")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

model = load_pretrained(args.ckpt) if args.ckpt else random_init_model(seed=0)
pfn_tag = "CausalPFN" if args.ckpt else "RandInitPFN"
skip = set(s for s in args.skip.split(",") if s)
suite = (make_suite(args.n_dgp, n=args.n) if args.suite == "base" else
         [(make_ric_hard if args.suite == "ric_hard" else make_ric_hard_clip)(i) for i in range(len(RIC_HARD_GRID))])
cache = os.path.join(args.out, "raw.csv")
done = set()
if os.path.exists(cache):
    old = pd.read_csv(cache); done = set(zip(old.dgp_id, old.rep))

t0 = time.time()
with open(cache, "a") as fh:
    header = not os.path.exists(cache) or os.path.getsize(cache) == 0
    for i, dgp in enumerate(suite):
        gid = f"{dgp.family}-{dgp.seed}"
        for r in range(args.reps):
            if (gid, r) in done:
                continue
            d = dgp.sample(r)
            X, t, y = d["X"], d["t"], d["y"]
            row = dict(dgp_id=gid, family=dgp.family, strength=dgp.strength, p=dgp.p, n=dgp.n, rep=r,
                       n1=int(t.sum()), psi=d["psi"], oracle=oracle_fluctuation(d))
            for name, f in BASELINES.items():
                if name not in skip:
                    row[name] = f(X, t, y)
            row[f"{pfn_tag}-S"], row[f"{pfn_tag}-RA"] = tau_both(model, X, t, y)
            pd.DataFrame([row]).to_csv(fh, header=header, index=False); header = False; fh.flush()
        print(f"[{i+1}/{len(suite)}] {gid} done  {time.time()-t0:.0f}s", flush=True)

df = pd.read_csv(cache)
ests = [c for c in df.columns if c not in ("dgp_id", "family", "strength", "p", "n", "rep", "n1", "psi", "oracle", "DiM")]
summ = []
for e in ests:
    pdg = per_dgp(df, e); pdg.to_csv(os.path.join(args.out, f"per_dgp_{e}.csv"), index=False)
    fs = family_summary(pdg); fs.insert(0, "estimator", e); summ.append(fs)
S = pd.concat(summ); S.to_csv(os.path.join(args.out, "summary.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(S[["estimator", "family", "n_dgp", "kappa_median", "kappa_median_ci", "kappa_pooled", "trend_strength", "trend_ci", "slope_median"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
