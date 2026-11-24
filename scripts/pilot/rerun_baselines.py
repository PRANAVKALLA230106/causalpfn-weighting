"""Recompute selected baselines on the exact draws of an existing raw.csv; writes raw_v2.csv.
python -I rerun_baselines.py --suite ric_hard --raw results/ric_hard/raw.csv --ests GP-S,AIPW2"""
import argparse, os, sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import pandas as pd
from cfm.dgp import make_suite, make_ric_hard, RIC_HARD_GRID
from cfm.baselines import BASELINES

ap = argparse.ArgumentParser()
ap.add_argument("--suite", required=True); ap.add_argument("--raw", required=True); ap.add_argument("--ests", required=True)
ap.add_argument("--n_dgp", type=int, default=20)
a = ap.parse_args()
suite = [make_ric_hard(i) for i in range(len(RIC_HARD_GRID))] if a.suite == "ric_hard" else make_suite(a.n_dgp, n=500)
by_id = {f"{g.family}-{g.seed}": g for g in suite}
df = pd.read_csv(a.raw); ests = a.ests.split(",")
stem = os.path.splitext(a.raw)[0]
out = stem + "_v2_partial.csv"
done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(columns=["dgp_id", "rep"] + ests)
keys = set(zip(done.dgp_id, done.rep)); t0 = time.time(); new = []
for i, (gid, rep) in enumerate(zip(df.dgp_id, df.rep)):
    if (gid, rep) in keys: continue
    d = by_id[gid].sample(int(rep))
    assert abs(d["psi"] - df.loc[i, "psi"]) < 1e-12
    row = {"dgp_id": gid, "rep": rep}
    for e in ests: row[e] = BASELINES[e](d["X"], d["t"], d["y"])
    pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
    if i % 20 == 0: print(f"{i}/{len(df)} {time.time()-t0:.0f}s", flush=True)
v = pd.read_csv(out)
m = df.drop(columns=[c for c in ests if c in df.columns]).merge(v, on=["dgp_id", "rep"])
m.to_csv(stem + "_v2.csv", index=False); print("wrote", len(m))
