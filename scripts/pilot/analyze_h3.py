"""Summarise H3 (results/h3.jsonl)."""
import json
import numpy as np, pandas as pd
R_all = [json.loads(l) for l in open("results/h3.jsonl")]
R = [r for r in R_all if abs(r["tau_S"] - r["DiM"]) >= 0.1]   # non-negligible correction (pre-specified)
print("excluded (|tau-DiM|<0.1):", [r["idx"] for r in R_all if r not in R])
pilot = json.load(open("results/h3_pilot.json"))
L = pilot["l_star"]; print(f"datasets: {len(R)}  l* = {L} (fixed on pilot datasets 0-2)")

# probes
rows = []
for r in R:
    for cond in ("probe_real", "probe_shuf", "probe_rand"):
        for c, prof in r[cond].items():
            rows.append(dict(idx=r["idx"], cond=cond, concept=c, peak=max(prof), at_lstar=prof[L], layer0=prof[0], last=prof[-1]))
    for c, v in r["rawx"].items():
        rows.append(dict(idx=r["idx"], cond="rawx_poly3", concept=c, peak=v, at_lstar=v, layer0=v, last=v))
P = pd.DataFrame(rows)
print("\n[probe R^2, mean over datasets]\n", P.groupby(["concept", "cond"])[["layer0", "at_lstar", "peak", "last"]].mean().round(3).to_string())
real = P[P.cond == "probe_real"].set_index(["idx", "concept"]).at_lstar
shuf = P[P.cond == "probe_shuf"].set_index(["idx", "concept"]).at_lstar
diff = (real - shuf).unstack()
print("\n[R^2 drop from shuffling T at l*: mean, min]\n", diff.agg(["mean", "min"]).round(3).to_string())

# erasure
rows = []
for r in R:
    base = r["tau_S"]; corr = abs(base - r["DiM"]); need = abs(r["psi"] - r["DiM"])
    for c, e in r["erasure"].items():
        rows.append(dict(idx=r["idx"], concept=c, rel_lstar=(e["lstar_S"] - base) / corr,
                         rel_all=((e["all_S"] - base) / corr) if "all_S" in e else np.nan,
                         abs_lstar=abs(e["lstar_S"] - base) / corr,
                         abs_all=(abs(e["all_S"] - base) / corr) if "all_S" in e else np.nan,
                         abs_all_need=(abs(e["all_S"] - base) / need) if "all_S" in e else np.nan,
                         abs_lstar_need=abs(e["lstar_S"] - base) / need))
E = pd.DataFrame(rows)
summ = E.groupby("concept")[["abs_lstar", "abs_all", "abs_lstar_need", "abs_all_need"]].agg(["mean", "median", "max"]).round(3)
print("\n[|delta tau| / |tau - DiM| under erasure]\n", summ.to_string())
# TOST-style: upper 90% bootstrap bound of mean |delta|/|corr| < 0.10 ?
rng = np.random.default_rng(0)
for c in ("e_perp", "nuisance", "random_rank1", "logit_e"):
    for col in ("abs_lstar", "abs_all"):
        v = E[E.concept == c][col].dropna().values
        if len(v) == 0: continue
        b = [rng.choice(v, len(v)).mean() for _ in range(4000)]
        print(f"  {c:13s} {col:9s} mean={v.mean():.3f}  90% upper={np.percentile(b,95):.3f}  below 0.10: {np.percentile(b,95) < 0.10}")
ep = E[E.concept == "e_perp"].set_index("idx").abs_all; nu = E[E.concept == "nuisance"].set_index("idx").abs_all
print(f"\n e_perp vs nuisance (all layers): mean diff {(ep-nu).mean():+.3f}; e_perp larger in {(ep>nu).sum()}/{len(ep)}")
E.to_csv("results/h3_erasure_flat.csv", index=False); P.to_csv("results/h3_probes_flat.csv", index=False)
