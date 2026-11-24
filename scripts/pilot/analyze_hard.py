"""H1 on the hard RIC suite: kappa by estimator, strength trend, breakdown by (p, n)."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from cfm.kappa import per_dgp

df = pd.read_csv("results/ric_hard/raw.csv")
print("rows", len(df), "dgps", df.dgp_id.nunique())
ests = ["CausalPFN-S", "CausalPFN-RA", "GP-S", "IPW", "Hajek", "S-OLS", "T-OLS"]
rng = np.random.default_rng(0)
meta = df.groupby("dgp_id")[["p", "n"]].first()
rows, per = [], {}
for e in ests:
    a = per_dgp(df, e).join(meta, on="dgp_id"); a = a[a.eligible].reset_index(drop=True); per[e] = a
    def st(h):
        w = h.bias_dim.abs()
        return np.array([(h.kappa * w).sum() / w.sum(), h.kappa.median(), np.polyfit(h.strength, h.kappa, 1)[0], h.slope.median()])
    pt = st(a); bs = np.array([st(a.iloc[rng.integers(0, len(a), len(a))]) for _ in range(4000)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0); lo90, hi90 = np.percentile(bs, [5, 95], axis=0)
    rows.append(dict(est=e, n_dgp=len(a), kappa_pooled=pt[0], ci=f"[{lo[0]:.3f},{hi[0]:.3f}]", kappa_median=pt[1],
                     upper95_below_0p9=hi[0] < 0.9, TOST_0p9_1p1=(lo90[0] > 0.9) & (hi90[0] < 1.1),
                     trend=pt[2], trend_ci=f"[{lo[2]:.3f},{hi[2]:.3f}]", slope=pt[3],
                     rmse_med=per_dgp(df, e).rmse_est.median()))
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).round(3).to_string(index=False))

# paired PFN - GP
j = per["CausalPFN-S"][["dgp_id", "kappa", "bias_dim"]].merge(per["GP-S"][["dgp_id", "kappa"]], on="dgp_id", suffixes=("", "_gp"))
def dd(h):
    w = h.bias_dim.abs(); return ((h.kappa - h.kappa_gp) * w).sum() / w.sum()
b = np.array([dd(j.iloc[rng.integers(0, len(j), len(j))]) for _ in range(4000)])
print(f"PFN-S minus GP-S (|bias|-weighted): {dd(j):+.3f}  95% CI [{np.percentile(b,2.5):+.3f},{np.percentile(b,97.5):+.3f}]  n={len(j)}")

# breakdown table: kappa_d per (p, strength) and n
for e in ("CausalPFN-S", "GP-S"):
    a = per[e]
    print(f"\n{e}: kappa_d by strength (rows) x (p,n) (cols)")
    print(a.pivot_table(index="strength", columns=["p", "n"], values="kappa").round(2).to_string())
pd.concat([per[e].assign(est=e) for e in ests]).to_csv("results/ric_hard/per_dgp_all.csv", index=False)
