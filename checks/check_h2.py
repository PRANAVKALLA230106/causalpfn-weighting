"""Validate the H2 toolkit on estimators whose answer is known:
 (1) OLS T-learner on the first K sieve columns -> K* must equal K, cos = 1,
     and its weights must exactly balance those K columns.
 (2) Hajek-IPW weights -> corr with 1/e_hat is 1 by construction.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np
from cfm.dgp import make_dgp
from cfm.h2 import sieve_basis, han_ding_weights, k_star, h2_report, imbalance
from cfm.baselines import _prop

d = make_dgp("ric", seed=3, n=500).sample(0)
X, t = d["X"], d["t"]; tr = t == 1
B = sieve_basis(X, 3)
for K_true in (4, 10, 20):
    w1 = han_ding_weights(B[:, :K_true], tr)
    K, c, _ = k_star(w1, X, tr)
    bal = np.abs(w1 @ B[tr][:, :K_true] - B[:, :K_true].mean(0)).max()
    print(f"K_true={K_true:2d}  K*={K}  cos={c:.6f}  max balance err={bal:.2e}  sum w={w1.sum():.6f}")

e_hat = _prop(X, t)
# build a fake 'ATE weight vector' from Hajek weights to exercise h2_report
w = np.zeros(len(t)); h1 = 1 / e_hat[tr]; h0 = 1 / (1 - e_hat[~tr])
w[tr] = h1 / h1.sum(); w[~tr] = -h0 / h0.sum()
rep = h2_report(w, X, t, e_hat, d["mu0"], d["mu1"])
print({k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in rep.items()})
