"""Numerical checks of the proposal's 'setup facts' and the 1e-6 reproduction gate.

Usage: python -I check_decomposition.py [--ckpt PATH]
Without --ckpt a randomly initialised CausalPFN-architecture network is used:
the setup facts follow from preprocessing, so they must hold for any weights.
"""
import argparse, sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from cfm.decomp import decompose_ate, random_init_model, load_pretrained, tau_only, cepo_standardised, fp64_mode
from cfm.dgp import make_dgp

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None)
ap.add_argument("--n", type=int, default=300)
args = ap.parse_args()

model = load_pretrained(args.ckpt) if args.ckpt else random_init_model(seed=0)
label = "pretrained" if args.ckpt else "random-init"
res = {"model": label}

for fam in ("ric", "overlap"):
    dgp = make_dgp(fam, seed=1, n=args.n)
    d = dgp.sample(0)
    X, t, y = d["X"], d["t"], d["y"]
    for mode in ("S", "RA"):
        D = decompose_ate(model, X, t, y, mode=mode)
        key = f"{fam}/{mode}"
        res[key] = dict(tau=D.tau, dim=D.dim, psi=d["psi"], euler_resid=D.euler_resid,
                        armsum_t=D.armsum_t, armsum_c=D.armsum_c, split_resid=D.split_resid,
                        correction=D.correction, cap=D.cap, within_cap=abs(D.correction) <= D.cap)

    # homogeneity: tau(c*y) = c*tau(y)
    t1 = tau_only(model, X, t, y); t2 = tau_only(model, X, t, 3.7 * y)
    res[f"{fam}/homog_resid"] = t2 - 3.7 * t1
    # within-arm shift: adding c to treated outcomes moves tau by exactly c
    y_sh = y + 2.5 * t
    res[f"{fam}/treated_shift_resid"] = tau_only(model, X, t, y_sh) - t1 - 2.5
    # mu1-hat invariant to positive affine map of control outcomes
    with fp64_mode(), torch.no_grad():
        mdl = model.double()
        def mu1hat(yv):
            Xt = torch.as_tensor(X); tt = torch.as_tensor(t); yt = torch.as_tensor(yv)
            tr, ct = tt == 1, tt == 0
            y1m = yt[tr].mean(); s1 = torch.sqrt(((yt[tr]-y1m)**2).mean()) + 1e-20
            y0m = yt[ct].mean(); s0 = torch.sqrt(((yt[ct]-y0m)**2).mean()) + 1e-20
            z = torch.where(tr, (yt-y1m)/s1, (yt-y0m)/s0)
            m1 = cepo_standardised(mdl, Xt, tt, z, Xt, torch.ones(len(yv)))
            return (y1m + s1*m1).numpy()
        y_aff = np.where(t == 0, 5.0 * y - 3.0, y)
        res[f"{fam}/mu1_affine_control_maxdiff"] = float(np.abs(mu1hat(y) - mu1hat(y_aff)).max())

# ---- 1e-6 reproduction gate vs the public ATEEstimator path (float32)
from causalpfn import ATEEstimator
class _Est(ATEEstimator):
    def load_model(self):
        self.icl_model = model.float()
        self.max_feature_size = self.icl_model.model.num_features - 1
from sklearn.decomposition import TruncatedSVD
dgp = make_dgp("ric", seed=2, n=args.n); d = dgp.sample(0)
X32, t32, y32 = d["X"].astype(np.float32), d["t"].astype(np.float32), d["y"].astype(np.float32)
est = _Est(device="cpu"); est.fit(X32, t32, y32)
for mode in ("S", "RA"):
    official = float(est.estimate_ate(algorithm=mode))
    direct32 = tau_only(model, X32, t32, y32, mode=mode, fp64=False)
    direct64 = tau_only(model, X32.astype(np.float64), t32.astype(np.float64), y32.astype(np.float64), mode=mode)
    res[f"repro/{mode}"] = dict(official=official, direct_fp32=direct32, direct_fp64=direct64,
                                absdiff_fp32=abs(official - direct32), absdiff_fp64=abs(official - direct64),
                                pass_1e6=abs(official - direct32) < 1e-6)

print(json.dumps(res, indent=1, default=float))
