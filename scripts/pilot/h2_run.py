"""H2 on a DGP suite: implied weights of pretrained CausalPFN (S mode).

Exact identity (tau_hat = sum w y, y = mu_T(x) + eps):
  tau_hat - psi = [sum_{t} w1 mu1 - mean mu1] - [sum_{c} w0 mu0 - mean mu0] + sum w eps
so the error is the weights' imbalance on the true outcome surfaces plus noise.
We split mu_a = L_a (best linear-in-X fit on the full sample) + N_a (remainder) and
report which part the weights fail to balance.  Same for oracle Hajek-IPW.

Per dataset we also record K*, rich-dictionary imbalance vs Hajek (estimated e),
corr(n_a w, 1/e_true), negative-weight share, label-dependent share, and the
label-dependence check: cosine of w with weights recomputed after a fresh noise draw.

python -I h2_run.py --suite ric_hard --reps 0 --n_filter 200 --out results/h2_hard_n200.jsonl
"""
import argparse, json, os, sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from cfm.dgp import make_ric_hard, RIC_HARD_GRID, make_suite
from cfm.decomp import load_pretrained, decompose_ate
from cfm.h2 import h2_report, cosine
from cfm.baselines import _prop

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="ric_hard"); ap.add_argument("--reps", default="0")
ap.add_argument("--n_filter", type=int, default=None); ap.add_argument("--out", required=True)
ap.add_argument("--ckpt", default=None)
ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args()
torch.set_num_threads(a.threads)
model = load_pretrained(a.ckpt)
import cfm.decomp as _dc; _dc.CHECKPOINT_LAYERS = True
suite = [make_ric_hard(i) for i in range(len(RIC_HARD_GRID))] if a.suite == "ric_hard" else make_suite(20, n=500)
if a.n_filter: suite = [g for g in suite if g.n == a.n_filter]
reps = [int(r) for r in a.reps.split(",")]
done = set()
if os.path.exists(a.out):
    for l in open(a.out): j = json.loads(l); done.add((j["dgp_id"], j["rep"]))


def lin_split(X, mu):
    A = np.column_stack([np.ones(len(X)), X]); c = np.linalg.lstsq(A, mu, rcond=None)[0]
    L = A @ c; return L, mu - L


def imb(wa, f, mask):           # signed imbalance of arm weights on function f
    return float(wa @ f[mask] - f.mean())


t0 = time.time()
for g in suite:
    gid = f"{g.family}-{g.seed}"
    for r in reps:
        if (gid, r) in done: continue
        d = g.sample(r); X, t, y = d["X"], d["t"], d["y"]; tr, ct = t == 1, t == 0
        D = decompose_ate(model, X, t, y, mode="S")
        w1, w0 = D.w[tr], -D.w[ct]
        # Hajek with oracle e (reference: what full IPW balance would do)
        h1 = 1 / d["e"][tr]; h1 /= h1.sum(); h0 = 1 / (1 - d["e"][ct]); h0 /= h0.sum()
        L1, N1 = lin_split(X, d["mu1"]); L0, N0 = lin_split(X, d["mu0"])
        noise_term = float(D.w @ (y - np.where(tr, d["mu1"], d["mu0"])))
        err = D.tau - d["psi"]
        attrib = {}
        for nm, a1, a0 in (("pfn", w1, w0), ("hajek_oracle", h1, h0)):
            attrib[nm] = dict(lin=imb(a1, L1, tr) - imb(a0, L0, ct), nonlin=imb(a1, N1, tr) - imb(a0, N0, ct))
        # label-dependence: fresh noise draw, same X, t
        rng = np.random.default_rng([g.seed, r, 4242])
        y2 = np.where(tr, d["mu1"], d["mu0"]) + g.noise * rng.standard_normal(len(y))
        D2 = decompose_ate(model, X, t, y2, mode="S")
        e_hat = _prop(X, t)
        rep = h2_report(D.w, X, t, e_hat, d["mu0"], d["mu1"], degree=2 if g.p > 10 else 3)
        rec = dict(dgp_id=gid, rep=r, p=g.p, n=g.n, strength=g.strength, psi=d["psi"], dim=D.dim, tau=D.tau,
                   kappa_single=(D.tau - D.dim) / (d["psi"] - D.dim), err=err, euler=D.euler_resid,
                   err_identity_resid=err - (attrib["pfn"]["lin"] + attrib["pfn"]["nonlin"] + noise_term),
                   noise_term=noise_term, attrib=attrib,
                   dim_imb_lin=float(L1[tr].mean() - L1.mean() - (L0[ct].mean() - L0.mean())),
                   dim_imb_nonlin=float(N1[tr].mean() - N1.mean() - (N0[ct].mean() - N0.mean())),
                   labeldep_share=float(np.linalg.norm(D.w_labeldep) / (np.linalg.norm(D.w_labelfree) + np.linalg.norm(D.w_labeldep))),
                   cos_fresh_noise=cosine(D.w_net, D2.w_net),
                   corr_w_inv_e_true_t=float(np.corrcoef(w1, 1 / d["e"][tr])[0, 1]),
                   corr_w_inv_e_true_c=float(np.corrcoef(w0, 1 / (1 - d["e"][ct]))[0, 1]),
                   ess_t=float(1 / (w1 ** 2).sum()), ess_c=float(1 / (w0 ** 2).sum()),
                   ess_hajek_t=float(1 / (h1 ** 2).sum()), ess_hajek_c=float(1 / (h0 ** 2).sum()),
                   report=rep)
        with open(a.out, "a") as fh: fh.write(json.dumps(rec, default=float) + "\n")
        print(f"{gid} rep{r} p={g.p} n={g.n} s={g.strength} kappa={rec['kappa_single']:.2f} "
              f"resid={rec['err_identity_resid']:.1e} {time.time()-t0:.0f}s", flush=True)
