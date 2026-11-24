"""Exact GP S-learner in PyTorch (GPU-capable), mirroring baselines.gp_s_learner:
per-arm z-scored outcomes; features [standardised X, t]; ARD-RBF * constant + white noise;
length-scales initialised at sqrt(p) (t at 1); type-II ML by Adam then L-BFGS.

With hyper-parameters frozen, the GP ATE is degree-1 homogeneous in y (per-arm z-scoring),
so `gp_implied_weights` gives exact weights tau = sum w y via autograd: the reference
smoother for H2 reference-to-reference distances.
"""
from __future__ import annotations

import math
import numpy as np
import torch


def _features(X, t, dev, dt):
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return torch.as_tensor(np.column_stack([Xs, t]), dtype=dt, device=dev)


def _kern(A, B, log_ls, log_sf):
    A_ = A / log_ls.exp(); B_ = B / log_ls.exp()
    d2 = (A_ * A_).sum(1)[:, None] + (B_ * B_).sum(1)[None, :] - 2 * A_ @ B_.T
    return (2 * log_sf).exp() * torch.exp(-0.5 * d2.clamp_min(0))


def fit_gp(Z, z, iters_adam=150, iters_lbfgs=30):
    n, d = Z.shape
    p = d - 1
    log_ls = torch.tensor(np.r_[np.full(p, 0.5 * math.log(p)), 0.0], dtype=Z.dtype, device=Z.device, requires_grad=True)
    log_sf = torch.zeros((), dtype=Z.dtype, device=Z.device, requires_grad=True)
    log_sn = torch.tensor(0.5 * math.log(0.5), dtype=Z.dtype, device=Z.device, requires_grad=True)
    params = [log_ls, log_sf, log_sn]
    eye = torch.eye(n, dtype=Z.dtype, device=Z.device)

    def nll():
        K = _kern(Z, Z, log_ls.clamp(math.log(1e-2), math.log(1e3)), log_sf) + ((2 * log_sn).exp() + 1e-6) * eye
        L = torch.linalg.cholesky(K)
        a = torch.cholesky_solve(z[:, None], L)
        return 0.5 * (z[:, None] * a).sum() + torch.log(torch.diagonal(L)).sum()

    opt = torch.optim.Adam(params, lr=0.05)
    for _ in range(iters_adam):
        opt.zero_grad(); loss = nll(); loss.backward(); opt.step()
    lb = torch.optim.LBFGS(params, max_iter=iters_lbfgs, line_search_fn="strong_wolfe")
    def closure():
        lb.zero_grad(); l = nll(); l.backward(); return l
    try:
        lb.step(closure)
    except RuntimeError:
        pass
    return {k: v.detach() for k, v in zip(("log_ls", "log_sf", "log_sn"), params)}


def _ate_from_y(yt, t_t, Z, hp, tr, ct):
    n = len(yt)
    y1m, y0m = yt[tr].mean(), yt[ct].mean()
    s1 = torch.sqrt(((yt[tr] - y1m) ** 2).mean()); s0 = torch.sqrt(((yt[ct] - y0m) ** 2).mean())
    z = torch.where(tr, (yt - y1m) / s1, (yt - y0m) / s0)
    log_ls = hp["log_ls"].clamp(math.log(1e-2), math.log(1e3))
    K = _kern(Z, Z, log_ls, hp["log_sf"]) + ((2 * hp["log_sn"]).exp() + 1e-6) * torch.eye(n, dtype=Z.dtype, device=Z.device)
    alpha = torch.cholesky_solve(z[:, None], torch.linalg.cholesky(K))[:, 0]
    Z1 = Z.clone(); Z1[:, -1] = 1; Z0 = Z.clone(); Z0[:, -1] = 0
    m1 = _kern(Z1, Z, log_ls, hp["log_sf"]) @ alpha; m0 = _kern(Z0, Z, log_ls, hp["log_sf"]) @ alpha
    return ((y1m + s1 * m1) - (y0m + s0 * m0)).mean()


def gp_s_learner_torch(X, t, y, device=None, dtype=torch.float64, return_hp=False):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Z = _features(X, t, dev, dtype)
    tr_np = t == 1
    yt = torch.as_tensor(y, dtype=dtype, device=dev)
    tr = torch.as_tensor(tr_np, device=dev); ct = ~tr
    z = torch.where(tr, (yt - yt[tr].mean()) / yt[tr].std(unbiased=False), (yt - yt[ct].mean()) / yt[ct].std(unbiased=False))
    hp = fit_gp(Z, z)
    with torch.no_grad():
        tau = float(_ate_from_y(yt, None, Z, hp, tr, ct))
    return (tau, hp) if return_hp else tau


def gp_implied_weights(X, t, y, hp=None, device=None, dtype=torch.float64):
    """Exact weights of the GP S-learner ATE (hyper-parameters frozen)."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if hp is None:
        _, hp = gp_s_learner_torch(X, t, y, dev, dtype, return_hp=True)
    Z = _features(X, t, dev, dtype)
    yt = torch.as_tensor(y, dtype=dtype, device=dev).clone().requires_grad_(True)
    tr = torch.as_tensor(t == 1, device=dev); ct = ~tr
    tau = _ate_from_y(yt, None, Z, hp, tr, ct)
    w = torch.autograd.grad(tau, yt)[0]
    return float(tau.detach()), w.detach().cpu().numpy(), float(tau.detach() - (w * yt.detach()).sum())
