"""H3: where does propensity information sit, and is it used?

Probes: ridge regression (5-fold CV R^2) from each layer's context-row residual
stream (rows carry x, t and the y-embedding) to
  * logit_e      : logit e(x_i)
  * e_perp       : logit e residualised on [1, x, mu0(x), t]  -> propensity info that is
                   not linear-in-x, not prognostic, and not the row's own treatment bit
  * nuisance     : a nonlinear function of covariates the DGP never uses for e or mu,
                   residualised the same way (matched-difficulty control concept)
Controls: raw-x probe (degree-3 polynomial of x, same residualised targets);
random-init network; T shuffled across context rows (destroys the T-x association that
defines propensity while leaving x and y intact).

Erasure: LEACE (Belrose et al. 2023), closed form for a k-dim concept Z:
  r(h) = h - W^+ P W (h - mu),   W = Sigma_hh^{-1/2},  P = proj onto col(W Sigma_hz)
applied to context rows only, at chosen layers, inside the forward pass.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import cfm.decomp as dc


def residualise(target, *covs):
    A = np.column_stack([np.ones(len(target))] + [c if c.ndim > 1 else c[:, None] for c in covs])
    return target - A @ np.linalg.lstsq(A, target, rcond=None)[0]


def concepts(d):
    X, t, e, mu0 = d["X"], d["t"], d["e"], d["mu0"]
    le = np.log(e / (1 - e))
    nuis = np.sin(2.0 * X[:, 2]) + 0.7 * (X[:, 3] ** 2 - 1) + 0.5 * X[:, 2] * X[:, 3]
    return dict(logit_e=le, e_perp=residualise(le, X, mu0, t), nuisance=residualise(nuis, X, mu0, t))


def cv_r2(H, z, seed=0):
    m = RidgeCV(alphas=np.logspace(-2, 5, 15))
    pred = cross_val_predict(m, StandardScaler().fit_transform(H), z, cv=KFold(5, shuffle=True, random_state=seed))
    return float(1 - ((z - pred) ** 2).mean() / z.var())


@torch.no_grad()
def context_streams(model, X, t, y):
    """Residual stream of every context row after embedding (layer -1) and after each layer."""
    n = len(y); acts = {}
    def mk(li):
        def f(src, L):
            acts[li] = src[:L, 0].detach().double().cpu().numpy().copy(); return src
        return f
    nl = len(model.model.transformer_encoder)
    dc.LAYER_HOOKS.clear(); dc.LAYER_HOOKS.update({li: mk(li) for li in range(nl)})
    try:
        dc.tau_both(model, X, t, y)
    finally:
        dc.LAYER_HOOKS.clear()
    return acts


def probe_profile(model, d, t_override=None):
    t = d["t"] if t_override is None else t_override
    acts = context_streams(model, d["X"], t, d["y"])
    C = concepts(d)
    return {name: [cv_r2(acts[l], z) for l in sorted(acts)] for name, z in C.items()}


def rawx_r2(d):
    P = PolynomialFeatures(3, include_bias=False).fit_transform(StandardScaler().fit_transform(d["X"]))
    return {name: cv_r2(P, z) for name, z in concepts(d).items()}


# ---------------- LEACE ----------------
class Leace:
    def __init__(self, H: np.ndarray, Z: np.ndarray, eig_floor: float = 1e-8):
        Z = Z.reshape(len(Z), -1)
        self.mu = H.mean(0)
        Hc, Zc = H - self.mu, Z - Z.mean(0)
        S = Hc.T @ Hc / len(H)
        evals, evecs = np.linalg.eigh(S)
        keep = evals > eig_floor * evals.max()
        V, lam = evecs[:, keep], evals[keep]
        self.W = V @ np.diag(lam ** -0.5) @ V.T           # Sigma^{-1/2} (pseudo)
        self.Winv = V @ np.diag(lam ** 0.5) @ V.T         # its pseudo-inverse
        Sxz = Hc.T @ Zc / len(H)
        U, s, _ = np.linalg.svd(self.W @ Sxz, full_matrices=False)
        U = U[:, s > 1e-12 * max(s.max(), 1e-300)]
        self.P = U @ U.T

    def __call__(self, H):
        return H - (H - self.mu) @ (self.Winv @ self.P @ self.W).T


def random_direction_eraser(H, rank, seed):
    """Control: erase a random rank-k direction (orthogonal projection in whitened space)."""
    rng = np.random.default_rng(seed)
    return Leace(H, rng.standard_normal((len(H), rank)) @ rng.standard_normal((rank, rank)))


@torch.no_grad()
def tau_with_erasure(model, X, t, y, erasers: dict):
    """erasers: {layer: callable on context-row array}.  Returns (S, RA)."""
    def mk(f):
        def hook(src, L):
            h = src[:L, 0].detach().double().cpu().numpy()
            src = src.clone(); src[:L, 0] = torch.as_tensor(f(h), dtype=src.dtype, device=src.device); return src
        return hook
    dc.LAYER_HOOKS.clear(); dc.LAYER_HOOKS.update({l: mk(f) for l, f in erasers.items()})
    try:
        return dc.tau_both(model, X, t, y)
    finally:
        dc.LAYER_HOOKS.clear()


def fit_erasers_sequential(model, d, concept: np.ndarray, layers):
    """Fit LEACE at each layer on the stream produced with erasure already applied at
    earlier layers (so the all-layer intervention is self-consistent)."""
    erasers = {}
    for l in sorted(layers):
        acts = {}
        def cap(src, L, l=l):
            acts[l] = src[:L, 0].detach().double().cpu().numpy().copy(); return src
        def mk(f):
            def hook(src, L):
                h = src[:L, 0].detach().double().cpu().numpy(); src = src.clone()
                src[:L, 0] = torch.as_tensor(f(h), dtype=src.dtype, device=src.device); return src
            return hook
        hooks = {k: mk(f) for k, f in erasers.items()}
        prev = hooks.get(l)
        hooks[l] = (lambda src, L, prev=prev: cap(prev(src, L) if prev else src, L))
        dc.LAYER_HOOKS.clear(); dc.LAYER_HOOKS.update(hooks)
        try:
            dc.tau_both(model, d["X"], d["t"], d["y"])
        finally:
            dc.LAYER_HOOKS.clear()
        erasers[l] = Leace(acts[l], concept)
    return erasers


# ---------------- attention mass on treated keys ----------------
@torch.no_grad()
def attention_treated_mass(model, X, t, y):
    """For every layer and head, the attention mass each query row places on treated context
    rows.  Returns array (layers, heads, n_queries) for queries [x at t=0 ; x at t=1]."""
    from causalpfn.models import transformer_layer as tl
    import math as _m, torch.nn.functional as F
    store = []
    tmask = torch.as_tensor(t == 1, device=dc.DEVICE)
    orig = tl.TransformerEncoderLayer.forward

    def fwd(self, x, context_length):
        xx = x.transpose(0, 1); B, L, _ = xx.size()
        h = self.attn_norm(xx)
        q, _, _ = self.q_proj(h).chunk(3, dim=-1)
        k, _ = self.kv_proj(h[:, :context_length]).chunk(2, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, context_length, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * _m.log2(context_length) / 10
        q = self.q_norm(q); k = self.k_norm(k)
        att = torch.softmax(q @ k.transpose(-1, -2) / _m.sqrt(self.head_dim), dim=-1)   # B,H,L,C
        store.append(att[0, :, context_length:, :][..., tmask].sum(-1).double().cpu().numpy())
        return orig(self, x, context_length)

    tl.TransformerEncoderLayer.forward = fwd
    try:
        dc.tau_both(model, X, t, y)
    finally:
        tl.TransformerEncoderLayer.forward = orig
    return np.stack(store)


def attention_propensity_stats(mass, d):
    """corr over query rows between treated-key mass and e(x), raw and after partialling out
    a degree-2 polynomial of x (i.e. beyond what similarity in x alone would give)."""
    from sklearn.preprocessing import PolynomialFeatures
    n = len(d["t"]); e = d["e"]; le = np.log(e / (1 - e))
    P = PolynomialFeatures(2, include_bias=True).fit_transform(d["X"])
    def partial(a, b):
        ra = a - P @ np.linalg.lstsq(P, a, rcond=None)[0]; rb = b - P @ np.linalg.lstsq(P, b, rcond=None)[0]
        return float(np.corrcoef(ra, rb)[0, 1])
    Lyr, H, Q = mass.shape
    raw = np.zeros((Lyr, H)); par = np.zeros((Lyr, H))
    for l in range(Lyr):
        for h in range(H):
            m = 0.5 * (mass[l, h, :n] + mass[l, h, n:])       # average over the t=0 / t=1 query copies
            raw[l, h] = np.corrcoef(m, le)[0, 1]; par[l, h] = partial(m, le)
    return raw, par
