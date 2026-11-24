"""H2 toolkit: what do the implied weights balance?

Regression-imputation ATE estimators with basis B are weighting estimators whose
treated-arm weights w1 = B1 (B1'B1)^+ bbar exactly balance B (sum w1 B1 = bbar,
the full-sample basis mean); as n grows they converge to the projection of
1/(n e(x)) onto span(B) (Han & Ding; Chattopadhyay & Zubizarreta).

Given a model's per-arm weights (w on treated rows, -w on control rows, each
summing to 1), we
  * find the sieve dimension K* whose Han-Ding weights best match them (cosine),
  * report imbalance on a rich dictionary vs Hajek-IPW,
  * report corr(n_arm * w, 1/e_hat).
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def sieve_basis(X: np.ndarray, degree: int = 3) -> np.ndarray:
    """Nested basis ordered by degree: [1, linear, quadratic, cubic ...], standardised.
    Columns are Gram-Schmidt-orthogonalised on the full sample so that 'first K
    columns' is a well-defined nested sieve."""
    Xs = StandardScaler().fit_transform(X)
    P = PolynomialFeatures(degree, include_bias=True).fit_transform(Xs)
    Q, R = np.linalg.qr(P)
    keep = np.abs(np.diag(R)) > 1e-8 * np.abs(np.diag(R)).max()
    return Q[:, keep] * np.sqrt(len(X))


def han_ding_weights(B: np.ndarray, arm: np.ndarray) -> np.ndarray:
    """Arm weights (summing to 1) of the OLS imputation estimator of E[mu_arm(X)]."""
    Ba = B[arm]
    bbar = B.mean(0)
    return Ba @ np.linalg.pinv(Ba.T @ Ba) @ bbar


def arm_weights_from_decomp(w: np.ndarray, t: np.ndarray):
    """Split full ATE weights into treated weights (sum 1) and control weights (sum 1)."""
    return w[t == 1], -w[t == 0]


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def k_star(w_arm: np.ndarray, X: np.ndarray, arm: np.ndarray, degree: int = 3, Ks=None):
    """Return (K*, cos at K*, list of (K, cos)) comparing de-meaned (net) weights."""
    B = sieve_basis(X, degree)
    Ks = Ks or list(range(1, B.shape[1] + 1))
    na = arm.sum()
    out = []
    for K in Ks:
        if K >= na:
            break
        hd = han_ding_weights(B[:, :K], arm)
        out.append((K, cosine(w_arm - 1 / na, hd - 1 / na) if K > 1 else np.nan))
    valid = [(k, c) for k, c in out if np.isfinite(c)]
    kb, cb = max(valid, key=lambda kc: kc[1]) if valid else (np.nan, np.nan)
    return kb, cb, out


def imbalance(w_arm: np.ndarray, F: np.ndarray, arm: np.ndarray) -> np.ndarray:
    """Per-feature standardised imbalance |sum w f - mean_all f| / sd_all f."""
    target = F.mean(0); sd = F.std(0) + 1e-12
    return np.abs(w_arm @ F[arm] - target) / sd


def rich_dictionary(X: np.ndarray, mu0=None, mu1=None, n_rff: int = 200, seed: int = 0) -> np.ndarray:
    """Degree-3 polynomials + random Fourier features (RBF, median bandwidth) + optional
    oracle outcome surfaces (the 'mu-relevant features')."""
    rng = np.random.default_rng(seed)
    Xs = StandardScaler().fit_transform(X)
    P = PolynomialFeatures(3, include_bias=False).fit_transform(Xs)
    d = np.sqrt(np.median(((Xs[:200, None] - Xs[None, :200]) ** 2).sum(-1)))
    W = rng.normal(0, 1 / max(d, 1e-6), (X.shape[1], n_rff)); b = rng.uniform(0, 2 * np.pi, n_rff)
    R = np.sqrt(2) * np.cos(Xs @ W + b)
    cols = [P, R] + [c[:, None] for c in (mu0, mu1) if c is not None]
    return np.column_stack(cols)


def hajek_arm_weights(e_hat: np.ndarray, t: np.ndarray):
    w1 = 1 / e_hat[t == 1]; w0 = 1 / (1 - e_hat[t == 0])
    return w1 / w1.sum(), w0 / w0.sum()


def h2_report(w: np.ndarray, X: np.ndarray, t: np.ndarray, e_hat: np.ndarray, mu0=None, mu1=None, degree=3):
    tr, ct = t == 1, t == 0
    w1, w0 = arm_weights_from_decomp(w, t)
    h1, h0 = hajek_arm_weights(e_hat, t)
    F = rich_dictionary(X, mu0, mu1)
    rep = {}
    for name, wa, ha, arm, inv in (("treated", w1, h1, tr, 1 / e_hat[tr]), ("control", w0, h0, ct, 1 / (1 - e_hat[ct]))):
        K, c, curve = k_star(wa, X, arm, degree)
        rep[name] = dict(
            K_star=K, cos_K_star=c,
            imb_model_mean=float(imbalance(wa, F, arm).mean()), imb_model_max=float(imbalance(wa, F, arm).max()),
            imb_hajek_mean=float(imbalance(ha, F, arm).mean()), imb_hajek_max=float(imbalance(ha, F, arm).max()),
            imb_uniform_mean=float(imbalance(np.full(arm.sum(), 1 / arm.sum()), F, arm).mean()),
            corr_w_inv_e=float(np.corrcoef(arm.sum() * wa, inv)[0, 1]),
            corr_hajek_inv_e=float(np.corrcoef(arm.sum() * ha, inv)[0, 1]),
            neg_weight_frac=float((wa < 0).mean()),
        )
    return rep
