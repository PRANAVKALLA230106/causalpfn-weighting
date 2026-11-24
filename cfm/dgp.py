"""Synthetic DGP families with oracle mu0, mu1, e and sample effect psi.

A DGP is a frozen set of random functional parameters; each replication redraws
(X, U, eps).  T = 1{U < e(x)}.  Coupled draws (same X, U, eps) are used only for
propensity swaps D1 -> D2 that share outcome surfaces but differ in e(x).

Families
--------
ric      : strong confounding, small tau (regularisation-induced-confounding regime,
           after Hahn et al. 2018).  Propensity is a function of the prognostic score
           mu0(x), so outcome-model shrinkage leaves bias.
overlap  : limited overlap with heterogeneous tau(x).  Propensity is steep in a few
           covariates, pushing e(x) toward {0,1}.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


def _sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


@dataclass
class DGP:
    family: str
    seed: int
    p: int
    n: int
    strength: float               # confounding strength knob
    noise: float
    beta: np.ndarray = field(repr=False)     # linear prognostic coefs
    nl: np.ndarray = field(repr=False)       # nonlinear prognostic coefs
    gamma: np.ndarray = field(repr=False)    # propensity direction (overlap family)
    tau0: float = 0.0
    tau_coef: np.ndarray = field(default=None, repr=False)
    e_clip: float = 0.01
    in_prior: bool | None = None  # filled once prior relationship is assessed
    dense_nl: np.ndarray | None = field(default=None, repr=False)  # (2, p): per-coordinate sin / quadratic coefs

    # --- oracle surfaces -------------------------------------------------
    def mu0(self, X):
        f = X @ self.beta
        if self.family != "prop_nl":
            f = f + self.nl[0] * np.sin(X[:, 0] * 1.5) + self.nl[1] * (X[:, 1] ** 2 - 1) + self.nl[2] * X[:, 0] * X[:, 2 % self.p]
        if self.dense_nl is not None:
            f = f + np.sin(1.5 * X) @ self.dense_nl[0] + (X ** 2 - 1) @ self.dense_nl[1]
        if self.family == "prop_nl":
            f = f + self.nl[0] * self._h(X)
        return f

    @staticmethod
    def _h(X):
        return np.sin(2.0 * X[:, 0]) + 0.7 * (X[:, 1] ** 2 - 1) + 0.5 * X[:, 0] * X[:, 1]

    def tau(self, X):
        out = np.full(len(X), self.tau0)
        if self.tau_coef is not None:
            out = out + X @ self.tau_coef
        return out

    def mu1(self, X):
        return self.mu0(X) + self.tau(X)

    def e(self, X):
        if self.family == "prop_nl":
            return np.clip(_sigmoid(self.strength * self._h(X)), self.e_clip, 1 - self.e_clip)
        if self.family in ("ric", "ric_hard", "ric_hard_clip"):
            s = self.mu0(X)
            s = (s - self._s_loc) / self._s_scale
            a = self.strength * s
        else:
            a = self.strength * (X @ self.gamma)
        return np.clip(_sigmoid(a), self.e_clip, 1 - self.e_clip)

    # --- sampling -------------------------------------------------------
    def draw_raw(self, rng):
        X = rng.standard_normal((self.n, self.p))
        U = rng.random(self.n)
        eps = rng.standard_normal(self.n)
        return X, U, eps

    def realise(self, X, U, eps):
        e = self.e(X)
        t = (U < e).astype(float)
        mu0, mu1 = self.mu0(X), self.mu1(X)
        y = np.where(t == 1, mu1, mu0) + self.noise * eps
        psi = float(np.mean(mu1 - mu0))
        return dict(X=X, t=t, y=y, e=e, mu0=mu0, mu1=mu1, psi=psi)

    def sample(self, rep: int, max_tries: int = 20):
        """Replication `rep`; re-draws if an arm is too small or > 1024 rows."""
        for k in range(max_tries):
            rng = np.random.default_rng([self.seed, rep, k])
            d = self.realise(*self.draw_raw(rng))
            n1 = int(d["t"].sum())
            if 20 <= n1 <= min(1024, self.n - 20) and self.n - n1 <= 1024:
                d["rep"] = rep
                return d
        raise RuntimeError("could not draw admissible arms")

    def __post_init__(self):
        # standardise prognostic score for the ric propensity
        rng = np.random.default_rng([self.seed, 10**6])
        s = self.mu0(rng.standard_normal((20000, self.p)))
        self._s_loc, self._s_scale = float(s.mean()), float(s.std() + 1e-12)


def make_dgp(family: str, seed: int, strength: float | None = None, n: int = 500) -> DGP:
    rng = np.random.default_rng([seed, 7])
    p = int(rng.choice([3, 5, 8]))
    beta = rng.normal(0, 1, p) * (rng.random(p) < 0.8)
    beta[0] = rng.choice([-1, 1]) * rng.uniform(0.8, 1.5)
    nl = rng.normal(0, 0.7, 3)
    noise = float(rng.uniform(0.3, 1.0))
    gamma = np.zeros(p)
    k = int(rng.integers(1, min(3, p) + 1))
    gamma[:k] = rng.normal(0, 1, k)
    gamma /= np.linalg.norm(gamma) + 1e-12
    if family == "ric":
        strength = float(rng.uniform(0.5, 3.0)) if strength is None else strength
        tau0 = float(rng.uniform(-0.3, 0.3))
        tau_coef = None
    elif family == "overlap":
        strength = float(rng.uniform(1.0, 4.0)) if strength is None else strength
        tau0 = float(rng.uniform(-1, 1))
        tau_coef = rng.normal(0, 0.5, p) * (rng.random(p) < 0.5)
    else:
        raise ValueError(family)
    return DGP(family=family, seed=seed, p=p, n=n, strength=strength, noise=noise, beta=beta, nl=nl,
               gamma=gamma, tau0=tau0, tau_coef=tau_coef)


def propensity_swap(d1: DGP, strength2: float) -> DGP:
    """D2: same outcome surfaces and noise as D1, different propensity strength.
    Use with identical (X, U, eps) draws for coupling."""
    d2 = replace(d1, strength=strength2)
    return d2


def coupled_pair(d1: DGP, d2: DGP, rep: int):
    rng = np.random.default_rng([d1.seed, rep, 999])
    X, U, eps = d1.draw_raw(rng)
    return d1.realise(X, U, eps), d2.realise(X, U, eps)


RIC_HARD_GRID = [(p, s, n) for p in (10, 20, 30) for s in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0) for n in (200, 1000)]


def make_ric_hard_clip(idx: int, clip: float = 0.1) -> DGP:
    """Same as make_ric_hard but with bounded overlap e in [clip, 1-clip] (positivity control).
    Same seeds, so outcome surfaces are identical to the unclipped suite."""
    g = make_ric_hard(idx)
    g.e_clip = clip
    g.family = "ric_hard_clip"
    return g


HARD_V2_GRID = [(p, s, n) for p in (5, 10, 20, 30, 50) for s in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0) for n in (200, 500, 1000)]


def make_hard_v2(cell: int, seed_rep: int = 0, clip: float = 0.01) -> DGP:
    """Full-scale hard grid (90 cells x seed replicates).  Same construction as make_ric_hard;
    clip=0.1 gives the bounded-overlap (positivity-control) twin with identical surfaces."""
    p, strength, n = HARD_V2_GRID[cell]
    seed = 60000 + 1000 * seed_rep + cell
    rng = np.random.default_rng([seed, 7])
    beta = rng.normal(0, 1, p) / np.sqrt(p)
    dense = rng.normal(0, 0.6, (2, p)) / np.sqrt(p)
    fam = "ric_hard" if clip <= 0.01 else "ric_hard_clip"
    return DGP(family=fam, seed=seed, p=p, n=n, strength=strength, noise=float(rng.uniform(0.3, 0.7)),
               beta=beta, nl=np.zeros(3), gamma=np.zeros(p), tau0=float(rng.uniform(-0.3, 0.3)),
               tau_coef=None, dense_nl=dense, e_clip=clip)


def make_ric_hard(idx: int) -> DGP:
    """Dense, weak, nonlinear confounding: every covariate enters mu0 a little,
    propensity = strength x standardised prognostic score.  Deterministic grid over
    (p, strength, n) so the strength trend is identified."""
    p, strength, n = RIC_HARD_GRID[idx]
    seed = 50000 + idx
    rng = np.random.default_rng([seed, 7])
    beta = rng.normal(0, 1, p) / np.sqrt(p)
    dense = rng.normal(0, 0.6, (2, p)) / np.sqrt(p)
    return DGP(family="ric_hard", seed=seed, p=p, n=n, strength=strength, noise=float(rng.uniform(0.3, 0.7)),
               beta=beta, nl=np.zeros(3), gamma=np.zeros(p), tau0=float(rng.uniform(-0.3, 0.3)),
               tau_coef=None, dense_nl=dense, e_clip=0.01)


def make_prop_nl(idx: int, strength: float | None = None, n: int = 500) -> DGP:
    """H3 family: nonlinear propensity h(x) on (x0, x1) that is NOT a function of mu0;
    mu0 is linear-plus-mild-nonlinear in all covariates.  Propensity information is
    therefore separable from prognostic information."""
    rng = np.random.default_rng([70000 + idx, 7]); p = 6
    beta = rng.normal(0, 1, p); nl = np.array([float(rng.uniform(0.4, 0.7)) * rng.choice([-1, 1]), 0.0, 0.0])
    dense = np.vstack([rng.normal(0, 0.3, p), rng.normal(0, 0.2, p)])
    dense[:, :2] = 0.0   # keep mu0's nonlinear part off the propensity coordinates
    s = float(rng.uniform(1.0, 2.5)) if strength is None else strength
    return DGP(family="prop_nl", seed=70000 + idx, p=p, n=n, strength=s, noise=0.5, beta=beta, nl=nl,
               gamma=np.zeros(p), tau0=float(rng.uniform(-0.5, 0.5)), tau_coef=None, dense_nl=dense, e_clip=0.02)


def make_suite(n_per_family: int, n: int = 500, seed0: int = 0):
    out = []
    for fam_i, fam in enumerate(("ric", "overlap")):
        for j in range(n_per_family):
            out.append(make_dgp(fam, seed=seed0 + 10000 * fam_i + j, n=n))
    return out
