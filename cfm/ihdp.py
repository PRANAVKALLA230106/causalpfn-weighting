"""IHDP semi-synthetic benchmark (Hill 2011) on the real IHDP covariates and treatment.

Covariates/treatment: CEVAE's IHDP NPCI csv (747 units, 139 treated; 6 continuous,
standardised, + 19 binary).  Response surfaces (Hill 2011, Sec. 4):
  A (linear):      Y0 ~ N(X b_A, 1),            Y1 ~ N(X b_A + 4, 1),
                   b_A entries from {0,1,2,3,4} w.p. {.5,.2,.15,.1,.05}
  B (nonlinear):   Y0 ~ N(exp((X + W) b_B), 1), Y1 ~ N(X b_B - omega_B, 1),
                   W = 0.5 offset, b_B from {0,.1,.2,.3,.4} w.p. {.6,.1,.1,.1,.1},
                   omega_B set so the effect on the treated (CATT) = 4.
Each 'surface' is one coefficient draw; replications redraw only the noise.
psi = sample ATE = mean(mu1 - mu0).  The true propensity is unknown (real assignment).
"""
from __future__ import annotations

import os
import numpy as np

from cfm.paths import ihdp_csv as _ihdp_csv
_CSV = _ihdp_csv()


def load_ihdp(path: str = _CSV):
    D = np.loadtxt(path, delimiter=",")
    t = D[:, 0].astype(float); X = D[:, 5:]
    return X, t, D


def npci_file(k: int, base=os.path.dirname(_CSV)):
    D = np.loadtxt(os.path.join(base, f"ihdp_npci_{k}.csv"), delimiter=",")
    t = D[:, 0]; yf = D[:, 1]; mu0, mu1 = D[:, 3], D[:, 4]; X = D[:, 5:]
    return dict(X=X, t=t, y=yf, mu0=mu0, mu1=mu1, psi=float(np.mean(mu1 - mu0)))


class IHDPSurface:
    def __init__(self, setting: str, surface: int, X, t):
        assert setting in ("A", "B")
        self.setting, self.surface, self.X, self.t = setting, surface, X, t
        rng = np.random.default_rng([90000 + (0 if setting == "A" else 1), surface])
        p = X.shape[1]
        if setting == "A":
            b = rng.choice([0, 1, 2, 3, 4], size=p, p=[.5, .2, .15, .1, .05])
            self.mu0 = X @ b; self.mu1 = self.mu0 + 4.0
        else:
            b = rng.choice([0, .1, .2, .3, .4], size=p, p=[.6, .1, .1, .1, .1])
            self.mu0 = np.exp((X + 0.5) @ b)
            m1 = X @ b
            omega = float(np.mean(m1[t == 1] - self.mu0[t == 1]) - 4.0)
            self.mu1 = m1 - omega
        self.psi = float(np.mean(self.mu1 - self.mu0))
        self.family = f"ihdp_{setting}"; self.seed = surface
        self.strength = np.nan; self.p = p; self.n = len(t)

    def sample(self, rep: int):
        rng = np.random.default_rng([91000 + (0 if self.setting == "A" else 1), self.surface, rep])
        y0 = self.mu0 + rng.standard_normal(self.n); y1 = self.mu1 + rng.standard_normal(self.n)
        y = np.where(self.t == 1, y1, y0)
        return dict(X=self.X, t=self.t, y=y, mu0=self.mu0, mu1=self.mu1, psi=self.psi, e=None)
