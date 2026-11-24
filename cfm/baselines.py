"""Reference ATE estimators.  Each takes (X, t, y) and returns a float.
Propensity models are fitted on degree-2 polynomial features; AIPW is 2-fold
cross-fitted.  The GP S-learner works on per-arm-standardised outcomes, like
CausalPFN, so it isolates 'regularised smoother under CausalPFN's preprocessing'."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")


def dim(X, t, y):
    return y[t == 1].mean() - y[t == 0].mean()


def _prop(X, t, Xq=None, clip=0.01):
    m = make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    m.fit(X, t)
    return np.clip(m.predict_proba(X if Xq is None else Xq)[:, 1], clip, 1 - clip)


def s_learner_ols(X, t, y):
    Z = np.column_stack([X, t])
    m = LinearRegression().fit(Z, y)
    return float(m.coef_[-1])


def t_learner_ols(X, t, y):
    m1 = LinearRegression().fit(X[t == 1], y[t == 1])
    m0 = LinearRegression().fit(X[t == 0], y[t == 0])
    return float(np.mean(m1.predict(X) - m0.predict(X)))


def ipw(X, t, y):
    e = _prop(X, t)
    return float(np.mean(t * y / e - (1 - t) * y / (1 - e)))


def hajek(X, t, y):
    e = _prop(X, t)
    w1, w0 = t / e, (1 - t) / (1 - e)
    return float((w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum())


def _gbm():
    return GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)


def aipw(X, t, y, seed=0):
    n = len(y); rng = np.random.default_rng(seed)
    folds = rng.permutation(n) % 2
    psi = np.zeros(n)
    for k in (0, 1):
        tr, te = folds != k, folds == k
        e = _prop(X[tr], t[tr], X[te])
        m1 = _gbm().fit(X[tr][t[tr] == 1], y[tr][t[tr] == 1]).predict(X[te])
        m0 = _gbm().fit(X[tr][t[tr] == 0], y[tr][t[tr] == 0]).predict(X[te])
        tt, yy = t[te], y[te]
        psi[te] = m1 - m0 + tt * (yy - m1) / e - (1 - tt) * (yy - m0) / (1 - e)
    return float(psi.mean())


def knn_match(X, t, y, k=5):
    Xs = StandardScaler().fit_transform(X)
    tr, ct = t == 1, t == 0
    nn1 = NearestNeighbors(n_neighbors=k).fit(Xs[tr]); nn0 = NearestNeighbors(n_neighbors=k).fit(Xs[ct])
    y1 = y[tr][nn1.kneighbors(Xs, return_distance=False)].mean(1)
    y0 = y[ct][nn0.kneighbors(Xs, return_distance=False)].mean(1)
    y1 = np.where(tr, y, y1); y0 = np.where(ct, y, y0)
    return float(np.mean(y1 - y0))


def gp_s_learner(X, t, y):
    tr, ct = t == 1, t == 0
    mu = np.where(tr, y[tr].mean(), y[ct].mean()); sd = np.where(tr, y[tr].std(), y[ct].std())
    z = (y - mu) / sd
    Z = np.column_stack([StandardScaler().fit_transform(X), t])
    # initialise X length-scales at sqrt(p) (median-distance scale); at ls=1 in high p the
    # kernel underflows to ~0, gradients vanish and the GP collapses to DiM.
    ls0 = np.r_[np.full(X.shape[1], np.sqrt(X.shape[1])), 1.0]
    k = ConstantKernel(1.0) * RBF(length_scale=ls0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(0.5)
    gp = GaussianProcessRegressor(kernel=k, normalize_y=False, n_restarts_optimizer=0, random_state=0).fit(Z, z)
    Z1 = Z.copy(); Z1[:, -1] = 1; Z0 = Z.copy(); Z0[:, -1] = 0
    m1, m0 = gp.predict(Z1), gp.predict(Z0)
    return float(np.mean((y[tr].mean() + y[tr].std() * m1) - (y[ct].mean() + y[ct].std() * m0)))


BASELINES = {
    "DiM": dim, "S-OLS": s_learner_ols, "T-OLS": t_learner_ols, "IPW": ipw, "Hajek": hajek,
    "AIPW": aipw, "kNN": knn_match, "GP-S": gp_s_learner,
}


def oracle_fluctuation(d):
    """psi + P_n phi with true mu, e: the oracle efficient (AIPW) estimate."""
    t, y, e, mu0, mu1 = d["t"], d["y"], d["e"], d["mu0"], d["mu1"]
    return float(np.mean(mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)))


def _poly_ridge():
    from sklearn.linear_model import RidgeCV
    return make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))


def _prop_cv(X, t, Xq, clip):
    from sklearn.linear_model import LogisticRegressionCV
    m = make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                      LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=3, max_iter=3000))
    m.fit(X, t)
    return np.clip(m.predict_proba(Xq)[:, 1], clip, 1 - clip)


def aipw_v2(X, t, y, seed=0, K=5, clip=0.05):
    """5-fold cross-fitted AIPW; CV-ridge outcome models and CV-logistic propensity on
    degree-2 features; propensity trimmed to [clip, 1-clip]."""
    n = len(y); folds = np.random.default_rng(seed).permutation(n) % K
    psi = np.zeros(n)
    for k in range(K):
        tr, te = folds != k, folds == k
        e = _prop_cv(X[tr], t[tr], X[te], clip)
        m1 = _poly_ridge().fit(X[tr][t[tr] == 1], y[tr][t[tr] == 1]).predict(X[te])
        m0 = _poly_ridge().fit(X[tr][t[tr] == 0], y[tr][t[tr] == 0]).predict(X[te])
        tt, yy = t[te], y[te]
        psi[te] = m1 - m0 + tt * (yy - m1) / e - (1 - tt) * (yy - m0) / (1 - e)
    return float(psi.mean())


BASELINES["AIPW2"] = aipw_v2
