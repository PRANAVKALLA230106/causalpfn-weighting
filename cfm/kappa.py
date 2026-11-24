"""kappa_d = E[tau_hat - DiM] / E[psi - DiM] per DGP, response slopes, and
DGP-clustered bootstrap summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd


def per_dgp(df: pd.DataFrame, est: str, min_z: float = 3.0) -> pd.DataFrame:
    """df: one row per (dgp_id, rep) with columns psi, DiM, oracle, <est>, family, strength."""
    rows = []
    for gid, g in df.groupby("dgp_id"):
        R = len(g)
        gap = g["psi"] - g["DiM"]                       # needed correction per rep
        se = gap.std(ddof=1) / np.sqrt(R)
        corr = g[est] - g["DiM"]
        kap = corr.mean() / gap.mean()
        # delta-method SE of the ratio
        cov = np.cov(corr, gap, ddof=1)
        a, b = corr.mean(), gap.mean()
        var_r = (cov[0, 0] / b**2 - 2 * a * cov[0, 1] / b**3 + a**2 * cov[1, 1] / b**4) / R
        slope = np.polyfit(g["oracle"], g[est], 1)[0] if g["oracle"].std() > 0 else np.nan
        rows.append(dict(dgp_id=gid, family=g["family"].iloc[0], strength=g["strength"].iloc[0], R=R,
                         bias_dim=-gap.mean(), bias_z=abs(gap.mean()) / se, kappa=kap, kappa_se=np.sqrt(max(var_r, 0)),
                         slope=slope, bias_est=(g[est] - g["psi"]).mean(),
                         rmse_est=np.sqrt(((g[est] - g["psi"]) ** 2).mean())))
    out = pd.DataFrame(rows)
    out["eligible"] = out["bias_z"] > min_z
    return out


def family_summary(pd_df: pd.DataFrame, B: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Cluster bootstrap over DGPs (eligible only): median kappa_d, pooled kappa,
    trend of kappa_d on confounding strength, median slope."""
    rng = np.random.default_rng(seed)
    rows = []
    for fam, g in pd_df[pd_df.eligible].groupby("family"):
        g = g.reset_index(drop=True); m = len(g)
        def stats(h):
            # |bias|-weighted mean of kappa_d (signed gaps would cancel across DGPs)
            wts = h["bias_dim"].abs()
            pooled = (h["kappa"] * wts).sum() / wts.sum()
            tr = np.polyfit(h["strength"], h["kappa"], 1)[0] if h["strength"].std() > 0 else np.nan
            return np.array([h["kappa"].median(), pooled, tr, h["slope"].median()])
        point = stats(g)
        boots = np.array([stats(g.iloc[rng.integers(0, m, m)]) for _ in range(B)])
        lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
        rows.append(dict(family=fam, n_dgp=m,
                         kappa_median=point[0], kappa_median_ci=(lo[0], hi[0]),
                         kappa_pooled=point[1], kappa_pooled_ci=(lo[1], hi[1]),
                         trend_strength=point[2], trend_ci=(lo[2], hi[2]),
                         slope_median=point[3], slope_ci=(lo[3], hi[3])))
    return pd.DataFrame(rows)
