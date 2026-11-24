"""Turn full-scale outputs (h1_*.csv, h2_*.jsonl, h3.jsonl) into the paper's tables.
python analyze_full.py RESULTS_DIR  ->  RESULTS_DIR/tables.md
"""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from cfm.kappa import per_dgp

RD = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/results"
rng = np.random.default_rng(0); B = 3000
out = []
P = lambda *s: out.append(" ".join(str(x) for x in s))


def boot(df, stat, B=B):
    pt = stat(df); bs = np.array([stat(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(B)])
    return pt, np.nanpercentile(bs, [2.5, 97.5], axis=0), np.nanpercentile(bs, [5, 95], axis=0)


def pooled(h):
    w = h.bias_dim.abs(); return (h.kappa * w).sum() / w.sum()


def fmt(pt, ci): return f"{pt:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


# ================================================================== H1
H1 = {os.path.basename(f)[3:-4]: pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RD, "h1_*.csv")))}
EST = ["CausalPFN-S", "CausalPFN-RA", "CausalPFN-HO", "GP-S", "AIPW2", "IPW", "Hajek", "S-OLS", "T-OLS", "kNN", "TabPFN-S"]
perd = {}
for sname, df in H1.items():
    meta = df.groupby("dgp_id")[["p", "n", "in_prior_mech", "prop_mechanism"]].first()
    evc = (df.fact_r2_pfn / df.fact_r2_oracle).groupby(df.dgp_id).mean().rename("evc")
    P(f"\n## H1 — suite `{sname}`: {df.dgp_id.nunique()} DGPs, {len(df)} rows\n")
    P("| estimator | family | DGPs | κ pooled [95% CI] | strength trend [95% CI] | median RMSE |"); P("|---|---|---|---|---|---|")
    for e in [e for e in EST if e in df.columns and df[e].notna().any()]:
        a = per_dgp(df.dropna(subset=[e]), e).join(meta, on="dgp_id").join(evc, on="dgp_id")
        perd[(sname, e)] = a; a = a[a.eligible]
        for fam, h in a.groupby("family"):
            h = h.reset_index(drop=True)
            pt, ci, _ = boot(h, lambda x: np.array([pooled(x), np.polyfit(x.strength, x.kappa, 1)[0] if x.strength.std() > 0 else np.nan]))
            P(f"| {e} | {fam} | {len(h)} | {fmt(pt[0], ci[:, 0])} | {fmt(pt[1], ci[:, 1])} | {h.rmse_est.median():.3f} |")
    # paired comparisons
    P("\nPaired on identical draws (|bias|-weighted mean of κ_d differences, DGP bootstrap):\n")
    for a_, b_ in (("CausalPFN-S", "GP-S"), ("CausalPFN-S", "AIPW2"), ("CausalPFN-S", "IPW"), ("CausalPFN-HO", "CausalPFN-S"), ("CausalPFN-S", "TabPFN-S")):
        if (sname, a_) not in perd or (sname, b_) not in perd: continue
        j = perd[(sname, a_)][["dgp_id", "family", "kappa", "bias_dim", "eligible"]].merge(perd[(sname, b_)][["dgp_id", "kappa"]], on="dgp_id", suffixes=("", "_b"))
        j = j[j.eligible]
        for fam, h in j.groupby("family"):
            h = h.reset_index(drop=True)
            pt, ci, ci90 = boot(h, lambda x: ((x.kappa - x.kappa_b) * x.bias_dim.abs()).sum() / x.bias_dim.abs().sum())
            P(f"- {fam}: {a_} − {b_} = {fmt(pt, ci)}; TOST ±0.1: {bool(ci90[0] > -0.1 and ci90[1] < 0.1)}")
    if (sname, "CausalPFN-S") in perd:
        a = perd[(sname, "CausalPFN-S")]; a = a[a.eligible]
        if a.p.nunique() > 1:
            Xd = np.column_stack([np.ones(len(a)), a.strength, a.p, np.log(a.n)])
            fit = lambda idx: np.linalg.lstsq(Xd[idx], a.kappa.values[idx], rcond=None)[0]
            pt = fit(np.arange(len(a))); bs = np.array([fit(rng.integers(0, len(a), len(a))) for _ in range(B)])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            P("\nJoint regression κ_d ~ strength + p + log n (CausalPFN-S): " + "; ".join(f"{nm} {b:+.4f} [{l:+.4f}, {u:+.4f}]" for nm, b, l, u in zip(["const", "strength", "p", "log n"], pt, lo, hi)))
        for lab, h in a.groupby("in_prior_mech"):
            P(f"- in_prior_mech={lab}: {len(h)} DGPs, κ pooled {pooled(h):.3f}")
        P(f"- corr(κ_d, held-out explained-variance captured EVC) = {np.corrcoef(a.kappa, a.evc)[0, 1]:.3f}; median EVC {a.evc.median():.3f}")

if "hard" in H1 and "clip" in H1 and ("hard", "CausalPFN-S") in perd:
    u = perd[("hard", "CausalPFN-S")].assign(seed=lambda x: x.dgp_id.str.split("-").str[-1])
    c = perd[("clip", "CausalPFN-S")].assign(seed=lambda x: x.dgp_id.str.split("-").str[-1])
    m = u.merge(c, on="seed", suffixes=("_u", "_c"))
    P(f"\nPositivity control (same surfaces): κ hard → clip median {m.kappa_u.median():.3f} → {m.kappa_c.median():.3f}; clip higher in {(m.kappa_c > m.kappa_u).sum()}/{len(m)}")

# ================================================================== H2
for f in sorted(glob.glob(os.path.join(RD, "h2_*.jsonl"))):
    R = [json.loads(l) for l in open(f)]
    if not R: continue
    sname = os.path.basename(f)[3:-6]
    P(f"\n## H2 — suite `{sname}`: {len(R)} decompositions\n")
    P(f"max |Euler residual| over all weight sets: {max(max(abs(v) for v in r['euler'].values()) for r in R):.1e}\n")
    rows = []
    for r in R:
        base = dict(family=r["family"], p=r["p"], n=r["n"], dimL=r["dim_attrib"][0], dimN=r["dim_attrib"][1], labeldep=r["labeldep_share"])
        for k, (l, nn) in r["attrib"].items(): base[f"{k}_L"] = l; base[f"{k}_N"] = nn
        for k, v in r["cos"].items(): base[f"cos:{k}"] = np.mean(v)
        rep = r["report"]
        base["imb_pfn"] = (rep["treated"]["imb_model_mean"] + rep["control"]["imb_model_mean"]) / 2
        base["imb_hajek"] = (rep["treated"]["imb_hajek_mean"] + rep["control"]["imb_hajek_mean"]) / 2
        base["neg_frac"] = (rep["treated"]["neg_weight_frac"] + rep["control"]["neg_weight_frac"]) / 2
        rows.append(base)
    D = pd.DataFrame(rows)
    est = ["pfn_S", "pfn_RA", "pfn_HO", "gp", "tols", "hajek_oracle"]
    P("Share of DiM's linear / nonlinear confounding removed (ratio of means, by p, n):\n")
    P("| p | n | " + " | ".join(f"{e} lin / nonlin" for e in est) + " |"); P("|---|---|" + "---|" * len(est))
    for (p, n), h in D.groupby(["p", "n"]):
        cells = [f"{1 - h[f'{e}_L'].mean() / h.dimL.mean():.0%} / {1 - h[f'{e}_N'].mean() / h.dimN.mean():.0%}" for e in est]
        P(f"| {p} | {n} | " + " | ".join(cells) + " |")
    cc = [c for c in D.columns if c.startswith("cos:")]
    P("\nWeight similarity (mean within-arm cosine of de-meaned net weights; noise floors = same estimator after a fresh noise draw):\n")
    for c in cc: P(f"- {c[4:]}: {D[c].mean():.3f}")
    P(f"\nRich-dictionary imbalance: PFN ≤ Hájek(ê) in {(D.imb_pfn <= D.imb_hajek).mean():.0%} of datasets; negative-weight share {D.neg_frac.median():.2f}; label-dependent share {D.labeldep.median():.3f}")

# ================================================================== H3
f = os.path.join(RD, "h3.jsonl")
if os.path.exists(f):
    Rall = [json.loads(l) for l in open(f)]; R = [r for r in Rall if abs(r["tau_S"] - r["DiM"]) >= 0.1]
    L = R[0]["l_star"] if R else None
    P(f"\n## H3: {len(R)} eval datasets (excluded for negligible correction: {[r['idx'] for r in Rall if r not in R]}), l* = {L}\n")
    pr = []
    for r in R:
        for cond in ("probe_real", "probe_shuf", "probe_rand"):
            for c, prof in r[cond].items(): pr.append(dict(cond=cond, concept=c, at_l=prof[L], peak=max(prof)))
        for c, v in r["rawx"].items(): pr.append(dict(cond="rawx", concept=c, at_l=v, peak=v))
    Pr = pd.DataFrame(pr).groupby(["concept", "cond"]).mean().round(3)
    P("Probe R² (mean):\n")
    try: P(Pr.to_markdown())
    except ImportError: P(Pr.to_string())
    er = []
    for r in R:
        c0 = abs(r["tau_S"] - r["DiM"])
        for k, e in r["erasure"].items():
            er.append(dict(concept=k, lstar=abs(e["lstar_S"] - r["tau_S"]) / c0, all=abs(e["all_S"] - r["tau_S"]) / c0 if "all_S" in e else np.nan))
    E = pd.DataFrame(er)
    P("\n|Δτ̂| / |τ̂ − DiM| under LEACE erasure (mean, 90% bootstrap upper bound):\n")
    for k, h in E.groupby("concept"):
        for col in ("lstar", "all"):
            v = h[col].dropna().values
            if len(v): b = [rng.choice(v, len(v)).mean() for _ in range(B)]; P(f"- {k} @ {col}: {v.mean():.3f} (≤ {np.percentile(b, 95):.3f})")
    A = {k: np.mean([np.abs(np.array(r[k])).max(1) for r in R], 0) for k in ("attn_corr_partial", "attn_corr_partial_randinit", "attn_corr_partial_shufT") if k in R[0]}
    P("\nAttention: max over heads of |partial corr(treated-key mass, logit e | poly2(x))|, mean over datasets, per layer:\n")
    for k, v in A.items(): P(f"- {k}: " + " ".join(f"{x:.2f}" for x in v))
    sw = pd.DataFrame([{k: r["swap"]["D2"][k] - r["swap"]["D1"][k] for k in r["swap"]["D1"]} for r in R])
    P("\nCoupled propensity swap: Δτ̂ = τ̂(D2) − τ̂(D1) (mean ± sd over datasets):\n")
    for k in sw.columns: P(f"- {k}: {sw[k].mean():+.3f} ± {sw[k].std():.3f}")
    if len(sw) > 3:
        Xd = np.column_stack([np.ones(len(sw)), sw["GP-S"], sw["IPW"]]); b = np.linalg.lstsq(Xd, sw["CausalPFN-S"], rcond=None)[0]
        P(f"- regression ΔPFN ~ ΔGP-S + ΔIPW: coefs GP {b[1]:+.3f}, IPW {b[2]:+.3f}; corr(ΔPFN, ΔGP) {np.corrcoef(sw['CausalPFN-S'], sw['GP-S'])[0,1]:.3f}, corr(ΔPFN, ΔIPW) {np.corrcoef(sw['CausalPFN-S'], sw['IPW'])[0,1]:.3f}")

open(os.path.join(RD, "tables.md"), "w").write("\n".join(out)); print("\n".join(out))
