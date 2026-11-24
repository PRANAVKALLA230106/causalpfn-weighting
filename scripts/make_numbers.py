"""Regenerate paper/numbers.tex from full-scale results.
python make_numbers.py RESULTS_DIR OUT_TEX [--final]
Macros missing from the results keep their pilot values (read from the existing OUT_TEX)
and stay blue; every macro computed here is printed black when --final is given."""
import glob, json, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from cfm.kappa import per_dgp

RD, OUT = sys.argv[1], sys.argv[2]; FINAL = "--final" in sys.argv
rng = np.random.default_rng(0); B = 3000
old = open(OUT).read() if os.path.exists(OUT) else ""
vals = {}


def f3(x): return f"{x:.3f}"
def sgn(x): return f"${x:+.3f}$".replace("+", "+") if x >= 0 else f"${x:.3f}$"
def ci(lo, hi, signed=False): return f"[{sgn(lo)}, {sgn(hi)}]" if signed else f"[{lo:.3f}, {hi:.3f}]"
def pooled(h): w = h.bias_dim.abs(); return (h.kappa * w).sum() / w.sum()
def boot(df, stat):
    pt = stat(df); bs = np.array([stat(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(B)])
    return pt, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


def h1(name):
    p = os.path.join(RD, f"h1_{name}.csv"); return pd.read_csv(p) if os.path.exists(p) else None


def kap(df, e, fam=None):
    a = per_dgp(df.dropna(subset=[e]), e); a = a[a.eligible]
    if fam: a = a[a.family == fam]
    return a.reset_index(drop=True)


def paired(df, a_, b_, fam=None):
    A, Bk = kap(df, a_, fam), kap(df, b_, fam)
    j = A[["dgp_id", "kappa", "bias_dim"]].merge(Bk[["dgp_id", "kappa"]], on="dgp_id", suffixes=("", "_b"))
    return boot(j, lambda x: ((x.kappa - x.kappa_b) * x.bias_dim.abs()).sum() / x.bias_dim.abs().sum())


E, Hd, Cl = h1("easy"), h1("hard"), h1("clip")
n_dgp = sum(d.dgp_id.nunique() for d in (E, Hd, Cl) if d is not None)
if n_dgp: vals["nDGPhone"] = str(n_dgp)
if E is not None:
    for fam, k in (("ric", "kEasyRic"), ("overlap", "kEasyOv")):
        pt, lo, hi = boot(kap(E, "CausalPFN-S", fam), pooled); vals[k] = f3(pt); vals[k + "CI"] = ci(lo, hi)
if Hd is not None:
    vals["nDGPhard"] = str(Hd.dgp_id.nunique())
    for e, k in (("CausalPFN-S", "kHardPFN"), ("GP-S", "kHardGP"), ("AIPW2", "kHardAIPW"), ("IPW", "kHardIPW"), ("S-OLS", "kHardOLS")):
        pt, lo, hi = boot(kap(Hd, e), pooled); vals[k] = f3(pt)
        if k in ("kHardPFN", "kHardGP"): vals[k + "CI"] = ci(lo, hi)
    for b_, k in (("GP-S", "dPFNGP"), ("AIPW2", "dPFNAIPW")):
        pt, lo, hi = paired(Hd, "CausalPFN-S", b_); vals[k] = sgn(pt); vals[k + "CI"] = ci(lo, hi, True)
    a = kap(Hd, "CausalPFN-S"); vals["trendHard"] = sgn(np.polyfit(a.strength, a.kappa, 1)[0])
if Cl is not None:
    pt, lo, hi = boot(kap(Cl, "CausalPFN-S"), pooled); vals["kClipPFN"] = f3(pt); vals["kClipPFNCI"] = ci(lo, hi)
    vals["kClipIPW"] = f3(pooled(kap(Cl, "IPW")))
    pt, lo, hi = paired(Cl, "CausalPFN-S", "IPW"); vals["dClipPFNIPW"] = sgn(pt); vals["dClipPFNIPWCI"] = ci(lo, hi, True)
    a = kap(Cl, "CausalPFN-S")
    meta = Cl.groupby("dgp_id")[["p", "n"]].first(); a = a.join(meta, on="dgp_id")
    Xd = np.column_stack([np.ones(len(a)), a.strength, a.p, np.log(a.n)]); b = np.linalg.lstsq(Xd, a.kappa, rcond=None)[0]
    vals["trendClip"] = sgn(b[1]); vals["regP"] = sgn(b[2]); vals["regLogN"] = sgn(b[3])

# ---- H2
R2 = []
for f in glob.glob(os.path.join(RD, "h2_*.jsonl")):
    R2 += [json.loads(l) for l in open(f)]
R2 = [r for r in R2 if r.get("family", "") == "ric_hard"]
if R2:
    dl = np.mean([r["dim_attrib"][0] for r in R2]); dn = np.mean([r["dim_attrib"][1] for r in R2])
    pl = np.mean([r["attrib"]["pfn_S"][0] for r in R2]); pn = np.mean([r["attrib"]["pfn_S"][1] for r in R2])
    vals["linRemoved"] = f"{1 - pl / dl:.0%}".replace("%", "\\%"); vals["nonlinRemoved"] = f"{1 - pn / dn:.0%}".replace("%", "\\%")
    vals["nonlinShareLeft"] = f"{pn / (pl + pn):.0%}".replace("%", "\\%")
    k = np.array([(r["tau"]["pfn_S"] - r["dim"]) / (r["psi"] - r["dim"]) for r in R2])
    nl = np.array([abs(r["attrib"]["pfn_S"][1]) / abs(r["dim"] - r["psi"]) for r in R2])
    vals["corrKappaNonlin"] = sgn(np.corrcoef(k, nl)[0, 1]).replace(".000", "")
    imb = np.array([(r["report"]["treated"]["imb_model_mean"] + r["report"]["control"]["imb_model_mean"]) / 2 for r in R2])
    imh = np.array([(r["report"]["treated"]["imb_hajek_mean"] + r["report"]["control"]["imb_hajek_mean"]) / 2 for r in R2])
    vals["hajekBetter"] = f"{(imh < imb).mean():.0%}".replace("%", "\\%")
    neg = np.median([(r["report"]["treated"]["neg_weight_frac"] + r["report"]["control"]["neg_weight_frac"]) / 2 for r in R2])
    vals["negWeights"] = f"{neg:.0%}".replace("%", "\\%")
    fr = [np.mean(r["cos"]["pfn_S|pfn_S_freshnoise"]) for r in R2]; vals["cosFresh"] = f"{np.percentile(fr, 10):.2f}--{np.percentile(fr, 90):.2f}"
    vals["labelDep"] = f"{np.median([r['labeldep_share'] for r in R2]):.1%}".replace("%", "\\%")
    vals["cosHOin"] = f"{np.mean([np.mean(r['cos']['pfn_S|pfn_HO']) for r in R2]):.2f}"
    vals["cosSRA"] = f"{np.mean([np.mean(r['cos']['pfn_S|pfn_RA']) for r in R2]):.2f}"

# ---- H3
f3p = os.path.join(RD, "h3.jsonl")
if os.path.exists(f3p):
    R = [json.loads(l) for l in open(f3p)]; R = [r for r in R if abs(r["tau_S"] - r["DiM"]) >= 0.1]
    if R:
        L = R[0]["l_star"]; vals["nHthree"] = str(len(R))
        for cond, k in (("probe_real", "probeReal"), ("probe_shuf", "probeShuf"), ("probe_rand", "probeRand")):
            vals[k] = f"{np.mean([r[cond]['e_perp'][L] for r in R]):.2f}"
        vals["probeRawx"] = f"{np.mean([r['rawx']['e_perp'] for r in R]):.2f}"
        vals["shufDrop"] = f"{float(vals['probeReal']) - float(vals['probeShuf']):.2f}"
        def er(c, col):
            v = np.array([abs(r["erasure"][c][col] - r["tau_S"]) / abs(r["tau_S"] - r["DiM"]) for r in R])
            b = [rng.choice(v, len(v)).mean() for _ in range(B)]; return v.mean(), np.percentile(b, 95)
        m, u = er("e_perp", "lstar_S"); vals["eraseOne"], vals["eraseOneUB"] = f3(m), f3(u)
        m, u = er("e_perp", "all_S"); vals["eraseAll"], vals["eraseAllUB"] = f3(m), f3(u)
        vals["eraseNuis"] = f3(er("nuisance", "all_S")[0]); vals["eraseRand"] = f3(er("random_rank1", "lstar_S")[0])

# ---- IHDP
fi = next((p for p in (os.path.join(RD, "ihdp", "h1_ihdp.csv"), os.path.join(RD, "h1_ihdp.csv")) if os.path.exists(p)), None)
if fi:
    I = pd.read_csv(fi)
    for s, k in (("ihdp_A", "ihdpA"), ("ihdp_B", "ihdpB")):
        a = kap(I, "CausalPFN-S", s)
        if len(a): vals[k] = f3(pooled(a))

# ---- write: replace computed macros, keep the rest
txt = old
for k, v in vals.items():
    new = (v if FINAL else f"\\prov{{{v}}}")
    pat = re.compile(r"(\\newcommand\{\\" + re.escape(k) + r"\}\{)(.*?)(\}\s*(?:%|\\newcommand|$))", re.M)
    def rep(m): return m.group(1) + new + m.group(3)
    txt, nsub = pat.subn(rep, txt, count=1)
    if nsub == 0: txt += f"\n\\newcommand{{\\{k}}}{{{new}}}"
open(OUT, "w").write(txt)
print(f"updated {len(vals)} macros:", ", ".join(sorted(vals)))
