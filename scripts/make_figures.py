"""Paper figures from result files.  python make_figures.py RESULTS_DIR [FIG_DIR]
Reads whichever of h1_hard.csv / h1_clip.csv / h2_*.jsonl / h3.jsonl / ihdp/h1_ihdp.csv exist.
Palette: validated categorical slots (blue, orange, aqua, yellow; CVD-separated); every
series also carries a distinct marker and a direct label (contrast relief rule)."""
import glob, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from cfm.kappa import per_dgp

RD = sys.argv[1]; FD = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RD, "figures")
os.makedirs(FD, exist_ok=True)
INK, INK2, GRID, SURF = "#1f1f1e", "#5f5e5a", "#e6e5e0", "#fcfcfb"
C = {"CausalPFN-S": "#2a78d6", "GP-S": "#eb6834", "IPW": "#1baf7a", "Hajek": "#1baf7a", "AIPW2": "#eda100",
     "pfn_S": "#2a78d6", "gp": "#eb6834", "hajek_oracle": "#1baf7a", "tols": "#eda100"}
MK = {"CausalPFN-S": "o", "GP-S": "s", "IPW": "^", "Hajek": "^", "AIPW2": "D", "pfn_S": "o", "gp": "s", "hajek_oracle": "^", "tols": "D"}
LAB = {"CausalPFN-S": "CausalPFN", "GP-S": "GP S-learner", "IPW": "IPW", "Hajek": "Hájek", "AIPW2": "AIPW",
       "pfn_S": "CausalPFN", "gp": "GP S-learner", "hajek_oracle": "Hájek (oracle e)", "tols": "T-learner (OLS)"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
                     "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False, "pdf.fonttype": 42})
made = []


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FD, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig); made.append(name)


def h1_file(name):
    for cand in (f"h1_{name}.csv", os.path.join("ric_hard" if name == "hard" else "ric_hard_clip", "raw_v2.csv"),
                 os.path.join("ric_hard" if name == "hard" else "ric_hard_clip", "raw.csv")):
        p = os.path.join(RD, cand)
        if os.path.exists(p): return pd.read_csv(p)
    return None


# ---------------- F1: kappa vs p, faceted by n (hard), with clip overlay for CausalPFN
H = h1_file("hard")
if H is not None and "p" in H:
    meta = H.groupby("dgp_id")[["p", "n"]].first()
    ests = [e for e in ("CausalPFN-S", "GP-S", "AIPW2", "IPW") if e in H]
    ns = sorted(meta.n.unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(2.4 * len(ns) + 0.6, 2.5), sharey=True)
    axes = np.atleast_1d(axes)
    Cl = h1_file("clip")
    for ax, n in zip(axes, ns):
        for e in ests:
            a = per_dgp(H, e).join(meta, on="dgp_id"); a = a[a.eligible & (a.n == n)]
            g = a.groupby("p").kappa
            m, q1, q3 = g.median(), g.quantile(.25), g.quantile(.75)
            ax.fill_between(m.index, q1, q3, color=C[e], alpha=0.12, lw=0)
            ax.plot(m.index, m.values, marker=MK[e], color=C[e], label=LAB[e], markeredgecolor=SURF, markeredgewidth=1)
        if Cl is not None and "p" in Cl:
            mc = Cl.groupby("dgp_id")[["p", "n"]].first()
            a = per_dgp(Cl, "CausalPFN-S").join(mc, on="dgp_id"); a = a[a.eligible & (a.n == n)]
            if len(a):
                m = a.groupby("p").kappa.median()
                ax.plot(m.index, m.values, ls="--", lw=1.5, color=C["CausalPFN-S"], label="CausalPFN, overlap ∈ [0.1, 0.9]")
        ax.axhline(1, color=INK2, lw=0.8, ls=":")
        ax.set_title(f"n = {n}", color=INK, fontsize=9); ax.set_xlabel("covariates p")
    axes[0].set_ylabel("κ (share of confounding bias removed)")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.12))
    save(fig, "fig1_kappa_vs_p")

# ---------------- F2: linear vs nonlinear confounding removed (H2)
def load_h2():
    rows = []
    for f in sorted(glob.glob(os.path.join(RD, "h2_*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            if "dim_attrib" in r:                       # full-scale format
                dl, dn = r["dim_attrib"]; at = r["attrib"]
            else:                                       # pilot format
                dl, dn = r["dim_imb_lin"], r["dim_imb_nonlin"]
                at = {"pfn_S": (r["attrib"]["pfn"]["lin"], r["attrib"]["pfn"]["nonlin"]),
                      "hajek_oracle": (r["attrib"]["hajek_oracle"]["lin"], r["attrib"]["hajek_oracle"]["nonlin"])}
            row = dict(p=r["p"], n=r["n"], family=r.get("family", "ric_hard"), dimL=dl, dimN=dn)
            for k, v in at.items(): row[f"{k}_L"], row[f"{k}_N"] = v[0], v[1]
            if "cos" in r:
                for k, v in r["cos"].items(): row[f"cos:{k}"] = float(np.mean(v))
            rows.append(row)
    return pd.DataFrame(rows)


D2 = load_h2()
if len(D2):
    D2 = D2[D2.family == "ric_hard"] if "family" in D2 else D2
    ests = [e for e in ("pfn_S", "gp", "tols", "hajek_oracle") if f"{e}_L" in D2]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, part, title in ((axes[0], "L", "linear part of μ"), (axes[1], "N", "nonlinear part of μ")):
        for e in ests:
            g = D2.groupby("p")
            vals = 1 - g[f"{e}_{part}"].mean() / g[f"dim{part}"].mean()
            ax.plot(vals.index, vals.values, marker=MK[e], color=C[e], label=LAB[e], markeredgecolor=SURF, markeredgewidth=1)
        ax.axhline(1, color=INK2, lw=0.8, ls=":"); ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel("covariates p"); ax.set_title(f"Confounding removed: {title}", fontsize=9, color=INK)
    axes[0].set_ylabel("share of DiM's imbalance removed")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.1))
    save(fig, "fig2_linear_vs_nonlinear_balance")

    # ---------------- F3: weight-similarity heatmap
    cc = [c for c in D2.columns if c.startswith("cos:") and "fresh" not in c]
    if cc:
        names = ["pfn_S", "pfn_RA", "pfn_HO", "gp", "tols", "hajek_oracle"]
        nice = ["CausalPFN S", "CausalPFN RA", "CausalPFN held-out", "GP", "T-OLS", "Hájek (oracle)"]
        M = np.full((6, 6), np.nan); np.fill_diagonal(M, 1.0)
        for c in cc:
            a, b = c[4:].split("|")
            if a in names and b in names:
                i, j = names.index(a), names.index(b); M[i, j] = M[j, i] = D2[c].mean()
        fig, ax = plt.subplots(figsize=(3.9, 3.3)); ax.grid(False)
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                if np.isfinite(M[i, j]): ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] > 0.6 else INK)
        ax.set_xticks(range(6)); ax.set_xticklabels(nice, rotation=40, ha="right", fontsize=7); ax.set_yticks(range(6)); ax.set_yticklabels(nice, fontsize=7)
        fl = [f"{k[4:].split('|')[0]} {D2[k].mean():.2f}" for k in D2.columns if "fresh" in k]
        ax.set_title("Within-arm cosine of net weights\nnoise floors: " + ", ".join(fl).replace("pfn_S", "CausalPFN").replace("gp", "GP"), fontsize=7.5, color=INK)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        save(fig, "fig3_weight_similarity")

# ---------------- F4-F6: H3
f3 = os.path.join(RD, "h3.jsonl")
if os.path.exists(f3):
    R = [json.loads(l) for l in open(f3)]; R = [r for r in R if abs(r["tau_S"] - r["DiM"]) >= 0.1]
    if R:
        L = R[0]["l_star"]
        fig, ax = plt.subplots(figsize=(3.8, 2.6))
        for cond, col, mk, lab in (("probe_real", "#2a78d6", "o", "CausalPFN"), ("probe_shuf", "#eb6834", "s", "CausalPFN, T shuffled"),
                                    ("probe_rand", "#1baf7a", "^", "random-init network")):
            Y = np.array([r[cond]["e_perp"] for r in R]); m = Y.mean(0)
            ax.plot(np.arange(len(m)), m, color=col, marker=mk, markevery=3, label=lab, markeredgecolor=SURF)
        ax.axhline(np.mean([r["rawx"]["e_perp"] for r in R]), color=INK2, ls=":", lw=1, label="cubic polynomial of x")
        ax.axvline(L, color=GRID, lw=3, zorder=0)
        ax.annotate(r"$l^*$ (fixed on pilot)", xy=(L, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -10), textcoords="offset points", fontsize=7, color=INK2)
        ax.set_xlabel("layer"); ax.set_ylabel(r"probe $R^2$ for $e_\perp$"); ax.legend(fontsize=7, loc="lower left")
        save(fig, "fig4_propensity_probes")

        er = {}
        for r in R:
            c0 = abs(r["tau_S"] - r["DiM"])
            for k, e in r["erasure"].items():
                er.setdefault((k, "one layer (l*)"), []).append(abs(e["lstar_S"] - r["tau_S"]) / c0)
                if "all_S" in e: er.setdefault((k, "all layers"), []).append(abs(e["all_S"] - r["tau_S"]) / c0)
        order = [("random_rank1", "random direction"), ("nuisance", "matched nuisance"), ("e_perp", "propensity e⊥")]
        fig, ax = plt.subplots(figsize=(3.8, 2.4))
        rng = np.random.default_rng(0)
        for i, (scope, col, mk) in enumerate((("one layer (l*)", "#2a78d6", "o"), ("all layers", "#eb6834", "s"))):
            for j, (k, lab) in enumerate(order):
                v = er.get((k, scope))
                if not v: continue
                v = np.array(v); b = [rng.choice(v, len(v)).mean() for _ in range(2000)]
                y = j + (i - 0.5) * 0.25
                ax.errorbar(v.mean(), y, xerr=[[v.mean() - np.percentile(b, 2.5)], [np.percentile(b, 97.5) - v.mean()]],
                            fmt=mk, color=col, markeredgecolor=SURF, capsize=2, label=scope if j == 0 or (j == 1 and i == 1 and not er.get((order[0][0], scope))) else None)
        ax.axvline(0.10, color=INK2, ls=":", lw=1)
        ax.annotate("10% margin", xy=(0.10, 0.5), xycoords=("data", "axes fraction"), xytext=(3, 0), textcoords="offset points", fontsize=7, color=INK2, rotation=90, va="center")
        ax.set_yticks(range(3)); ax.set_yticklabels([o[1] for o in order]); ax.set_xlabel(r"$|\Delta\hat\tau|\,/\,|\hat\tau - \mathrm{DiM}|$ after LEACE erasure")
        ax.legend(fontsize=7, loc="lower right"); save(fig, "fig5_erasure")

        if "attn_corr_partial" in R[0]:
            fig, ax = plt.subplots(figsize=(3.8, 2.5))
            for k, col, mk, lab in (("attn_corr_partial", "#2a78d6", "o", "CausalPFN"), ("attn_corr_partial_shufT", "#eb6834", "s", "T shuffled"),
                                    ("attn_corr_partial_randinit", "#1baf7a", "^", "random-init")):
                if k not in R[0]: continue
                Y = np.array([np.abs(np.array(r[k])).max(1) for r in R]); m = Y.mean(0)
                ax.plot(np.arange(len(m)), m, color=col, marker=mk, markevery=2, label=lab, markeredgecolor=SURF)
            ax.set_xlabel("layer"); ax.set_ylabel("max over heads |partial r|\n(treated-key mass, logit e | x²)"); ax.legend(fontsize=7)
            save(fig, "fig6_attention_propensity")

# ---------------- F7: IHDP
fi = os.path.join(RD, "ihdp", "h1_ihdp.csv")
if not os.path.exists(fi): fi = os.path.join(RD, "h1_ihdp.csv")
if os.path.exists(fi):
    I = pd.read_csv(fi); I = I[I.family.isin(["ihdp_A", "ihdp_B"])]
    ests = [e for e in ("CausalPFN-S", "GP-S", "AIPW2", "Hajek") if e in I]
    fig, ax = plt.subplots(figsize=(3.8, 2.4))
    for i, fam in enumerate(("ihdp_A", "ihdp_B")):
        for j, e in enumerate(ests):
            a = per_dgp(I[I.family == fam], e); a = a[a.eligible]
            if not len(a): continue
            w = a.bias_dim.abs(); k = (a.kappa * w).sum() / w.sum()
            ax.bar(i + (j - (len(ests) - 1) / 2) * 0.19, k, width=0.17, color=C[e], edgecolor=SURF, label=LAB[e] if i == 0 else None)
            ax.text(i + (j - (len(ests) - 1) / 2) * 0.19, k + 0.02, f"{k:.2f}", ha="center", fontsize=6.5, color=INK)
    ax.axhline(1, color=INK2, ls=":", lw=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Surface A (linear)", "Surface B (nonlinear)"]); ax.set_ylabel("κ pooled (eligible surfaces)")
    ax.set_ylim(0, 1.15)
    h, l = ax.get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=len(l), fontsize=7, bbox_to_anchor=(0.5, 1.08))
    save(fig, "fig7_ihdp")

print("figures:", made, "->", FD)
