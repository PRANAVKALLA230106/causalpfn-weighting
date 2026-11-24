"""Summarise H2 results from results/h2_hard.jsonl."""
import json, sys
import numpy as np, pandas as pd
recs = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "results/h2_hard.jsonl")]
rows = []
for r in recs:
    row = {k: v for k, v in r.items() if k not in ("attrib", "report")}
    for est, a in r["attrib"].items():
        row[f"{est}_lin"] = a["lin"]; row[f"{est}_nonlin"] = a["nonlin"]
    for arm, rep in r["report"].items():
        for k, v in rep.items(): row[f"{arm}_{k}"] = v
    rows.append(row)
df = pd.DataFrame(rows)
print("datasets:", len(df), " max |identity resid|:", f"{df.err_identity_resid.abs().max():.1e}", " max |euler|:", f"{df.euler.abs().max():.1e}")

# 1) error attribution: DiM bias split into linear/nonlinear confounding vs what PFN leaves
df["dim_bias"] = df.dim - df.psi
g = df.groupby(["p", "n"])
att = pd.DataFrame({
    "kappa": g.kappa_single.mean(),
    "DiM_bias_lin": g.dim_imb_lin.mean(), "DiM_bias_nonlin": g.dim_imb_nonlin.mean(),
    "PFN_left_lin": g.pfn_lin.mean(), "PFN_left_nonlin": g.pfn_nonlin.mean(), "PFN_noise": g.noise_term.mean(),
    "HajekOr_left_lin": g.hajek_oracle_lin.mean(), "HajekOr_left_nonlin": g.hajek_oracle_nonlin.mean(),
})
att["frac_lin_removed"] = 1 - att.PFN_left_lin / att.DiM_bias_lin
att["frac_nonlin_removed"] = 1 - att.PFN_left_nonlin / att.DiM_bias_nonlin
pd.set_option("display.width", 250)
print("\n[error attribution: signed imbalance of weights on linear / nonlinear parts of mu]\n", att.round(3).to_string())
tot = df[["dim_imb_lin", "dim_imb_nonlin", "pfn_lin", "pfn_nonlin"]].mean()
print(f"\nOverall: PFN removes {1-tot.pfn_lin/tot.dim_imb_lin:.1%} of DiM's linear confounding and {1-tot.pfn_nonlin/tot.dim_imb_nonlin:.1%} of its nonlinear confounding")
print(f"Share of PFN's remaining bias that is linear: {tot.pfn_lin/(tot.pfn_lin+tot.pfn_nonlin):.1%}  (DiM's bias: {tot.dim_imb_lin/(tot.dim_imb_lin+tot.dim_imb_nonlin):.1%} linear)")

# 2) H2 falsification criteria
c = pd.DataFrame({
    "imb_pfn": (df.treated_imb_model_mean + df.control_imb_model_mean) / 2,
    "imb_hajek_ehat": (df.treated_imb_hajek_mean + df.control_imb_hajek_mean) / 2,
    "imb_uniform": (df.treated_imb_uniform_mean + df.control_imb_uniform_mean) / 2,
    "corr_inv_e_true": (df.corr_w_inv_e_true_t + df.corr_w_inv_e_true_c) / 2,
    "K_star_t": df.treated_K_star, "cos_Kstar_t": df.treated_cos_K_star,
    "neg_frac": (df.treated_neg_weight_frac + df.control_neg_weight_frac) / 2,
    "labeldep": df.labeldep_share, "cos_fresh_noise": df.cos_fresh_noise,
    "ess_ratio": (df.ess_t + df.ess_c) / (df.ess_hajek_t + df.ess_hajek_c),
    "p": df.p, "n": df.n})
print("\n[H2 criteria by (p,n), medians]\n", c.groupby(["p", "n"]).median().round(3).to_string())
print("\nfraction of datasets where PFN rich-dict imbalance <= Hajek(e_hat):", f"{(c.imb_pfn <= c.imb_hajek_ehat).mean():.2f}")
# 3) does kappa track balance?
print("corr(kappa, imbalance on mu-nonlinear left):", round(np.corrcoef(df.kappa_single, df.pfn_nonlin.abs() / df.dim_bias.abs())[0, 1], 3))
df.to_csv("results/h2_hard_flat.csv", index=False)
