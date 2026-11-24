# FINAL full-scale results (Kaggle T4, 22 Nov 2026; `results/full/`, tables in `results/full/tables.md`)
Completeness: H1 7,800 rows (390 DGPs × 20 reps, none missing or duplicated); H2 570 exact decompositions (max Euler residual 1.1e-15); H3 37 datasets (35 eligible). Random rows recomputed locally agree to ≤ 4.5e-7.

| Hypothesis | Pre-registered test | Verdict |
|---|---|---|
| H1 under-correction | upper 95% bound of κ < 0.9; negative trend | **Supported on hard DGPs:** κ = 0.604 [0.574, 0.635], GP 0.740, paired −0.135 [−0.146, −0.124]. Clip: 0.675 vs GP 0.784, IPW 0.852. **Falsified on easy DGPs:** κ 0.948 / 0.993, inside [0.9, 1.1]. Robust drivers: p (−0.010/covariate), log n (+0.13). The strength trend is mostly positivity (−0.033 → −0.013). κ tracks held-out fit (r = 0.87 / 0.95). |
| H2 low-dimensional balance | PFN balances a rich dictionary as well as Hájek and tracks 1/ê → falsified | **Supported:** linear confounding removed 89%, nonlinear 32% (GP 87/51, OLS 100/−48, oracle Hájek 91/95). 79% of the remaining bias is nonlinear; corr(κ, nonlinear residual) = −0.91 vs −0.59 for linear. Hájek(ê) balances better in 61% of datasets. Weights are closest to the GP's (cosine 0.55; noise floor 0.66; Hájek 0.36). |
| H3 propensity known but unused | erasure changes τ̂ < 10% of the correction | **Falsified (partial use):** 0.104 at l* (UB 0.132), 0.239 at all layers (UB 0.287), vs nuisance 0.051 and random 0.007. Probe e⊥ R² 0.73 vs shuffled-T 0.55 / random 0.61 / raw-x 0.38. Attention-to-treated tracks logit e beyond the controls in 12/20 layers. **Swaps:** ΔPFN ~ 1.18·ΔGP + 0.05·ΔIPW, so it responds like an outcome model. |
| IHDP | — | κ 0.944 (A) / 1.008 (B); confounding is ~95% linear, so the failure mode never shows. |

# What Estimator Does CausalPFN Learn? Results so far and paper plan
*Updated 22 Nov 2026. All numbers come from code in this repo (`cfm/`), run on pretrained `causalpfn_v0.pt` (18.8M params) on CPU.*

## Headline story (draft abstract)
CausalPFN's ATE is exactly a weighting estimator, τ̂ = DiM + Σ w_net,i yᵢ, and its error is exactly the weights' imbalance on the true outcome surfaces plus a noise term. Using this, we find:

1. **Under-correction depends on the regime, and it is CausalPFN-specific.** κ ≈ 0.95–0.99 on low-dimensional sparse DGPs, but 0.57 [0.51, 0.63] on dense, weak, nonlinear confounding. On identical draws, a GP S-learner with the same per-arm standardisation reaches 0.73, a gap of −0.16 [−0.18, −0.14]. With overlap bounded to [0.1, 0.9], CausalPFN still reaches only 0.64 [0.58, 0.70] and trails IPW by 0.12, so this is not a positivity artifact. Robust drivers: dimension (−0.012 per covariate) and sample size (+0.16 per log n). The confounding-strength trend is mostly positivity-driven (−0.035 → −0.013 once overlap is bounded).
2. **Mechanism: the implied weights balance linear functions of X, not nonlinear ones.** They remove 84% of DiM's linear confounding but only 15% of its nonlinear confounding (≈0% at n = 200, p ≥ 20). 76% of the remaining bias is nonlinear. κ_d correlates −0.78 with the leftover nonlinear imbalance. The weights are not IPW-like: corr with true 1/e is 0.18–0.50, ~20% are negative, and rich-dictionary balance is worse than Hájek in 78% of datasets.
3. **Propensity is computed but barely used.** Propensity-specific information (logit e residualised on x, μ₀, t) is decodable from the context streams from layer 0 onward. The decodability depends on the T–x association: shuffling T cuts R² by 0.16 for this concept only. Erasing it with LEACE moves τ̂ by 4% of the correction at one layer and 13% at all layers, against 3.6% for a matched nuisance concept.

## 1. Setup facts (verified, pretrained and random-init)
| Check | Result |
|---|---|
| Euler: τ̂ − Σ w y | ≤ 1.2e-15 (72 hard datasets) |
| Σ w_net within each arm | ≤ 1.8e-16 |
| label-free + label-dependent split | exact to 9e-18 |
| μ̂₁ invariant to affine maps of control y | 8e-15 |
| error identity τ̂ − ψ = imbalance(μ₁, μ₀) + Σ w ε | ≤ 1.2e-15 |
| direct `predict_cepo` vs `ATEEstimator.estimate_ate` | 9e-8 (S), 1.8e-7 (RA) |
| per-layer activation checkpointing vs plain backward | bit-identical |

Implementation notes for the paper's appendix:
- CausalPFN's RMSNorm casts to fp32, so exact decompositions need the dtype-preserving patch.
- In S mode, queries duplicate context rows.
- With each arm at 1,024 rows or fewer, retrieval keeps every row.

## 2. H1: how much correction?
**Easy suite** (40 DGPs × 20 reps, p ∈ {3, 5, 8}, sparse): CausalPFN-S κ = 0.951 [0.940, 0.961] on RIC and 0.994 [0.975, 1.010] on overlap. TOST puts κ in [0.9, 1.1], so H1 is falsified there. There is still a small, precise under-correction on RIC versus the GP (−0.032).

**Hard suite** (36 DGPs: p ∈ {10, 20, 30} × strength 1–6 × n ∈ {200, 1000}, × 10 reps):

| Estimator | κ pooled [95% CI] | trend in strength |
|---|---|---|
| CausalPFN-S | 0.572 [0.514, 0.632] | −0.035 [−0.068, −0.002] |
| CausalPFN-RA | 0.578 [0.518, 0.637] | −0.037 [−0.069, −0.002] |
| IPW | 0.591 [0.544, 0.642] | n.s. |
| S-OLS | 0.441 [0.404, 0.480] | −0.032 |
| GP-S (init fixed) | 0.733 [0.675, 0.788] | −0.027 [−0.059, 0.006] |
| AIPW2 (5-fold, CV ridge/logit, trim 0.05) | 0.620 [0.551, 0.689] | −0.039 [−0.078, 0.002] |

H1 is supported here (upper bound < 0.9; negative trend). Paired on identical draws (|bias|-weighted, DGP bootstrap):
- CausalPFN-S − GP-S = **−0.160 [−0.177, −0.142]**
- − AIPW2 = −0.048 [−0.072, −0.024]
- − IPW = −0.018 [−0.056, 0.018]

GP-S median κ by p is 0.85 / 0.82 / 0.65, versus CausalPFN 0.71 / 0.60 / 0.47.

**Baseline fixes.** The original GP collapsed to DiM at p = 30: its length scales started at 1, the RBF kernel underflowed in 31 dimensions, and gradients vanished. It is now initialised at √p. The original AIPW (GBM, 2-fold, 0.01 clip) over-corrected (κ 1.25); AIPW2 gives 0.96–1.04 on the easy suite.

**Positivity control** (`ric_hard_clip`: same seeds and surfaces, e clipped to [0.1, 0.9]):

| Estimator | κ pooled [95% CI] | trend in strength |
|---|---|---|
| CausalPFN-S | 0.639 [0.578, 0.698] | −0.013 [−0.047, 0.020] |
| CausalPFN-RA | 0.648 [0.588, 0.705] | −0.014 |
| IPW | 0.758 [0.644, 0.880] | +0.022 |
| S-OLS | 0.494 [0.460, 0.530] | −0.016 |

CausalPFN − IPW = −0.120 [−0.213, −0.035]. CausalPFN's κ rises in 36/36 DGPs when overlap is bounded (median 0.60 → 0.68).
Joint regression on the clipped suite: strength −0.013 [−0.026, −0.001], p −0.012 [−0.015, −0.010], log n +0.163 [+0.135, +0.192].

**Framing.** Lead with the *relative* comparisons on identical draws (vs the GP with the same preprocessing; vs IPW under bounded overlap) and the *mechanism* (H2), not with κ < 1 alone. Every estimator falls short of κ = 1 on the hard suite. Do not claim the strength trend as a CausalPFN property: under bounded overlap it is small.

## 3. H2: what do the weights balance? (72 hard datasets; 3 draws at n = 200, 1 at n = 1000)
| (p, n) | κ | linear confounding removed | nonlinear removed | rich-dict imbalance PFN / Hájek(ê) | corr(w, 1/e true) | cos(w_net, w_net after new noise) |
|---|---|---|---|---|---|---|
| 10, 200 | 0.62 | 90% | 40% | 0.089 / 0.088 | 0.42 | 0.64 |
| 10, 1000 | 0.80 | 93% | 72% | 0.044 / 0.073 | 0.50 | 0.77 |
| 20, 200 | 0.38 | 81% | −3% | 0.104 / 0.065 | 0.21 | 0.51 |
| 20, 1000 | 0.72 | 92% | 51% | 0.047 / 0.062 | 0.43 | 0.72 |
| 30, 200 | 0.37 | 77% | −35% | 0.099 / 0.061 | 0.18 | 0.48 |
| 30, 1000 | 0.61 | 83% | 37% | 0.046 / 0.031 | 0.27 | 0.68 |

Reading:
- **Supported:** low-dimensional (essentially linear) balance. Nonlinear balance appears only with more data.
- **K\* is not clean:** the best Han–Ding polynomial sieve matches with cosine only 0.44–0.70, so "projection onto a fixed sieve" is too strong. Prefer "linear-span balance plus a data-hungry nonlinear component".
- **Label dependence matters at small n:** w_net's cosine under a fresh noise draw is 0.48–0.77. The weights are a *local* (first-order) description; the error identity is exact regardless.

## 4. H3: where is propensity, and is it used? (prop_nl family, p = 6, n = 500; pilot idx 0–2 fixed l* = 10; eval idx 3–11, idx 11 excluded for negligible correction)
| Probe R² (mean) | real T | T shuffled | random-init net | raw-x poly-3 |
|---|---|---|---|---|
| e⊥ (logit e residualised on x, μ₀, t) at l* | 0.70 | 0.54 | 0.54 | 0.30 |
| nuisance (matched) at l* | 0.79 | 0.79 | 0.75 | 0.73 |

| LEACE erasure | \|Δτ̂\| / \|τ̂ − DiM\| mean (90% upper) |
|---|---|
| e⊥ at l* | 0.041 (0.055) |
| e⊥ all layers | 0.133 (0.184) |
| nuisance all layers | 0.036 (0.054) |
| random rank-1 at l* | 0.010 (0.017) |
| logit e all layers | 0.89 (not interpretable: logit e is ~60% prognostic) |

H3 verdict: *partly* supported. Propensity-specific information is present and computed from the T–x association, but it carries only ~10–15% of the correction. Most of the correction does not flow through a linearly decodable e(x). Per the pre-registered caveat, nonlinear use (e.g., through treated-vs-control attention mass) is not ruled out.


## IHDP semi-synthetic benchmark (23 Nov; `run_ihdp.py`, `results/ihdp/`)
Real IHDP covariates and treatment (747 units, 139 treated); Hill's surfaces A (linear) and B (nonlinear), 40 surfaces × 3 noise draws each, plus the 10 official NPCI files.

| Estimator | κ, A (32/40 eligible) [95% CI] | κ, B (20/40 eligible) [95% CI] |
|---|---|---|
| CausalPFN-S | 0.944 [0.911, 0.974] | 1.008 [0.877, 1.112] |
| CausalPFN-RA | 0.947 | 0.916 |
| CausalPFN held-out | 0.924 | 0.904 |
| GP-S | 0.983 [0.952, 1.010] | 0.961 |
| AIPW2 | 0.898 | 0.879 |
| Hájek / IPW | 0.19 / unstable (κ ≈ 5) | 0.45 / unstable |

- **IHDP does not reproduce the hard-suite shortfall.** The error identity shows why: even under surface B only ~5% of DiM's imbalance is on the nonlinear part of μ₀ (0.015 vs 0.30 linear), because the real assignment aligns with the linear part. So IHDP never exercises the failure mode. This is consistent with the mechanism, and it is a useful point for the paper: standard benchmarks can hide it.
- On surface A, CausalPFN leaves ~3× the GP's linear imbalance (mean |·| 0.089 vs 0.032).
- Official NPCI files: median |error| CausalPFN 0.070, GP 0.070, AIPW2 0.092, DiM 0.111.

## Gaps: status after 22 Nov (all implemented and smoke-tested on CPU; full runs go on Kaggle via `cfm-bundle.zip`)
| Gap | Status | Where |
|---|---|---|
| 1. Scale (power analysis) | **Design fixed.** Between-DGP sd ≈ 0.18 dominates within-DGP noise (≈ 0.02 at R = 10), so we use more DGPs, not more reps: 390 DGPs × R = 20 for H1 (easy 120, hard 180, clip 90). Projected 95% half-width ≈ ±0.03 on pooled κ; paired PFN−GP ≈ ±0.01. | `cfm/experiments.py`, `run_kaggle.py --phase h1` |
| 2. Positivity control | **Done on the pilot**; the full run adds GP-S and AIPW2 on the clipped twin | `suite("clip")` |
| 3. In- vs out-of-prior | **Implemented.** Mechanism labels follow CausalPFN App. D.1 (prior propensity is constant/linear/MLP drawn independently of the outcome columns; noise var ≤ Var(μ_t); context ≤ 2048; ≤ 99 features). RIC families are labelled out-of-prior ("propensity aligned with prognostic score"). Data-driven proxy: held-out factual R² of CausalPFN as a fraction of the oracle's (EVC); report corr(κ_d, EVC). | `mechanism_label`, `heldout_fit` |
| 4. Reference-to-reference distances | **Implemented.** Exact implied weights for the GP S-learner (new PyTorch GP; matches sklearn's κ to 3 d.p.; Euler 1e-15), T-OLS and oracle Hájek; within-arm cosines between all pairs; fresh-noise floors for PFN and GP | `cfm/gp_torch.py`, `run_h2` |
| 5. H3 extensions | **Implemented.** Coupled propensity swaps (Δτ̂ for PFN, PFN with e⊥ erased, GP-S, IPW, Hájek, AIPW2, S-OLS); attention mass on treated keys vs logit e (partial on poly2(x)), with random-init and shuffled-T controls; eval set raised to 37 | `run_h3`, `cfm/h3.py` |
| 6. Held-out queries and RA-mode H2 | **Implemented.** Cross-fitted held-out decomposition (exact, Euler 0) | `decompose_ate_heldout` |
| 7. Second model | TabPFN-as-S-learner contrast is optional in H1 (needs Internet on Kaggle). Do-PFN weights not located; stays a stretch goal | `run_kaggle.py --tabpfn` |

Smoke-test observations (single datasets; not results):
- In-context vs held-out CausalPFN weights have cosine ≈ 0.46–0.57, so the duplicate own-row queries change the weights substantially. RA vs S ≈ 0.98.
- Attention mass on treated keys tracks logit e beyond poly2(x) (partial |r| up to 0.8 at layer 3), but controls also reach 0.3–0.7 at some layers, so only the full run can say whether this is propensity-specific.

## Proposed paper outline (ICML/NeurIPS style)
1. Introduction: causal foundation models are black boxes; we open one exactly.
2. Exact weighting representation (Euler; within-arm contrast; error identity). Make it Proposition 1.
3. Experimental design: DGP families, κ, oracle fluctuation, eligibility, clustered bootstrap.
4. How much correction? (H1: the easy vs hard regimes; trend regression.)
5. What is balanced? (H2: linear vs nonlinear attribution; Han–Ding comparison; label dependence.)
6. Where is propensity? (H3: probes with controls, LEACE.)
7. Implications: why debiasing (SC-PFN, debiased PFNs) is needed, which specific failure it targets (nonlinear imbalance), and a diagnostic practitioners can run (the error identity on a simulated surrogate).
8. Limitations: synthetic DGPs, one model, local weights, linear erasure.
