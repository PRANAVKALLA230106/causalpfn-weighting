
## H1 — suite `clip`: 90 DGPs, 1800 rows

| estimator | family | DGPs | κ pooled [95% CI] | strength trend [95% CI] | median RMSE |
|---|---|---|---|---|---|
| CausalPFN-S | ric_hard_clip | 90 | 0.675 [0.634, 0.716] | -0.013 [-0.036, 0.011] | 0.469 |
| CausalPFN-RA | ric_hard_clip | 90 | 0.683 [0.639, 0.724] | -0.014 [-0.037, 0.008] | 0.465 |
| CausalPFN-HO | ric_hard_clip | 90 | 0.575 [0.532, 0.618] | -0.015 [-0.041, 0.011] | 0.688 |
| GP-S | ric_hard_clip | 90 | 0.784 [0.743, 0.821] | -0.010 [-0.031, 0.010] | 0.260 |
| AIPW2 | ric_hard_clip | 90 | 0.722 [0.667, 0.777] | -0.006 [-0.036, 0.024] | 0.519 |
| IPW | ric_hard_clip | 90 | 0.852 [0.752, 0.955] | 0.046 [-0.003, 0.099] | 0.750 |
| Hajek | ric_hard_clip | 90 | 0.560 [0.429, 0.695] | 0.029 [-0.041, 0.101] | 1.207 |
| S-OLS | ric_hard_clip | 90 | 0.487 [0.464, 0.511] | -0.023 [-0.035, -0.009] | 0.860 |
| T-OLS | ric_hard_clip | 90 | 0.484 [0.461, 0.509] | -0.023 [-0.035, -0.009] | 0.867 |
| kNN | ric_hard_clip | 90 | 0.289 [0.261, 0.318] | -0.016 [-0.034, 0.002] | 1.167 |

Paired on identical draws (|bias|-weighted mean of κ_d differences, DGP bootstrap):

- ric_hard_clip: CausalPFN-S − GP-S = -0.108 [-0.122, -0.095]; TOST ±0.1: False
- ric_hard_clip: CausalPFN-S − AIPW2 = -0.047 [-0.070, -0.023]; TOST ±0.1: True
- ric_hard_clip: CausalPFN-S − IPW = -0.177 [-0.252, -0.105]; TOST ±0.1: False
- ric_hard_clip: CausalPFN-HO − CausalPFN-S = -0.100 [-0.111, -0.090]; TOST ±0.1: False

Joint regression κ_d ~ strength + p + log n (CausalPFN-S): const +0.1758 [-0.0056, +0.3578]; strength -0.0133 [-0.0232, -0.0032]; p -0.0098 [-0.0110, -0.0087]; log n +0.1265 [+0.0988, +0.1532]
- in_prior_mech=False: 90 DGPs, κ pooled 0.675
- corr(κ_d, held-out explained-variance captured EVC) = 0.953; median EVC 0.724

## H1 — suite `easy`: 120 DGPs, 2400 rows

| estimator | family | DGPs | κ pooled [95% CI] | strength trend [95% CI] | median RMSE |
|---|---|---|---|---|---|
| CausalPFN-S | overlap | 58 | 0.993 [0.983, 1.003] | 0.018 [-0.010, 0.062] | 0.104 |
| CausalPFN-S | ric | 60 | 0.948 [0.938, 0.956] | -0.009 [-0.022, 0.005] | 0.136 |
| CausalPFN-RA | overlap | 58 | 0.996 [0.985, 1.006] | 0.028 [-0.007, 0.081] | 0.103 |
| CausalPFN-RA | ric | 60 | 0.954 [0.946, 0.962] | -0.009 [-0.021, 0.004] | 0.125 |
| CausalPFN-HO | overlap | 58 | 0.986 [0.976, 0.996] | -0.008 [-0.025, 0.008] | 0.108 |
| CausalPFN-HO | ric | 60 | 0.917 [0.900, 0.930] | -0.013 [-0.037, 0.010] | 0.205 |
| GP-S | overlap | 58 | 0.992 [0.982, 1.004] | 0.045 [0.000, 0.120] | 0.096 |
| GP-S | ric | 60 | 0.984 [0.981, 0.987] | 0.002 [-0.005, 0.009] | 0.078 |
| AIPW2 | overlap | 58 | 1.010 [0.957, 1.066] | 0.233 [-0.010, 0.704] | 0.176 |
| AIPW2 | ric | 60 | 0.948 [0.930, 0.963] | -0.004 [-0.032, 0.030] | 0.155 |
| IPW | overlap | 58 | 0.841 [0.815, 0.865] | -0.057 [-0.093, -0.017] | 0.377 |
| IPW | ric | 60 | 0.877 [0.858, 0.895] | -0.035 [-0.062, -0.003] | 0.487 |
| Hajek | overlap | 58 | 0.809 [0.780, 0.836] | -0.066 [-0.106, -0.022] | 0.388 |
| Hajek | ric | 60 | 0.859 [0.838, 0.880] | -0.049 [-0.076, -0.015] | 0.484 |
| S-OLS | overlap | 58 | 0.976 [0.926, 1.019] | -0.210 [-0.648, 0.013] | 0.167 |
| S-OLS | ric | 60 | 0.698 [0.643, 0.749] | 0.004 [-0.050, 0.062] | 0.642 |
| T-OLS | overlap | 58 | 0.978 [0.931, 1.023] | -0.204 [-0.639, 0.016] | 0.158 |
| T-OLS | ric | 60 | 0.691 [0.636, 0.744] | 0.001 [-0.053, 0.059] | 0.658 |
| kNN | overlap | 58 | 0.554 [0.523, 0.586] | -0.082 [-0.153, -0.027] | 0.466 |
| kNN | ric | 60 | 0.557 [0.533, 0.581] | -0.028 [-0.058, 0.001] | 0.962 |

Paired on identical draws (|bias|-weighted mean of κ_d differences, DGP bootstrap):

- overlap: CausalPFN-S − GP-S = 0.001 [-0.011, 0.012]; TOST ±0.1: True
- ric: CausalPFN-S − GP-S = -0.036 [-0.045, -0.029]; TOST ±0.1: True
- overlap: CausalPFN-S − AIPW2 = -0.018 [-0.070, 0.032]; TOST ±0.1: True
- ric: CausalPFN-S − AIPW2 = -0.001 [-0.013, 0.012]; TOST ±0.1: True
- overlap: CausalPFN-S − IPW = 0.152 [0.126, 0.178]; TOST ±0.1: False
- ric: CausalPFN-S − IPW = 0.070 [0.054, 0.088]; TOST ±0.1: True
- overlap: CausalPFN-HO − CausalPFN-S = -0.007 [-0.013, -0.001]; TOST ±0.1: True
- ric: CausalPFN-HO − CausalPFN-S = -0.030 [-0.038, -0.024]; TOST ±0.1: True

Joint regression κ_d ~ strength + p + log n (CausalPFN-S): const +0.0230 [+0.0211, +0.0246]; strength +0.0176 [+0.0001, +0.0409]; p +0.0035 [-0.0034, +0.0114]; log n +0.1431 [+0.1310, +0.1527]
- in_prior_mech=False: 61 DGPs, κ pooled 0.948
- in_prior_mech=True: 57 DGPs, κ pooled 0.993
- corr(κ_d, held-out explained-variance captured EVC) = 0.241; median EVC 0.972

## H1 — suite `hard`: 180 DGPs, 3600 rows

| estimator | family | DGPs | κ pooled [95% CI] | strength trend [95% CI] | median RMSE |
|---|---|---|---|---|---|
| CausalPFN-S | ric_hard | 180 | 0.604 [0.574, 0.635] | -0.033 [-0.049, -0.017] | 0.659 |
| CausalPFN-RA | ric_hard | 180 | 0.610 [0.580, 0.642] | -0.035 [-0.051, -0.017] | 0.664 |
| CausalPFN-HO | ric_hard | 180 | 0.502 [0.472, 0.535] | -0.033 [-0.051, -0.016] | 0.904 |
| GP-S | ric_hard | 180 | 0.740 [0.707, 0.772] | -0.021 [-0.038, -0.004] | 0.336 |
| AIPW2 | ric_hard | 180 | 0.616 [0.577, 0.653] | -0.035 [-0.056, -0.013] | 0.669 |
| IPW | ric_hard | 180 | 0.603 [0.582, 0.626] | -0.019 [-0.034, -0.004] | 0.807 |
| Hajek | ric_hard | 180 | 0.302 [0.252, 0.351] | -0.042 [-0.071, -0.011] | 1.348 |
| S-OLS | ric_hard | 180 | 0.439 [0.422, 0.456] | -0.034 [-0.044, -0.025] | 1.030 |
| T-OLS | ric_hard | 180 | 0.434 [0.417, 0.452] | -0.034 [-0.043, -0.025] | 1.037 |
| kNN | ric_hard | 180 | 0.259 [0.241, 0.276] | -0.027 [-0.037, -0.016] | 1.408 |

Paired on identical draws (|bias|-weighted mean of κ_d differences, DGP bootstrap):

- ric_hard: CausalPFN-S − GP-S = -0.135 [-0.146, -0.124]; TOST ±0.1: False
- ric_hard: CausalPFN-S − AIPW2 = -0.012 [-0.025, 0.002]; TOST ±0.1: True
- ric_hard: CausalPFN-S − IPW = 0.001 [-0.018, 0.020]; TOST ±0.1: True
- ric_hard: CausalPFN-HO − CausalPFN-S = -0.102 [-0.108, -0.096]; TOST ±0.1: False

Joint regression κ_d ~ strength + p + log n (CausalPFN-S): const +0.1816 [+0.0636, +0.3056]; strength -0.0332 [-0.0400, -0.0264]; p -0.0097 [-0.0104, -0.0090]; log n +0.1257 [+0.1072, +0.1427]
- in_prior_mech=False: 180 DGPs, κ pooled 0.604
- corr(κ_d, held-out explained-variance captured EVC) = 0.867; median EVC 0.727

Positivity control (same surfaces): κ hard → clip median 0.649 → 0.698; clip higher in 88/90

## H2 — suite `clip`: 90 decompositions

max |Euler residual| over all weight sets: 4.4e-16

Share of DiM's linear / nonlinear confounding removed (ratio of means, by p, n):

| p | n | pfn_S lin / nonlin | pfn_RA lin / nonlin | pfn_HO lin / nonlin | gp lin / nonlin | tols lin / nonlin | hajek_oracle lin / nonlin |
|---|---|---|---|---|---|---|---|
| 5 | 200 | 96% / 93% | 97% / 95% | 96% / 66% | 95% / 80% | 100% / -40% | 87% / 107% |
| 5 | 500 | 100% / 93% | 100% / 94% | 102% / 92% | 99% / 88% | 100% / -30% | 99% / 92% |
| 5 | 1000 | 101% / 91% | 101% / 93% | 101% / 92% | 100% / 95% | 100% / -22% | 97% / 101% |
| 10 | 200 | 93% / 64% | 93% / 65% | 86% / 31% | 97% / 69% | 100% / -36% | 102% / 103% |
| 10 | 500 | 97% / 74% | 98% / 75% | 99% / 53% | 99% / 78% | 100% / -38% | 93% / 95% |
| 10 | 1000 | 96% / 85% | 97% / 85% | 92% / 82% | 99% / 87% | 100% / -24% | 99% / 93% |
| 20 | 200 | 93% / -1% | 92% / -2% | 80% / -24% | 83% / 26% | 100% / -45% | 85% / 84% |
| 20 | 500 | 94% / 49% | 93% / 51% | 92% / 13% | 94% / 67% | 100% / -32% | 93% / 96% |
| 20 | 1000 | 97% / 69% | 97% / 69% | 99% / 47% | 99% / 76% | 100% / -27% | 99% / 102% |
| 30 | 200 | 79% / -11% | 79% / -11% | 66% / -34% | 76% / 12% | 100% / -47% | 100% / 92% |
| 30 | 500 | 90% / 17% | 91% / 17% | 95% / -39% | 94% / 52% | 100% / -36% | 98% / 110% |
| 30 | 1000 | 96% / 35% | 95% / 38% | 91% / 6% | 97% / 58% | 100% / -32% | 89% / 115% |
| 50 | 200 | 84% / -56% | 84% / -57% | 55% / -26% | 47% / 3% | 100% / -62% | 93% / 94% |
| 50 | 500 | 98% / -25% | 98% / -24% | 94% / -59% | 86% / 20% | 100% / -44% | 86% / 112% |
| 50 | 1000 | 93% / 20% | 93% / 21% | 95% / -15% | 94% / 38% | 100% / -31% | 96% / 97% |

Weight similarity (mean within-arm cosine of de-meaned net weights; noise floors = same estimator after a fresh noise draw):

- pfn_S|pfn_RA: 0.975
- pfn_S|pfn_HO: 0.533
- pfn_S|gp: 0.602
- pfn_S|tols: 0.504
- pfn_S|hajek_oracle: 0.428
- pfn_RA|pfn_HO: 0.540
- pfn_RA|gp: 0.639
- pfn_RA|tols: 0.539
- pfn_RA|hajek_oracle: 0.463
- pfn_HO|gp: 0.421
- pfn_HO|tols: 0.408
- pfn_HO|hajek_oracle: 0.311
- gp|tols: 0.597
- gp|hajek_oracle: 0.493
- tols|hajek_oracle: 0.430
- pfn_S|pfn_S_freshnoise: 0.693
- gp|gp_freshnoise: 0.875

Rich-dictionary imbalance: PFN ≤ Hájek(ê) in 47% of datasets; negative-weight share 0.15; label-dependent share 0.068

## H2 — suite `easy`: 120 decompositions

max |Euler residual| over all weight sets: 1.1e-15

Share of DiM's linear / nonlinear confounding removed (ratio of means, by p, n):

| p | n | pfn_S lin / nonlin | pfn_RA lin / nonlin | pfn_HO lin / nonlin | gp lin / nonlin | tols lin / nonlin | hajek_oracle lin / nonlin |
|---|---|---|---|---|---|---|---|
| 3 | 500 | 101% / 95% | 101% / 97% | 100% / 93% | 98% / 96% | 100% / -25% | 99% / 103% |
| 5 | 500 | 99% / 93% | 99% / 95% | 99% / 90% | 99% / 90% | 100% / -18% | 90% / 115% |
| 8 | 500 | 100% / 93% | 100% / 94% | 98% / 73% | 100% / 95% | 100% / -28% | 102% / 115% |

Weight similarity (mean within-arm cosine of de-meaned net weights; noise floors = same estimator after a fresh noise draw):

- pfn_S|pfn_RA: 0.987
- pfn_S|pfn_HO: 0.742
- pfn_S|gp: 0.745
- pfn_S|tols: 0.592
- pfn_S|hajek_oracle: 0.670
- pfn_RA|pfn_HO: 0.748
- pfn_RA|gp: 0.774
- pfn_RA|tols: 0.612
- pfn_RA|hajek_oracle: 0.692
- pfn_HO|gp: 0.633
- pfn_HO|tols: 0.525
- pfn_HO|hajek_oracle: 0.531
- gp|tols: 0.786
- gp|hajek_oracle: 0.670
- tols|hajek_oracle: 0.551
- pfn_S|pfn_S_freshnoise: 0.815
- gp|gp_freshnoise: 0.986

Rich-dictionary imbalance: PFN ≤ Hájek(ê) in 82% of datasets; negative-weight share 0.10; label-dependent share 0.031

## H2 — suite `hard`: 360 decompositions

max |Euler residual| over all weight sets: 8.9e-16

Share of DiM's linear / nonlinear confounding removed (ratio of means, by p, n):

| p | n | pfn_S lin / nonlin | pfn_RA lin / nonlin | pfn_HO lin / nonlin | gp lin / nonlin | tols lin / nonlin | hajek_oracle lin / nonlin |
|---|---|---|---|---|---|---|---|
| 5 | 200 | 93% / 81% | 94% / 84% | 92% / 54% | 95% / 74% | 100% / -45% | 87% / 91% |
| 5 | 500 | 98% / 90% | 98% / 91% | 98% / 80% | 98% / 86% | 100% / -50% | 97% / 103% |
| 5 | 1000 | 98% / 87% | 99% / 89% | 97% / 88% | 99% / 95% | 100% / -29% | 93% / 99% |
| 10 | 200 | 87% / 55% | 87% / 55% | 83% / 18% | 94% / 57% | 100% / -48% | 91% / 89% |
| 10 | 500 | 93% / 68% | 94% / 68% | 94% / 47% | 98% / 76% | 100% / -46% | 104% / 90% |
| 10 | 1000 | 94% / 78% | 96% / 77% | 91% / 71% | 99% / 85% | 100% / -40% | 96% / 91% |
| 20 | 200 | 80% / 1% | 81% / -2% | 69% / -27% | 76% / 21% | 100% / -54% | 76% / 95% |
| 20 | 500 | 87% / 38% | 88% / 37% | 82% / 6% | 94% / 53% | 100% / -43% | 99% / 93% |
| 20 | 1000 | 90% / 55% | 93% / 53% | 90% / 31% | 98% / 69% | 100% / -39% | 88% / 91% |
| 30 | 200 | 65% / -19% | 66% / -20% | 53% / -27% | 62% / 5% | 100% / -68% | 80% / 93% |
| 30 | 500 | 91% / -3% | 91% / -3% | 81% / -29% | 92% / 38% | 100% / -50% | 84% / 97% |
| 30 | 1000 | 88% / 27% | 91% / 26% | 79% / 10% | 96% / 52% | 100% / -42% | 88% / 90% |
| 50 | 200 | 79% / -63% | 79% / -62% | 48% / -42% | 45% / -1% | 100% / -74% | 88% / 99% |
| 50 | 500 | 98% / -42% | 99% / -42% | 84% / -51% | 82% / 7% | 100% / -56% | 95% / 101% |
| 50 | 1000 | 93% / 3% | 93% / 3% | 96% / -30% | 92% / 28% | 100% / -43% | 100% / 105% |

Weight similarity (mean within-arm cosine of de-meaned net weights; noise floors = same estimator after a fresh noise draw):

- pfn_S|pfn_RA: 0.977
- pfn_S|pfn_HO: 0.543
- pfn_S|gp: 0.551
- pfn_S|tols: 0.459
- pfn_S|hajek_oracle: 0.364
- pfn_RA|pfn_HO: 0.550
- pfn_RA|gp: 0.581
- pfn_RA|tols: 0.491
- pfn_RA|hajek_oracle: 0.385
- pfn_HO|gp: 0.399
- pfn_HO|tols: 0.381
- pfn_HO|hajek_oracle: 0.271
- gp|tols: 0.588
- gp|hajek_oracle: 0.355
- tols|hajek_oracle: 0.299
- pfn_S|pfn_S_freshnoise: 0.658
- gp|gp_freshnoise: 0.870

Rich-dictionary imbalance: PFN ≤ Hájek(ê) in 39% of datasets; negative-weight share 0.20; label-dependent share 0.072

## H3: 35 eval datasets (excluded for negligible correction: [11, 26]), l* = 10

Probe R² (mean):

|                            |   at_l |   peak |
|:---------------------------|-------:|-------:|
| ('e_perp', 'probe_rand')   |  0.61  |  0.645 |
| ('e_perp', 'probe_real')   |  0.728 |  0.814 |
| ('e_perp', 'probe_shuf')   |  0.547 |  0.611 |
| ('e_perp', 'rawx')         |  0.377 |  0.377 |
| ('logit_e', 'probe_rand')  |  0.81  |  0.826 |
| ('logit_e', 'probe_real')  |  0.871 |  0.92  |
| ('logit_e', 'probe_shuf')  |  0.859 |  0.915 |
| ('logit_e', 'rawx')        |  0.632 |  0.632 |
| ('nuisance', 'probe_rand') |  0.739 |  0.789 |
| ('nuisance', 'probe_real') |  0.81  |  0.92  |
| ('nuisance', 'probe_shuf') |  0.792 |  0.902 |
| ('nuisance', 'rawx')       |  0.748 |  0.748 |

|Δτ̂| / |τ̂ − DiM| under LEACE erasure (mean, 90% bootstrap upper bound):

- e_perp @ lstar: 0.104 (≤ 0.133)
- e_perp @ all: 0.239 (≤ 0.287)
- logit_e @ lstar: 0.454 (≤ 0.563)
- logit_e @ all: 0.769 (≤ 0.914)
- nuisance @ lstar: 0.021 (≤ 0.027)
- nuisance @ all: 0.051 (≤ 0.066)
- random_rank1 @ lstar: 0.007 (≤ 0.009)

Attention: max over heads of |partial corr(treated-key mass, logit e | poly2(x))|, mean over datasets, per layer:

- attn_corr_partial: 0.32 0.59 0.65 0.80 0.60 0.50 0.32 0.32 0.39 0.50 0.50 0.41 0.40 0.54 0.44 0.62 0.48 0.46 0.41 0.28
- attn_corr_partial_randinit: 0.22 0.26 0.12 0.32 0.28 0.26 0.26 0.20 0.16 0.20 0.22 0.24 0.17 0.24 0.21 0.16 0.21 0.27 0.16 0.21
- attn_corr_partial_shufT: 0.43 0.22 0.19 0.65 0.29 0.27 0.28 0.29 0.27 0.24 0.28 0.37 0.32 0.35 0.26 0.25 0.25 0.44 0.31 0.28

Coupled propensity swap: Δτ̂ = τ̂(D2) − τ̂(D1) (mean ± sd over datasets):

- psi: +0.000 ± 0.000
- DiM: +0.015 ± 0.187
- CausalPFN-S: +0.006 ± 0.100
- S-OLS: -0.001 ± 0.143
- IPW: -0.059 ± 0.391
- Hajek: -0.050 ± 0.343
- AIPW2: +0.002 ± 0.164
- GP-S: +0.005 ± 0.051
- CausalPFN-S_eperp_erased: -0.001 ± 0.130
- regression ΔPFN ~ ΔGP-S + ΔIPW: coefs GP +1.183, IPW +0.055; corr(ΔPFN, ΔGP) 0.657, corr(ΔPFN, ΔIPW) 0.375