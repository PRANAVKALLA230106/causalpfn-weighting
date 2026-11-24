# What Estimator Does CausalPFN Learn?
### An Exact Weighting View of a Causal Foundation Model

**Pranav Kalla** · paper: [`paper/paper.pdf`](paper/paper.pdf)

CausalPFN estimates treatment effects in one forward pass. This repository shows that its per-arm outcome standardisation makes its ATE estimate **exactly** a weighting estimator,

$$\hat\tau = \mathrm{DiM} + \sum_i w^{\mathrm{net}}_i\, y_i, \qquad w = \nabla_{\mathbf y}\hat\tau \quad(\text{Euler; residual} \le 10^{-15}),$$

and that its error equals the **imbalance of these weights on the true outcome surfaces** plus noise. Using this, together with probes, LEACE erasure, attention analysis and coupled propensity swaps, we characterise how CausalPFN corrects for confounding.

## Main results
All results are on the public `causalpfn_v0.pt` checkpoint: 390 synthetic DGPs × 20 replications, 570 exact decompositions, 37 propensity-probe datasets, and IHDP.

| Question | Finding |
|---|---|
| **How much bias is removed?** (share κ) | Easy DGPs: **0.95–0.99**. Dense nonlinear confounding: **0.604** [0.574, 0.635], against **0.740** for a GP S-learner with the same preprocessing (paired −0.135). With overlap bounded to [0.1, 0.9]: 0.675, against GP 0.784 and IPW 0.852. κ falls with dimension, rises with n, and tracks held-out fit (r = 0.87–0.95). |
| **What do the weights balance?** | **89%** of the linear confounding but only **32%** of the nonlinear part (GP 87 / 51, OLS 100 / −48, oracle Hájek 91 / 95). 79% of the remaining bias is nonlinear; it predicts the shortfall (r = −0.91). |
| **Is propensity used?** | It is computed (probe R² 0.73 vs 0.55 with T shuffled) and partly used: LEACE erasure moves τ̂ by **10–24%** of the correction, which falsifies the pre-registered "known but unused" hypothesis. Under coupled propensity swaps, though, CausalPFN responds like the GP outcome model (coefficient 1.18), not like IPW (0.05). |
| **IHDP** | κ 0.94 (linear surface) / 1.01 (nonlinear surface), because IHDP's confounding is about 95% linear, so the failure mode never shows. |

Full tables: [`results/full/tables.md`](results/full/tables.md). Verdicts and run history: [`docs/RESULTS_LOG.md`](docs/RESULTS_LOG.md).

## Repository layout
```
cfm/                    the analysis package
  decomp.py             exact weight decomposition (S / RA / cross-fitted held-out), float64,
                        dtype-preserving RMSNorm patch, layer hooks, activation checkpointing
  dgp.py, ihdp.py       synthetic DGP families (easy, hard, clip, prop-NL) and IHDP surfaces
  baselines.py          DiM, OLS S/T, IPW, Hájek, cross-fitted AIPW, kNN, sklearn GP S-learner
  gp_torch.py           GPU GP S-learner + its exact implied weights
  kappa.py              κ_d, response slopes, DGP-clustered bootstrap
  h2.py, h3.py          balance diagnostics; probes, LEACE, attention mass
  experiments.py        resumable full-scale runners (H1, H2, H3)
  paths.py              checkpoint / data resolution
checks/                 numerical checks of every setup fact and the 1e-6 reproduction gate
scripts/
  run_kaggle.py         entry point for all phases (repo or Kaggle bundle; CPU or GPU)
  run_ihdp.py           IHDP benchmark
  analyze_full.py       results -> tables.md
  make_figures.py       results -> paper figures
  make_numbers.py       results -> paper/numbers.tex (every number in the paper)
  build_kaggle_bundle.sh, get_ihdp.sh
  pilot/                the original pilot scripts (kept for provenance)
notebooks/kaggle_causalpfn.ipynb   Kaggle driver notebook (see docs/KAGGLE.md)
results/full/           all full-scale outputs (h1_*.csv, h2_*.jsonl, h3.jsonl, ihdp/, figures/)
results/pilot/          pilot-stage outputs
paper/                  LaTeX source (ICLR 2027 style), figures, numbers.tex, paper.pdf
```

## Setup
```bash
pip install -r requirements.txt          # includes CausalPFN from GitHub
pip install -e .
```
**Checkpoint.** `cfm.paths.get_checkpoint()` uses, in order: a `--ckpt`-style path, `$CAUSALPFN_CKPT`, `checkpoints/causalpfn_v0.pt`, or an automatic download from [huggingface.co/vdblm/causalpfn](https://huggingface.co/vdblm/causalpfn). The weights are not redistributed here.
**IHDP.** `bash scripts/get_ihdp.sh` fetches the covariates into `data/IHDP/`.

## Reproduce
```bash
# 1. setup facts (Euler, arm sums, invariances, split) and the 1e-6 gate (~2 min on CPU)
python checks/check_decomposition.py --ckpt checkpoints/causalpfn_v0.pt
python scripts/run_kaggle.py --phase checks --out results/run          # adds a GPU-vs-CPU check

# 2. full study (resumable; ~12-20 GPU-hours on a T4; see docs/KAGGLE.md for free Kaggle GPUs)
python scripts/run_kaggle.py --phase h1 --out results/run --hours 100   # κ, 390 DGPs x 20 reps
python scripts/run_kaggle.py --phase h2 --out results/run --hours 100   # exact weights, 570 datasets
python scripts/run_kaggle.py --phase h3 --out results/run --hours 100   # probes, LEACE, attention, swaps
python scripts/run_ihdp.py --out results/run/ihdp                       # IHDP (~1 h on CPU)

# 3. tables, figures, numbers, paper (works directly on the shipped results/full)
python scripts/analyze_full.py results/full
python scripts/make_figures.py results/full paper/figures
python scripts/make_numbers.py results/full paper/numbers.tex --final
cd paper && latexmk -pdf paper.tex
```
Every number in the paper is a macro in `paper/numbers.tex`. Most are generated by `make_numbers.py`; the rest are listed at the end of that file, computed from the same result files. Recomputing random rows of `results/full` on a different machine agrees to ≤ 5e-7 (float32), and exact weights agree across CPU/GPU to 3e-16.

## Implementation notes
- With each arm at most 1,024 rows, CausalPFN's retrieval keeps every row, so the direct network call equals `ATEEstimator.estimate_ate` to < 1e-6.
- CausalPFN's `RMSNorm` casts to float32. `cfm.decomp.fp64_mode()` patches it to preserve dtype for exact decompositions; the patch changes float32 predictions by < 1e-6.
- The weights depend on y, so they are a *local* description. The error identity is exact at the observed data.

## Citation
```bibtex
@misc{kalla2026causalpfnweighting,
  title  = {What Estimator Does {CausalPFN} Learn? An Exact Weighting View of a Causal Foundation Model},
  author = {Kalla, Pranav},
  year   = {2026}
}
```

## Licence
Code: MIT (see `LICENSE`). CausalPFN's source and weights belong to their authors under the CausalPFN License and are not included. The ICLR style files in `paper/` belong to ICLR.
