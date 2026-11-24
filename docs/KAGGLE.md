# Running the full study on Kaggle (free tier: 30 GPU-h/week, 12 h per session)

## One-time setup (about 5 minutes)
0. **Build the bundle** (repo code + CausalPFN source + checkpoint; the latter two are not in this repo):
   `bash scripts/build_kaggle_bundle.sh /path/to/causalpfn_v0.pt` → `cfm-bundle.zip`.
1. **Upload the bundle as a dataset.** On kaggle.com go to *Datasets → New Dataset*. Drag in `cfm-bundle.zip` (Kaggle extracts it), name it `cfm-bundle`, keep it Private, and click *Create*.
2. **Create the notebook.** Go to *Code → New Notebook → File → Import Notebook* and choose `notebooks/kaggle_causalpfn.ipynb`.
3. In the notebook's right panel:
   - *Add Input → Datasets → Your Datasets → cfm-bundle*
   - *Settings → Accelerator → **GPU T4 x2*** (the code uses one of the two GPUs).
   - *Internet*: Off is fine. Turn it On only if you set `TABPFN = True`. This needs a phone-verified account.

## Runs (in order)
In the first cell, set `PHASE`, then click **Save Version → Save & Run All (Commit)**. Commit mode keeps running after you close the browser, and it saves `/kaggle/working/results` as the version's output.

| Run | PHASE | What it does | Expected time (T4) |
|---|---|---|---|
| 1 | `checks` | setup facts, 1e-6 gate, GPU-vs-CPU agreement | < 10 min |
| 2 | `h1` | κ for 11 estimators on 390 DGPs × 20 reps (easy 120, hard 180, clip 90), plus held-out-query CausalPFN and prior-typicality | ~5–8 h |
| 3 | `h2` | exact weights for CausalPFN S/RA/held-out and the GP, plus fresh-noise floors, on 570 datasets | ~5–9 h |
| 4 | `h3` | probes, LEACE, attention mass and propensity swaps on 37 datasets | ~2–4 h |
| (opt.) | `h1` with `TABPFN=True`, `SUITES="hard"` | TabPFN-as-S-learner contrast | ~2 h |

Total ≈ 12–20 GPU-hours, within one week's quota.

**Resuming.** Each phase stops cleanly at `HOURS` (default 11.3) and skips finished work when rerun. To continue, or to start the next phase on top of earlier results:
1. Open the notebook and choose *Add Input → Your Work → (this notebook)*, picking the latest version. Its `results/` folder is detected automatically.
2. Commit again.

Every version's output also carries `results_<phase>.zip` and `results/tables.md`.

## After the runs
Download each version's output (*Output → Download*), unzip all phases into one folder, then:
```bash
python scripts/analyze_full.py RESULTS_DIR          # tables.md
python scripts/make_figures.py RESULTS_DIR paper/figures
python scripts/make_numbers.py RESULTS_DIR paper/numbers.tex --final
```

## Local equivalent (any CUDA box)
```bash
python scripts/run_kaggle.py --phase h1 --out results/run --hours 100   # from a clone of this repo
python scripts/analyze_full.py results/run
```
