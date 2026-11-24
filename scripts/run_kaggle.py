"""Entry point for the full-scale study (any CUDA/CPU machine, or a Kaggle GPU via the bundle).

From a clone of this repo:
  python scripts/run_kaggle.py --phase checks --out results/run
On Kaggle (bundle built by scripts/build_kaggle_bundle.sh):
  python run_kaggle.py --bundle /kaggle/input/cfm-bundle --phase checks
  python run_kaggle.py --bundle ... --phase h1 --hours 11
  python run_kaggle.py --bundle ... --phase h2 --hours 11
  python run_kaggle.py --bundle ... --phase h3 --hours 11

Every phase is resumable: re-run with the previous outputs copied into --out and it skips
finished (dataset, rep) pairs.  Results go to --out (default /kaggle/working/results on Kaggle, else results/run).
"""
import argparse, json, os, shutil, sys, time, platform

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", default=None, help="Kaggle bundle dir; default: this repository")
ap.add_argument("--phase", required=True, choices=["checks", "h1", "h2", "h3"])
ap.add_argument("--out", default="/kaggle/working/results" if os.path.isdir("/kaggle/working") else "results/run")
ap.add_argument("--resume_from", action="append", default=[], help="folder(s) with earlier results to copy into --out first")
ap.add_argument("--hours", type=float, default=11.0)
ap.add_argument("--R", type=int, default=20)
ap.add_argument("--suites", default="easy,hard,clip")
ap.add_argument("--n_jobs", type=int, default=4)
ap.add_argument("--tabpfn", action="store_true", help="also run TabPFN as an S-learner (needs internet + `pip install tabpfn`)")
a = ap.parse_args()
deadline = time.time() + a.hours * 3600

if a.bundle is None:
    a.bundle = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path.insert(0, os.path.join(a.bundle, "causalpfn_src"))   # only present in Kaggle bundles
sys.path.insert(0, a.bundle)
os.makedirs(a.out, exist_ok=True)
for rf in a.resume_from:
    if not os.path.isdir(rf): continue
    for f in os.listdir(rf):
        src = os.path.join(rf, f); dst = os.path.join(a.out, f)
        if os.path.isfile(src) and not os.path.exists(dst): shutil.copy(src, dst)

import numpy as np, torch
import cfm.decomp as dc
import cfm.experiments as ex
from cfm.decomp import load_pretrained, random_init_model

dev = "cuda" if torch.cuda.is_available() else "cpu"
dc.DEVICE = dev
info = dict(device=dev, gpu=torch.cuda.get_device_name() if dev == "cuda" else None, torch=torch.__version__,
            python=platform.python_version(), gp_dtype=str(ex._gp_dtype()))
print(info, flush=True)
json.dump(info, open(os.path.join(a.out, f"env_{a.phase}.json"), "w"))
from cfm.paths import get_checkpoint
ckpt = os.path.join(a.bundle, "causalpfn_v0.pt")
ckpt = ckpt if os.path.exists(ckpt) else get_checkpoint()
model = load_pretrained(ckpt)

if a.phase == "checks":
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(a.bundle, "checks", "check_decomposition.py"), "--ckpt", ckpt],
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.pathsep.join([a.bundle, os.path.join(a.bundle, "causalpfn_src"), os.environ.get("PYTHONPATH", "")])})
    print(r.stdout[-4000:], r.stderr[-2000:])
    open(os.path.join(a.out, "checks_cpu.json"), "w").write(r.stdout)
    # GPU vs CPU agreement of the forward pass and of exact weights
    from cfm.dgp import make_hard_v2
    d = make_hard_v2(40).sample(0)
    dc.DEVICE = "cpu"; S_cpu = dc.tau_both(model, d["X"], d["t"], d["y"]); D_cpu = dc.decompose_ate(model, d["X"], d["t"], d["y"])
    dc.DEVICE = dev;   S_dev = dc.tau_both(model, d["X"], d["t"], d["y"]); D_dev = dc.decompose_ate(model, d["X"], d["t"], d["y"])
    res = dict(tau_fp32_cpu=S_cpu, tau_fp32_dev=S_dev, max_abs_w_diff_fp64=float(np.abs(D_cpu.w - D_dev.w).max()),
               euler_dev=D_dev.euler_resid)
    print("GPU/CPU agreement:", res); json.dump(res, open(os.path.join(a.out, "checks_gpu.json"), "w"), default=float)

elif a.phase == "h1":
    tab = None
    if a.tabpfn:
        try:
            from tabpfn import TabPFNRegressor
            def tab(X, t, y):
                m = TabPFNRegressor(device=dev).fit(np.column_stack([X, t]), y)
                return float(np.mean(m.predict(np.column_stack([X, np.ones(len(t))])) - m.predict(np.column_stack([X, np.zeros(len(t))]))))
            print("TabPFN enabled", flush=True)
        except Exception as e:
            print("TabPFN unavailable:", e, flush=True)
    for s in a.suites.split(","):
        ex.run_h1(model, s, a.R, a.out, n_jobs=a.n_jobs, deadline=deadline, tabpfn=tab)

elif a.phase == "h2":
    for s, reps, filt in (("easy", [0], None), ("hard", [0, 1], None), ("clip", [0], None)):
        if s in a.suites.split(","):
            ex.run_h2(model, s, reps, a.out, dgp_filter=filt, deadline=deadline)

elif a.phase == "h3":
    cfg = model.model_config["model"]
    rand = random_init_model(0, cfg["emsize"], cfg["nlayers"], cfg["nhead"], cfg["nhid_factor"],
                             cfg["max_num_features"], cfg["nbins"], cfg["max_num_classes"])
    ex.run_h3(model, rand, a.out, n_eval=37, deadline=deadline)

print("phase done", a.phase, flush=True)
shutil.make_archive("/kaggle/working/results_" + a.phase if os.path.isdir("/kaggle/working") else a.out.rstrip("/") + "_" + a.phase, "zip", a.out)
