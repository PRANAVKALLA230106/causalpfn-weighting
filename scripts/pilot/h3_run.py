"""H3 run on the prop_nl family.

Pilot datasets 0..2 fix l* = argmax over layers of mean [R2(e_perp | real T) - R2(e_perp | shuffled T)].
Evaluation datasets 3..(3+n_eval-1):
  probes: real T, shuffled T, random-init network; raw-x polynomial probe
  erasure at l* and at all layers (sequential fit) of: logit_e, e_perp, nuisance, random rank-1
  outcome: delta tau / |tau - DiM|   (pre-registered margin 0.10)
python -I h3_run.py --n_eval 9 --out results/h3.jsonl
"""
import argparse, json, os, sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import numpy as np, torch
from cfm.dgp import make_prop_nl
from cfm.decomp import load_pretrained, random_init_model
from cfm.h3 import probe_profile, rawx_r2, concepts, context_streams, Leace, random_direction_eraser, tau_with_erasure, fit_erasers_sequential

ap = argparse.ArgumentParser(); ap.add_argument("--n_eval", type=int, default=9); ap.add_argument("--out", required=True)
ap.add_argument("--ckpt", default=None); ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args(); torch.set_num_threads(a.threads)
model = load_pretrained(a.ckpt)
cfg = model.model_config["model"]
rand = random_init_model(seed=0, emsize=cfg["emsize"], nlayers=cfg["nlayers"], nhead=cfg["nhead"], nhid_factor=cfg["nhid_factor"],
                         max_num_features=cfg["max_num_features"], nbins=cfg["nbins"], n_out=cfg["max_num_classes"])
t0 = time.time()


def shuffled_t(d, seed):
    return np.random.default_rng(seed).permutation(d["t"])


pilot_path = a.out.replace(".jsonl", "_pilot.json")
if os.path.exists(pilot_path):
    pilot = json.load(open(pilot_path))
else:
    diffs = []
    for i in range(3):
        d = make_prop_nl(i).sample(0)
        real = probe_profile(model, d)["e_perp"]; shuf = probe_profile(model, d, shuffled_t(d, i))["e_perp"]
        diffs.append(np.array(real) - np.array(shuf))
    md = np.mean(diffs, 0)
    pilot = dict(l_star=int(np.argmax(md)), mean_diff_by_layer=md.tolist())
    json.dump(pilot, open(pilot_path, "w"))
    print("pilot l* =", pilot["l_star"], np.round(md, 3), f"{time.time()-t0:.0f}s", flush=True)
L = pilot["l_star"]; nlay = cfg["nlayers"]

done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["idx"] for l in open(a.out)}
for idx in range(3, 3 + a.n_eval):
    if idx in done: continue
    g = make_prop_nl(idx); d = g.sample(0); X, t, y = d["X"], d["t"], d["y"]
    DiM = y[t == 1].mean() - y[t == 0].mean()
    rec = dict(idx=idx, strength=g.strength, psi=d["psi"], DiM=DiM, l_star=L)
    rec["probe_real"] = probe_profile(model, d)
    rec["probe_shuf"] = probe_profile(model, d, shuffled_t(d, 100 + idx))
    rec["probe_rand"] = probe_profile(rand, d)
    rec["rawx"] = rawx_r2(d)
    S0, RA0 = tau_with_erasure(model, X, t, y, {})
    rec["tau_S"], rec["tau_RA"] = S0, RA0
    C = concepts(d)
    acts = context_streams(model, X, t, y)
    er = {}
    for cname in ("logit_e", "e_perp", "nuisance"):
        S1, RA1 = tau_with_erasure(model, X, t, y, {L: Leace(acts[L], C[cname])})
        seq = fit_erasers_sequential(model, d, C[cname], range(nlay))
        S2, RA2 = tau_with_erasure(model, X, t, y, seq)
        er[cname] = dict(lstar_S=S1, lstar_RA=RA1, all_S=S2, all_RA=RA2)
    rs = []
    for s in range(5):
        S1, RA1 = tau_with_erasure(model, X, t, y, {L: random_direction_eraser(acts[L], 1, s)})
        rs.append((S1, RA1))
    er["random_rank1"] = dict(lstar_S=float(np.mean([r[0] for r in rs])), lstar_RA=float(np.mean([r[1] for r in rs])),
                              lstar_S_absdev=float(np.mean([abs(r[0] - S0) for r in rs])))
    rec["erasure"] = er
    with open(a.out, "a") as fh: fh.write(json.dumps(rec, default=float) + "\n")
    corr = abs(S0 - DiM)
    print(f"idx {idx}: tau_S={S0:.3f} DiM={DiM:.3f} psi={d['psi']:.3f} | rel delta e_perp@l*={(er['e_perp']['lstar_S']-S0)/corr:+.3f} "
          f"all={(er['e_perp']['all_S']-S0)/corr:+.3f} logit_e all={(er['logit_e']['all_S']-S0)/corr:+.3f} "
          f"nuis all={(er['nuisance']['all_S']-S0)/corr:+.3f}  {time.time()-t0:.0f}s", flush=True)
