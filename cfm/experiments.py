"""Full-scale experiment runners (resumable; CPU or GPU).

Suites (final design, set by the power analysis in RESULTS.md):
  easy     : 60 'ric' + 60 'overlap' DGPs, n = 500
  hard     : 90 cells (p in {5,10,20,30,50} x strength 1-6 x n in {200,500,1000}) x 2 seeds
  clip     : the 90 hard cells (seed 0) with e clipped to [0.1, 0.9]
  prop_nl  : 40 DGPs for H3 (0-2 = pilot used to fix l*)
"""
from __future__ import annotations

import json, os, time
import numpy as np
import pandas as pd
import torch

import cfm.decomp as dc
from cfm.dgp import make_suite, make_hard_v2, HARD_V2_GRID, make_prop_nl
from cfm.baselines import BASELINES, oracle_fluctuation, _prop
from cfm.gp_torch import gp_s_learner_torch, gp_implied_weights


# ------------------------------------------------------------------ suites
def suite(name: str):
    if name == "easy":
        return make_suite(60, n=500)
    if name == "hard":
        return [make_hard_v2(c, r) for r in (0, 1) for c in range(len(HARD_V2_GRID))]
    if name == "clip":
        return [make_hard_v2(c, 0, clip=0.1) for c in range(len(HARD_V2_GRID))]
    if name == "prop_nl":
        return [make_prop_nl(i) for i in range(40)]
    raise ValueError(name)


def gid(g):
    return f"{g.family}-{g.seed}"


# ------------------------------------------------------------------ prior labels
def mechanism_label(g, n_mc: int = 20000) -> dict:
    """Compare a DGP to CausalPFN's training prior (Balazadeh et al. 2025, App. D.1):
    prior propensity logit is constant / linear / random-MLP in x drawn independently of the
    outcome columns; noise variance <= Var(mu_t); causal-phase context <= 2048; <= 99 features.
    'aligned' = propensity is a function of the prognostic score (not how the prior generates
    confounding)."""
    rng = np.random.default_rng([g.seed, 31337])
    X = rng.standard_normal((n_mc, g.p))
    snr_ok = (g.noise ** 2) <= min(np.var(g.mu0(X)), np.var(g.mu1(X)))
    mech = {"overlap": "linear", "ric": "aligned", "ric_hard": "aligned", "ric_hard_clip": "aligned",
            "prop_nl": "nonlinear+partly-aligned"}[g.family]
    in_prior = bool(snr_ok and g.n <= 2048 and g.p <= 99 and mech == "linear")
    return dict(prop_mechanism=mech, noise_le_signal=bool(snr_ok), in_prior_mech=in_prior)


# ------------------------------------------------------------------ helpers
def _gp_dtype():
    # always float64: the GP's Cholesky steps are cheap even at T4 fp64 throughput, and
    # float32 risks Cholesky failures at n = 1000
    return torch.float64


@torch.no_grad()
def heldout_fit(model, X, t, y, seed=0):
    """Fast (no-grad, fp32) cross-fitted held-out ATE and factual predictions."""
    n = len(y); rng = np.random.default_rng(seed); fold = np.zeros(n, dtype=int)
    for arm in (0, 1):
        idx = np.where(t == arm)[0]; rng.shuffle(idx); fold[idx[: len(idx) // 2]] = 1
    model = model.float().to(dc.DEVICE)
    taus, fact = [], np.zeros(n)
    for a in (0, 1):
        A, B = fold == a, fold != a
        Xa = torch.as_tensor(X[A], dtype=torch.float32, device=dc.DEVICE); ta = torch.as_tensor(t[A], dtype=torch.float32, device=dc.DEVICE)
        ya = torch.as_tensor(y[A], dtype=torch.float32, device=dc.DEVICE)
        tr, ct = ta == 1, ta == 0
        y1m, y0m, s1, s0 = dc._arm_stats(ya, tr, ct)
        z = torch.where(tr, (ya - y1m) / s1, (ya - y0m) / s0)
        Xb = torch.as_tensor(X[B], dtype=torch.float32, device=dc.DEVICE); nb = Xb.shape[0]
        m = dc.cepo_standardised(model, Xa, ta, z, torch.cat([Xb, Xb]),
                                 torch.cat([torch.zeros(nb, device=dc.DEVICE), torch.ones(nb, device=dc.DEVICE)]))
        mu0, mu1 = (y0m + s0 * m[:nb]).cpu().numpy(), (y1m + s1 * m[nb:]).cpu().numpy()
        taus.append(float(np.mean(mu1 - mu0)))
        fact[B] = np.where(t[B] == 1, mu1, mu0)
    return float(np.mean(taus)), fact


def _cpu_baselines(X, t, y, names):
    return {k: BASELINES[k](X, t, y) for k in names}


# ------------------------------------------------------------------ H1
CPU_BASELINES = ["DiM", "S-OLS", "T-OLS", "IPW", "Hajek", "AIPW2", "kNN"]


def run_h1(model, suite_name, R, out_dir, n_jobs=4, deadline=None, tabpfn=None):
    from joblib import Parallel, delayed
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"h1_{suite_name}.csv")
    done = set()
    if os.path.exists(path):
        old = pd.read_csv(path); done = set(zip(old.dgp_id, old.rep))
    gp_dt = _gp_dtype(); t0 = time.time()
    with Parallel(n_jobs=n_jobs) as par:
        for g in suite(suite_name):
            if deadline and time.time() > deadline:
                print("deadline reached; resume later", flush=True); return
            reps = [r for r in range(R) if (gid(g), r) not in done]
            if not reps: continue
            draws = [g.sample(r) for r in reps]
            cpu = par(delayed(_cpu_baselines)(d["X"], d["t"], d["y"], CPU_BASELINES) for d in draws)
            lab = mechanism_label(g)
            rows = []
            for r, d, cb in zip(reps, draws, cpu):
                X, t, y = d["X"], d["t"], d["y"]
                S, RA = dc.tau_both(model, X, t, y)
                tau_ho, fact = heldout_fit(model, X, t, y, seed=r)
                mu_f = np.where(t == 1, d["mu1"], d["mu0"])
                r2 = lambda p_: 1 - ((y - p_) ** 2).mean() / y.var()
                row = dict(dgp_id=gid(g), family=g.family, strength=g.strength, p=g.p, n=g.n, rep=r, n1=int(t.sum()),
                           psi=d["psi"], oracle=oracle_fluctuation(d), **cb,
                           **{"CausalPFN-S": S, "CausalPFN-RA": RA, "CausalPFN-HO": tau_ho},
                           GP_S=gp_s_learner_torch(X, t, y, device=dc.DEVICE, dtype=gp_dt),
                           fact_r2_pfn=r2(fact), fact_r2_oracle=r2(mu_f), **lab)
                if tabpfn is not None:
                    try: row["TabPFN-S"] = tabpfn(X, t, y)
                    except Exception as e: row["TabPFN-S"] = np.nan
                rows.append(row)
            df = pd.DataFrame(rows).rename(columns={"GP_S": "GP-S"})
            df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            print(f"[h1 {suite_name}] {gid(g)} p={g.p} n={g.n} s={g.strength} {len(reps)} reps  {time.time()-t0:.0f}s", flush=True)


# ------------------------------------------------------------------ H2
def _hajek_oracle_w(d):
    t = d["t"]; w = np.zeros(len(t))
    h1 = 1 / d["e"][t == 1]; h0 = 1 / (1 - d["e"][t == 0])
    w[t == 1] = h1 / h1.sum(); w[t == 0] = -h0 / h0.sum(); return w


def _tols_w(X, t):
    A = np.column_stack([np.ones(len(X)), X]); w = np.zeros(len(t)); bbar = A.mean(0)
    for arm, sgn in ((1, 1.0), (0, -1.0)):
        Aa = A[t == arm]; w[t == arm] = sgn * Aa @ np.linalg.pinv(Aa.T @ Aa) @ bbar
    return w


def _net(w, t):
    return w - np.where(t == 1, 1 / (t == 1).sum(), -1 / (t == 0).sum())


def _armcos(a, b, t):
    out = []
    for arm in (1, 0):
        m = t == arm; x, z = a[m] - a[m].mean(), b[m] - b[m].mean()
        out.append(float(x @ z / (np.linalg.norm(x) * np.linalg.norm(z) + 1e-300)))
    return out


def _lin_split(X, mu):
    A = np.column_stack([np.ones(len(X)), X]); L = A @ np.linalg.lstsq(A, mu, rcond=None)[0]; return L, mu - L


def _attrib(w, d):
    """Signed imbalance of ATE weights w on the linear / nonlinear parts of mu1, mu0."""
    t, X = d["t"], d["X"]; tr, ct = t == 1, t == 0
    L1, N1 = _lin_split(X, d["mu1"]); L0, N0 = _lin_split(X, d["mu0"])
    w1, w0 = w[tr], -w[ct]
    lin = (w1 @ L1[tr] - L1.mean()) - (w0 @ L0[ct] - L0.mean())
    non = (w1 @ N1[tr] - N1.mean()) - (w0 @ N0[ct] - N0.mean())
    return float(lin), float(non)


def run_h2(model, suite_name, reps, out_dir, dgp_filter=None, deadline=None):
    from cfm.h2 import h2_report
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"h2_{suite_name}.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path): j = json.loads(l); done.add((j["dgp_id"], j["rep"]))
    gp_dt = _gp_dtype(); t0 = time.time()
    for g in suite(suite_name):
        if dgp_filter and not dgp_filter(g): continue
        for r in reps:
            if (gid(g), r) in done: continue
            if deadline and time.time() > deadline:
                print("deadline reached; resume later", flush=True); return
            d = g.sample(r); X, t, y = d["X"], d["t"], d["y"]
            dc.CHECKPOINT_LAYERS = g.n >= 800
            S = dc.decompose_ate(model, X, t, y, "S"); RA = dc.decompose_ate(model, X, t, y, "RA")
            HO = dc.decompose_ate_heldout(model, X, t, y, seed=r)
            gp_tau, gp_w, gp_res = gp_implied_weights(X, t, y, device=dc.DEVICE, dtype=gp_dt)
            rng = np.random.default_rng([g.seed, r, 4242])
            y2 = np.where(t == 1, d["mu1"], d["mu0"]) + g.noise * rng.standard_normal(len(y))
            S2 = dc.decompose_ate(model, X, t, y2, "S")
            _, gp_w2, _ = gp_implied_weights(X, t, y2, device=dc.DEVICE, dtype=gp_dt)
            W = {"pfn_S": S.w, "pfn_RA": RA.w, "pfn_HO": HO["w"], "gp": gp_w, "tols": _tols_w(X, t), "hajek_oracle": _hajek_oracle_w(d)}
            cos = {}
            names = list(W)
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    cos[f"{a}|{b}"] = _armcos(_net(W[a], t), _net(W[b], t), t)
            cos["pfn_S|pfn_S_freshnoise"] = _armcos(S.w_net, S2.w_net, t)
            cos["gp|gp_freshnoise"] = _armcos(_net(gp_w, t), _net(gp_w2, t), t)
            DiM = S.dim
            rec = dict(dgp_id=gid(g), rep=r, family=g.family, p=g.p, n=g.n, strength=g.strength, psi=d["psi"], dim=DiM,
                       tau={"pfn_S": S.tau, "pfn_RA": RA.tau, "pfn_HO": HO["tau"], "gp": gp_tau},
                       euler={"pfn_S": S.euler_resid, "pfn_RA": RA.euler_resid, "pfn_HO": HO["euler_resid"], "gp": gp_res},
                       attrib={k: _attrib(v, d) for k, v in W.items()},
                       dim_attrib=_attrib(np.where(t == 1, 1 / t.sum(), -1 / (1 - t).sum()), d),
                       noise_term={k: float(v @ (y - np.where(t == 1, d["mu1"], d["mu0"]))) for k, v in W.items()},
                       labeldep_share=float(np.linalg.norm(S.w_labeldep) / (np.linalg.norm(S.w_labelfree) + np.linalg.norm(S.w_labeldep))),
                       cos=cos, report=h2_report(S.w, X, t, _prop(X, t), d["mu0"], d["mu1"], degree=2 if g.p > 10 else 3),
                       **mechanism_label(g))
            with open(path, "a") as fh: fh.write(json.dumps(rec, default=float) + "\n")
            print(f"[h2 {suite_name}] {gid(g)} r{r} p={g.p} n={g.n} kappa_S={(S.tau-DiM)/(d['psi']-DiM):.2f} "
                  f"euler={max(abs(v) for v in rec['euler'].values()):.1e} {time.time()-t0:.0f}s", flush=True)


# ------------------------------------------------------------------ H3
def run_h3(model, rand_model, out_dir, n_eval=37, swap_strength=2.5, deadline=None):
    from cfm.h3 import (probe_profile, rawx_r2, concepts, context_streams, Leace, random_direction_eraser,
                        tau_with_erasure, fit_erasers_sequential, attention_treated_mass, attention_propensity_stats)
    from cfm.dgp import propensity_swap, coupled_pair
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "h3.jsonl"); pilot_path = os.path.join(out_dir, "h3_pilot.json")
    t0 = time.time()
    if os.path.exists(pilot_path):
        pilot = json.load(open(pilot_path))
    else:
        diffs = []
        for i in range(3):
            d = make_prop_nl(i).sample(0)
            real = probe_profile(model, d)["e_perp"]
            shuf = probe_profile(model, d, np.random.default_rng(i).permutation(d["t"]))["e_perp"]
            diffs.append(np.array(real) - np.array(shuf))
        md = np.mean(diffs, 0); pilot = dict(l_star=int(np.argmax(md)), mean_diff_by_layer=md.tolist())
        json.dump(pilot, open(pilot_path, "w")); print("[h3] pilot l* =", pilot["l_star"], flush=True)
    L = pilot["l_star"]; nlay = len(model.model.transformer_encoder)
    done = {json.loads(l)["idx"] for l in open(path)} if os.path.exists(path) else set()
    for idx in range(3, 3 + n_eval):
        if idx in done: continue
        if deadline and time.time() > deadline:
            print("deadline reached; resume later", flush=True); return
        g = make_prop_nl(idx); d = g.sample(0); X, t, y = d["X"], d["t"], d["y"]
        DiM = y[t == 1].mean() - y[t == 0].mean()
        rec = dict(idx=idx, strength=g.strength, psi=d["psi"], DiM=DiM, l_star=L)
        rec["probe_real"] = probe_profile(model, d)
        rec["probe_shuf"] = probe_profile(model, d, np.random.default_rng(100 + idx).permutation(t))
        rec["probe_rand"] = probe_profile(rand_model, d)
        rec["rawx"] = rawx_r2(d)
        S0, RA0 = tau_with_erasure(model, X, t, y, {}); rec["tau_S"], rec["tau_RA"] = S0, RA0
        C = concepts(d); acts = context_streams(model, X, t, y); er = {}; seq_eperp = None
        for cname in ("logit_e", "e_perp", "nuisance"):
            S1, RA1 = tau_with_erasure(model, X, t, y, {L: Leace(acts[L], C[cname])})
            seq = fit_erasers_sequential(model, d, C[cname], range(nlay))
            if cname == "e_perp": seq_eperp = seq
            S2, RA2 = tau_with_erasure(model, X, t, y, seq)
            er[cname] = dict(lstar_S=S1, lstar_RA=RA1, all_S=S2, all_RA=RA2)
        rs = [tau_with_erasure(model, X, t, y, {L: random_direction_eraser(acts[L], 1, s)}) for s in range(5)]
        er["random_rank1"] = dict(lstar_S=float(np.mean([r[0] for r in rs])), lstar_RA=float(np.mean([r[1] for r in rs])))
        rec["erasure"] = er
        # attention mass vs propensity
        mass = attention_treated_mass(model, X, t, y)
        raw, par = attention_propensity_stats(mass, d)
        rec["attn_corr_raw"] = raw.tolist(); rec["attn_corr_partial"] = par.tolist()
        rraw, rpar = attention_propensity_stats(attention_treated_mass(rand_model, X, t, y), d)
        rec["attn_corr_partial_randinit"] = rpar.tolist()
        sraw, spar = attention_propensity_stats(attention_treated_mass(model, X, np.random.default_rng(100 + idx).permutation(t), y), d)
        rec["attn_corr_partial_shufT"] = spar.tolist()
        # coupled propensity swap D1 (own strength) -> D2 (swap_strength)
        g2 = propensity_swap(g, swap_strength)
        d1, d2 = coupled_pair(g, g2, rep=0)
        sw = {}
        for nm, dd in (("D1", d1), ("D2", d2)):
            Xs, ts, ys = dd["X"], dd["t"], dd["y"]
            S_, _ = dc.tau_both(model, Xs, ts, ys)
            ent = dict(psi=dd["psi"], DiM=ys[ts == 1].mean() - ys[ts == 0].mean(), **{"CausalPFN-S": S_},
                       **{k: BASELINES[k](Xs, ts, ys) for k in ("S-OLS", "IPW", "Hajek", "AIPW2")},
                       **{"GP-S": gp_s_learner_torch(Xs, ts, ys, device=dc.DEVICE, dtype=_gp_dtype())})
            # PFN with e_perp erased at all layers (erasers refit on this dataset)
            Cs = concepts(dd); seq = fit_erasers_sequential(model, dd, Cs["e_perp"], range(nlay))
            ent["CausalPFN-S_eperp_erased"] = tau_with_erasure(model, Xs, ts, ys, seq)[0]
            sw[nm] = ent
        rec["swap"] = sw
        with open(path, "a") as fh: fh.write(json.dumps(rec, default=float) + "\n")
        corr = abs(S0 - DiM)
        print(f"[h3] idx {idx}: rel e_perp all={(er['e_perp']['all_S']-S0)/max(corr,1e-9):+.3f} "
              f"max|attn partial corr|={np.abs(par).max():.2f}  {time.time()-t0:.0f}s", flush=True)
