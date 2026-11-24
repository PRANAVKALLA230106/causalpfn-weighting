"""Exact gradient-x-input decomposition of CausalPFN's ATE estimate.

tau_hat(y) is degree-1 positively homogeneous in y (per-arm z-scoring), so by
Euler's theorem tau_hat = sum_i y_i * d tau_hat / d y_i exactly.  Subtracting
DiM's weights (+1/n1 treated, -1/n0 control) leaves w_net, which sums to zero
within each arm.

Write the correction as C = s1*M1(z) - s0*M0(z), where M_a is the (mode-specific)
average of the network's standardised predictions m_a.  With G_k = dC/dz_k
(holding s fixed) the chain rule through z-scoring gives, for j in arm a with
sign sigma_a = +1 (treated) / -1 (control):

    w_net_j = (G_j - mean_{arm a} G) / s_a                                  [label-free part]
            + z_j * (sigma_a * s_a * M_a - sum_{k in a} G_k z_k) / (n_a s_a)   [label-dependent]

(population SD, as in CausalPFN).  decompose_ate verifies this split against
the direct autograd weights.

All analysis runs in float64.  CausalPFN's RMSNorm casts to float32 internally;
fp64_mode() makes it respect the input dtype.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import torch

from causalpfn.models import transformer_layer as _tl
from causalpfn.models.icl_model import InContextModel
from causalpfn.models.model import TabDPTLongContextModel

_ORIG_RMS_FORWARD = _tl.RMSNorm.forward


def _rms_forward_native(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


@contextlib.contextmanager
def fp64_mode():
    """Run CausalPFN in float64 with dtype-preserving RMSNorm."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    _tl.RMSNorm.forward = _rms_forward_native
    try:
        yield
    finally:
        _tl.RMSNorm.forward = _ORIG_RMS_FORWARD
        torch.set_default_dtype(prev)


def random_init_model(seed: int = 0, emsize: int = 128, nlayers: int = 4, nhead: int = 4,
                      nhid_factor: int = 2, max_num_features: int = 32, nbins: int = 1000,
                      n_out: int = 10) -> InContextModel:
    """Randomly initialised CausalPFN-architecture network.  Used (i) as the
    random-init control from the proposal and (ii) to test the machinery before
    the pretrained checkpoint is available.  Default sizes are small for CPU."""
    torch.manual_seed(seed)
    base = TabDPTLongContextModel(dropout=0.0, n_out=n_out, nhead=nhead, nhid=emsize * nhid_factor,
                                  ninp=emsize, nlayers=nlayers, num_features=max_num_features, nbins=nbins)
    cfg = {"model": {"nbins": nbins, "max_num_features": max_num_features}, "model_type": "tabdpt"}
    return InContextModel(model=base, model_config=cfg).eval()


def load_pretrained(path: str | None = None) -> InContextModel:
    """Load the public CausalPFN checkpoint (tensors only; weights_only=True)."""
    from cfm.paths import get_checkpoint
    ckpt = torch.load(get_checkpoint(path), weights_only=True, map_location="cpu")
    m = InContextModel.load(model_state=ckpt["model_state_dict"], model_config=ckpt["model_config"])
    return m.eval()


CHECKPOINT_LAYERS = False
DEVICE = "cpu"   # set to "cuda" on GPU machines (cfm.decomp.DEVICE = "cuda")
# H3 instrumentation: {layer_index: fn(src[T,B,H], context_len) -> src}; applied AFTER that layer.
LAYER_HOOKS = {}


def _tabdpt_forward(mm, x_src, y_src):
    """Re-implementation of TabDPTLongContextModel.forward (eval mode), optionally with
    per-layer activation checkpointing to cut backward-pass memory."""
    from causalpfn.models.model import normalize_data, clip_outliers
    from torch.utils.checkpoint import checkpoint
    L = y_src.shape[0]
    x_src = normalize_data(x_src, L)
    x_src = clip_outliers(x_src, L, n_sigma=10)
    x_src = torch.nan_to_num(x_src, nan=0)
    x_src = mm.xnorm(mm.encoder(x_src))
    y_e = mm.ynorm(mm.y_encoder(y_src.unsqueeze(-1)))
    src = torch.cat([x_src[:L] + y_e, x_src[L:]], 0)
    for li, layer in enumerate(mm.transformer_encoder):
        src = checkpoint(layer, src, L, use_reentrant=False) if CHECKPOINT_LAYERS else layer(src, L)
        if LAYER_HOOKS.get(li) is not None:
            src = LAYER_HOOKS[li](src, L)
    pred = mm.head(src)
    pred = 30 * torch.tanh(pred / (7.5 * src.size(-1) ** 0.5))
    return pred[L:]


def cepo_standardised(model: InContextModel, X, t, z, Xq, tq, temperature: float = 1.0):
    """Network output m (standardised units, mean over bins) for queries (Xq, tq).
    Mirrors InContextModel.predict_cepo before un-standardisation."""
    xt_c = torch.cat([t.unsqueeze(-1), X], dim=-1)[None]
    xt_q = torch.cat([tq.unsqueeze(-1), Xq], dim=-1)[None]
    x_src, y_src = model.prepare_input(torch.cat([xt_c, xt_q], dim=1), z[None])
    logits = _tabdpt_forward(model.model, x_src.transpose(0, 1), y_src.transpose(0, 1)).transpose(0, 1)
    logits = logits[:, :, -model.model.nbins:] / temperature
    return model._predict_mean(logits)[0]


@dataclass
class Decomposition:
    mode: str
    tau: float
    dim: float
    w: np.ndarray            # d tau / d y
    w_net: np.ndarray        # w - DiM weights
    euler_resid: float       # tau - sum(w*y)
    armsum_t: float          # sum of w_net over treated (should be 0)
    armsum_c: float
    w_labelfree: np.ndarray
    w_labeldep: np.ndarray
    split_resid: float       # max |w_net - (labelfree + labeldep)|
    correction: float        # tau - dim
    cap: float               # 10*(s1+s0) style bound for this mode
    m1: np.ndarray
    m0: np.ndarray
    s1: float
    s0: float


def _arm_stats(y, tr, ct):
    y1m, y0m = y[tr].mean(), y[ct].mean()
    s1 = torch.sqrt(((y[tr] - y1m) ** 2).mean()) + 1e-20
    s0 = torch.sqrt(((y[ct] - y0m) ** 2).mean()) + 1e-20
    return y1m, y0m, s1, s0


def _M(m0, m1, tr, ct, n, mode):
    if mode == "S":
        return m1.mean(), m0.mean()
    # RA: factual outcomes replace predictions; only counterfactual predictions enter
    return m1[ct].sum() / n, m0[tr].sum() / n


def decompose_ate(model: InContextModel, X: np.ndarray, t: np.ndarray, y: np.ndarray,
                  mode: str = "S", temperature: float = 1.0) -> Decomposition:
    """Exact decomposition of CausalPFN's ATE with in-context queries (the training
    rows, at t=0 and t=1), as in ATEEstimator.estimate_ate(algorithm=mode)."""
    assert mode in ("S", "RA")
    with fp64_mode():
        model = model.double().to(DEVICE)
        Xt = torch.as_tensor(X, dtype=torch.float64, device=DEVICE)
        tt = torch.as_tensor(t, dtype=torch.float64, device=DEVICE)
        n = len(y)
        tr, ct = tt == 1, tt == 0
        n1, n0 = int(tr.sum()), int(ct.sum())
        Xq = torch.cat([Xt, Xt], 0)
        tq = torch.cat([torch.zeros(n, device=DEVICE), torch.ones(n, device=DEVICE)])

        # ---- pass 1: full graph y -> z -> network -> tau ; w = d tau / d y
        yt = torch.as_tensor(y, dtype=torch.float64, device=DEVICE).clone().requires_grad_(True)
        y1m, y0m, s1, s0 = _arm_stats(yt, tr, ct)
        z = torch.where(tr, (yt - y1m) / s1, (yt - y0m) / s0)
        m = cepo_standardised(model, Xt, tt, z, Xq, tq, temperature)
        m0, m1 = m[:n], m[n:]
        if mode == "S":
            mu1 = y1m + s1 * m1
            mu0 = y0m + s0 * m0
            tau = (mu1 - mu0).mean()
        else:
            mu1 = torch.where(tr, yt, y1m + s1 * m1)
            mu0 = torch.where(ct, yt, y0m + s0 * m0)
            tau = (mu1 - mu0).mean()
        w = torch.autograd.grad(tau, yt)[0]

        # ---- pass 2: z as leaf, G = dC/dz with s fixed
        zl = z.detach().clone().requires_grad_(True)
        m_ = cepo_standardised(model, Xt, tt, zl, Xq, tq, temperature)
        M1, M0 = _M(m_[:n], m_[n:], tr, ct, n, mode)
        s1d, s0d = s1.detach(), s0.detach()
        C = s1d * M1 - s0d * M0
        G = torch.autograd.grad(C, zl)[0]

        yv, zv = yt.detach(), z.detach()
        tau_v, dim_v = float(tau.detach()), float((y1m - y0m).detach())
        w = w.detach()
        dim_w = torch.where(tr, 1.0 / n1, -1.0 / n0)
        w_net = w - dim_w

        lf = torch.zeros(n, device=DEVICE); ld = torch.zeros(n, device=DEVICE)
        for mask, s_a, n_a, sign, Ma in ((tr, s1d, n1, 1.0, M1.detach()), (ct, s0d, n0, -1.0, M0.detach())):
            Ga, za = G[mask], zv[mask]
            lf[mask] = (Ga - Ga.mean()) / s_a
            ld[mask] = za * (sign * s_a * Ma - (Ga * za).sum()) / (n_a * s_a)

        vmax = float(model.vmax)
        cap = vmax * (float(s1d) + float(s0d)) if mode == "S" else vmax * (n0 * float(s1d) + n1 * float(s0d)) / n

        return Decomposition(
            mode=mode, tau=tau_v, dim=dim_v, w=w.cpu().numpy(), w_net=w_net.cpu().numpy(),
            euler_resid=tau_v - float((w * yv).sum()),
            armsum_t=float(w_net[tr].sum()), armsum_c=float(w_net[ct].sum()),
            w_labelfree=lf.cpu().numpy(), w_labeldep=ld.cpu().numpy(),
            split_resid=float((w_net - lf - ld).abs().max()),
            correction=tau_v - dim_v, cap=cap,
            m1=m1.detach().cpu().numpy(), m0=m0.detach().cpu().numpy(), s1=float(s1d), s0=float(s0d),
        )


def tau_only(model: InContextModel, X, t, y, mode="S", temperature=1.0, fp64=True) -> float:
    """Fast forward-only ATE (no grad)."""
    ctx = fp64_mode() if fp64 else contextlib.nullcontext()
    with ctx, torch.no_grad():
        dt = torch.float64 if fp64 else torch.float32
        model = model.to(dt).to(DEVICE)
        Xt = torch.as_tensor(X, dtype=dt, device=DEVICE); tt = torch.as_tensor(t, dtype=dt, device=DEVICE); yt = torch.as_tensor(y, dtype=dt, device=DEVICE)
        n = len(y); tr, ct = tt == 1, tt == 0
        y1m, y0m, s1, s0 = _arm_stats(yt, tr, ct)
        z = torch.where(tr, (yt - y1m) / s1, (yt - y0m) / s0)
        m = cepo_standardised(model, Xt, tt, z, torch.cat([Xt, Xt]), torch.cat([torch.zeros(n, dtype=dt, device=DEVICE), torch.ones(n, dtype=dt, device=DEVICE)]), temperature)
        m0, m1 = m[:n], m[n:]
        if mode == "S":
            return float((y1m + s1 * m1 - y0m - s0 * m0).mean())
        mu1 = torch.where(tr, yt, y1m + s1 * m1); mu0 = torch.where(ct, yt, y0m + s0 * m0)
        return float((mu1 - mu0).mean())


def tau_both(model: InContextModel, X, t, y, temperature=1.0, fp64=False):
    """S and RA ATEs from one forward pass (fp32 by default: matches the public path to ~1e-7)."""
    ctx = fp64_mode() if fp64 else contextlib.nullcontext()
    with ctx, torch.no_grad():
        dt = torch.float64 if fp64 else torch.float32
        model = model.to(dt).to(DEVICE)
        Xt = torch.as_tensor(X, dtype=dt, device=DEVICE); tt = torch.as_tensor(t, dtype=dt, device=DEVICE); yt = torch.as_tensor(y, dtype=dt, device=DEVICE)
        n = len(y); tr, ct = tt == 1, tt == 0
        y1m, y0m, s1, s0 = _arm_stats(yt, tr, ct)
        z = torch.where(tr, (yt - y1m) / s1, (yt - y0m) / s0)
        m = cepo_standardised(model, Xt, tt, z, torch.cat([Xt, Xt]), torch.cat([torch.zeros(n, dtype=dt, device=DEVICE), torch.ones(n, dtype=dt, device=DEVICE)]), temperature)
        mu0, mu1 = y0m + s0 * m[:n], y1m + s1 * m[n:]
        S = float((mu1 - mu0).mean())
        RA = float((torch.where(tr, yt, mu1) - torch.where(ct, yt, mu0)).mean())
        return S, RA


def decompose_ate_heldout(model: InContextModel, X, t, y, seed: int = 0, temperature: float = 1.0):
    """Cross-fitted held-out-query ATE: tau = 1/2 [tau_{A->B} + tau_{B->A}], where
    tau_{A->B} = mean over rows in B of mu1_hat - mu0_hat with context A.  Each half is
    degree-1 homogeneous in its context outcomes, so tau = sum w y exactly (Euler).
    Also returns held-out factual predictions (for prior-typicality / fit diagnostics)."""
    n = len(y); rng = np.random.default_rng(seed)
    # stratified split so both arms appear in both halves
    fold = np.zeros(n, dtype=int)
    for arm in (0, 1):
        idx = np.where(t == arm)[0]; rng.shuffle(idx); fold[idx[: len(idx) // 2]] = 1
    with fp64_mode():
        model = model.double().to(DEVICE)
        yt = torch.as_tensor(y, dtype=torch.float64, device=DEVICE).clone().requires_grad_(True)
        Xt = torch.as_tensor(X, dtype=torch.float64, device=DEVICE); tt = torch.as_tensor(t, dtype=torch.float64, device=DEVICE)
        taus, fact = [], np.zeros(n)
        for a_ in (0, 1):
            A = torch.as_tensor(fold == a_, device=DEVICE); B = ~A
            yA, tA, XA = yt[A], tt[A], Xt[A]
            trA, ctA = tA == 1, tA == 0
            y1m, y0m, s1, s0 = _arm_stats(yA, trA, ctA)
            zA = torch.where(trA, (yA - y1m) / s1, (yA - y0m) / s0)
            XB = Xt[B]; nB = XB.shape[0]
            m = cepo_standardised(model, XA, tA, zA, torch.cat([XB, XB]),
                                  torch.cat([torch.zeros(nB, device=DEVICE), torch.ones(nB, device=DEVICE)]), temperature)
            mu0, mu1 = y0m + s0 * m[:nB], y1m + s1 * m[nB:]
            taus.append((mu1 - mu0).mean())
            tB = tt[B]
            fact[(fold != a_)] = torch.where(tB == 1, mu1, mu0).detach().cpu().numpy()
        tau = 0.5 * (taus[0] + taus[1])
        w = torch.autograd.grad(tau, yt)[0].detach()
        tau_v = float(tau.detach())
        return dict(tau=tau_v, w=w.cpu().numpy(), euler_resid=tau_v - float((w * yt.detach()).sum()), fact_pred=fact, fold=fold)
