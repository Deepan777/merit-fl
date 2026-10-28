"""Federated core of MERIT-FL: client training, MR-Select (Algorithm 1),
TW-Agg with update clipping (Algorithm 2), plus FedAvg/FedProx/SCAFFOLD.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..data.partition import kl_usefulness


# --------------------------- client ---------------------------
def _loader(X, y, batch):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                      batch_size=batch, shuffle=True)


def local_train(model, X, y, device, epochs=1, lr=1e-3, batch=256,
                prox_mu=0.0, global_params=None):
    """One client's local update (Adam). Returns delta (state_dict of diffs,
    on CPU), n_samples, n_steps."""
    model = copy.deepcopy(model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.train()
    steps = 0
    for _ in range(epochs):
        for xb, yb in _loader(X, y, batch):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            if prox_mu > 0 and global_params is not None:   # FedProx
                prox = sum((p - global_params[k].to(device)).pow(2).sum()
                           for k, p in model.named_parameters())
                loss = loss + 0.5 * prox_mu * prox
            loss.backward()
            opt.step()
            steps += 1
    delta = {k: (model.state_dict()[k] - g0[k]).cpu() for k in g0}
    return delta, len(y), steps


def local_train_scaffold(model, X, y, device, c_global, c_local, epochs=1,
                         lr=0.05, batch=256):
    """SCAFFOLD client (Karimireddy et al., option II) with plain SGD:
    g <- grad - c_u + c;  c_u+ = c_u - c + (theta_g - theta_u) / (K * lr).
    Control variates cover parameters only (BN buffers are averaged)."""
    model = copy.deepcopy(model).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    g0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    corr = {k: (c_global[k] - c_local[k]).to(device) for k in c_global}
    model.train()
    steps = 0
    for _ in range(epochs):
        for xb, yb in _loader(X, y, batch):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            for k, p in model.named_parameters():
                p.grad += corr[k]
            opt.step()
            steps += 1
    sd = model.state_dict()
    delta = {k: (sd[k] - g0[k]).cpu() for k in g0}
    c_new = {k: c_local[k] - c_global[k] - delta[k] / (max(steps, 1) * lr)
             for k in c_global}
    delta_c = {k: c_new[k] - c_local[k] for k in c_global}
    return delta, len(y), steps, c_new, delta_c


def zero_controls(model):
    return {k: torch.zeros_like(p, device="cpu") for k, p in model.named_parameters()}


def scaffold_server_update(c_global, delta_cs, n_total):
    """c <- c + (|S|/N) * mean_u(delta c_u) = c + sum_u(delta c_u) / N."""
    return {k: c_global[k] + sum(dc[k] for dc in delta_cs) / n_total for k in c_global}


# ----------------------- Algorithm 1: MR-Select -----------------------
def mr_select(candidates, tau, stability, energy_need, battery, phi, s_max,
              w=(0.4, 0.3, 0.3), rare_classes=(), rare_mass=0.01,
              energy_gate=True):
    """Mobility- and Resource-Aware Client Selection.
    feasible = energy gate; score = w1*tau + w2*s + w3*usefulness;
    diversity guard: for every rare class with no holder in S, swap in the
    best feasible holder for the lowest-scoring member that does not hold an
    already-guarded rare class."""
    feas = [u for u in candidates
            if not energy_gate or energy_need[u] <= battery[u]]
    if not feas:
        return []
    phi_mean = np.mean([phi[u] for u in feas], axis=0)
    mu = np.array([kl_usefulness(phi[u], phi_mean) for u in feas])
    mu = mu / (mu.max() + 1e-9)
    score = {u: w[0] * tau[u] + w[1] * stability[u] + w[2] * mu[j]
             for j, u in enumerate(feas)}
    S = sorted(feas, key=lambda u: -score[u])[:s_max]
    guarded = []
    for k in rare_classes:
        holds = lambda u, c=k: phi[u][c] > rare_mass
        if not any(holds(u) for u in S):
            outside = [u for u in feas if u not in S and holds(u)]
            if outside:
                protected = {u for u in S if any(phi[u][j] > rare_mass for j in guarded)}
                swappable = [u for u in S if u not in protected] or S
                worst = min(swappable, key=score.get)
                S[S.index(worst)] = max(outside, key=score.get)
        guarded.append(k)
    return S


# ------------------- Algorithm 2: TW-Agg (+clipping) -------------------
def tw_agg(global_model, deltas, weights_raw, clip_norm=10.0):
    """theta <- theta + sum_u w_u * clip(delta_u); w normalised to sum 1."""
    w = np.array(weights_raw, dtype=np.float64)
    w = w / (w.sum() + 1e-12)
    gsd = global_model.state_dict()
    agg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in gsd.items()
           if v.dtype.is_floating_point}
    for wi, delta in zip(w, deltas):
        flat = torch.cat([delta[k].flatten().float() for k in agg])
        norm = flat.norm().item()
        scale = min(1.0, clip_norm / (norm + 1e-12))     # bound outliers
        for k in agg:                                    # deltas live on CPU
            agg[k] += float(wi) * delta[k].to(agg[k].device).float() * scale
    for k in gsd:
        if k in agg:
            gsd[k] = gsd[k] + agg[k].to(gsd[k].dtype)
    global_model.load_state_dict(gsd)
    return global_model


def agg_weights(n_samples, selected, tau=None, stab=None, mult=None):
    """|D_u| [* tau_u] [* s_u] [* DHRM multiplier]; normalised in tw_agg."""
    w = np.asarray(n_samples, dtype=np.float64)
    sel = np.asarray(selected)
    if tau is not None:
        w = w * np.asarray(tau)[sel]
    if stab is not None:
        w = w * np.asarray(stab)[sel]
    if mult is not None:
        w = w * np.asarray(mult)
    return w.tolist()


def fedavg_weights(n_samples):
    return list(n_samples)                              # ∝ |D_u|


def twagg_weights(tau, stab, n_samples, selected):
    return agg_weights(n_samples, selected, tau=tau, stab=stab)
