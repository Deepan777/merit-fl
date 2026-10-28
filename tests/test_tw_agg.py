"""Algorithm 2 (TW-Agg): FedAvg equivalence, clipping, trust weighting."""
import copy

import numpy as np
import torch

from src.federated.core import agg_weights, fedavg_weights, tw_agg, twagg_weights
from src.models.fib import build_model


def _delta(model, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return {k: (torch.randn(v.shape, generator=g) * 0.01 * scale if v.dtype.is_floating_point
                else torch.zeros_like(v)) for k, v in model.state_dict().items()}


def _flat(model):
    return torch.cat([v.flatten().float() for v in model.state_dict().values()
                      if v.dtype.is_floating_point])


def _norm(d):
    return torch.cat([v.flatten().float() for v in d.values()]).norm().item()


def test_reduces_to_fedavg_when_trust_and_stability_uniform():
    torch.manual_seed(0)
    m = build_model("mlp", 8, 3)
    deltas = [_delta(m, s) for s in range(4)]
    sizes, sel = [100, 300, 50, 250], [0, 1, 2, 3]
    tau, stab = np.full(4, 0.7), np.full(4, 0.4)
    a = tw_agg(copy.deepcopy(m), deltas, twagg_weights(tau, stab, sizes, sel), clip_norm=1e9)
    b = tw_agg(copy.deepcopy(m), deltas, fedavg_weights(sizes), clip_norm=1e9)
    assert torch.allclose(_flat(a), _flat(b), atol=1e-7)


def test_clipping_bounds_a_100x_update():
    torch.manual_seed(0)
    m = build_model("mlp", 8, 3)
    benign = [_delta(m, s) for s in range(3)]
    attack = _delta(m, 99, scale=100.0)
    clip = 2 * max(_norm(d) for d in benign)
    assert _norm(attack) > 10 * clip
    before = _flat(m)
    out = tw_agg(copy.deepcopy(m), benign + [attack], [1, 1, 1, 1], clip_norm=clip)
    step = (_flat(out) - before).norm().item()
    assert step <= clip + 1e-5                            # convex combination of clipped updates
    raw = tw_agg(copy.deepcopy(m), benign + [attack], [1, 1, 1, 1], clip_norm=1e9)
    assert (_flat(raw) - before).norm().item() > 5 * step


def test_zero_trust_client_has_no_influence():
    torch.manual_seed(0)
    m = build_model("mlp", 8, 3)
    good, bad = _delta(m, 1), _delta(m, 2)
    w = agg_weights([100, 100], [0, 1], tau=np.array([0.9, 0.0]))
    out = tw_agg(copy.deepcopy(m), [good, bad], w, clip_norm=1e9)
    ref = tw_agg(copy.deepcopy(m), [good], [1.0], clip_norm=1e9)
    assert torch.allclose(_flat(out), _flat(ref), atol=1e-7)


def test_dhrm_multiplier_scales_weights():
    w = agg_weights([10, 10], [0, 1], mult=[1.0, 0.25])
    assert w[0] == 4 * w[1]
