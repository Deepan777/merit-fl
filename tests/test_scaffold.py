"""SCAFFOLD: reduces to SGD with zero controls; option-II control update."""
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.federated.core import (local_train_scaffold, scaffold_server_update,
                                zero_controls)
from src.models.fib import build_model


def _data(n=300, f=8, k=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, f)).astype(np.float32), rng.integers(0, k, n)


def test_zero_controls_equal_plain_sgd():
    X, y = _data()
    torch.manual_seed(0)
    m = build_model("mlp", 8, 3)
    c0 = zero_controls(m)
    torch.manual_seed(1)
    d1, n, steps, *_ = local_train_scaffold(m, X, y, torch.device("cpu"), c0, c0, 1, 0.1, 128)
    torch.manual_seed(1)
    mm = copy.deepcopy(m)
    opt = torch.optim.SGD(mm.parameters(), lr=0.1)
    mm.train()
    for xb, yb in DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                             batch_size=128, shuffle=True):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(mm(xb), yb).backward()
        opt.step()
    for k in d1:
        assert torch.allclose(d1[k], mm.state_dict()[k] - m.state_dict()[k], atol=1e-6)
    assert n == 300 and steps == 3


def test_control_variate_update_option_ii():
    X, y = _data()
    torch.manual_seed(0)
    m = build_model("mlp", 8, 3)
    c = {k: torch.full_like(v, 0.01) for k, v in zero_controls(m).items()}
    c_u = zero_controls(m)
    delta, _, steps, c_new, dc = local_train_scaffold(m, X, y, torch.device("cpu"), c, c_u, 1, 0.1, 128)
    for k in c:
        expect = c_u[k] - c[k] - delta[k] / (steps * 0.1)
        assert torch.allclose(c_new[k], expect, atol=1e-6)
        assert torch.allclose(dc[k], c_new[k] - c_u[k])


def test_server_control_update():
    c = {"w": torch.zeros(2)}
    out = scaffold_server_update(c, [{"w": torch.ones(2)}, {"w": 3 * torch.ones(2)}], n_total=8)
    assert torch.allclose(out["w"], torch.full((2,), 0.5))
