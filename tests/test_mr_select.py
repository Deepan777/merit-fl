"""Algorithm 1 (MR-Select): energy gate and rare-class diversity guard."""
import numpy as np

from src.federated.core import mr_select

U = 6
COMMON = np.array([0.5, 0.2, 0.2, 0.1, 0.0])


def _phi(rare_holder=None):
    phi = [COMMON.copy() for _ in range(U)]
    if rare_holder is not None:
        phi[rare_holder] = np.array([0.4, 0.2, 0.1, 0.1, 0.2])
    return phi


def test_energy_gate_excludes_infeasible():
    battery = np.array([10, 1, 10, 1, 10, 10.0])
    S = mr_select(range(U), np.ones(U), np.ones(U), np.full(U, 5.0), battery, _phi(),
                  s_max=6)
    assert set(S) == {0, 2, 4, 5}


def test_energy_gate_off_includes_infeasible():
    battery = np.array([10, 1, 10, 1, 10, 10.0])
    S = mr_select(range(U), np.ones(U), np.ones(U), np.full(U, 5.0), battery, _phi(),
                  s_max=6, energy_gate=False)
    assert set(S) == set(range(U))


def test_score_orders_and_caps():
    tau = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
    S = mr_select(range(U), tau, np.ones(U), np.zeros(U), np.ones(U), _phi(),
                  s_max=3, w=(1.0, 0.0, 0.0))
    assert S == [0, 2, 4]


def test_diversity_guard_swaps_in_rare_holder():
    tau = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.1])       # rare holder 5 scores lowest
    S = mr_select(range(U), tau, np.ones(U), np.zeros(U), np.ones(U),
                  _phi(rare_holder=5), s_max=3, w=(1.0, 0.0, 0.0), rare_classes=(4,))
    assert 5 in S and len(S) == 3
    assert 2 not in S                                     # lowest-scoring member swapped out


def test_diversity_guard_covers_every_rare_class():
    phi = [np.array([0.6, 0.2, 0.2, 0.0, 0.0]) for _ in range(U)]
    phi[4] = np.array([0.4, 0.2, 0.2, 0.0, 0.2])          # holds class 4
    phi[5] = np.array([0.4, 0.2, 0.2, 0.2, 0.0])          # holds class 3
    tau = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])
    S = mr_select(range(U), tau, np.ones(U), np.zeros(U), np.ones(U), phi,
                  s_max=3, w=(1.0, 0.0, 0.0), rare_classes=(4, 3))
    assert 4 in S and 5 in S                              # second swap keeps the first holder


def test_infeasible_rare_holder_is_not_forced_in():
    battery = np.ones(U)
    battery[5] = 0.0
    S = mr_select(range(U), np.ones(U), np.ones(U), np.full(U, 0.5), battery,
                  _phi(rare_holder=5), s_max=3, rare_classes=(4,))
    assert 5 not in S
