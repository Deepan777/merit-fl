"""R6 robustness simulation: state placements, edge-node blocks, threshold search."""
import numpy as np

from src.v2.robust import (edge_view, first_crossing, first_crossings, honest_hazard, longest_residence,
                           node_of_cell, path)


def test_node_of_cell_blocks():
    assert (node_of_cell(4, 1) == np.arange(16)).all()
    n = node_of_cell(4, 2)
    assert n[0] == n[1] == n[4] == n[5] == 0 and n[2] == n[3] == n[6] == n[7] == 1 and n[15] == 3


def test_edge_view_ignores_handover_within_a_block():
    serving = np.array([[0], [1], [4], [2]])                     # cells 0,1,4 in block 0; cell 2 in block 1
    node, ho = edge_view({"serving": serving}, 4, 2)
    assert node[:, 0].tolist() == [0, 0, 0, 1] and ho[:, 0].tolist() == [False, False, False, True]


def _walk():
    # one device: node 0 for 3 windows, node 1 for 2, back to node 0 for 2; increment +1 per window
    node = np.array([[0], [0], [0], [1], [1], [0], [0]])
    ho = np.zeros_like(node, dtype=bool)
    ho[1:] = node[1:] != node[:-1]
    inc = np.ones((7, 1))
    return inc, np.ones((7, 1), dtype=bool), node, ho


def test_placements():
    inc, ok, node, ho = _walk()
    carry = path("share", 0, inc, ok, node, ho, "carry", 2)[:, 0]
    reset = path("share", 0, inc, ok, node, ho, "reset", 2)[:, 0]
    mem = path("share", 0, inc, ok, node, ho, "memory", 2)[:, 0]
    assert carry.tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert reset.tolist() == [1, 2, 3, 1, 2, 1, 2]
    assert mem.tolist() == [1, 2, 3, 1, 2, 4, 5]                # resumes node 0's value (3) on return


def test_threshold_search_and_hazard():
    rng = np.random.default_rng(0)
    p = np.maximum.accumulate(rng.normal(size=(50, 6)).cumsum(0), axis=0)
    hs = np.array([0.5, 2.0, 5.0])
    many = first_crossings(p, hs)
    for i, h in enumerate(hs):
        assert (many[i] == first_crossing(p, h)).all()
    first = np.array([0, 10, 20, 50, 50, 30])                     # W = 50: 50 means never
    ev, hrs = honest_hazard(first, np.ones(6, bool), 5, 50, 30)
    assert ev == 3 and np.isclose(hrs, (5 + 15 + 45 + 45 + 25) * 30 / 3600)


def test_alarms_restart_and_vectorised_thresholds():
    from src.v2.robust import run_alarms
    inc, ok, node, ho = _walk()                                   # +1 per window, 7 windows
    al, first = run_alarms("share", 0, inc, ok, node, ho, "carry", 2, [3.0], 0, {0: 4})
    assert al[0, 0] == 2 and first[0, 0] == 5                    # alarms at windows 2 and 5 (restart at 0)
    al_r, _ = run_alarms("share", 0, inc, ok, node, ho, "reset", 2, [3.0], 0, {})
    assert al_r[0, 0] == 1                                        # reset at windows 3 and 5: only window 2
    al_m, _ = run_alarms("share", 0, inc, ok, node, ho, "memory", 2, [3.0], 0, {})
    assert al_m[0, 0] == 1                                        # window 2 alarm clears node 0's memory too
    both, _ = run_alarms("share", 0, inc, ok, node, ho, "carry", 2, [2.0, 3.0], 0, {})
    one, _ = run_alarms("share", 0, inc, ok, node, ho, "carry", 2, [2.0], 0, {})
    assert both[0, 0] == one[0, 0] == 3 and both[1, 0] == 2


def test_r7_reference_llr_floor_and_lossy_carry():
    from src.v2.robust import hourly_type_reference, llr_floor, llr_increment, run_alarms
    rng = np.random.default_rng(0)
    W, U = 60, 8
    seen = np.full((W, U), 60)
    p = np.array([0.1] * 4 + [0.3] * 4)                          # two types with different flag rates
    flagged = rng.binomial(seen, p[None, :])
    tr = {"seen": seen, "flagged": flagged}
    dev_type = np.array([0] * 4 + [1] * 4)
    p0, valid = hourly_type_reference(tr, dev_type, horizon=30, min_flows=30, warm=5)
    assert not valid[:5].any() and valid[10:].all()
    assert abs(p0[40, 0] - 0.1) < 0.03 and abs(p0[40, 7] - 0.3) < 0.04
    inc = llr_increment(tr, p0, valid, 0.05)
    assert inc[:5].sum() == 0 and inc[10:].mean() < 0          # honest devices drift down under a correct null
    rho = llr_floor(0.1, 0.9, 0.05)
    assert 0 < rho < 0.05                                       # p* lies between p0 and p0 + delta
    node = np.zeros((W, 1), int)
    node[::2, 0] = 1
    ho = np.zeros((W, 1), bool)
    ho[1:] = node[1:] != node[:-1]
    one = np.ones((W, 1))
    a_c, _ = run_alarms("llr", 0, one, np.ones((W, 1), bool), node, ho, "carry", 2, [5.0], 0, {})
    a_l, _ = run_alarms("llr", 0, one, np.ones((W, 1), bool), node, ho, "carry_loss", 2, [5.0], 0, {},
                        loss=1.0, rng=np.random.default_rng(0))
    assert a_c[0, 0] == W // 5 and a_l[0, 0] == 0              # losing context at every change = reset


def test_longest_residence():
    node = np.array([[0], [0], [1], [1], [1], [0]])
    assert longest_residence(node, 0, 0, 6) == 3 and longest_residence(node, 0, 4, 6) == 1


def test_r8_flag_models_keep_the_mean():
    from src.v2.robust import benign_rates
    rng = np.random.default_rng(1)
    p = np.array([0.0, 0.02, 0.3, 1.0])
    assert benign_rates(p, 5, None, rng).shape == (1, 4)
    q = benign_rates(p, 20000, {"kind": "betabin", "rho": 0.1}, rng)
    assert (q[:, 0] == 0).all() and (q[:, 3] == 1).all()
    assert abs(q[:, 1].mean() - 0.02) < 0.002 and abs(q[:, 2].mean() - 0.3) < 0.01 and q[:, 2].std() > 0.1
    b = benign_rates(p, 20000, {"kind": "burst", "m": 10, "p_in": 1 / 240, "p_stay": 0.75}, rng)
    share = (b[:, 1] > 0.02).mean()                             # stationary burst share (1/240) / (1/240 + 0.25)
    assert abs(share - (1 / 240) / (1 / 240 + 0.25)) < 0.008 and b[:, 2].max() == 1.0


def test_r8_device_reference_lag_and_shrinkage():
    from src.v2.robust import device_reference, hourly_pooled_reference
    W, U = 40, 2
    seen = np.full((W, U), 100)
    flagged = np.zeros((W, U), dtype=int)
    flagged[:, 1] = 10                                          # device 1: own share 0.1
    flagged[30:, 0] = 50                                        # device 0 starts attacking at window 30
    tr = {"seen": seen, "flagged": flagged}
    p_type = np.full((W, U), 0.05)
    p0, ok = device_reference(tr, p_type, np.ones((W, U), bool), lag=5, hist=10, m=200, warm=2, z=0.0)
    assert not ok[6].any() and ok[7].all()                      # learning period: w - lag < warm
    assert abs(p0[20, 1] - (100 + 200 * 0.05) / (1000 + 200)) < 1e-12   # 10 windows of own history, shrunk
    assert p0[35, 0] == p0[34, 0] == (0 + 10) / (1000 + 200)    # the attack (from 30) is not yet in the reference
    assert p0[36, 0] > p0[35, 0]                                # ... and enters it after the lag
    pz, okz = device_reference(tr, p_type, np.ones((W, U), bool), lag=5, hist=10, m=200, warm=2, z=2.0, ceiling=0.09)
    se = np.sqrt(p0[20, 1] * (1 - p0[20, 1]) / 1200)
    assert abs(pz[20, 1] - p0[20, 1] - 2 * se) < 1e-12 and not okz[20, 1] and okz[20, 0]   # upper bound; ceiling
    pr, v = hourly_pooled_reference(tr, horizon=4, warm=2)
    assert not v[1, 0] and v[5, 0] and abs(pr[5, 0] - 0.05) < 1e-12


def test_r8_sprt_barrier_and_beta_reputation():
    from src.v2.robust import run_alarms, run_alarms_beta
    W = 6
    node = np.zeros((W, 1), dtype=int)
    ho = np.zeros((W, 1), dtype=bool)
    ok = np.ones((W, 1), dtype=bool)
    inc = np.array([[-3.0], [-3.0], [2.0], [2.0], [2.0], [2.0]])
    al, first = run_alarms("sprt", -4.6, inc, ok, node, ho, "carry", 1, [5.0], 0, {0: 0})
    assert first[0, 0] == 4                                     # restart at 0 after -6 <= -4.6, then 2, 4, 6 >= 5
    al, first = run_alarms("llr", 0, inc, ok, node, ho, "carry", 1, [5.0], 0, {0: 0})
    assert first[0, 0] == 4                                     # CUSUM: 0, 0, 2, 4, 6
    seen = np.full((W, 1), 100)
    fl = np.array([[5], [5], [50], [50], [50], [50]])
    p_ref = np.full((W, 1), 0.05)
    al, first = run_alarms_beta(fl, seen, p_ref, ok, 0.9, node, ho, "carry", [0.2], 0, {0: 2})
    assert first[0, 0] == 3 and al[0, 0] >= 1                   # score 0.165 at window 2, 0.247 at window 3
    al0, _ = run_alarms_beta(fl[:2], seen[:2], p_ref[:2], ok[:2], 0.9, node[:2], ho[:2], "carry", [1e-6], 0, {})
    assert al0.sum() == 0                                       # at the base rate the score stays at 0


def test_r9_dispersion_estimator_and_robust_increment():
    from src.v2.robust import dispersion, llr_increment, robust_increment, type_dispersion
    rng = np.random.default_rng(0)
    W, U = 400, 2
    n = np.full((W, U), 50)
    p = np.column_stack([np.full(W, 0.1), rng.beta(0.1 * 19, 0.9 * 19, size=W)])     # rho = 1 / (1 + 19) = 0.05
    tr = {"seen": n, "flagged": rng.binomial(n, p)}
    rho, k = dispersion(tr, lag=10, hist=300, warm=2, min_k=10)
    assert (rho[:12] == 0).all() and k[11, 0] == 0              # learning period
    assert rho[399, 0] < 0.01 and 0.03 < rho[399, 1] < 0.08     # binomial vs beta-binomial(0.05)
    assert k[399, 0] == 300
    tr1 = {"seen": n[:, :1], "flagged": np.zeros((W, 1), int)}
    assert (dispersion(tr1, lag=10, hist=300, warm=2)[0] == 0).all()   # p = 0: no estimate
    td = type_dispersion(rho, k, np.array([0, 0]))
    assert abs(td[399, 0] - np.median(rho[399])) < 1e-12 and td[0, 0] == 0
    p0, ok = np.full((W, U), 0.1), np.ones((W, U), bool)
    raw = llr_increment(tr, p0, ok, 0.05)
    x = robust_increment(tr, p0, ok, 0.05, rho, np.inf)
    assert np.allclose(x, raw / (1 + 49 * rho))
    assert robust_increment(tr, p0, ok, 0.05, np.zeros((W, U)), 0.5).max() <= 0.5


def test_r9_replay_keeps_counts_and_adds_attack_flows():
    from src.v2.robust import simulate_replay
    rng = np.random.default_rng(1)
    W, U, ws = 50, 3, 30
    node = np.zeros((W * ws, U), dtype=int)
    ho = np.zeros((W * ws, U), dtype=bool)
    outage = np.zeros((W * ws, U), dtype=bool)
    outage[:ws, 2] = True                                       # device 2 out of coverage in window 0
    sn = rng.integers(0, 20, size=(W, U))
    sf = rng.binomial(sn, 0.2)
    tr = simulate_replay(node, ho, outage, sn, sf, {1: {"start": 10 * ws, "p_det": 1.0}}, ws, rng)
    assert (tr["seen"][:, 0] == sn[:, 0]).all() and (tr["flagged"][:, 0] == sf[:, 0]).all()
    assert tr["seen"][0, 2] == 0 and (tr["seen"][1:, 2] == sn[1:, 2]).all()
    assert tr["attacks"][:10, 1].sum() == 0 and tr["attacks"][10:, 1].sum() > 0
    assert (tr["attack_flagged"] == tr["attacks"]).all() and (tr["benign"][:, 1] == sn[:, 1]).all()
