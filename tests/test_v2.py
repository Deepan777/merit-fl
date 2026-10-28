"""v2 modules: radio layer, containment logic, hierarchical FL smoke run."""
import os

import numpy as np
import pytest

from src.v2.containment import contain, evidence
from src.v2.network import compute_links, edge_grid, p_stay, time_to_exit

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _trace(T=200):
    """Device 0 static next to edge 0; device 1 drives from edge 0 to edge 1 along the x axis."""
    edges = edge_grid(1000.0, 2)                              # centres at 250 and 750
    pos = np.zeros((T, 2, 2))
    pos[:, 0] = (250.0, 250.0)
    pos[:, 1, 0] = np.linspace(150.0, 850.0, T)
    pos[:, 1, 1] = 250.0
    return pos, edges


def test_handover_happens_once_for_a_crossing_and_never_for_a_static_device():
    pos, edges = _trace()
    L = compute_links(pos, edges, seed=0)
    assert not L["handover"][:, 0].any()
    assert 1 <= L["handover"][:, 1].sum() <= 3                 # hysteresis + time-to-trigger: no ping-pong storm
    assert L["serving"][0, 1] == 0 and L["serving"][-1, 1] == 1


def test_time_to_exit_and_p_stay():
    pos, edges = _trace()
    vel = np.array([[0.0, 0.0], [3.5, 0.0]])
    tte = time_to_exit(pos[0], vel, np.array([0, 0]), edges)
    assert np.isinf(tte[0]) and 0 < tte[1] < 200               # crosses the boundary at x ~ 500 + margin
    assert p_stay(np.array([5.0]), 30)[0] < 0.5 < p_stay(np.array([60.0]), 30)[0]


def _traffic(W=40, U=6, attacker=5, excess=0.3, ho_every=None):
    seen = np.full((W, U), 60.0)
    flagged = np.full((W, U), 3.0)                            # 5% benign flagging for everyone
    flagged[10:, attacker] += excess * 60
    ho = np.zeros((W, U), dtype=bool)
    if ho_every:
        ho[::ho_every, attacker] = True
    return {"seen": seen, "flagged": flagged, "ho_win": ho, "cell": np.zeros((W, U), dtype=int),
            "attacks": np.zeros((W, U)), "benign_flows": seen}


P = {"k": 0.01, "h": 0.5, "tau0": 0.5, "eps": 0.2, "delta": 0.85, "q": 0.6}


def test_cusum_contains_a_persistent_attacker_and_no_honest_device():
    tr = _traffic()
    yhat, x = evidence(tr)
    first = contain("cusum_carry", tr, yhat, x, P)
    assert first[5] >= 10 and (first[:5] == -1).all()


def test_reset_at_frequent_handovers_launders_a_weak_attacker():
    tr = _traffic(excess=0.05, ho_every=3)
    yhat, x = evidence(tr)
    assert contain("cusum_carry", tr, yhat, x, P)[5] >= 0
    assert contain("cusum_reset", tr, yhat, x, P)[5] == -1


def test_recalibrated_reset_uses_its_own_threshold_and_carry_dominates():
    tr = _traffic(excess=0.05, ho_every=3)
    yhat, x = evidence(tr)
    low = {**P, "h_reset": 0.05}
    assert contain("cusum_reset_cal", tr, yhat, x, low)[5] >= 0          # low enough h_reset catches it
    assert contain("cusum_reset_cal", tr, yhat, x, {**P, "h_reset": P["h"]})[5] == -1
    for m in ("cusum_reset", "cusum_reset_cal"):                       # pathwise: carry never later than reset
        a, b = contain("cusum_carry", tr, yhat, x, low), contain(m, tr, yhat, x, {**low, "h_reset": low["h"]})
        assert ((a >= 0) | (b < 0)).all() and ((b < 0) | (a <= b)).all()


def test_forged_token_contains_and_withheld_token_resets_v1_trust():
    tr = _traffic(excess=0.0, ho_every=4)
    yhat, x = evidence(tr)
    forge = contain("cusum_carry", tr, yhat, x, P, token={5: ("forge", 10)})
    assert forge[5] >= 10 and forge[5] % 4 == 0
    tr2 = _traffic(excess=0.0, ho_every=1)
    y2, x2 = evidence(tr2)
    # v1 trust: a withheld token gives the prior at every handover (no containment either way here)
    assert (contain("trust_carry", tr2, y2, x2, P, token={5: ("omit", 0)}) == -1).all()


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                    "data", "5g_nidd", "processed", "family.npz")),
                    reason="needs the preprocessed 5G-NIDD data: scripts/fetch_5g_nidd.py, then scripts/preprocess.py")
def test_hfl_smoke_run(tmp_path):
    from src.v2.config import load_v2
    from src.v2.hfl import run_hfl
    if not os.path.exists(os.path.join(REPO, "data", "v2_traces", "seed100.npz")):
        pytest.skip("trace not generated")
    os.environ["MERIT_FL_FORBID_TEST"] = "1"
    cfg = load_v2(overrides={"seed": 100, "eval.split": "val", "output_dir": str(tmp_path),
                             "net.horizon_s": 300, "net.trace_dir": os.path.join(REPO, "data", "v2_traces"),
                             "data.root": os.path.join(REPO, "data"), "fl.forwarding": True,
                             "fl.selection": "residence", "eval.every_rounds": 2})
    out = run_hfl(cfg, log=lambda *a: None)
    s = out["stats"]
    assert s["selected"] == s["delivered"] + s["interrupted"] + s["lost_outage"]
    assert s["forwarded"] == s["interrupted"] and out["curve"]
    assert os.path.exists(os.path.join(out["run_dir"], "model.pt"))
