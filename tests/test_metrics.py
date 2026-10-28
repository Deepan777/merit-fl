"""Evaluation metrics: decision rule, FAR with any benign index, H2 report."""
import numpy as np
import torch

from src.evaluation.metrics import (decide, detection_metrics, evaluate,
                                    handover_report, j_sec)

NAMES = ["A", "Normal", "B", "C"]
BENIGN = 1


def test_decide_threshold_and_attack_label():
    probs = np.array([[0.1, 0.6, 0.2, 0.1],     # attack score 0.4
                      [0.5, 0.2, 0.2, 0.1]])    # attack score 0.8, top attack A
    alert, pred = decide(probs, np.array([0.5, 0.5]), BENIGN)
    assert alert.tolist() == [False, True] and pred.tolist() == [BENIGN, 0]
    alert, pred = decide(probs, np.array([0.3, 0.9]), BENIGN)
    assert alert.tolist() == [True, False] and pred.tolist() == [2, BENIGN]


def test_single_class_set_is_not_capped_at_one_over_k():
    y = np.zeros(50, dtype=int)                 # all class A, all detected correctly
    m = detection_metrics(y, y.copy(), np.ones(50, bool), BENIGN, NAMES)
    assert m["macro_f1"] == 1.0
    assert m["attack_detection"] == 1.0 and m["far"] is None


def test_far_uses_the_benign_index():
    y = np.array([1, 1, 1, 1, 0, 2])
    alert = np.array([True, False, False, False, True, False])
    pred = np.where(alert, 0, BENIGN)
    m = detection_metrics(y, pred, alert, BENIGN, NAMES)
    assert m["far"] == 0.25
    assert m["per_attack"]["A"] == {"rate": 1.0, "n": 1}
    assert m["per_attack"]["B"] == {"rate": 0.0, "n": 1}
    assert m["allowed_attacks"] == 1


class _Fixed(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.logits[x[:, 0].long()]


def test_holdout_far_with_benign_not_first():
    logits = [[9, 0, 0, 0], [0, 9, 0, 0], [0, 0, 9, 0], [9, 0, 0, 0]]
    X = np.array([[0], [1], [2], [3]], dtype=np.float32)    # row 3: benign -> predicted A
    y = np.array([0, 1, 2, 1])
    m = evaluate(_Fixed(logits), X, y, torch.device("cpu"), 4, benign=BENIGN)
    assert m["far"] == 0.5
    assert m["per_class_detection"][0] == 1.0


def test_handover_report_counts_and_dip():
    n = 100
    s = {"y": np.r_[np.zeros(50, int), np.ones(50, int)],
         "in_window": np.r_[np.ones(20, bool), np.zeros(30, bool), np.ones(10, bool), np.zeros(40, bool)],
         "gap": np.zeros(n, bool), "injected": np.zeros(n, bool)}
    s["alert"] = s["y"] == 0
    s["alert"][:10] = False                               # half the in-window attacks missed
    s["pred"] = np.where(s["alert"], 0, BENIGN)
    rep = handover_report(s, BENIGN, NAMES, min_window_flows=25, n_events=3)
    assert rep["inside"]["n"] == 30 and rep["outside"]["n"] == 70
    assert rep["window_flows_ok"]
    assert rep["detection_dip"] == 0.5
    assert j_sec(rep["outside"], (1.0, 1.0)) == 1.0
