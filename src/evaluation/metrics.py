"""Evaluation for MERIT-FL (Part 12 metrics).

Two evaluation sets per run (both from the held-out split, disjoint):
  holdout  global detector quality (argmax decision), all classes present;
  stream   observed traffic scored by the deployed model with the fast-path
           decision rule (alert iff 1 - p(benign) >= theta); used for the
           handover-window analysis (H2), FAR and residual exposure.
"""
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)


# Inference batch: the cuDNN BiLSTM workspace grows with the batch (about 950 MiB
# at 4096), and on Windows GPU memory is charged to system commit. 1024 keeps a
# worker near 3 GB of commit; predictions match 4096 (argmax identical,
# probabilities within 5e-5).
PRED_BATCH = 1024


@torch.no_grad()
def predict_proba(model, X, device, batch=PRED_BATCH):
    model.eval().to(device)
    out = [torch.softmax(model(torch.from_numpy(X[i:i + batch]).to(device)), 1).cpu().numpy()
           for i in range(0, len(X), batch)]
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def evaluate(model, X, y, device, n_classes, benign=0, batch=PRED_BATCH):
    """Holdout metrics with the argmax decision."""
    yp = predict_proba(model, X, device, batch).argmax(1)
    labels = np.arange(n_classes)
    p, r, f1, _ = precision_recall_fscore_support(
        y, yp, average="macro", labels=labels, zero_division=0)
    cm = confusion_matrix(y, yp, labels=labels)
    others = [k for k in labels if k != benign]
    far = cm[benign, others].sum() / max(cm[benign].sum(), 1)    # benign flagged
    det = {int(k): float((yp[y == k] != benign).mean()) if (y == k).any() else None
           for k in others}
    return {"acc": accuracy_score(y, yp), "macro_f1": f1, "precision": p, "recall": r,
            "far": float(far),
            "per_class_f1": f1_score(y, yp, average=None, labels=labels, zero_division=0).tolist(),
            "per_class_detection": det}


def decide(probs, theta, benign):
    """Fast-path decision: alert iff attack score 1 - p(benign) >= theta; an
    alerted flow is labelled with its most likely attack class."""
    alert = (1.0 - probs[:, benign]) >= theta
    masked = probs.copy()
    masked[:, benign] = -1.0
    return alert, np.where(alert, masked.argmax(1), benign)


def detection_metrics(y, pred, alert, benign, names):
    """Detection on a set of stream flows. Macro-F1 averages the classes
    present in y or pred, so a set holding one class is not capped at 1/K."""
    n = len(y)
    if n == 0:
        return {"n": 0}
    att = y != benign
    labels = np.union1d(np.unique(y), np.unique(pred))
    return {
        "n": int(n), "n_attack": int(att.sum()), "n_benign": int((~att).sum()),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "attack_detection": float(alert[att].mean()) if att.any() else None,
        "far": float(alert[~att].mean()) if (~att).any() else None,
        "allowed_attacks": int((att & ~alert).sum()),
        "per_attack": {names[k]: {"rate": float(alert[y == k].mean()), "n": int((y == k).sum())}
                       for k in np.unique(y) if k != benign},
    }


def handover_report(s, benign, names, min_window_flows, n_events):
    """H2: detection inside vs outside handover windows, with sample counts.
    `s` holds stream arrays: y, pred, alert, in_window, gap, injected."""
    part = lambda m: detection_metrics(s["y"][m], s["pred"][m], s["alert"][m], benign, names)
    w = s["in_window"]
    inside, outside = part(w), part(~w)
    diff = lambda a, b, key: (None if a.get(key) is None or b.get(key) is None
                              else a[key] - b[key])
    return {
        "inside": inside, "outside": outside, "gap": part(s["gap"]),
        "injected_inside": part(s["injected"] & w), "injected_outside": part(s["injected"] & ~w),
        "n_handover_events": int(n_events),
        "min_window_flows": int(min_window_flows),
        "window_flows_ok": inside["n"] >= min_window_flows,
        "detection_dip": diff(outside, inside, "attack_detection"),
        "far_rise": diff(inside, outside, "far"),
        "macro_f1_dip": diff(outside, inside, "macro_f1"),
    }


def j_sec(m, xi):
    """E17 on the stream: F1 - xi1*FAR - xi2*residual exposure, with residual
    exposure = fraction of attack flows allowed through (normalised count)."""
    if m.get("n", 0) == 0 or m.get("attack_detection") is None or m.get("far") is None:
        return None
    return m["macro_f1"] - xi[0] * m["far"] - xi[1] * (1.0 - m["attack_detection"])


def trust_stability(tau_history):
    """Stab = -sum_t sum_u |tau(t)-tau(t-1)| (higher = more stable)."""
    T = np.asarray(tau_history)              # (T, U)
    return float(-np.abs(np.diff(T, axis=0)).sum()) if len(T) > 1 else 0.0


def comm_bytes(deltas):
    return int(sum(sum(v.numel() * 4 for v in d.values()) for d in deltas))
