"""C1: handover-continuous sequential evidence and containment (the design notes, Section 2).

The detector is fixed (a trained model per dataset and seed); its attack scores for the stream pool
are computed once. Each device sends flows_per_s outgoing flows per second while connected (flows sent
in outage are not observed). Honest devices send benign flows; an attacker sends, from its attack
start, attack flows of its family with probability rho(t), benign flows otherwise. The edge node
attributes outgoing flows to their source device; external attacks are not part of this experiment.

Evidence per window of window_s seconds: x_u = flagged share of u's outgoing flows (attack score >=
0.5) minus the median share of the devices served by the same edge node (all devices if fewer than
min_peers). Methods (all on the same traffic):
  cusum_carry    S = max(0, S + x - k); contained at S >= h; S is kept across handover (THCP v2:
                 authenticated token, or the network-held value if the token is missing or expired;
                 a forged token contains the device)
  cusum_reset    as above, S = 0 after every handover (cold start), same threshold h
  cusum_reset_cal  reset with its own threshold h_reset, calibrated to the same false-containment
                 budget as cusum_carry (the equal-budget comparator)
  trust_carry    v1 smoothed trust: tau = (1-eps) tau + eps (1 - yhat); quarantine (contain) when the
                 risk (1 - tau) yhat >= q; THCP v1 carry-over tau = delta tau + (1 - delta) tau0; a
                 withheld token gives tau0
  trust_cold     as above, tau = tau0 after every handover
  none           per-flow detection only, no containment
Containment blocks every later outgoing flow of the device for the rest of the run (operator review
is outside the horizon). Metrics per run: for attackers, contained or not, time to containment from
the attack start, attack flows that got through (exposure); for honest devices, false containment and
benign flows blocked.
"""
import numpy as np

METHODS = ("cusum_carry", "cusum_reset", "cusum_reset_cal", "trust_carry", "trust_cold", "none")


def attack_rate(kind, par, t, start, since_ho, window_after_ho=5):
    """rho for one device over time t (array of seconds)."""
    on = t >= start
    if kind == "persistent":
        return np.where(on, par, 0.0)
    if kind == "handover":                                  # high right after a handover (cold-start window)
        hi, lo = par
        return np.where(on, np.where(since_ho < window_after_ho, hi, lo), 0.0)
    if kind == "onoff":                                     # rate par in alternate blocks of 120 s
        return np.where(on & (((t - start) // 120) % 2 == 0), par, 0.0)
    raise ValueError(kind)


def simulate_traffic(links, scores_benign, scores_attack, attackers, flows_per_s, window_s, rng):
    """Per window and device: flows observed, flagged flows, attack flows sent (connected)."""
    serving, handover, outage = links["serving"], links["handover"], links["outage"]
    T, U = serving.shape
    W = T // window_s
    since_ho = np.zeros((T, U))
    last = np.full(U, -10_000)
    for t in range(T):
        last = np.where(handover[t], t, last)
        since_ho[t] = t - last
    seen = np.zeros((W, U))
    flagged = np.zeros((W, U))
    attacks = np.zeros((W, U))
    attack_flagged = np.zeros((W, U))
    t_all = np.arange(T)
    rho = np.zeros((T, U))
    fam = {}
    for u, spec in attackers.items():
        rho[:, u] = attack_rate(spec["kind"], spec["par"], t_all, spec["start"], since_ho[:, u])
        fam[u] = spec["family"]
    conn = ~outage
    n = np.where(conn, flows_per_s, 0)
    n_att = rng.binomial(n, np.clip(rho, 0, 1))
    n_ben = n - n_att
    fb = (scores_benign >= 0.5)
    p_fb = fb.mean()
    for w in range(W):
        sl = slice(w * window_s, (w + 1) * window_s)
        nb, na = n_ben[sl].sum(0), n_att[sl].sum(0)
        # flagged benign flows: exact draws from the benign pool's decisions
        flagged_b = rng.binomial(nb, p_fb)
        flagged_a = np.zeros(U)
        for u, f in fam.items():
            if na[u]:
                pool = scores_attack[f]
                flagged_a[u] = (pool[rng.integers(len(pool), size=int(na[u]))] >= 0.5).sum()
        seen[w], flagged[w], attacks[w], attack_flagged[w] = nb + na, flagged_b + flagged_a, na, flagged_a
    ho_win = handover[: W * window_s].reshape(W, window_s, U).any(1)
    ho_count = handover[: W * window_s].reshape(W, window_s, U).sum(1)
    cell = serving[np.minimum((np.arange(W) + 1) * window_s - 1, T - 1)]
    return {"seen": seen, "flagged": flagged, "attacks": attacks, "attack_flagged": attack_flagged,
            "ho_win": ho_win, "ho_count": ho_count, "cell": cell, "benign_flows": seen - attacks}


def evidence(tr, min_peers=3):
    seen, flagged, cell = tr["seen"], tr["flagged"], tr["cell"]
    yhat = np.where(seen > 0, flagged / np.maximum(seen, 1), 0.0)
    x = np.zeros_like(yhat)
    W = yhat.shape[0]
    for w in range(W):
        ok = seen[w] > 0
        if not ok.any():
            continue
        glob = np.median(yhat[w, ok])
        for c in np.unique(cell[w, ok]):
            m = ok & (cell[w] == c)
            x[w, m] = yhat[w, m] - (np.median(yhat[w, m]) if m.sum() >= min_peers else glob)
    return yhat, x


def contain(method, tr, yhat, x, p, token=None):
    """First window at which each device is contained (-1 if never).
    token: {u: (strategy, start_window)} with strategy 'omit' or 'forge', applied from the device's
    attack start (before it, the device presents its authentic token)."""
    W, U = yhat.shape
    first = np.full(U, -1)
    token = token or {}
    strat = {u: s_ for u, (s_, _) in token.items()}
    start_w = np.array([token[u][1] if u in token else W for u in range(U)])
    if method == "none":
        return first
    if method.startswith("cusum"):
        S = np.zeros(U)
        h = p["h_reset"] if method == "cusum_reset_cal" else p["h"]
        for w in range(W):
            ho = tr["ho_win"][w]
            if method in ("cusum_reset", "cusum_reset_cal"):
                S = np.where(ho, 0.0, S)
            elif method == "cusum_carry":                    # THCP v2: forged token at handover contains
                forged = np.array([strat.get(u) == "forge" for u in range(U)]) & (w >= start_w)
                hit_f = ho & forged & (first < 0)
                first[hit_f] = w
            seen = tr["seen"][w] > 0
            S = np.where(seen, np.maximum(0.0, S + x[w] - p["k"]), S)
            hit = (S >= h) & (first < 0)
            first[hit] = w
        return first
    tau = np.full(U, p["tau0"])
    for w in range(W):
        ho = tr["ho_win"][w]
        if method == "trust_cold":
            tau = np.where(ho, p["tau0"], tau)
        else:                                              # THCP v1: carry, or prior if the token is withheld
            carried = p["delta"] * tau + (1 - p["delta"]) * p["tau0"]
            omit = np.array([strat.get(u) == "omit" for u in range(U)]) & (w >= start_w)
            tau = np.where(ho, np.where(omit, p["tau0"], carried), tau)
        seen = tr["seen"][w] > 0
        tau = np.where(seen, (1 - p["eps"]) * tau + p["eps"] * (1 - yhat[w]), tau)
        risk = (1 - tau) * yhat[w]
        hit = (risk >= p["q"]) & (first < 0) & seen
        first[hit] = w
    return first


def outcomes(first, tr, attackers, window_s, honest):
    W, U = tr["seen"].shape
    res = {"att": [], "hon": []}
    for u, spec in attackers.items():
        sw = spec["start"] // window_s
        f = first[u]
        ok = f >= sw
        upto = f if ok else W
        res["att"].append({"u": u, "vclass": spec["vclass"], "contained": bool(ok),
                           "delay_s": float((f - sw + 1) * window_s) if ok else np.nan,
                           "exposure": float(tr["attacks"][sw:upto, u].sum()),
                           "attack_total": float(tr["attacks"][sw:, u].sum()),
                           "false_before_attack": bool(0 <= f < sw),
                           "ho_per_h": float(tr["ho_count"][sw:, u].sum() / max((W - sw) * window_s, 1) * 3600)})
    for u in honest:
        f = first[u]
        res["hon"].append({"u": u, "contained": bool(f >= 0),
                           "blocked_benign": float(tr["benign_flows"][f:, u].sum()) if f >= 0 else 0.0})
    return res
