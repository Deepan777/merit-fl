"""Robustness simulation of the containment experiment (post hoc); the pre-registered experiment is src/v2/containment.py and stays unchanged.

Differences from the pre-registered experiment:
- multi-hour operation: the 1-h link trace is chained K times with fresh traffic, so honest false
  containment can be measured as a steady-state hazard (per honest device-hour after a burn-in);
- heterogeneous honest devices (benign types with their own detector flag rates, per-device flow rates);
- edge nodes that may serve a block of cells (evidence moves only on edge-node changes);
- three state placements (carry, per-node memory, reset) x three statistics (share-CUSUM against the
  edge node's median, binomial log-likelihood-ratio CUSUM against the device type's median, EWMA);
- every rule calibrated with the same objective and budget.
Statistic paths do not depend on the threshold, so a threshold sweep is a search over running maxima.
"""
import numpy as np

PLACEMENTS = ("carry", "memory", "reset")
STATS = {"share": (0.005, 0.01, 0.02, 0.04), "llr": (0.02, 0.05, 0.1), "ewma": (0.02, 0.05, 0.1)}


def chain_links(links, K):
    """Repeat a link trace K times (serving, outage); edge changes are recomputed by edge_view."""
    return {k: np.concatenate([links[k]] * K, axis=0) for k in ("serving", "outage")}


def node_of_cell(n_side, block):
    """Edge node of each cell when one edge node serves a block x block square of cells
    (cells numbered row-major, as src/v2/network.edge_grid)."""
    ids = np.arange(n_side * n_side)
    iy, ix = ids // n_side, ids % n_side
    return (iy // block) * (n_side // block) + (ix // block)


def edge_view(links, n_side, block):
    """Edge node per second and edge-change indicator per second."""
    node = node_of_cell(n_side, block)[links["serving"].astype(int)]
    ho = np.zeros(node.shape, dtype=bool)
    ho[1:] = node[1:] != node[:-1]
    return node, ho


def since_change(ho):
    T = ho.shape[0]
    t = np.arange(T)[:, None]
    last = np.maximum.accumulate(np.where(ho, t, -10 ** 6), axis=0)
    return t - last


def attack_rate(kind, par, T, start, since, window_after_ho=5):
    t = np.arange(T)
    on = t >= start
    if kind == "persistent":
        return np.where(on, par, 0.0)
    if kind == "handover":
        hi, lo = par
        return np.where(on, np.where(since < window_after_ho, hi, lo), 0.0)
    if kind == "onoff":
        return np.where(on & (((t - start) // 120) % 2 == 0), par, 0.0)
    raise ValueError(kind)


def simulate(node, ho, outage, lam, p_benign, attackers, window_s, rng, flag_model=None):
    """Per window and device: flows seen, flagged, attack flows, flagged attack flows; node at the end
    of the window and whether the edge node changed within it.
    lam: flows/s per device; p_benign: benign flag probability per device; attackers: {u: dict(kind,
    par, start, p_det)}; flag_model: None (independent flags, R6/R7) or an R8 benign flag model (benign_rates)."""
    T, U = node.shape
    W = T // window_s
    T = W * window_s
    n = rng.poisson(np.where(outage[:T], 0.0, lam[None, :]))
    rho = np.zeros((T, U))
    if attackers:
        since = since_change(ho[:T])
        for u, a in attackers.items():
            rho[:, u] = attack_rate(a["kind"], a["par"], T, a["start"], since[:, u])
    na = rng.binomial(n, np.clip(rho, 0, 1))
    nb = n - na
    nb_w = nb.reshape(W, window_s, U).sum(1)
    na_w = na.reshape(W, window_s, U).sum(1)
    p_det = np.zeros(U)
    for u, a in attackers.items():
        p_det[u] = a["p_det"]
    fb = rng.binomial(nb_w, benign_rates(p_benign, W, flag_model, rng))
    fa = rng.binomial(na_w, p_det[None, :])
    return {"seen": nb_w + na_w, "flagged": fb + fa, "attacks": na_w, "attack_flagged": fa,
            "benign": nb_w, "node": node[window_s - 1: T: window_s], "ho": ho[:T].reshape(W, window_s, U).any(1)}


def benign_rates(p_benign, W, flag_model, rng):
    """Benign flag probability per window and device (R8). None: the device's rate in every window (independent
    flags). {"kind": "betabin", "rho": r}: a Beta draw per window with the device's mean and intra-window
    correlation r. {"kind": "burst", "m": m, "p_in": a, "p_stay": b}: a two-state Markov chain per device; in a
    burst the rate is multiplied by m (clipped to 1)."""
    if flag_model is None:
        return p_benign[None, :]
    U = len(p_benign)
    p = np.broadcast_to(p_benign[None, :], (W, U))
    if flag_model["kind"] == "betabin":
        c = 1.0 / flag_model["rho"] - 1.0
        inner = (p > 0) & (p < 1)
        q = rng.beta(np.where(inner, p * c, 1.0), np.where(inner, (1 - p) * c, 1.0))
        return np.where(inner, q, p)
    if flag_model["kind"] == "burst":
        draw = rng.random((W, U))
        state = np.zeros(U, dtype=bool)
        out = np.empty((W, U))
        hi = np.minimum(1.0, p_benign * flag_model["m"])
        for w in range(W):
            state = np.where(state, draw[w] < flag_model["p_stay"], draw[w] < flag_model["p_in"])
            out[w] = np.where(state, hi, p_benign)
        return out
    raise ValueError(flag_model["kind"])


def _median_by(groups, values, ok, min_n=3):
    """Median of values over ok devices with the same group label (per row); rows are windows. Groups
    with fewer than min_n members fall back to the row median."""
    W, U = values.shape
    out = np.zeros((W, U))
    for w in range(W):
        m_ok = ok[w]
        if not m_ok.any():
            continue
        glob = np.median(values[w, m_ok])
        g = groups[w]
        for c in np.unique(g[m_ok]):
            m = m_ok & (g == c)
            out[w, g == c] = np.median(values[w, m]) if m.sum() >= min_n else glob
    return out


def evidence(tr, dev_type):
    """Share excess over the edge node's median (pre-registered Eq. 2), and the type reference p0 (median
    flagged share of devices of the same type, network-wide, per window)."""
    seen, flagged = tr["seen"], tr["flagged"]
    ok = seen > 0
    yhat = np.where(ok, flagged / np.maximum(seen, 1), 0.0)
    x_share = yhat - _median_by(tr["node"], yhat, ok)
    types = np.broadcast_to(dev_type[None, :], yhat.shape)
    p0 = _median_by(types, yhat, ok)
    return yhat, ok, x_share, p0


def hourly_type_reference(tr, dev_type, horizon=120, min_flows=30, warm=10):
    """R7 reference: each device's flagged share over the previous `horizon` windows; p0 of a device is the
    median of these shares over the devices of its type (network-wide) with at least `min_flows` flows in
    that period. Returns p0 (W x U) and a validity mask (False in the first `warm` windows or when a type has
    no eligible device)."""
    seen, flagged = tr["seen"].astype(float), tr["flagged"].astype(float)
    W, U = seen.shape
    cs = np.vstack([np.zeros((1, U)), np.cumsum(seen, 0)])
    cf = np.vstack([np.zeros((1, U)), np.cumsum(flagged, 0)])
    p0 = np.zeros((W, U))
    valid = np.zeros((W, U), dtype=bool)
    types = np.unique(dev_type)
    for w in range(warm, W):
        lo = max(0, w - horizon)
        n = cs[w] - cs[lo]
        f = cf[w] - cf[lo]
        share = np.where(n > 0, f / np.maximum(n, 1), np.nan)
        for t in types:
            m = (dev_type == t)
            elig = m & (n >= min_flows)
            if elig.any():
                p0[w, m] = np.median(share[elig])
                valid[w, m] = True
    return p0, valid


def device_reference(tr, p_type, valid, lag=120, hist=360, m=200.0, warm=10, z=2.0, ceiling=0.5):
    """R8 reference `llrd` (plan amendment A1): a device's own flagged share in windows [w - lag - hist, w - lag),
    shrunk towards the type reference at w - lag with m pseudo-flows, plus z standard errors (a conservative
    reference). The lag keeps an attacker's own recent traffic out of its reference. The statistic does not move
    during the learning period (w - lag < warm) or while the reference is at least `ceiling` (a device whose benign
    flows are mostly flagged cannot be monitored by its flags)."""
    seen, flagged = tr["seen"].astype(float), tr["flagged"].astype(float)
    W, U = seen.shape
    cs = np.vstack([np.zeros((1, U)), np.cumsum(seen, 0)])
    cf = np.vstack([np.zeros((1, U)), np.cumsum(flagged, 0)])
    p0 = np.array(p_type, dtype=float, copy=True)
    ok = np.zeros((W, U), dtype=bool)
    for w in range(max(0, lag + warm), W):
        e = w - lag
        lo = max(0, e - hist)
        n, f = cs[e] - cs[lo], cf[e] - cf[lo]
        p = (f + m * p_type[e]) / (n + m)
        p0[w] = p + z * np.sqrt(p * (1 - p) / (n + m))
        ok[w] = valid[e] & (p0[w] < ceiling)
    return p0, ok


def hourly_pooled_reference(tr, horizon=120, warm=10):
    """Network-wide flagged share of all devices over the previous `horizon` windows (TRW-style comparator,
    R8): one reference for every device, no device types."""
    seen, flagged = tr["seen"].sum(1).astype(float), tr["flagged"].sum(1).astype(float)
    W, U = tr["seen"].shape
    cs, cf = np.concatenate([[0.0], np.cumsum(seen)]), np.concatenate([[0.0], np.cumsum(flagged)])
    w = np.arange(W)
    lo = np.maximum(0, w - horizon)
    n, f = cs[w] - cs[lo], cf[w] - cf[lo]
    p = np.where(n > 0, f / np.maximum(n, 1), 0.0)
    valid = (w >= warm) & (n > 0)
    return np.repeat(p[:, None], U, 1), np.repeat(valid[:, None], U, 1)


def llr_increment(tr, p0, valid, delta):
    """Binomial log-likelihood ratio of a device's window counts, p1 = p0 + delta against p0; zero where the
    reference is not yet valid."""
    q0 = np.clip(p0, 1e-4, 1 - delta - 1e-4)
    q1 = q0 + delta
    f, n = tr["flagged"], tr["seen"]
    inc = f * np.log(q1 / q0) + (n - f) * np.log((1 - q1) / (1 - q0))
    return np.where(valid, inc, 0.0)


def dispersion(tr, lag=120, hist=360, warm=10, min_k=10, rho_max=0.5):
    """R9: intra-window correlation rho of a device's flags, the beta-binomial moment estimator over its windows with
    at least two flows in [w - lag - hist, w - lag) (the history of device_reference). With k such windows, N flows and
    p = sum f / N: S = sum (f - n p)^2 / n, rho = (S / (p (1 - p)) - (k - 1)) / (N - k - sum n^2 / N + 1), clipped to
    [0, rho_max]. Zero during the learning period, with fewer than min_k windows or p in {0, 1}. Returns rho and the
    window count k (W x U)."""
    seen, flagged = tr["seen"].astype(float), tr["flagged"].astype(float)
    W, U = seen.shape
    m = seen >= 2
    n = np.where(m, seen, 0.0)
    f = np.where(m, flagged, 0.0)

    def cum(x):
        return np.vstack([np.zeros((1, U)), np.cumsum(x, 0)])
    ck, cn, cf, cn2, cf2 = cum(m.astype(float)), cum(n), cum(f), cum(n * n), cum(np.where(m, f * f / np.maximum(n, 1), 0.0))
    rho = np.zeros((W, U))
    kk = np.zeros((W, U))
    w = np.arange(max(0, lag + warm), W)
    if len(w) == 0:
        return rho, kk
    e = w - lag
    lo = np.maximum(0, e - hist)
    k, N, F = ck[e] - ck[lo], cn[e] - cn[lo], cf[e] - cf[lo]
    N2, F2 = cn2[e] - cn2[lo], cf2[e] - cf2[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        p = F / N
        S = F2 - F * F / N
        den = N - k - N2 / N + 1
        r = (S / (p * (1 - p)) - (k - 1)) / den
    ok = (k >= min_k) & (p > 0) & (p < 1) & (den > 0) & np.isfinite(r)
    rho[w] = np.where(ok, np.clip(r, 0.0, rho_max), 0.0)
    kk[w] = k
    return rho, kk


def type_dispersion(rho, k, dev_type, min_k=10):
    """R9: per window, the median of rho over the devices of each type with at least min_k windows (0 if none)."""
    out = np.zeros_like(rho)
    for t in np.unique(dev_type):
        cols = np.flatnonzero(dev_type == t)
        r = np.where(k[:, cols] >= min_k, rho[:, cols], np.nan)
        has = np.isfinite(r).any(1)
        med = np.zeros(rho.shape[0])
        if has.any():
            med[has] = np.nanmedian(r[has], axis=1)
        out[:, cols] = med[:, None]
    return out


def robust_increment(tr, p0, valid, delta, rho, kappa):
    """R9 `llrr`: the binomial LLR of llr_increment divided by the design effect D = 1 + (n - 1) rho and capped at
    kappa from above. Both preserve E[exp(x)] <= 1 under the null whenever it holds, so h = ln ARL0 is kept."""
    inc = llr_increment(tr, p0, valid, delta)
    D = 1.0 + np.maximum(tr["seen"] - 1, 0) * rho
    return np.minimum(inc / D, kappa)


def simulate_replay(node, ho, outage, seq_n, seq_f, attackers, window_s, rng, att_share=0.05):
    """R9: replayed benign flows and flags per window and device (seq_n, seq_f: W x U), thinned by the share of the
    window the device is in outage (flows binomially, flags hypergeometrically). Attackers add attack flows at rate
    att_share / (1 - att_share) per benign flow from their start (Poisson per window), flagged with p_det. Same output
    as simulate."""
    T, U = node.shape
    W = min(T // window_s, seq_n.shape[0])
    T = W * window_s
    up = 1.0 - outage[:T].reshape(W, window_s, U).mean(1)
    sn, sf = seq_n[:W].astype(np.int64), seq_f[:W].astype(np.int64)
    nb = rng.binomial(sn, up)
    fb = np.zeros((W, U), dtype=np.int64)
    pos = nb > 0
    fb[pos] = rng.hypergeometric(sf[pos], sn[pos] - sf[pos], nb[pos])
    na = np.zeros((W, U), dtype=np.int64)
    fa = np.zeros((W, U), dtype=np.int64)
    w_idx = np.arange(W)
    for u, a in attackers.items():
        on = w_idx >= a["start"] // window_s
        lam = np.where(on, nb[:, u] * att_share / (1 - att_share), 0.0)
        na[:, u] = rng.poisson(lam)
        fa[:, u] = rng.binomial(na[:, u], a["p_det"])
    return {"seen": nb + na, "flagged": fb + fa, "attacks": na, "attack_flagged": fa, "benign": nb,
            "node": node[window_s - 1: T: window_s], "ho": ho[:T].reshape(W, window_s, U).any(1)}


def llr_floor(p0, p_det, delta):
    """Attack-flow share rho* below which the LLR increment of an attacker has non-positive mean: its flag rate
    (1 - rho) p0 + rho p_det equals p*, the rate at which E[LLR per flow] = 0."""
    q0 = min(max(p0, 1e-4), 1 - delta - 1e-4)
    q1 = q0 + delta
    a, b = np.log(q1 / q0), np.log((1 - q0) / (1 - q1))
    p_star = b / (a + b)
    return (p_star - q0) / max(p_det - q0, 1e-9)


def increments(stat, par, tr, yhat, ok, x_share, p0):
    if stat == "share":
        return x_share - par
    if stat == "llr":
        q0 = np.clip(p0, 1e-3, 1 - par - 1e-3)
        q1 = q0 + par
        f, n = tr["flagged"], tr["seen"]
        return f * np.log(q1 / q0) + (n - f) * np.log((1 - q1) / (1 - q0))
    if stat == "ewma":
        return yhat - p0
    raise ValueError(stat)


def path(stat, par, inc, ok, node_w, ho_w, placement, n_nodes):
    """Statistic of every device over windows under a state placement. CUSUM: S = max(0, S + inc);
    EWMA: Z = (1 - lam) Z + lam inc. Windows without flows leave the statistic unchanged."""
    W, U = inc.shape
    S = np.zeros(U)
    out = np.empty((W, U))
    mem = np.zeros((n_nodes, U)) if placement == "memory" else None
    prev = node_w[0].astype(int).copy()
    cols = np.arange(U)
    for w in range(W):
        cur = node_w[w].astype(int)
        if placement == "reset":
            S = np.where(ho_w[w], 0.0, S)
        elif placement == "memory":
            ch = cur != prev
            if ch.any():
                u = cols[ch]
                mem[prev[u], u] = S[u]
                S = S.copy()
                S[u] = mem[cur[u], u]
        if stat == "ewma":
            S = np.where(ok[w], (1 - par) * S + par * inc[w], S)
        else:
            S = np.where(ok[w], np.maximum(0.0, S + inc[w]), S)
        out[w] = S
        prev = cur
    return out


def run_alarms(stat, par, inc, ok, node_w, ho_w, placement, n_nodes, hs, w_burn, att_sw, loss=0.0, rng=None):
    """Operate the statistic with a vector of thresholds hs at once. An alarm (statistic >= h) is a
    containment; the operator reviews the device and the statistic restarts at 0 (an honest device is
    released), so honest alarms after the burn-in measure the steady-state false-alarm rate (1 / ARL0).
    att_sw: {attacker: first window of its attack}. Returns honest-or-not alarm counts after the burn-in
    (len(hs) x U) and, per attacker column, the first alarm at or after its attack start (W if none)."""
    W, U = inc.shape
    hs = np.asarray(hs, dtype=float)[:, None]
    H = hs.shape[0]
    S = np.zeros((H, U))
    mem = np.zeros((n_nodes, H, U)) if placement == "memory" else None
    prev = node_w[0].astype(int)
    alarms = np.zeros((H, U), dtype=np.int64)
    first_att = np.full((H, U), W)
    sw = np.full(U, W)
    for u, s in att_sw.items():
        sw[u] = s
    for w in range(W):
        cur = node_w[w].astype(int)
        if placement == "reset":
            S[:, ho_w[w]] = 0.0
        elif placement == "carry_loss":                      # context lost, token withheld, no late merge
            lost = ho_w[w] & (rng.random(U) < loss)
            S[:, lost] = 0.0
        elif placement == "memory":
            ch = np.flatnonzero(cur != prev)
            if len(ch):
                mem[prev[ch], :, ch] = S[:, ch].T
                S[:, ch] = mem[cur[ch], :, ch].T
        upd = ok[w]
        if stat == "ewma":
            S[:, upd] = (1 - par) * S[:, upd] + par * inc[w, upd]
        elif stat == "sprt":                                  # SPRT: restart at 0 below the lower barrier par
            x = S[:, upd] + inc[w, upd]
            S[:, upd] = np.where(x <= par, 0.0, x)
        else:
            S[:, upd] = np.maximum(0.0, S[:, upd] + inc[w, upd])
        hit = S >= hs
        if hit.any():
            if w >= w_burn:
                alarms += hit
            new = hit & (w >= sw)[None, :] & (first_att == W)
            first_att[new] = w
            S[hit] = 0.0
            if placement == "memory":
                hi, ui = np.nonzero(hit)
                mem[:, hi, ui] = 0.0
        prev = cur
    return alarms, first_att


def run_alarms_beta(flagged, seen, p_ref, ok, lam, node_w, ho_w, placement, hs, w_burn, att_sw):
    """Beta reputation with forgetting (Josang and Ismail, 2002) and base rate p_ref: r <- lam r + flags,
    N <- lam N + flows, score (r + 2 p_ref) / (N + 2) - p_ref; an alarm when the score reaches h. Placements carry
    and reset; counting as in run_alarms."""
    W, U = flagged.shape
    hs = np.asarray(hs, dtype=float)[:, None]
    H = hs.shape[0]
    r, N = np.zeros((H, U)), np.zeros((H, U))
    alarms = np.zeros((H, U), dtype=np.int64)
    first_att = np.full((H, U), W)
    sw = np.full(U, W)
    for u, s in att_sw.items():
        sw[u] = s
    for w in range(W):
        if placement == "reset":
            r[:, ho_w[w]] = 0.0
            N[:, ho_w[w]] = 0.0
        upd = ok[w]
        r[:, upd] = lam * r[:, upd] + flagged[w, upd]
        N[:, upd] = lam * N[:, upd] + seen[w, upd]
        a = p_ref[w][None, :]
        score = np.where(upd[None, :], (r + 2 * a) / (N + 2) - a, -np.inf)
        hit = score >= hs
        if hit.any():
            if w >= w_burn:
                alarms += hit
            new = hit & (w >= sw)[None, :] & (first_att == W)
            first_att[new] = w
            r[hit] = 0.0
            N[hit] = 0.0
    return alarms, first_att


def first_crossing(running_max, h):
    """First window at which the running maximum reaches h (W if never), per device."""
    W = running_max.shape[0]
    hit = running_max >= h
    return np.where(hit.any(0), hit.argmax(0), W)


def first_crossings(running_max, hs):
    """first_crossing for an array of thresholds: (len(hs), U)."""
    W, U = running_max.shape
    out = np.empty((len(hs), U), dtype=int)
    for u in range(U):
        out[:, u] = np.searchsorted(running_max[:, u], hs, side="left")
    return np.minimum(out, W)


def honest_hazard(first, honest, w_burn, W, window_s):
    """Events and device-hours at risk after the burn-in: devices first contained before w_burn are
    excluded; first: (..., U) first-crossing windows."""
    f = first[..., honest]
    at_risk = f >= w_burn
    events = (at_risk & (f < W)).sum(-1)
    hours = (np.where(at_risk, np.minimum(f, W) - w_burn, 0)).sum(-1) * window_s / 3600
    return events, hours


def longest_residence(node_w, u, w0, w1):
    """Longest run of consecutive windows at one edge node for device u in [w0, w1)."""
    seg = node_w[w0:w1, u]
    if len(seg) == 0:
        return 0
    best = cur = 1
    for i in range(1, len(seg)):
        cur = cur + 1 if seg[i] == seg[i - 1] else 1
        best = max(best, cur)
    return best
