"""Radio access for v2: edge nodes, link model, association with hysteresis, link-derived
outage, and the residence predictor (the design notes, Sections 3-4).

Everything is computed per second from the SUMO positions of a trace and does not depend on
training, so it is computed once per (trace seed, setting) and shared by all methods.

- Edge nodes: an n x n grid over the area (default 4 x 4 = 16).
- Path loss (dB): 32.4 + 21 log10(f_GHz) + 31.9 log10(d_m) (urban micro-cell, non-line-of-sight
  form; the optional 3GPP TR 38.901 formula of this form uses 20 log10 f and 8.2 dB), f = 3.5 GHz, d >= 10 m.
- Shadowing: log-normal, sigma 6 dB, per device and edge node, correlated in distance travelled
  (Gudmundson model, decorrelation distance 50 m): s(t) = r s(t-1) + sqrt(1 - r^2) N(0, sigma^2),
  r = exp(-step / 50 m); static devices keep their shadowing.
- Association (downlink power, 30 dBm per edge node): measurements are smoothed by layer-3
  filtering (exponential, weight 0.3 per second), and a device hands over when a neighbour is stronger
  than its serving node by the hysteresis (3 dB) for the time-to-trigger (3 s), as in the cellular
  A3 event.
- Uplink SNR = 23 dBm - path loss - shadowing - noise (-174 dBm/Hz + 10 log10(B) + 9 dB noise
  figure, B = 10 MHz); outage when the SNR is below -10 dB (below the most robust modulation and
  coding scheme); rate = B log2(1 + SNR).
- Residence predictor: from the device's current position and velocity (both observable from its
  position reports), the time until it would come closer to another edge node than to its serving
  node by the equivalent of the hysteresis; P(stay for a round of T seconds) is a logistic function of
  (time to exit - T).
"""
import numpy as np

FREQ_GHZ, SIGMA_DB, DECORR_M = 3.5, 6.0, 50.0
P_BS_DBM, P_UE_DBM, BW_HZ, NF_DB = 30.0, 23.0, 10e6, 9.0
HYST_DB, TTT_S, OUTAGE_SNR_DB, L3_ALPHA = 3.0, 3, -10.0, 0.3


def edge_grid(side, n=4):
    step = side / n
    c = (np.arange(n) + 0.5) * step
    return np.array([(x, y) for y in c for x in c], dtype=np.float64)


def path_loss_db(d):
    return 32.4 + 21 * np.log10(FREQ_GHZ) + 31.9 * np.log10(np.maximum(d, 10.0))


def compute_links(pos, edges, seed, hyst_db=HYST_DB, ttt_s=TTT_S, outage_db=OUTAGE_SNR_DB, l3=L3_ALPHA):
    """pos: T x U x 2. Returns serving (T x U), handover (T x U bool), outage (T x U bool),
    snr_db (T x U, uplink to the serving node), speed (T x U, m/s), vel (T x U x 2)."""
    T, U, _ = pos.shape
    E = len(edges)
    rng = np.random.default_rng([seed, 7])
    vel = np.zeros_like(pos, dtype=np.float64)
    vel[1:] = pos[1:] - pos[:-1]
    speed = np.linalg.norm(vel, axis=2)
    shadow = rng.normal(0, SIGMA_DB, size=(U, E))
    serving = np.zeros((T, U), dtype=np.int16)
    handover = np.zeros((T, U), dtype=bool)
    snr = np.zeros((T, U), dtype=np.float32)
    noise = -174 + 10 * np.log10(BW_HZ) + NF_DB
    better = np.zeros((U, E), dtype=np.int16)            # seconds a neighbour has been better by hyst
    for t in range(T):
        if t > 0:
            r = np.exp(-speed[t] / DECORR_M)[:, None]
            shadow = r * shadow + np.sqrt(1 - r ** 2) * rng.normal(0, SIGMA_DB, size=(U, E))
        d = np.linalg.norm(pos[t][:, None, :] - edges[None, :, :], axis=2)
        loss = path_loss_db(d) + shadow                    # U x E
        meas = P_BS_DBM - loss
        rsrp = meas if t == 0 else (1 - l3) * rsrp + l3 * meas      # layer-3 filtered measurement
        if t == 0:
            cur = rsrp.argmax(1)
        else:
            cur = serving[t - 1].astype(int)
            margin = rsrp - rsrp[np.arange(U), cur][:, None]
            better = np.where(margin > hyst_db, better + 1, 0)
            better[np.arange(U), cur] = 0
            cand = np.where(better >= ttt_s, rsrp, -np.inf).argmax(1)
            switch = np.isfinite(np.where(better >= ttt_s, rsrp, -np.inf).max(1))
            handover[t] = switch
            cur = np.where(switch, cand, cur)
            better[switch] = 0
        serving[t] = cur
        snr[t] = P_UE_DBM - loss[np.arange(U), cur] - noise
    outage = snr < outage_db
    return {"serving": serving, "handover": handover, "outage": outage, "snr_db": snr,
            "speed": speed.astype(np.float32), "vel": vel.astype(np.float32)}


def time_to_exit(pos_t, vel_t, serving_t, edges, hyst_db=HYST_DB):
    """Seconds until each device, moving at its current velocity, would be closer to another edge
    node than to its serving one by the distance equivalent of the hysteresis (inf if never)."""
    U = len(pos_t)
    e = edges[serving_t]                                   # U x 2
    tte = np.full(U, np.inf)
    # distance margin equivalent to the hysteresis at the cell edge (31.9 log10 slope)
    k = 10 ** (hyst_db / 31.9)
    for j in range(len(edges)):
        o = edges[j]
        mask = serving_t != j
        # leaving towards o: d_e >= k d_o, i.e. f(t) = |p+vt-e|^2 - k^2 |p+vt-o|^2 >= 0,
        # a quadratic a t^2 + b t + c in t
        p, v = pos_t[mask], vel_t[mask]
        a = (1 - k ** 2) * (v ** 2).sum(1)
        pe, po = p - e[mask], p - o
        b = 2 * ((pe * v).sum(1) - k ** 2 * (po * v).sum(1))
        c = (pe ** 2).sum(1) - k ** 2 * (po ** 2).sum(1)
        t_hit = np.full(len(p), np.inf)
        disc = b ** 2 - 4 * a * c
        lin = np.abs(a) < 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            t_lin = np.where(lin & (b > 0), -c / b, np.inf)
            sq = np.sqrt(np.maximum(disc, 0))
            r1, r2 = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
        for r_ in (r1, r2):
            ok = ~lin & (disc >= 0) & (r_ > 0)
            t_hit = np.where(ok, np.minimum(t_hit, r_), t_hit)
        t_hit = np.where(lin, np.where(t_lin > 0, t_lin, np.inf), t_hit)
        t_hit = np.where(c >= 0, 0.0, t_hit)                  # already beyond the boundary
        cur = tte[mask]
        tte[mask] = np.minimum(cur, t_hit)
    return tte


def p_stay(tte, round_s, scale_s=10.0):
    """Predicted probability of staying in the cell for a round of round_s seconds."""
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-(np.minimum(tte, 1e6) - round_s) / scale_s))
