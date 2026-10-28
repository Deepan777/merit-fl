"""Client partitioning for MERIT-FL.

Two regimes (Sec. 3 of the proposal):
  1) static Dirichlet label-skew non-IID (the baseline regime in prior work);
  2) trajectory-conditioned NON-STATIONARY non-IID: each client's class
     mixture is tied to the CELL it currently occupies, so its local data
     distribution drifts as it moves (the regime no prior FL-IDS models).
"""
import numpy as np


def dirichlet_partition(y, n_clients, alpha=0.5, seed=0):
    """Standard Dirichlet(alpha) label-skew split. Returns list[np.ndarray]."""
    rng = np.random.default_rng(seed)
    n_classes = int(y.max()) + 1
    idx_by_class = [np.where(y == k)[0] for k in range(n_classes)]
    for idx in idx_by_class:
        rng.shuffle(idx)
    client_idx = [[] for _ in range(n_clients)]
    for k, idx in enumerate(idx_by_class):
        props = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for c, part in enumerate(np.split(idx, cuts)):
            client_idx[c].extend(part.tolist())
    return [np.array(sorted(ci)) for ci in client_idx]


def cell_class_profiles(n_cells, n_classes, concentration=0.3, seed=0):
    """Each cell gets its own class-mixture profile (Dirichlet on the simplex).
    A device drawing data from its current cell realises trajectory-conditioned
    drift: change cell => change distribution."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet([concentration] * n_classes, size=n_cells)


def sample_window(y_pool_by_class, cell_profile, window, rng):
    """Draw exactly `window` sample indices, classes ~ Multinomial(profile),
    members uniform with replacement within each class pool, returned in
    random order (v0.2 truncated rare classes and returned class-sorted)."""
    p = np.asarray(cell_profile, dtype=np.float64).copy()
    p[[len(pool) == 0 for pool in y_pool_by_class]] = 0.0   # classes absent from the pool
    counts = rng.multinomial(window, p / p.sum())
    out = [y_pool_by_class[k][rng.integers(len(y_pool_by_class[k]), size=c)]
           for k, c in enumerate(counts) if c > 0]
    idx = np.concatenate(out).astype(np.int64) if out else np.zeros(0, np.int64)
    rng.shuffle(idx)
    return idx


def rare_classes(prior, frac, benign):
    """Attack classes whose training prior is below `frac` (rarest first);
    falls back to the single least-frequent attack class."""
    attacks = [k for k in np.argsort(prior) if k != benign]
    rare = [int(k) for k in attacks if prior[k] < frac]
    return rare or [int(attacks[0])]


def attack_signature(y_local, n_classes):
    """phi_u: empirical class histogram on the simplex (Sec. 3.1)."""
    h = np.bincount(y_local, minlength=n_classes).astype(np.float64)
    s = h.sum()
    return h / s if s > 0 else np.full(n_classes, 1.0 / n_classes)


def signature_drift(phi_now, phi_prev):
    """d_u(t) = L1 distance between successive signatures (DHRM input)."""
    return float(np.abs(phi_now - phi_prev).sum())


def kl_usefulness(phi_u, phi_mean, eps=1e-9):
    """mu_u = KL(phi_u || mean of selected set) — marginal value of a client's
    local attack mixture to the global mixture (MR-Select input)."""
    p = phi_u + eps
    q = phi_mean + eps
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))
