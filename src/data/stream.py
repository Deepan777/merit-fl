"""Per-device data streams on top of the mobility simulator (Part 10).

ClassMixer   class mixture of device u in cell c:
             global prior x per-cell tilt x optional Dirichlet(alpha) device skew.
TrainBuffers each device keeps a FIFO of its most recent `window` labelled
             training flows; every slot it observes new flows from its current
             cell, so a handover shifts phi_u gradually and leaves part of the
             buffer "stale" (collected in the previous cell).
FlowStream   the traffic each device emits per slot, drawn from the held-out
             stream pool. Flows within K slots of a handover are in-window;
             flows in the first `gap_slots` slots lose a fraction of their
             features (flow split across edge nodes); adversarial devices
             inject attack flows (A1). The deployed model scores these flows.
"""
from dataclasses import dataclass, fields

import numpy as np
from sklearn.model_selection import train_test_split

from .partition import attack_signature, sample_window


def class_pools(y, n_classes):
    return [np.flatnonzero(y == k) for k in range(n_classes)]


def stratified_halves(X, y, seed):
    """Split a held-out split into (stream pool, holdout pool)."""
    idx = np.arange(len(y))
    try:
        a, b = train_test_split(idx, test_size=0.5, stratify=y, random_state=seed)
    except ValueError:                       # a class with a single member
        a, b = train_test_split(idx, test_size=0.5, random_state=seed)
    return (X[a], y[a]), (X[b], y[b])


class ClassMixer:
    def __init__(self, n_cells, n_devices, prior, data_cfg, rng):
        K = len(prior)
        if data_cfg["cell_profile"] == "dirichlet":      # v0.2: prior-free profiles
            self.cell = rng.dirichlet([data_cfg["cell_concentration"]] * K, size=n_cells)
        else:                                            # global prior, tilted per cell
            z = rng.normal(size=(n_cells, K))
            self.cell = np.asarray(prior) * np.exp(data_cfg["cell_spread"] * z)
        alpha = data_cfg["dirichlet_alpha"]
        # Dirichlet(alpha) label skew: share of class k held by device u (mean 1)
        self.dev = (n_devices * rng.dirichlet([alpha] * n_devices, size=K).T
                    if alpha else np.ones((n_devices, K)))

    def mix(self, u, cell):
        p = self.cell[cell] * self.dev[u] + 1e-12
        return p / p.sum()


class TrainBuffers:
    def __init__(self, pools, mixer, cells, window, per_slot, rng):
        self.pools, self.mixer, self.rng = pools, mixer, rng
        self.window, self.per_slot = window, min(per_slot, window)
        self.idx = np.stack([sample_window(pools, mixer.mix(u, c), window, rng)
                             for u, c in enumerate(cells)])
        self.cell = np.repeat(np.asarray(cells)[:, None], window, axis=1)
        self.ptr = np.zeros(len(cells), dtype=int)

    def observe_slot(self, cells):
        for u, c in enumerate(cells):
            new = sample_window(self.pools, self.mixer.mix(u, c), self.per_slot, self.rng)
            pos = (self.ptr[u] + np.arange(len(new))) % self.window
            self.idx[u, pos] = new
            self.cell[u, pos] = c
            self.ptr[u] = (self.ptr[u] + len(new)) % self.window

    def signatures(self, y, n_classes):
        """phi_u over each buffer (client-side, from the client's own labels)."""
        return np.stack([attack_signature(y[row], n_classes) for row in self.idx])

    def staleness(self, cells):
        """Fraction of each buffer collected in a cell other than the current
        one (label-free: uses cell IDs only)."""
        return (self.cell != np.asarray(cells)[:, None]).mean(1)


@dataclass
class FlowBatch:
    X: np.ndarray            # observed features (gaps applied)
    y: np.ndarray            # ground truth: metrics only, never trust/risk
    dev: np.ndarray
    rnd: np.ndarray
    slot: np.ndarray
    in_window: np.ndarray
    gap: np.ndarray
    injected: np.ndarray
    tau: np.ndarray          # device trust when the flow was observed (nan: none)
    quarantined: np.ndarray

    @staticmethod
    def concat(batches):
        return FlowBatch(**{f.name: np.concatenate([getattr(b, f.name) for b in batches])
                            for f in fields(FlowBatch)})


class FlowStream:
    def __init__(self, X, y, n_classes, mixer, attack_classes, flows_per_slot,
                 ho_cfg, adv_cfg, rng):
        self.X, self.y, self.mixer, self.rng = X, y, mixer, rng
        self.pools = class_pools(y, n_classes)
        self.attack_classes = [k for k in attack_classes if len(self.pools[k]) > 0]
        self.F = flows_per_slot
        self.K = ho_cfg["window_slots"]
        self.gap_slots = ho_cfg["gap_slots"]
        self.gap_frac = ho_cfg["gap_feature_frac"]
        self.rates = {"handover_timed": (adv_cfg["inject_rate_window"], adv_cfg["inject_rate_base"]),
                      "uniform": (adv_cfg["inject_rate_uniform"], adv_cfg["inject_rate_uniform"])}

    def emit(self, cells, since_ho, inject_mode, rnd, slot):
        """One slot of traffic for every device. since_ho[u]: slots since the
        device's last handover; inject_mode[u]: None or an injection mode."""
        F, out = self.F, []
        for u, c in enumerate(cells):
            idx = sample_window(self.pools, self.mixer.mix(u, c), F, self.rng)
            inw, gap = since_ho[u] < self.K, since_ho[u] < self.gap_slots
            inj = np.zeros(F, dtype=bool)
            if inject_mode[u] and self.attack_classes:
                inj = self.rng.random(F) < self.rates[inject_mode[u]][0 if inw else 1]
                cls = self.rng.choice(self.attack_classes, size=int(inj.sum()))
                idx[inj] = [self.pools[k][self.rng.integers(len(self.pools[k]))] for k in cls]
            X = self.X[idx].copy()
            if gap and self.gap_frac > 0:
                X[self.rng.random(X.shape) < self.gap_frac] = 0.0   # mean-imputed
            out.append((X, self.y[idx], u, inw, gap, inj))
        n = len(out) * F
        return FlowBatch(
            X=np.concatenate([o[0] for o in out]), y=np.concatenate([o[1] for o in out]),
            dev=np.repeat([o[2] for o in out], F), rnd=np.full(n, rnd), slot=np.full(n, slot),
            in_window=np.repeat([o[3] for o in out], F), gap=np.repeat([o[4] for o in out], F),
            injected=np.concatenate([o[5] for o in out]),
            tau=np.full(n, np.nan), quarantined=np.zeros(n, dtype=bool))
