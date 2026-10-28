"""Data and traffic for v2 (the design notes, Section 1).

- Splits: the training split feeds the devices' training buffers; the evaluation split (validation
  for calibration, test for final runs) is cut into a stream pool (traffic) and a holdout pool
  (accuracy), stratified, as in v1.
- Training buffers: every device keeps a first-in-first-out buffer of the labelled flows it observes;
  each round it adds flows drawn by its current cell's class mix (benign traffic and the external
  attacks seen in that cell; cell-dependent, as in v1), so local data is non-IID and drifts with
  mobility.
- Outgoing traffic (containment experiments, src/v2/containment.py): honest devices send benign
  flows; a compromised device mixes in flows of its attack family. External attacks are not
  attributed to any device and are never evidence against one.
"""
import numpy as np

from ..data.loaders import find_benign, get_dataset
from ..data.stream import ClassMixer, class_pools, stratified_halves


class V2Data:
    def __init__(self, cfg, seed):
        d = get_dataset(cfg)
        self.classes = d["classes"]
        self.K = len(self.classes)
        self.X_tr, self.y_tr = d["train"]
        Xe, ye = d[cfg["eval"]["split"]]
        (self.X_stream, self.y_stream), (self.X_hold, self.y_hold) = stratified_halves(Xe, ye, seed)
        counts = np.bincount(self.y_tr, minlength=self.K)
        self.benign = find_benign(self.classes, counts)
        self.attack_classes = [k for k in range(self.K) if k != self.benign]
        self.prior = counts / counts.sum()
        self.pools_tr = class_pools(self.y_tr, self.K)
        self.pools_stream = class_pools(self.y_stream, self.K)
        self.n_features = self.X_tr.shape[1]


class Buffers:
    """FIFO training buffers of indices into the training split."""

    def __init__(self, data, n_devices, n_cells, window, per_round, data_cfg, rng):
        self.data, self.rng = data, rng
        self.window, self.per_round = window, min(per_round, window)
        self.mixer = ClassMixer(n_cells, n_devices, data.prior, data_cfg, rng)
        self.idx = np.zeros((n_devices, window), dtype=np.int64)
        self.ptr = np.zeros(n_devices, dtype=int)
        self.filled = np.zeros(n_devices, dtype=int)

    def _draw(self, u, cell, n):
        p = self.mixer.mix(u, cell)
        cls = self.rng.choice(self.data.K, size=n, p=p)
        out = np.empty(n, dtype=np.int64)
        for k in np.unique(cls):
            pool = self.data.pools_tr[k]
            m = cls == k
            out[m] = pool[self.rng.integers(len(pool), size=int(m.sum()))] if len(pool) else 0
        return out

    def observe(self, cells, n=None):
        """Each device adds n flows (default per_round) by its current cell's mix."""
        n = n or self.per_round
        for u, c in enumerate(cells):
            new = self._draw(u, int(c), n)
            pos = (self.ptr[u] + np.arange(n)) % self.window
            self.idx[u, pos] = new
            self.ptr[u] = (self.ptr[u] + n) % self.window
            self.filled[u] = min(self.window, self.filled[u] + n)

    def data_of(self, u):
        i = self.idx[u, : self.filled[u]]
        return self.data.X_tr[i], self.data.y_tr[i]
