"""Dataset loading for MERIT-FL.

Real datasets (CICIDS2017, Edge-IIoTset, CICIoT2023) are read from the cache
written by src/data/preprocess.py (fixed split, scaler fit on train). The
synthetic workload is generated per seed so the pipeline runs without data.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .preprocess import assert_test_allowed, load_processed

def make_synthetic(n=20000, n_features=40, n_classes=6, rare_frac=0.02, seed=0):
    """Synthetic IDS-like data: Gaussian class clusters with overlapping
    covariance, one rare attack class (for the diversity-guard experiments)."""
    rng = np.random.default_rng(seed)
    means = rng.normal(0, 2.0, size=(n_classes, n_features))
    # class priors: benign-heavy, one rare class (last)
    priors = np.full(n_classes, (1 - 0.45 - rare_frac) / (n_classes - 2))
    priors[0] = 0.45
    priors[-1] = rare_frac
    y = rng.choice(n_classes, size=n, p=priors)
    X = means[y] + rng.normal(0, 1.0, size=(n, n_features))
    # inject mild nonlinearity so the CNN-BiLSTM has structure to learn
    X[:, ::4] = np.tanh(X[:, ::4]) * 3
    names = ["Benign"] + [f"Attack_{i}" for i in range(1, n_classes - 1)] + ["RareAttack"]
    return X.astype(np.float32), y.astype(np.int64), names


def find_benign(classes, counts):
    """Benign class index: by name (benign/normal), else the most frequent."""
    return next((k for k, c in enumerate(classes)
                 if str(c).strip().lower() in ("benign", "normal", "0")),
                int(np.argmax(counts)))


def get_dataset(cfg: dict, splits=None):
    """Entry point. cfg['data']['name'] in {synthetic, cicids2017,
    edge_iiotset, ciciot2023}. Returns {"train": (X, y), <eval split>: (X, y),
    "classes": [...], "features": [...]}. Only train and the run's evaluation
    split are loaded, so tuning runs (eval.split = val) never read test.
    """
    d, seed = cfg["data"], cfg.get("seed", 0)
    splits = splits or ("train", cfg["eval"]["split"])
    assert_test_allowed(splits)
    if d["name"] != "synthetic":
        data, meta = load_processed(d["root"], d["name"], d["taxonomy"], splits)
        return {**data, "classes": meta["classes"], "features": meta["features"]}
    X, y, names = make_synthetic(n=d["n_samples"], n_features=d["n_features"],
                                 n_classes=d["n_classes"], seed=seed)
    held = d["val_size"] + d["test_size"]
    tr, rest = train_test_split(np.arange(len(y)), test_size=held,
                                stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=d["test_size"] / held,
                              stratify=y[rest], random_state=seed)
    scaler = StandardScaler().fit(X[tr])
    idx = {"train": tr, "val": va, "test": te}
    out = {s_: (scaler.transform(X[idx[s_]]).astype(np.float32), y[idx[s_]]) for s_ in splits}
    return {**out, "classes": names, "features": [f"f{i}" for i in range(X.shape[1])]}
