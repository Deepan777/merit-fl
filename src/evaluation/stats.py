"""Statistics: summaries with 95% bootstrap CIs,
paired Wilcoxon signed-rank tests with Holm correction, and effect sizes.

Pairing: runs of different methods with the same (dataset, seed) see the same
traffic, mobility and adversaries (per-component RNG streams), so comparisons
are paired by (dataset, seed).

- Summary: n, mean, std (ddof=1), median, percentile-bootstrap 95% CI of the
  mean (B = 10,000, fixed RNG seed).
- Paired test: two-sided Wilcoxon signed-rank on the per-pair differences
  (zero differences dropped, scipy `zero_method="wilcox"`; exact p-values when
  possible). Effect size: matched-pairs rank-biserial correlation
  r = (R+ - R-) / (R+ + R-) over the non-zero differences, in [-1, 1]; plus the
  mean difference with its bootstrap 95% CI. Signs are oriented so that a
  positive difference / r favours the first method.
- Holm step-down adjustment within a family of comparisons.
"""
import numpy as np
from scipy import stats as st

B_DEFAULT = 10_000
SEED = 0


def bootstrap_ci(x, stat=np.mean, B=B_DEFAULT, alpha=0.05, seed=SEED):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    if len(x) == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    boot = stat(x[idx], axis=1)
    return (float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2)))


def summarize(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    lo, hi = bootstrap_ci(x)
    return {"n": int(len(x)), "mean": float(x.mean()) if len(x) else np.nan,
            "std": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
            "median": float(np.median(x)) if len(x) else np.nan, "ci_lo": lo, "ci_hi": hi}


def rank_biserial(d):
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = st.rankdata(np.abs(d))
    rp, rm = r[d > 0].sum(), r[d < 0].sum()
    return float((rp - rm) / (rp + rm))


def hodges_lehmann(d, axis=None):
    """One-sample Hodges-Lehmann estimate: median of the Walsh averages (the location
    estimate that matches the Wilcoxon signed-rank test). axis=1: one estimate per row."""
    d = np.asarray(d, dtype=float)
    if d.shape[-1] == 0:
        return np.nan
    i, j = np.triu_indices(d.shape[-1])
    walsh = (d[..., i] + d[..., j]) / 2
    return np.median(walsh, axis=-1) if axis is not None else float(np.median(walsh))


def paired_test(a, b, higher_is_better=True):
    """a, b: paired values (same order). Returns the comparison of a vs b with
    differences oriented so that positive = a better."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[ok] if higher_is_better else (b - a)[ok]
    out = {"n": int(len(d)), "mean_diff": float(d.mean()) if len(d) else np.nan,
           "median_diff": float(np.median(d)) if len(d) else np.nan,
           "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum())}
    out["diff_ci_lo"], out["diff_ci_hi"] = bootstrap_ci(d)
    out["hl"] = hodges_lehmann(d)
    out["hl_ci_lo"], out["hl_ci_hi"] = bootstrap_ci(d, stat=hodges_lehmann, B=2000)
    out["rank_biserial"] = rank_biserial(d)
    if (d != 0).sum() == 0:
        out.update(W=np.nan, p=1.0, note="identical in every pair")
        return out
    res = st.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    out.update(W=float(res.statistic), p=float(res.pvalue), note="")
    return out


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        adj[i] = running
    return adj


def compare_family(table, metric, ref, others, pair_keys=("dataset", "seed"), higher_is_better=True,
                   fixed_p=None):
    """ref vs each of `others` on `metric`, paired by `pair_keys`; Holm over the
    family. `table`: pandas DataFrame with a 'label' column naming the arm.
    fixed_p: {other: (p, note)} for comparisons whose p-value the analysis plan fixes
    (e.g. p = 1 for arms that share the reference baseline's training)."""
    rows = []
    base = table[table["label"] == ref].set_index(list(pair_keys))[metric]
    for o in others:
        other = table[table["label"] == o].set_index(list(pair_keys))[metric]
        j = base.to_frame("a").join(other.to_frame("b"), how="inner")
        res = paired_test(j["a"].to_numpy(), j["b"].to_numpy(), higher_is_better)
        if fixed_p and o in fixed_p:
            res.update(p=fixed_p[o][0], note=fixed_p[o][1])
        rows.append({"ref": ref, "vs": o, "metric": metric, **res})
    adj = holm([r["p"] for r in rows]) if rows else []
    for r, pa in zip(rows, adj):
        r["p_holm"] = float(pa)
    return rows
