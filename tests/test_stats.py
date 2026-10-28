"""Phase 5 statistics: bootstrap CI, paired Wilcoxon, Holm, rank-biserial."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as st

from src.evaluation import stats as S


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    x = np.random.default_rng(1).normal(0.6, 0.05, 30)
    lo, hi = S.bootstrap_ci(x)
    assert lo < x.mean() < hi
    assert (lo, hi) == S.bootstrap_ci(x)                     # fixed seed
    assert S.bootstrap_ci([0.5]) == (0.5, 0.5)


def test_summarize_skips_missing_values():
    s = S.summarize([0.5, None, 0.7, float("nan")])
    assert s["n"] == 2 and s["mean"] == pytest.approx(0.6) and s["std"] == pytest.approx(np.std([0.5, 0.7], ddof=1))


def test_paired_test_matches_scipy_and_orients_signs():
    rng = np.random.default_rng(2)
    b = rng.normal(0.6, 0.05, 10)
    a = b + rng.normal(0.03, 0.01, 10)
    r = S.paired_test(a, b)
    assert r["p"] == pytest.approx(st.wilcoxon(a - b).pvalue)
    assert r["wins"] == 10 and r["rank_biserial"] == 1.0 and r["mean_diff"] > 0
    lower = S.paired_test(a, b, higher_is_better=False)       # e.g. a detection dip: lower is better
    assert lower["mean_diff"] < 0 and lower["rank_biserial"] == -1.0 and lower["p"] == pytest.approx(r["p"])


def test_identical_arms_give_p_one():
    r = S.paired_test([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])
    assert r["p"] == 1.0 and r["ties"] == 3 and r["note"]


def test_rank_biserial_known_value():
    # ranks of |d| = 1,2,3,4 ; positives get 1+3, negatives 2+4 -> (4-6)/10
    assert S.rank_biserial([1, -2, 3, -4]) == pytest.approx(-0.2)


def test_holm_known_example():
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.06
    assert S.holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_compare_family_pairs_by_dataset_and_seed():
    rows = []
    for d in ("x", "y"):
        for s in range(6):
            rows += [{"label": "m", "dataset": d, "seed": s, "f1": 0.7 + 0.01 * s},
                     {"label": "f", "dataset": d, "seed": s, "f1": 0.6 + 0.01 * s},
                     {"label": "g", "dataset": d, "seed": s, "f1": 0.7 + 0.01 * s}]
    out = S.compare_family(pd.DataFrame(rows), "f1", "m", ["f", "g"])
    assert [o["vs"] for o in out] == ["f", "g"]
    assert out[0]["n"] == 12 and out[0]["wins"] == 12 and out[0]["p_holm"] < 0.01
    assert out[1]["p"] == 1.0 and out[1]["p_holm"] == 1.0


def test_hodges_lehmann_and_fixed_p():
    import pandas as pd
    from src.evaluation.stats import compare_family, hodges_lehmann
    assert hodges_lehmann([1.0, 2.0, 9.0]) == 3.5           # Walsh averages 1,1.5,2,5,5.5,9 -> median
    rows = np.vstack([hodges_lehmann(np.array([[1.0, 2.0, 9.0], [0.0, 0.0, 4.0]]), axis=1)])
    assert rows.tolist() == [[3.5, 1.0]]
    t = pd.DataFrame({"label": ["a"] * 4 + ["b"] * 4 + ["c"] * 4, "dataset": ["x"] * 12,
                      "seed": [1, 2, 3, 4] * 3, "m": [5, 6, 7, 8, 1, 2, 3, 4, 1, 2, 3, 4]})
    r = compare_family(t, "m", "a", ["b", "c"], fixed_p={"c": (1.0, "shares b's training")})
    assert r[1]["p"] == 1.0 and r[1]["note"] == "shares b's training" and r[0]["hl"] == 4.0
