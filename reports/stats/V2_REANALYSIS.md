# v2 re-analyses after the pre-specified analysis (post-hoc; scripts/v2_reanalysis.py)

## C1: unit of analysis

The three datasets share the mobility traces and the attacker devices of a seed, so the independent replicates are the 10 trace seeds.

**Pre-registered detectors (random participation): carry-over minus recalibrated reset, contained share, seed level (n = 10, datasets averaged; Holm over 6)**

| scenario | mean Δ [95% CI] | W / L | p | p (Holm) |
|---|---|---|---|---|
| persistent_0.02 | 0.157 [0.117, 0.197] | 10 / 0 | 0.0020 | 0.0117 |
| persistent_0.05 | 0.677 [0.610, 0.737] | 10 / 0 | 0.0020 | 0.0117 |
| persistent_0.1 | 0.577 [0.543, 0.613] | 10 / 0 | 0.0020 | 0.0117 |
| persistent_0.2 | 0.440 [0.403, 0.477] | 10 / 0 | 0.0020 | 0.0117 |
| handover_timed | 0.710 [0.677, 0.743] | 10 / 0 | 0.0020 | 0.0117 |
| on_off_0.2 | 0.563 [0.527, 0.597] | 10 / 0 | 0.0020 | 0.0117 |

**R5: detectors of residence-aware training with forwarding: carry-over minus recalibrated reset, contained share, seed level (n = 10, datasets averaged; Holm over 6)**

| scenario | mean Δ [95% CI] | W / L | p | p (Holm) |
|---|---|---|---|---|
| persistent_0.02 | 0.140 [0.107, 0.173] | 10 / 0 | 0.0020 | 0.0117 |
| persistent_0.05 | 0.677 [0.617, 0.733] | 10 / 0 | 0.0020 | 0.0117 |
| persistent_0.1 | 0.573 [0.537, 0.610] | 10 / 0 | 0.0020 | 0.0117 |
| persistent_0.2 | 0.440 [0.397, 0.480] | 10 / 0 | 0.0020 | 0.0117 |
| handover_timed | 0.733 [0.687, 0.773] | 10 / 0 | 0.0020 | 0.0117 |
| on_off_0.2 | 0.563 [0.533, 0.590] | 10 / 0 | 0.0020 | 0.0117 |

## C1: mobility dependence

**Mobility dependence: Spearman correlation of (carry − recalibrated reset) with the attacker's handover rate, by class (seed-cluster bootstrap 95% CI, 2,000 resamples)**

| class | n attacker-runs | ρ [95% CI] |
|---|---|---|
| vehicles | 600 | 0.20 [0.12, 0.28] |
| pedestrians | 360 | 0.08 [-0.01, 0.15] |
| mobile (vehicles and pedestrians) | 960 | 0.38 [0.33, 0.43] |
| all | 1200 | 0.57 [0.55, 0.60] |

## C1: false containment

**False containment of honest devices (% within the 1-h run), per dataset and class (primary scenarios; last column: stealthy scenario, all classes)**

| dataset | method | all | vehicles | pedestrians | static | stealthy |
|---|---|---|---|---|---|---|
| 5g_nidd | cusum_carry | 0.01 | 0.00 | 0.04 | 0.00 | 0.00 |
| 5g_nidd | cusum_reset_cal | 0.08 | 0.00 | 0.04 | 0.36 | 0.36 |
| 5g_nidd | cusum_reset | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5g_nidd | trust_carry | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5g_nidd | trust_cold | 0.01 | 0.00 | 0.04 | 0.00 | 0.00 |
| ciciot2023 | cusum_carry | 1.38 | 1.55 | 1.47 | 0.83 | 2.14 |
| ciciot2023 | cusum_reset_cal | 1.58 | 0.00 | 0.20 | 7.62 | 3.07 |
| ciciot2023 | cusum_reset | 0.17 | 0.00 | 0.00 | 0.83 | 0.43 |
| ciciot2023 | trust_carry | 0.06 | 0.00 | 0.20 | 0.00 | 0.00 |
| ciciot2023 | trust_cold | 0.10 | 0.00 | 0.32 | 0.00 | 0.00 |
| cicids2017 | cusum_carry | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cicids2017 | cusum_reset_cal | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cicids2017 | cusum_reset | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cicids2017 | trust_carry | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cicids2017 | trust_cold | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## C1: time to containment

Pre-registered (conditional on containment; positive = carry-over faster; Holm over 6): persistent_0.02: -7.3 min (n = 27, Holm p = 0.0015); persistent_0.05: -5.0 min (n = 29, Holm p = 0.0001); persistent_0.1: +1.5 min (n = 30, Holm p = 0.3135); persistent_0.2: +1.6 min (n = 30, Holm p = 0.0419); handover_timed: +24.0 min (n = 10, Holm p = 0.0156); on_off_0.2: +1.8 min (n = 30, Holm p = 0.0691).

**Exploratory: restricted mean time to containment within 30 min (attackers not contained within 30 min count as 30); carry-over minus recalibrated reset, seed level (n = 10)**

| scenario | carry (min) | reset (min) | mean Δ [95% CI] | p |
|---|---|---|---|---|
| persistent_0.02 | 29.4 | 29.3 | 0.1 [-0.3, 0.5] | 0.8438 |
| persistent_0.05 | 19.8 | 26.8 | -7.0 [-8.0, -6.0] | 0.0020 |
| persistent_0.1 | 11.6 | 23.0 | -11.4 [-12.3, -10.5] | 0.0020 |
| persistent_0.2 | 6.4 | 17.9 | -11.5 [-12.6, -10.4] | 0.0020 |
| handover_timed | 15.9 | 29.9 | -14.1 [-15.0, -13.0] | 0.0020 |
| on_off_0.2 | 9.6 | 21.9 | -12.3 [-13.2, -11.3] | 0.0020 |

## C2

**C2: uplink to target with censored runs ranked worst (as pre-registered)**

n = 30, residence+fw better / worse / tied: 15 / 12 / 3, Wilcoxon p = 0.171.

**C2 per dataset, residence+fw − random macro-F1, Holm over 3 datasets**

| dataset | mean Δ | p | p (Holm) |
|---|---|---|---|
| 5g_nidd | 0.015 | 0.1934 | 0.1934 |
| ciciot2023 | 0.022 | 0.0020 | 0.0059 |
| cicids2017 | 0.040 | 0.0371 | 0.0742 |

**C2 equivalence check (margin ±0.02 macro-F1; 90% bootstrap CI of the paired difference, n = 30)**

| residence+fw vs | mean Δ | 90% CI | equivalent |
|---|---|---|---|
| random+fw | 0.009 | [-0.000, 0.018] | yes |
| speed | 0.009 | [-0.001, 0.021] | no |
| speed+fw | 0.008 | [-0.004, 0.020] | no |
| residence | 0.006 | [-0.002, 0.014] | yes |

