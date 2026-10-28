# v2 statistics (scripts/v2_analyze.py; plan: the analysis plan

## V3: mobility-aware participation (C2)

| dataset | arm | macro-F1 (last 3) | interrupted | lost | energy (J) | uplink (MB) | rounds to target |
|---|---|---|---|---|---|---|---|
| 5g_nidd | oracle | 0.749 ± 0.070 | 0.011 | 0.013 | 36022 | 415 | 14.0 |
| 5g_nidd | random | 0.731 ± 0.061 | 0.366 | 0.417 | 36044 | 394 | 16.0 |
| 5g_nidd | random+fw | 0.740 ± 0.056 | 0.366 | 0.051 | 36044 | 394 | 14.0 |
| 5g_nidd | residence | 0.747 ± 0.042 | 0.153 | 0.190 | 36030 | 400 | 12.0 |
| 5g_nidd | residence+fw | 0.747 ± 0.048 | 0.153 | 0.037 | 36030 | 400 | 13.0 |
| 5g_nidd | speed | 0.738 ± 0.055 | 0.116 | 0.131 | 36013 | 409 | 15.0 |
| 5g_nidd | speed+fw | 0.740 ± 0.064 | 0.116 | 0.015 | 36013 | 409 | 12.0 |
| ciciot2023 | oracle | 0.474 ± 0.027 | 0.012 | 0.013 | 19690 | 419 | 61.0 |
| ciciot2023 | random | 0.451 ± 0.030 | 0.370 | 0.416 | 19714 | 400 | 75.1 |
| ciciot2023 | random+fw | 0.465 ± 0.025 | 0.370 | 0.046 | 19714 | 400 | 56.0 |
| ciciot2023 | residence | 0.467 ± 0.018 | 0.160 | 0.196 | 19702 | 404 | 66.0 |
| ciciot2023 | residence+fw | 0.473 ± 0.022 | 0.160 | 0.036 | 19702 | 404 | 67.0 |
| ciciot2023 | speed | 0.486 ± 0.021 | 0.123 | 0.137 | 19691 | 413 | 56.0 |
| ciciot2023 | speed+fw | 0.486 ± 0.021 | 0.123 | 0.015 | 19691 | 413 | 59.0 |
| cicids2017 | oracle | 0.618 ± 0.046 | 0.011 | 0.012 | 32757 | 418 | 72.1 |
| cicids2017 | random | 0.593 ± 0.055 | 0.372 | 0.424 | 32780 | 397 | 82.2 |
| cicids2017 | random+fw | 0.621 ± 0.065 | 0.372 | 0.051 | 32780 | 397 | 75.1 |
| cicids2017 | residence | 0.620 ± 0.061 | 0.154 | 0.190 | 32765 | 403 | 74.1 |
| cicids2017 | residence+fw | 0.632 ± 0.070 | 0.154 | 0.036 | 32765 | 403 | 78.1 |
| cicids2017 | speed | 0.600 ± 0.063 | 0.117 | 0.132 | 32751 | 412 | 77.1 |
| cicids2017 | speed+fw | 0.603 ± 0.078 | 0.117 | 0.014 | 32751 | 412 | 75.1 |

**Primary: residence+fw − random, macro-F1 (pooled, n = 30; non-inferiority margin 0.02)**

| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) | non-inferior |
|---|---|---|---|---|---|---|---|---|
| random | 30 | 0.026 [0.013, 0.039] | 0.025 | 26 / 4 / 0 | 0.72 | 0.0003 | 0.0003 | yes |

**Secondary: residence+fw − other arms, macro-F1 (Holm over 5)**

| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) | non-inferior |
|---|---|---|---|---|---|---|---|---|
| random | 30 | 0.026 [0.013, 0.039] | 0.025 | 26 / 4 / 0 | 0.72 | 0.0003 | 0.0016 | yes |
| random+fw | 30 | 0.009 [-0.002, 0.020] | 0.008 | 18 / 12 / 0 | 0.31 | 0.1403 | 0.4208 | yes |
| speed | 30 | 0.009 [-0.003, 0.023] | 0.004 | 16 / 14 / 0 | 0.14 | 0.5291 | 0.9542 | yes |
| speed+fw | 30 | 0.008 [-0.006, 0.023] | 0.003 | 16 / 14 / 0 | 0.15 | 0.4771 | 0.9542 | yes |
| residence | 30 | 0.006 [-0.003, 0.016] | 0.007 | 20 / 10 / 0 | 0.34 | 0.1048 | 0.4194 | yes |

**Interrupted share of selected updates (positive = residence+fw lower)**

| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| random | 30 | 0.214 [0.209, 0.219] | 0.214 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |

**Interrupted share, residence (no forwarding) vs random (positive = residence lower)**

| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| random | 30 | 0.214 [0.209, 0.219] | 0.214 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |

**Rounds to 0.9 x random's median final macro-F1 (positive = residence+fw faster)**

| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| random | 30 | 5.1 [-0.3, 10.5] | 5.0 | 13 / 4 / 13 | 0.46 | 0.0939 | 0.0939 |

**Uplink MB to target (positive = residence+fw less; censored runs ranked worst as +inf)**

| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| random | 27 | 12.8 [-6.6, 32.3] | 14.3 | 13 / 12 / 2 | 0.19 | 0.4118 | 0.4118 |

**Forwarding on − off within each selection rule, macro-F1 (Holm over 3)**

| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| random+fw vs random | 30 | 0.017 [0.007, 0.029] | 0.010 | 19 / 11 / 0 | 0.59 | 0.0037 | 0.0112 |
| speed+fw vs speed | 30 | 0.002 [-0.006, 0.009] | 0.001 | 17 / 13 / 0 | 0.11 | 0.6120 | 0.6120 |
| residence+fw vs residence | 30 | 0.006 [-0.003, 0.016] | 0.007 | 20 / 10 / 0 | 0.34 | 0.1048 | 0.2097 |

**Device training energy (positive = residence+fw less)**

| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| random | 30 | 14 [12, 17] | 13 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |

**Per dataset 5g_nidd: residence+fw − random, macro-F1 (n = 10)**

| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) | non-inferior |
|---|---|---|---|---|---|---|---|---|
| random | 10 | 0.015 [-0.006, 0.035] | 0.018 | 7 / 3 / 0 | 0.49 | 0.1934 | 0.1934 | yes |

**Per dataset ciciot2023: residence+fw − random, macro-F1 (n = 10)**

| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) | non-inferior |
|---|---|---|---|---|---|---|---|---|
| random | 10 | 0.022 [0.010, 0.035] | 0.020 | 10 / 0 / 0 | 1.00 | 0.0020 | 0.0020 | yes |

**Per dataset cicids2017: residence+fw − random, macro-F1 (n = 10)**

| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) | non-inferior |
|---|---|---|---|---|---|---|---|---|
| random | 10 | 0.040 [0.013, 0.069] | 0.032 | 9 / 1 / 0 | 0.75 | 0.0371 | 0.0371 | yes |

**Round length: residence+fw − random (n = 15 per length)**

| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| random @ 15 s (macro-F1) | 15 | -0.013 [-0.035, 0.005] | -0.009 | 6 / 9 / 0 | -0.32 | 0.3028 | 0.3028 |
| random @ 15 s (interrupted) | 15 | 0.112 [0.106, 0.117] | 0.112 | 15 / 0 / 0 | 1.00 | 0.0001 | 0.0002 |
| random @ 60 s (macro-F1) | 15 | 0.054 [0.031, 0.080] | 0.051 | 14 / 1 / 0 | 0.93 | 0.0004 | 0.0009 |
| random @ 60 s (interrupted) | 15 | 0.271 [0.260, 0.282] | 0.271 | 15 / 0 / 0 | 1.00 | 0.0001 | 0.0002 |

**Rule outcome (plan): partially supported** (non-inferior True; fewer interruptions True; faster or cheaper to target False).

## V1: carry-over of sequential evidence (C1)

| scenario | method | contained | delay (s) | exposure | false containment |
|---|---|---|---|---|---|
| handover_timed | cusum_carry | 0.753 | 695 | 0.249 | 0.0045 |
| handover_timed | cusum_reset | 0.000 | – | 1.000 | 0.0000 |
| handover_timed | cusum_reset_cal | 0.043 | 2128 | 0.999 | 0.0060 |
| handover_timed | none | 0.000 | – | 1.000 | 0.0000 |
| handover_timed | trust_carry | 0.000 | – | 1.000 | 0.0000 |
| handover_timed | trust_cold | 0.003 | 1530 | 0.999 | 0.0000 |
| on_off_0.2 | cusum_carry | 0.977 | 546 | 0.244 | 0.0033 |
| on_off_0.2 | cusum_reset | 0.337 | 764 | 0.760 | 0.0005 |
| on_off_0.2 | cusum_reset_cal | 0.413 | 654 | 0.676 | 0.0040 |
| on_off_0.2 | none | 0.000 | – | 1.000 | 0.0000 |
| on_off_0.2 | trust_carry | 0.000 | – | 1.000 | 0.0002 |
| on_off_0.2 | trust_cold | 0.000 | – | 1.000 | 0.0005 |
| persistent_0.02 | cusum_carry | 0.267 | 1976 | 0.934 | 0.0057 |
| persistent_0.02 | cusum_reset | 0.060 | 1863 | 0.983 | 0.0002 |
| persistent_0.02 | cusum_reset_cal | 0.110 | 1574 | 0.956 | 0.0055 |
| persistent_0.02 | none | 0.000 | – | 1.000 | 0.0000 |
| persistent_0.02 | trust_carry | 0.000 | – | 1.000 | 0.0005 |
| persistent_0.02 | trust_cold | 0.000 | – | 1.000 | 0.0007 |
| persistent_0.05 | cusum_carry | 0.877 | 1130 | 0.524 | 0.0067 |
| persistent_0.05 | cusum_reset | 0.173 | 1177 | 0.907 | 0.0012 |
| persistent_0.05 | cusum_reset_cal | 0.200 | 839 | 0.862 | 0.0071 |
| persistent_0.05 | none | 0.000 | – | 1.000 | 0.0000 |
| persistent_0.05 | trust_carry | 0.000 | – | 1.000 | 0.0000 |
| persistent_0.05 | trust_cold | 0.000 | – | 1.000 | 0.0002 |
| persistent_0.05_forge | cusum_carry | 0.973 | 273 | 0.129 | 0.0067 |
| persistent_0.05_forge | cusum_reset | 0.173 | 1177 | 0.907 | 0.0012 |
| persistent_0.05_forge | cusum_reset_cal | 0.200 | 839 | 0.862 | 0.0071 |
| persistent_0.05_forge | none | 0.000 | – | 1.000 | 0.0000 |
| persistent_0.05_forge | trust_carry | 0.000 | – | 1.000 | 0.0000 |
| persistent_0.05_forge | trust_cold | 0.000 | – | 1.000 | 0.0002 |
| persistent_0.05_omit | cusum_carry | 0.877 | 1130 | 0.524 | 0.0067 |
| persistent_0.05_omit | cusum_reset | 0.173 | 1177 | 0.907 | 0.0012 |
| persistent_0.05_omit | cusum_reset_cal | 0.200 | 839 | 0.862 | 0.0071 |
| persistent_0.05_omit | none | 0.000 | – | 1.000 | 0.0000 |
| persistent_0.05_omit | trust_carry | 0.000 | – | 1.000 | 0.0000 |
| persistent_0.05_omit | trust_cold | 0.000 | – | 1.000 | 0.0002 |
| persistent_0.1 | cusum_carry | 0.963 | 663 | 0.281 | 0.0040 |
| persistent_0.1 | cusum_reset | 0.290 | 917 | 0.814 | 0.0010 |
| persistent_0.1 | cusum_reset_cal | 0.387 | 753 | 0.718 | 0.0055 |
| persistent_0.1 | none | 0.000 | – | 1.000 | 0.0000 |
| persistent_0.1 | trust_carry | 0.000 | – | 1.000 | 0.0002 |
| persistent_0.1 | trust_cold | 0.000 | – | 1.000 | 0.0005 |
| persistent_0.2 | cusum_carry | 0.970 | 339 | 0.146 | 0.0036 |
| persistent_0.2 | cusum_reset | 0.410 | 561 | 0.671 | 0.0005 |
| persistent_0.2 | cusum_reset_cal | 0.530 | 432 | 0.550 | 0.0052 |
| persistent_0.2 | none | 0.000 | – | 1.000 | 0.0000 |
| persistent_0.2 | trust_carry | 0.000 | – | 1.000 | 0.0002 |
| persistent_0.2 | trust_cold | 0.000 | – | 1.000 | 0.0002 |
| stealthy_0.005 | cusum_carry | 0.013 | 1260 | 0.994 | 0.0071 |
| stealthy_0.005 | cusum_reset | 0.000 | – | 1.000 | 0.0014 |
| stealthy_0.005 | cusum_reset_cal | 0.003 | 1830 | 0.999 | 0.0114 |
| stealthy_0.005 | none | 0.000 | – | 1.000 | 0.0000 |
| stealthy_0.005 | trust_carry | 0.000 | – | 1.000 | 0.0000 |
| stealthy_0.005 | trust_cold | 0.000 | – | 1.000 | 0.0000 |

**Primary: cusum_carry − cusum_reset_cal (equal false-containment budget), contained share (pooled n = 30 per scenario; Holm over 6)**

| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| cusum_reset_cal, persistent_0.02 | 30 | 0.157 [0.120, 0.193] | 0.150 | 24 / 0 / 6 | 1.00 | 0.0000 | 0.0000 |
| cusum_reset_cal, persistent_0.05 | 30 | 0.677 [0.623, 0.723] | 0.700 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| cusum_reset_cal, persistent_0.1 | 30 | 0.577 [0.540, 0.613] | 0.600 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| cusum_reset_cal, persistent_0.2 | 30 | 0.440 [0.397, 0.483] | 0.450 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| cusum_reset_cal, handover_timed | 30 | 0.710 [0.667, 0.747] | 0.700 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| cusum_reset_cal, on_off_0.2 | 30 | 0.563 [0.530, 0.597] | 0.550 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |

**Secondary: cusum_carry − cusum_reset (same k, h), contained share (Holm over 6)**

| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| cusum_reset, persistent_0.02 | 30 | 0.207 [0.167, 0.247] | 0.200 | 28 / 0 / 2 | 1.00 | 0.0000 | 0.0000 |
| cusum_reset, persistent_0.05 | 30 | 0.703 [0.657, 0.743] | 0.700 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| cusum_reset, persistent_0.1 | 30 | 0.673 [0.637, 0.710] | 0.700 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| cusum_reset, persistent_0.2 | 30 | 0.560 [0.530, 0.590] | 0.550 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| cusum_reset, handover_timed | 30 | 0.753 [0.713, 0.783] | 0.800 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| cusum_reset, on_off_0.2 | 30 | 0.640 [0.597, 0.683] | 0.650 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |

**Secondary: delay, cusum_carry vs cusum_reset_cal (positive = carry better)**

| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| cusum_reset_cal, persistent_0.02 | 27 | -440.167 [-620.928, -261.370] | -420.000 | 4 / 22 / 1 | -0.81 | 0.0003 | 0.0015 |
| cusum_reset_cal, persistent_0.05 | 29 | -297.466 [-375.416, -217.082] | -308.625 | 3 / 26 / 0 | -0.91 | 0.0000 | 0.0001 |
| cusum_reset_cal, persistent_0.1 | 30 | 89.992 [-16.199, 199.129] | 87.250 | 15 / 15 / 0 | 0.21 | 0.3135 | 0.3135 |
| cusum_reset_cal, persistent_0.2 | 30 | 93.120 [29.943, 164.398] | 65.786 | 20 / 10 / 0 | 0.51 | 0.0140 | 0.0419 |
| cusum_reset_cal, handover_timed | 10 | 1442.732 [1021.926, 1734.964] | 1579.821 | 9 / 1 / 0 | 0.96 | 0.0039 | 0.0156 |
| cusum_reset_cal, on_off_0.2 | 30 | 108.211 [8.915, 213.287] | 98.000 | 21 / 9 / 0 | 0.44 | 0.0345 | 0.0691 |

**Secondary: exposure, cusum_carry vs cusum_reset_cal (positive = carry better)**

| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| cusum_reset_cal, persistent_0.02 | 30 | 0.022 [0.010, 0.035] | 0.023 | 24 / 6 / 0 | 0.64 | 0.0016 | 0.0016 |
| cusum_reset_cal, persistent_0.05 | 30 | 0.338 [0.304, 0.369] | 0.337 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| cusum_reset_cal, persistent_0.1 | 30 | 0.437 [0.408, 0.464] | 0.437 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| cusum_reset_cal, persistent_0.2 | 30 | 0.404 [0.372, 0.437] | 0.402 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| cusum_reset_cal, handover_timed | 30 | 0.751 [0.712, 0.782] | 0.768 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| cusum_reset_cal, on_off_0.2 | 30 | 0.433 [0.407, 0.460] | 0.430 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |

**Secondary: cusum_carry − v1 smoothed trust, contained share (Holm over 12)**

| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| trust_carry, persistent_0.02 | 30 | 0.267 [0.227, 0.303] | 0.250 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| trust_cold, persistent_0.02 | 30 | 0.267 [0.227, 0.303] | 0.250 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| trust_carry, persistent_0.05 | 30 | 0.877 [0.817, 0.930] | 0.900 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| trust_cold, persistent_0.05 | 30 | 0.877 [0.817, 0.930] | 0.900 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| trust_carry, persistent_0.1 | 30 | 0.963 [0.937, 0.987] | 1.000 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| trust_cold, persistent_0.1 | 30 | 0.963 [0.937, 0.987] | 1.000 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| trust_carry, persistent_0.2 | 30 | 0.970 [0.940, 0.993] | 1.000 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| trust_cold, persistent_0.2 | 30 | 0.970 [0.940, 0.993] | 1.000 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| trust_carry, handover_timed | 30 | 0.753 [0.713, 0.783] | 0.800 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| trust_cold, handover_timed | 30 | 0.750 [0.710, 0.780] | 0.750 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| trust_carry, on_off_0.2 | 30 | 0.977 [0.953, 0.993] | 1.000 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |
| trust_cold, on_off_0.2 | 30 | 0.977 [0.953, 0.993] | 1.000 | 30 / 0 / 0 | 1.00 | 0.0000 | 0.0000 |

**False containment of honest devices (positive = carry lower; non-inferiority margin 0.005)**

| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) | non-inferior |
|---|---|---|---|---|---|---|---|---|
| cusum_reset_cal | 30 | 0.0009 [-0.0001, 0.0019] | 0.0006 | 11 / 2 / 17 | 0.70 | 0.0248 | 0.0248 | yes |

Contained share by velocity class (persistent scenarios):

| class | carry | reset (recalibrated) | reset (same threshold) |
|---|---|---|---|
| veh | 0.767 | 0.035 | 0.000 |
| ped | 0.778 | 0.406 | 0.269 |
| static | 0.762 | 0.838 | 0.762 |

Spearman correlation of (carry − recalibrated reset) containment with the attacker's handover rate: rho = 0.575 (p = 1.6e-106, n = 1200 attacker-runs).

**Rule outcome (plan): supported** (6/6 primary scenarios significant; false containment non-inferior True).

**Per dataset 5g_nidd: cusum_carry − cusum_reset_cal, contained share (n = 10; Holm over 6)**

| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| reset_cal, persistent_0.02 | 10 | 0.190 [0.130, 0.250] | 0.200 | 9 / 0 / 1 | 1.00 | 0.0039 | 0.0117 |
| reset_cal, persistent_0.05 | 10 | 0.750 [0.710, 0.790] | 0.750 | 10 / 0 / 0 | 1.00 | 0.0020 | 0.0117 |
| reset_cal, persistent_0.1 | 10 | 0.560 [0.520, 0.600] | 0.550 | 10 / 0 / 0 | 1.00 | 0.0020 | 0.0117 |
| reset_cal, persistent_0.2 | 10 | 0.420 [0.370, 0.460] | 0.400 | 10 / 0 / 0 | 1.00 | 0.0020 | 0.0117 |
| reset_cal, handover_timed | 10 | 0.760 [0.710, 0.800] | 0.800 | 10 / 0 / 0 | 1.00 | 0.0020 | 0.0117 |
| reset_cal, on_off_0.2 | 10 | 0.570 [0.530, 0.610] | 0.550 | 10 / 0 / 0 | 1.00 | 0.0020 | 0.0117 |

**Per dataset ciciot2023: cusum_carry − cusum_reset_cal, contained share (n = 10; Holm over 6)**

| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| reset_cal, persistent_0.02 | 10 | 0.170 [0.110, 0.220] | 0.150 | 9 / 0 / 1 | 1.00 | 0.0039 | 0.0117 |
| reset_cal, persistent_0.05 | 10 | 0.630 [0.550, 0.710] | 0.650 | 10 / 0 / 0 | 1.00 | 0.0020 | 0.0117 |
| reset_cal, persistent_0.1 | 10 | 0.660 [0.610, 0.720] | 0.650 | 10 / 0 / 0 | 1.00 | 0.0020 | 0.0117 |
| reset_cal, persistent_0.2 | 10 | 0.520 [0.460, 0.590] | 0.500 | 10 / 0 / 0 | 1.00 | 0.0020 | 0.0117 |
| reset_cal, handover_timed | 10 | 0.660 [0.620, 0.700] | 0.650 | 10 / 0 / 0 | 1.00 | 0.0020 | 0.0117 |
| reset_cal, on_off_0.2 | 10 | 0.630 [0.590, 0.670] | 0.650 | 10 / 0 / 0 | 1.00 | 0.0020 | 0.0117 |

**Per dataset cicids2017: cusum_carry − cusum_reset_cal, contained share (n = 10; Holm over 6)**

| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| reset_cal, persistent_0.02 | 10 | 0.110 [0.050, 0.180] | 0.100 | 6 / 0 / 4 | 1.00 | 0.0312 | 0.0312 |
| reset_cal, persistent_0.05 | 10 | 0.650 [0.540, 0.740] | 0.700 | 10 / 0 / 0 | 1.00 | 0.0020 | 0.0117 |
| reset_cal, persistent_0.1 | 10 | 0.510 [0.450, 0.560] | 0.500 | 10 / 0 / 0 | 1.00 | 0.0020 | 0.0117 |
| reset_cal, persistent_0.2 | 10 | 0.380 [0.300, 0.460] | 0.400 | 10 / 0 / 0 | 1.00 | 0.0020 | 0.0117 |
| reset_cal, handover_timed | 10 | 0.710 [0.600, 0.790] | 0.750 | 10 / 0 / 0 | 1.00 | 0.0020 | 0.0117 |
| reset_cal, on_off_0.2 | 10 | 0.490 [0.430, 0.550] | 0.500 | 10 / 0 / 0 | 1.00 | 0.0020 | 0.0117 |

## V2: token attacks

**Contained share with a withheld or forged token minus with the authentic token (non-inferiority margin 0.05)**

| vs | n | mean Δ [95% CI] | HL | W / L / T | r | p | p (Holm) | non-inferior |
|---|---|---|---|---|---|---|---|---|
| omit vs authentic (THCP v2) | 30 | 0.000 [0.000, 0.000] | 0.000 | 0 / 0 / 30 | 0.00 | 1.0000 | 1.0000 | yes |
| forge vs authentic (THCP v2) | 30 | 0.097 [0.057, 0.143] | 0.100 | 15 / 0 / 15 | 1.00 | 0.0005 | 0.0011 | yes |

v1 smoothed trust with THCP v1, contained share: authentic 0.000, withheld token 0.000.

**Rule outcome (plan): supported**.

## V4: boundaries

Stealthy attackers (rate 0.005), contained share: cusum_carry 0.013, cusum_reset 0.000, cusum_reset_cal 0.003, none 0.000, trust_carry 0.000, trust_cold 0.000

| dataset | detector p_fa | p_det | margin | carry contained (persistent 0.05) |
|---|---|---|---|---|
| 5g_nidd | 0.085 | 0.869 | 0.784 | 0.970 |
| cicids2017 | 0.024 | 0.731 | 0.706 | 0.840 |
| ciciot2023 | 0.261 | 0.834 | 0.573 | 0.820 |

