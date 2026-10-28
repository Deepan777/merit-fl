# R6 robustness of C1 (post-hoc; scripts/v2_robust_analyze.py)

## Contained share of persistent 5% attackers within 60 min, and steady-state honest false-alarm rate on test (% per device-hour), at the main budget (0.1% per device-hour)

| world | dataset | rule | contained | false alarms %/h [95% CI] |
|---|---|---|---|---|
| W0 | 5g_nidd | share/carry | 0.985 | 0.273 [0.242, 0.308] |
| W0 | 5g_nidd | share/memory | 0.465 | 0.141 [0.118, 0.166] |
| W0 | 5g_nidd | share/reset | 0.275 | 0.108 [0.089, 0.131] |
| W0 | 5g_nidd | llr/carry | 1.000 | 0.085 [0.067, 0.105] |
| W0 | 5g_nidd | llr/memory | 0.795 | 0.114 [0.094, 0.138] |
| W0 | 5g_nidd | llr/reset | 0.380 | 0.106 [0.087, 0.129] |
| W0 | 5g_nidd | ewma/carry | 0.980 | 0.284 [0.251, 0.319] |
| W0 | 5g_nidd | ewma/memory | 0.440 | 0.079 [0.062, 0.098] |
| W0 | 5g_nidd | ewma/reset | 0.290 | 0.104 [0.085, 0.126] |
| W0 | ciciot2023 | share/carry | 0.640 | 0.073 [0.057, 0.093] |
| W0 | ciciot2023 | share/memory | 0.150 | 0.166 [0.142, 0.194] |
| W0 | ciciot2023 | share/reset | 0.150 | 0.149 [0.126, 0.175] |
| W0 | ciciot2023 | llr/carry | 0.670 | 0.036 [0.025, 0.050] |
| W0 | ciciot2023 | llr/memory | 0.140 | 0.016 [0.009, 0.027] |
| W0 | ciciot2023 | llr/reset | 0.165 | 0.045 [0.033, 0.060] |
| W0 | ciciot2023 | ewma/carry | 0.640 | 0.064 [0.049, 0.082] |
| W0 | ciciot2023 | ewma/memory | 0.145 | 0.051 [0.038, 0.067] |
| W0 | ciciot2023 | ewma/reset | 0.160 | 0.055 [0.041, 0.072] |
| W0 | cicids2017 | share/carry | 0.890 | 0.064 [0.049, 0.082] |
| W0 | cicids2017 | share/memory | 0.730 | 0.061 [0.047, 0.079] |
| W0 | cicids2017 | share/reset | 0.405 | 0.079 [0.062, 0.098] |
| W0 | cicids2017 | llr/carry | 0.850 | 0.000 [0.000, 0.004] |
| W0 | cicids2017 | llr/memory | 0.520 | 0.000 [0.000, 0.004] |
| W0 | cicids2017 | llr/reset | 0.385 | 0.027 [0.017, 0.039] |
| W0 | cicids2017 | ewma/carry | 0.890 | 0.060 [0.046, 0.078] |
| W0 | cicids2017 | ewma/memory | 0.725 | 0.052 [0.039, 0.068] |
| W0 | cicids2017 | ewma/reset | 0.395 | 0.087 [0.069, 0.107] |
| W1 | 5g_nidd | share/carry | 0.185 | 17.277 [17.017, 17.539] |
| W1 | 5g_nidd | share/memory | 0.060 | 4.372 [4.243, 4.505] |
| W1 | 5g_nidd | share/reset | 0.050 | 3.049 [2.941, 3.160] |
| W1 | 5g_nidd | llr/carry | 0.770 | 0.532 [0.487, 0.579] |
| W1 | 5g_nidd | llr/memory | 0.620 | 0.511 [0.467, 0.558] |
| W1 | 5g_nidd | llr/reset | 0.520 | 0.485 [0.442, 0.530] |
| W1 | 5g_nidd | ewma/carry | 0.480 | 0.145 [0.122, 0.171] |
| W1 | 5g_nidd | ewma/memory | 0.135 | 0.177 [0.151, 0.205] |
| W1 | 5g_nidd | ewma/reset | 0.135 | 0.088 [0.070, 0.108] |
| W1 | ciciot2023 | share/carry | 0.265 | 41.386 [40.984, 41.790] |
| W1 | ciciot2023 | share/memory | 0.140 | 13.313 [13.086, 13.544] |
| W1 | ciciot2023 | share/reset | 0.060 | 9.387 [9.196, 9.581] |
| W1 | ciciot2023 | llr/carry | 0.420 | 0.309 [0.275, 0.346] |
| W1 | ciciot2023 | llr/memory | 0.210 | 0.237 [0.207, 0.269] |
| W1 | ciciot2023 | llr/reset | 0.155 | 0.299 [0.266, 0.335] |
| W1 | ciciot2023 | ewma/carry | 0.045 | 0.064 [0.049, 0.082] |
| W1 | ciciot2023 | ewma/memory | 0.000 | 0.079 [0.062, 0.098] |
| W1 | ciciot2023 | ewma/reset | 0.045 | 0.094 [0.076, 0.115] |
| W1 | cicids2017 | share/carry | 0.010 | 3.242 [3.130, 3.357] |
| W1 | cicids2017 | share/memory | 0.015 | 1.354 [1.282, 1.429] |
| W1 | cicids2017 | share/reset | 0.010 | 0.948 [0.888, 1.011] |
| W1 | cicids2017 | llr/carry | 0.715 | 1.879 [1.794, 1.966] |
| W1 | cicids2017 | llr/memory | 0.465 | 0.999 [0.937, 1.064] |
| W1 | cicids2017 | llr/reset | 0.275 | 0.447 [0.406, 0.491] |
| W1 | cicids2017 | ewma/carry | 0.355 | 0.071 [0.056, 0.090] |
| W1 | cicids2017 | ewma/memory | 0.105 | 0.042 [0.030, 0.057] |
| W1 | cicids2017 | ewma/reset | 0.110 | 0.016 [0.009, 0.027] |
| W2 | 5g_nidd | share/carry | 0.105 | 15.237 [14.993, 15.483] |
| W2 | 5g_nidd | share/memory | 0.050 | 7.153 [6.987, 7.323] |
| W2 | 5g_nidd | share/reset | 0.005 | 3.020 [2.913, 3.131] |
| W2 | 5g_nidd | llr/carry | 0.735 | 0.278 [0.246, 0.313] |
| W2 | 5g_nidd | llr/memory | 0.655 | 0.286 [0.253, 0.321] |
| W2 | 5g_nidd | llr/reset | 0.535 | 0.378 [0.340, 0.418] |
| W2 | 5g_nidd | ewma/carry | 0.700 | 0.840 [0.783, 0.899] |
| W2 | 5g_nidd | ewma/memory | 0.280 | 0.852 [0.795, 0.912] |
| W2 | 5g_nidd | ewma/reset | 0.200 | 0.439 [0.398, 0.482] |
| W2 | ciciot2023 | share/carry | 0.350 | 45.976 [45.552, 46.402] |
| W2 | ciciot2023 | share/memory | 0.270 | 23.671 [23.368, 23.978] |
| W2 | ciciot2023 | share/reset | 0.070 | 8.645 [8.462, 8.831] |
| W2 | ciciot2023 | llr/carry | 0.310 | 0.280 [0.247, 0.315] |
| W2 | ciciot2023 | llr/memory | 0.245 | 0.267 [0.236, 0.302] |
| W2 | ciciot2023 | llr/reset | 0.145 | 0.122 [0.102, 0.146] |
| W2 | ciciot2023 | ewma/carry | 0.030 | 0.065 [0.050, 0.083] |
| W2 | ciciot2023 | ewma/memory | 0.010 | 0.067 [0.052, 0.086] |
| W2 | ciciot2023 | ewma/reset | 0.020 | 0.016 [0.009, 0.027] |
| W2 | cicids2017 | share/carry | 0.060 | 3.520 [3.404, 3.640] |
| W2 | cicids2017 | share/memory | 0.050 | 1.478 [1.402, 1.556] |
| W2 | cicids2017 | share/reset | 0.020 | 0.399 [0.360, 0.441] |
| W2 | cicids2017 | llr/carry | 0.710 | 1.508 [1.432, 1.587] |
| W2 | cicids2017 | llr/memory | 0.630 | 1.703 [1.622, 1.787] |
| W2 | cicids2017 | llr/reset | 0.470 | 1.192 [1.124, 1.262] |
| W2 | cicids2017 | ewma/carry | 0.560 | 0.229 [0.200, 0.261] |
| W2 | cicids2017 | ewma/memory | 0.125 | 0.076 [0.059, 0.095] |
| W2 | cicids2017 | ewma/reset | 0.175 | 0.156 [0.132, 0.183] |
| W3 | 5g_nidd | share/carry | 0.160 | 13.327 [13.099, 13.557] |
| W3 | 5g_nidd | share/memory | 0.050 | 3.900 [3.777, 4.026] |
| W3 | 5g_nidd | share/reset | 0.050 | 2.811 [2.707, 2.918] |
| W3 | 5g_nidd | llr/carry | 0.745 | 2.610 [2.510, 2.713] |
| W3 | 5g_nidd | llr/memory | 0.595 | 1.844 [1.760, 1.931] |
| W3 | 5g_nidd | llr/reset | 0.510 | 1.953 [1.867, 2.043] |
| W3 | 5g_nidd | ewma/carry | 0.490 | 0.223 [0.195, 0.255] |
| W3 | 5g_nidd | ewma/memory | 0.100 | 0.181 [0.155, 0.209] |
| W3 | 5g_nidd | ewma/reset | 0.115 | 0.214 [0.186, 0.245] |
| W3 | ciciot2023 | share/carry | 0.350 | 40.329 [39.932, 40.728] |
| W3 | ciciot2023 | share/memory | 0.150 | 11.672 [11.460, 11.888] |
| W3 | ciciot2023 | share/reset | 0.090 | 7.756 [7.583, 7.932] |
| W3 | ciciot2023 | llr/carry | 0.260 | 0.017 [0.010, 0.028] |
| W3 | ciciot2023 | llr/memory | 0.125 | 0.015 [0.009, 0.025] |
| W3 | ciciot2023 | llr/reset | 0.140 | 0.114 [0.094, 0.138] |
| W3 | ciciot2023 | ewma/carry | 0.020 | 0.043 [0.031, 0.058] |
| W3 | ciciot2023 | ewma/memory | 0.010 | 0.046 [0.033, 0.061] |
| W3 | ciciot2023 | ewma/reset | 0.025 | 0.033 [0.022, 0.046] |
| W3 | cicids2017 | share/carry | 0.005 | 2.534 [2.435, 2.635] |
| W3 | cicids2017 | share/memory | 0.000 | 1.211 [1.143, 1.282] |
| W3 | cicids2017 | share/reset | 0.000 | 0.860 [0.803, 0.920] |
| W3 | cicids2017 | llr/carry | 0.700 | 0.308 [0.274, 0.345] |
| W3 | cicids2017 | llr/memory | 0.400 | 0.233 [0.203, 0.265] |
| W3 | cicids2017 | llr/reset | 0.300 | 0.241 [0.211, 0.274] |
| W3 | cicids2017 | ewma/carry | 0.215 | 0.031 [0.021, 0.044] |
| W3 | cicids2017 | ewma/memory | 0.110 | 0.046 [0.033, 0.061] |
| W3 | cicids2017 | ewma/reset | 0.180 | 0.082 [0.065, 0.102] |

## R1: carry vs memory and vs reset, persistent 5%, per world, dataset and statistic (seed level, n = 10; Holm within world over 18)

**W0** — verdict: **robust** (5g_nidd: best share, carry beats both True; ciciot2023: best ewma, carry beats both True; cicids2017: best share, carry beats both True)

| dataset | statistic | vs | mean Δ [95% CI] | W / L | p (Holm) |
|---|---|---|---|---|---|
| 5g_nidd | share | memory | 0.520 [0.455, 0.590] | 10 / 0 | 0.0352 |
| 5g_nidd | share | reset | 0.710 [0.670, 0.750] | 10 / 0 | 0.0352 |
| 5g_nidd | llr | memory | 0.205 [0.120, 0.295] | 9 / 0 | 0.0352 |
| 5g_nidd | llr | reset | 0.620 [0.530, 0.710] | 10 / 0 | 0.0352 |
| 5g_nidd | ewma | memory | 0.540 [0.475, 0.600] | 10 / 0 | 0.0352 |
| 5g_nidd | ewma | reset | 0.690 [0.655, 0.725] | 10 / 0 | 0.0352 |
| ciciot2023 | share | memory | 0.490 [0.440, 0.545] | 10 / 0 | 0.0352 |
| ciciot2023 | share | reset | 0.490 [0.440, 0.540] | 10 / 0 | 0.0352 |
| ciciot2023 | llr | memory | 0.530 [0.480, 0.570] | 10 / 0 | 0.0352 |
| ciciot2023 | llr | reset | 0.505 [0.460, 0.550] | 10 / 0 | 0.0352 |
| ciciot2023 | ewma | memory | 0.495 [0.455, 0.530] | 10 / 0 | 0.0352 |
| ciciot2023 | ewma | reset | 0.480 [0.445, 0.515] | 10 / 0 | 0.0352 |
| cicids2017 | share | memory | 0.160 [0.090, 0.225] | 8 / 0 | 0.0352 |
| cicids2017 | share | reset | 0.485 [0.425, 0.535] | 10 / 0 | 0.0352 |
| cicids2017 | llr | memory | 0.330 [0.220, 0.430] | 9 / 0 | 0.0352 |
| cicids2017 | llr | reset | 0.465 [0.325, 0.595] | 9 / 0 | 0.0352 |
| cicids2017 | ewma | memory | 0.165 [0.085, 0.250] | 8 / 0 | 0.0352 |
| cicids2017 | ewma | reset | 0.495 [0.430, 0.555] | 10 / 0 | 0.0352 |

**W1** — verdict: **not robust** (5g_nidd: best llr, carry beats both False; ciciot2023: best llr, carry beats both False; cicids2017: best llr, carry beats both False)

| dataset | statistic | vs | mean Δ [95% CI] | W / L | p (Holm) |
|---|---|---|---|---|---|
| 5g_nidd | share | memory | 0.125 [0.060, 0.195] | 7 / 0 | 0.1094 |
| 5g_nidd | share | reset | 0.135 [0.065, 0.210] | 7 / 0 | 0.1094 |
| 5g_nidd | llr | memory | 0.150 [0.100, 0.195] | 9 / 0 | 0.0547 |
| 5g_nidd | llr | reset | 0.250 [0.170, 0.320] | 9 / 0 | 0.0547 |
| 5g_nidd | ewma | memory | 0.345 [0.285, 0.400] | 10 / 0 | 0.0352 |
| 5g_nidd | ewma | reset | 0.345 [0.280, 0.405] | 10 / 0 | 0.0352 |
| ciciot2023 | share | memory | 0.125 [0.080, 0.170] | 8 / 0 | 0.0859 |
| ciciot2023 | share | reset | 0.205 [0.155, 0.255] | 10 / 0 | 0.0352 |
| ciciot2023 | llr | memory | 0.210 [0.110, 0.315] | 7 / 0 | 0.1094 |
| ciciot2023 | llr | reset | 0.265 [0.160, 0.365] | 8 / 0 | 0.0859 |
| ciciot2023 | ewma | memory | 0.045 [0.000, 0.100] | 3 / 0 | 1.0000 |
| ciciot2023 | ewma | reset | -0.000 [-0.040, 0.045] | 3 / 4 | 1.0000 |
| cicids2017 | share | memory | -0.005 [-0.015, 0.000] | 0 / 1 | 1.0000 |
| cicids2017 | share | reset | 0.000 [0.000, 0.000] | 0 / 0 | 1.0000 |
| cicids2017 | llr | memory | 0.250 [0.180, 0.310] | 9 / 0 | 0.0547 |
| cicids2017 | llr | reset | 0.440 [0.365, 0.520] | 10 / 0 | 0.0352 |
| cicids2017 | ewma | memory | 0.250 [0.145, 0.350] | 8 / 0 | 0.0859 |
| cicids2017 | ewma | reset | 0.245 [0.135, 0.350] | 8 / 1 | 0.0859 |

**W2** — verdict: **not robust** (5g_nidd: best llr, carry beats both False; ciciot2023: best llr, carry beats both False; cicids2017: best llr, carry beats both False)

| dataset | statistic | vs | mean Δ [95% CI] | W / L | p (Holm) |
|---|---|---|---|---|---|
| 5g_nidd | share | memory | 0.055 [0.020, 0.090] | 5 / 0 | 0.3125 |
| 5g_nidd | share | reset | 0.100 [0.040, 0.170] | 6 / 0 | 0.2500 |
| 5g_nidd | llr | memory | 0.080 [0.035, 0.130] | 8 / 0 | 0.0859 |
| 5g_nidd | llr | reset | 0.200 [0.100, 0.300] | 9 / 0 | 0.0508 |
| 5g_nidd | ewma | memory | 0.420 [0.340, 0.500] | 10 / 0 | 0.0352 |
| 5g_nidd | ewma | reset | 0.500 [0.420, 0.575] | 10 / 0 | 0.0352 |
| ciciot2023 | share | memory | 0.080 [0.035, 0.125] | 6 / 0 | 0.2500 |
| ciciot2023 | share | reset | 0.280 [0.210, 0.350] | 10 / 0 | 0.0352 |
| ciciot2023 | llr | memory | 0.065 [0.020, 0.115] | 6 / 0 | 0.2500 |
| ciciot2023 | llr | reset | 0.165 [0.080, 0.250] | 7 / 0 | 0.1562 |
| ciciot2023 | ewma | memory | 0.020 [-0.005, 0.050] | 3 / 1 | 1.0000 |
| ciciot2023 | ewma | reset | 0.010 [-0.025, 0.045] | 3 / 2 | 1.0000 |
| cicids2017 | share | memory | 0.010 [-0.030, 0.060] | 1 / 1 | 1.0000 |
| cicids2017 | share | reset | 0.040 [0.000, 0.100] | 2 / 0 | 1.0000 |
| cicids2017 | llr | memory | 0.080 [0.040, 0.120] | 7 / 0 | 0.1562 |
| cicids2017 | llr | reset | 0.240 [0.170, 0.305] | 9 / 0 | 0.0508 |
| cicids2017 | ewma | memory | 0.435 [0.330, 0.530] | 10 / 0 | 0.0352 |
| cicids2017 | ewma | reset | 0.385 [0.295, 0.480] | 10 / 0 | 0.0352 |

**W3** — verdict: **not robust** (5g_nidd: best llr, carry beats both False; ciciot2023: best llr, carry beats both False; cicids2017: best llr, carry beats both True)

| dataset | statistic | vs | mean Δ [95% CI] | W / L | p (Holm) |
|---|---|---|---|---|---|
| 5g_nidd | share | memory | 0.110 [0.040, 0.190] | 5 / 0 | 0.4375 |
| 5g_nidd | share | reset | 0.110 [0.040, 0.190] | 5 / 0 | 0.4375 |
| 5g_nidd | llr | memory | 0.150 [0.085, 0.215] | 8 / 0 | 0.0859 |
| 5g_nidd | llr | reset | 0.235 [0.185, 0.285] | 10 / 0 | 0.0352 |
| 5g_nidd | ewma | memory | 0.390 [0.315, 0.450] | 10 / 0 | 0.0352 |
| 5g_nidd | ewma | reset | 0.375 [0.305, 0.430] | 10 / 0 | 0.0352 |
| ciciot2023 | share | memory | 0.200 [0.130, 0.295] | 10 / 0 | 0.0352 |
| ciciot2023 | share | reset | 0.260 [0.195, 0.350] | 10 / 0 | 0.0352 |
| ciciot2023 | llr | memory | 0.135 [0.055, 0.225] | 6 / 0 | 0.3125 |
| ciciot2023 | llr | reset | 0.120 [0.045, 0.205] | 6 / 0 | 0.3125 |
| ciciot2023 | ewma | memory | 0.010 [-0.020, 0.035] | 4 / 1 | 1.0000 |
| ciciot2023 | ewma | reset | -0.005 [-0.050, 0.030] | 4 / 2 | 1.0000 |
| cicids2017 | share | memory | 0.005 [0.000, 0.015] | 1 / 0 | 1.0000 |
| cicids2017 | share | reset | 0.005 [0.000, 0.015] | 1 / 0 | 1.0000 |
| cicids2017 | llr | memory | 0.300 [0.210, 0.395] | 10 / 0 | 0.0352 |
| cicids2017 | llr | reset | 0.400 [0.320, 0.490] | 10 / 0 | 0.0352 |
| cicids2017 | ewma | memory | 0.105 [0.030, 0.185] | 7 / 1 | 0.3125 |
| cicids2017 | ewma | reset | 0.035 [-0.020, 0.100] | 5 / 3 | 1.0000 |

## R2: steady-state honest false-alarm rate on test vs budget (pooled over datasets per world)

| world | budget | rule | events | device-hours | %/h [95% CI] | within budget |
|---|---|---|---|---|---|---|
| W0 | b01 | share/carry | 403 | 294000 | 0.137 [0.124, 0.151] | no |
| W0 | b01 | share/memory | 361 | 294000 | 0.123 [0.110, 0.136] | no |
| W0 | b01 | share/reset | 329 | 294000 | 0.112 [0.100, 0.125] | no |
| W0 | b01 | llr/carry | 118 | 294000 | 0.040 [0.033, 0.048] | yes |
| W0 | b01 | llr/memory | 128 | 294000 | 0.044 [0.036, 0.052] | yes |
| W0 | b01 | llr/reset | 174 | 294000 | 0.059 [0.051, 0.069] | yes |
| W0 | b01 | ewma/carry | 400 | 294000 | 0.136 [0.123, 0.150] | no |
| W0 | b01 | ewma/memory | 178 | 294000 | 0.061 [0.052, 0.070] | yes |
| W0 | b01 | ewma/reset | 241 | 294000 | 0.082 [0.072, 0.093] | yes |
| W0 | b1 | share/carry | 3430 | 294000 | 1.167 [1.128, 1.206] | no |
| W0 | b1 | share/memory | 4267 | 294000 | 1.451 [1.408, 1.496] | no |
| W0 | b1 | share/reset | 4643 | 294000 | 1.579 [1.534, 1.625] | no |
| W0 | b1 | llr/carry | 848 | 294000 | 0.288 [0.269, 0.309] | yes |
| W0 | b1 | llr/memory | 1029 | 294000 | 0.350 [0.329, 0.372] | yes |
| W0 | b1 | llr/reset | 1698 | 294000 | 0.578 [0.550, 0.606] | yes |
| W0 | b1 | ewma/carry | 3315 | 294000 | 1.128 [1.089, 1.167] | no |
| W0 | b1 | ewma/memory | 3987 | 294000 | 1.356 [1.314, 1.399] | no |
| W0 | b1 | ewma/reset | 4881 | 294000 | 1.660 [1.614, 1.707] | no |
| W1 | b01 | share/carry | 60666 | 294000 | 20.635 [20.471, 20.800] | no |
| W1 | b01 | share/memory | 18659 | 294000 | 6.347 [6.256, 6.438] | no |
| W1 | b01 | share/reset | 13116 | 294000 | 4.461 [4.385, 4.538] | no |
| W1 | b01 | llr/carry | 2665 | 294000 | 0.906 [0.872, 0.942] | no |
| W1 | b01 | llr/memory | 1712 | 294000 | 0.582 [0.555, 0.611] | no |
| W1 | b01 | llr/reset | 1206 | 294000 | 0.410 [0.387, 0.434] | no |
| W1 | b01 | ewma/carry | 275 | 294000 | 0.094 [0.083, 0.105] | yes |
| W1 | b01 | ewma/memory | 291 | 294000 | 0.099 [0.088, 0.111] | yes |
| W1 | b01 | ewma/reset | 194 | 294000 | 0.066 [0.057, 0.076] | yes |
| W1 | b1 | share/carry | 60666 | 294000 | 20.635 [20.471, 20.800] | no |
| W1 | b1 | share/memory | 18659 | 294000 | 6.347 [6.256, 6.438] | no |
| W1 | b1 | share/reset | 13116 | 294000 | 4.461 [4.385, 4.538] | no |
| W1 | b1 | llr/carry | 10364 | 294000 | 3.525 [3.458, 3.594] | no |
| W1 | b1 | llr/memory | 8185 | 294000 | 2.784 [2.724, 2.845] | no |
| W1 | b1 | llr/reset | 7178 | 294000 | 2.441 [2.385, 2.499] | no |
| W1 | b1 | ewma/carry | 3595 | 294000 | 1.223 [1.183, 1.263] | no |
| W1 | b1 | ewma/memory | 3030 | 294000 | 1.031 [0.994, 1.068] | yes |
| W1 | b1 | ewma/reset | 3326 | 294000 | 1.131 [1.093, 1.170] | no |
| W2 | b01 | share/carry | 63438 | 294000 | 21.578 [21.410, 21.746] | no |
| W2 | b01 | share/memory | 31656 | 294000 | 10.767 [10.649, 10.887] | no |
| W2 | b01 | share/reset | 11823 | 294000 | 4.021 [3.949, 4.095] | no |
| W2 | b01 | llr/carry | 2024 | 294000 | 0.688 [0.659, 0.719] | no |
| W2 | b01 | llr/memory | 2211 | 294000 | 0.752 [0.721, 0.784] | no |
| W2 | b01 | llr/reset | 1658 | 294000 | 0.564 [0.537, 0.592] | no |
| W2 | b01 | ewma/carry | 1111 | 294000 | 0.378 [0.356, 0.401] | no |
| W2 | b01 | ewma/memory | 975 | 294000 | 0.332 [0.311, 0.353] | no |
| W2 | b01 | ewma/reset | 599 | 294000 | 0.204 [0.188, 0.221] | no |
| W2 | b1 | share/carry | 63438 | 294000 | 21.578 [21.410, 21.746] | no |
| W2 | b1 | share/memory | 31656 | 294000 | 10.767 [10.649, 10.887] | no |
| W2 | b1 | share/reset | 12473 | 294000 | 4.243 [4.168, 4.318] | no |
| W2 | b1 | llr/carry | 8933 | 294000 | 3.038 [2.976, 3.102] | no |
| W2 | b1 | llr/memory | 7685 | 294000 | 2.614 [2.556, 2.673] | no |
| W2 | b1 | llr/reset | 7736 | 294000 | 2.631 [2.573, 2.691] | no |
| W2 | b1 | ewma/carry | 6462 | 294000 | 2.198 [2.145, 2.252] | no |
| W2 | b1 | ewma/memory | 5826 | 294000 | 1.982 [1.931, 2.033] | no |
| W2 | b1 | ewma/reset | 5395 | 294000 | 1.835 [1.786, 1.885] | no |
| W3 | b01 | share/carry | 55065 | 294000 | 18.730 [18.573, 18.887] | no |
| W3 | b01 | share/memory | 16448 | 294000 | 5.595 [5.509, 5.681] | no |
| W3 | b01 | share/reset | 11199 | 294000 | 3.809 [3.739, 3.880] | no |
| W3 | b01 | llr/carry | 2877 | 294000 | 0.979 [0.943, 1.015] | no |
| W3 | b01 | llr/memory | 2050 | 294000 | 0.697 [0.667, 0.728] | no |
| W3 | b01 | llr/reset | 2262 | 294000 | 0.769 [0.738, 0.802] | no |
| W3 | b01 | ewma/carry | 291 | 294000 | 0.099 [0.088, 0.111] | yes |
| W3 | b01 | ewma/memory | 267 | 294000 | 0.091 [0.080, 0.102] | yes |
| W3 | b01 | ewma/reset | 322 | 294000 | 0.110 [0.098, 0.122] | yes |
| W3 | b1 | share/carry | 55065 | 294000 | 18.730 [18.573, 18.887] | no |
| W3 | b1 | share/memory | 16448 | 294000 | 5.595 [5.509, 5.681] | no |
| W3 | b1 | share/reset | 11199 | 294000 | 3.809 [3.739, 3.880] | no |
| W3 | b1 | llr/carry | 9196 | 294000 | 3.128 [3.064, 3.192] | no |
| W3 | b1 | llr/memory | 7725 | 294000 | 2.628 [2.569, 2.687] | no |
| W3 | b1 | llr/reset | 7974 | 294000 | 2.712 [2.653, 2.772] | no |
| W3 | b1 | ewma/carry | 2999 | 294000 | 1.020 [0.984, 1.057] | yes |
| W3 | b1 | ewma/memory | 3164 | 294000 | 1.076 [1.039, 1.114] | no |
| W3 | b1 | ewma/reset | 3350 | 294000 | 1.139 [1.101, 1.179] | no |

## R3: contained share by class (persistent scenarios, main budget, pooled datasets)

| world | rule | vehicles | pedestrians | static |
|---|---|---|---|---|
| W0 | share/carry | 0.820 | 0.786 | 0.817 |
| W0 | share/memory | 0.441 | 0.535 | 0.787 |
| W0 | share/reset | 0.163 | 0.435 | 0.802 |
| W0 | llr/carry | 0.765 | 0.750 | 0.781 |
| W0 | llr/memory | 0.447 | 0.562 | 0.756 |
| W0 | llr/reset | 0.332 | 0.532 | 0.692 |
| W0 | ewma/carry | 0.828 | 0.792 | 0.825 |
| W0 | ewma/memory | 0.439 | 0.583 | 0.815 |
| W0 | ewma/reset | 0.152 | 0.487 | 0.804 |
| W1 | share/carry | 0.172 | 0.181 | 0.208 |
| W1 | share/memory | 0.020 | 0.061 | 0.248 |
| W1 | share/reset | 0.000 | 0.000 | 0.248 |
| W1 | llr/carry | 0.604 | 0.576 | 0.577 |
| W1 | llr/memory | 0.406 | 0.464 | 0.588 |
| W1 | llr/reset | 0.243 | 0.442 | 0.625 |
| W1 | ewma/carry | 0.511 | 0.458 | 0.458 |
| W1 | ewma/memory | 0.135 | 0.260 | 0.463 |
| W1 | ewma/reset | 0.000 | 0.147 | 0.527 |
| W2 | share/carry | 0.197 | 0.222 | 0.158 |
| W2 | share/memory | 0.092 | 0.139 | 0.158 |
| W2 | share/reset | 0.000 | 0.008 | 0.158 |
| W2 | llr/carry | 0.600 | 0.561 | 0.654 |
| W2 | llr/memory | 0.505 | 0.517 | 0.658 |
| W2 | llr/reset | 0.323 | 0.489 | 0.677 |
| W2 | ewma/carry | 0.538 | 0.522 | 0.556 |
| W2 | ewma/memory | 0.372 | 0.399 | 0.537 |
| W2 | ewma/reset | 0.069 | 0.358 | 0.619 |
| W3 | share/carry | 0.151 | 0.194 | 0.148 |
| W3 | share/memory | 0.013 | 0.053 | 0.175 |
| W3 | share/reset | 0.000 | 0.000 | 0.175 |
| W3 | llr/carry | 0.573 | 0.568 | 0.608 |
| W3 | llr/memory | 0.356 | 0.443 | 0.621 |
| W3 | llr/reset | 0.241 | 0.442 | 0.650 |
| W3 | ewma/carry | 0.503 | 0.446 | 0.450 |
| W3 | ewma/memory | 0.137 | 0.253 | 0.492 |
| W3 | ewma/reset | 0.000 | 0.212 | 0.573 |

## Contained share by scenario (main budget, pooled datasets)

| world | rule | handover_timed | on_off_0.2 | persistent_0.02 | persistent_0.05 | persistent_0.1 | persistent_0.2 | stealthy_0.005 |
|---|---|---|---|---|---|---|---|---|
| W0 | ewma/carry | 0.755 | 0.953 | 0.493 | 0.837 | 0.958 | 0.977 | 0.017 |
| W0 | ewma/memory | 0.573 | 0.775 | 0.118 | 0.437 | 0.738 | 0.937 | 0.003 |
| W0 | ewma/reset | 0.223 | 0.580 | 0.083 | 0.282 | 0.445 | 0.722 | 0.002 |
| W0 | llr/carry | 0.743 | 0.962 | 0.280 | 0.840 | 0.958 | 0.977 | 0.008 |
| W0 | llr/memory | 0.503 | 0.752 | 0.080 | 0.485 | 0.737 | 0.872 | 0.000 |
| W0 | llr/reset | 0.465 | 0.755 | 0.032 | 0.310 | 0.695 | 0.818 | 0.000 |
| W0 | share/carry | 0.760 | 0.955 | 0.455 | 0.838 | 0.967 | 0.977 | 0.015 |
| W0 | share/memory | 0.527 | 0.732 | 0.078 | 0.448 | 0.728 | 0.898 | 0.003 |
| W0 | share/reset | 0.312 | 0.592 | 0.082 | 0.277 | 0.413 | 0.717 | 0.002 |
| W1 | ewma/carry | 0.560 | 0.775 | 0.007 | 0.293 | 0.753 | 0.885 | 0.002 |
| W1 | ewma/memory | 0.123 | 0.285 | 0.002 | 0.080 | 0.225 | 0.645 | 0.002 |
| W1 | ewma/reset | 0.002 | 0.183 | 0.002 | 0.097 | 0.160 | 0.340 | 0.002 |
| W1 | llr/carry | 0.638 | 0.753 | 0.185 | 0.635 | 0.735 | 0.807 | 0.033 |
| W1 | llr/memory | 0.485 | 0.648 | 0.100 | 0.432 | 0.595 | 0.712 | 0.020 |
| W1 | llr/reset | 0.352 | 0.572 | 0.092 | 0.317 | 0.498 | 0.610 | 0.010 |
| W1 | share/carry | 0.185 | 0.207 | 0.142 | 0.153 | 0.188 | 0.245 | 0.158 |
| W1 | share/memory | 0.065 | 0.083 | 0.053 | 0.072 | 0.078 | 0.108 | 0.063 |
| W1 | share/reset | 0.028 | 0.052 | 0.028 | 0.040 | 0.053 | 0.077 | 0.037 |
| W2 | ewma/carry | 0.368 | 0.807 | 0.012 | 0.430 | 0.793 | 0.913 | 0.010 |
| W2 | ewma/memory | 0.157 | 0.645 | 0.003 | 0.138 | 0.633 | 0.877 | 0.007 |
| W2 | ewma/reset | 0.005 | 0.348 | 0.017 | 0.132 | 0.363 | 0.552 | 0.007 |
| W2 | llr/carry | 0.502 | 0.782 | 0.218 | 0.585 | 0.727 | 0.867 | 0.008 |
| W2 | llr/memory | 0.463 | 0.730 | 0.183 | 0.510 | 0.648 | 0.815 | 0.013 |
| W2 | llr/reset | 0.268 | 0.623 | 0.142 | 0.383 | 0.537 | 0.713 | 0.012 |
| W2 | share/carry | 0.187 | 0.160 | 0.147 | 0.172 | 0.218 | 0.250 | 0.142 |
| W2 | share/memory | 0.110 | 0.113 | 0.088 | 0.123 | 0.120 | 0.147 | 0.080 |
| W2 | share/reset | 0.032 | 0.043 | 0.023 | 0.032 | 0.045 | 0.037 | 0.035 |
| W3 | ewma/carry | 0.410 | 0.778 | 0.003 | 0.242 | 0.723 | 0.933 | 0.000 |
| W3 | ewma/memory | 0.032 | 0.295 | 0.003 | 0.073 | 0.237 | 0.657 | 0.000 |
| W3 | ewma/reset | 0.002 | 0.235 | 0.005 | 0.107 | 0.218 | 0.383 | 0.002 |
| W3 | llr/carry | 0.525 | 0.745 | 0.168 | 0.568 | 0.738 | 0.840 | 0.012 |
| W3 | llr/memory | 0.378 | 0.633 | 0.095 | 0.373 | 0.555 | 0.717 | 0.012 |
| W3 | llr/reset | 0.257 | 0.577 | 0.095 | 0.317 | 0.470 | 0.650 | 0.012 |
| W3 | share/carry | 0.140 | 0.168 | 0.158 | 0.172 | 0.143 | 0.180 | 0.120 |
| W3 | share/memory | 0.045 | 0.055 | 0.055 | 0.067 | 0.053 | 0.055 | 0.067 |
| W3 | share/reset | 0.027 | 0.033 | 0.033 | 0.047 | 0.028 | 0.032 | 0.038 |

## Exposure (share of attack flows not blocked within 60 min), persistent 5%, with and without per-flow blocking of flagged flows

| world | rule | exposure | exposure_blocking |
|---|---|---|---|
| W0 | ewma/carry | 0.396 | 0.118 |
| W0 | ewma/memory | 0.754 | 0.162 |
| W0 | ewma/reset | 0.788 | 0.167 |
| W0 | llr/carry | 0.428 | 0.120 |
| W0 | llr/memory | 0.713 | 0.158 |
| W0 | llr/reset | 0.783 | 0.169 |
| W0 | share/carry | 0.464 | 0.127 |
| W0 | share/memory | 0.743 | 0.163 |
| W0 | share/reset | 0.797 | 0.168 |
| W1 | ewma/carry | 0.911 | 0.186 |
| W1 | ewma/memory | 0.972 | 0.189 |
| W1 | ewma/reset | 0.957 | 0.188 |
| W1 | llr/carry | 0.432 | 0.121 |
| W1 | llr/memory | 0.573 | 0.142 |
| W1 | llr/reset | 0.635 | 0.147 |
| W1 | share/carry | 0.926 | 0.179 |
| W1 | share/memory | 0.963 | 0.180 |
| W1 | share/reset | 0.974 | 0.184 |
| W2 | ewma/carry | 0.839 | 0.179 |
| W2 | ewma/memory | 0.946 | 0.186 |
| W2 | ewma/reset | 0.936 | 0.184 |
| W2 | llr/carry | 0.489 | 0.118 |
| W2 | llr/memory | 0.529 | 0.124 |
| W2 | llr/reset | 0.587 | 0.132 |
| W2 | share/carry | 0.882 | 0.173 |
| W2 | share/memory | 0.923 | 0.175 |
| W2 | share/reset | 0.984 | 0.185 |
| W3 | ewma/carry | 0.947 | 0.194 |
| W3 | ewma/memory | 0.983 | 0.196 |
| W3 | ewma/reset | 0.947 | 0.192 |
| W3 | llr/carry | 0.507 | 0.125 |
| W3 | llr/memory | 0.654 | 0.151 |
| W3 | llr/reset | 0.678 | 0.155 |
| W3 | share/carry | 0.913 | 0.185 |
| W3 | share/memory | 0.962 | 0.191 |
| W3 | share/reset | 0.973 | 0.194 |

## Amendment A2: placements at equal false-alarm rates matched on test (persistent 5%)

**A2 W0, equal test false-alarm rate 0.1%/h** — verdict: **robust** (5g_nidd: best llr, beats both True; ciciot2023: best llr, beats both True; cicids2017: best share, beats both True)

| dataset | statistic | vs | carry | other | mean Δ (seed) | W / L | p (Holm) |
|---|---|---|---|---|---|---|---|
| 5g_nidd | share | memory | 0.965 | 0.380 | 0.585 | 10 / 0 | 0.0352 |
| 5g_nidd | share | reset | 0.965 | 0.275 | 0.690 | 10 / 0 | 0.0352 |
| 5g_nidd | llr | memory | 1.000 | 0.795 | 0.205 | 9 / 0 | 0.0352 |
| 5g_nidd | llr | reset | 1.000 | 0.360 | 0.640 | 10 / 0 | 0.0352 |
| 5g_nidd | ewma | memory | 0.945 | 0.470 | 0.475 | 10 / 0 | 0.0352 |
| 5g_nidd | ewma | reset | 0.945 | 0.310 | 0.635 | 10 / 0 | 0.0352 |
| ciciot2023 | share | memory | 0.645 | 0.140 | 0.505 | 10 / 0 | 0.0352 |
| ciciot2023 | share | reset | 0.645 | 0.140 | 0.505 | 10 / 0 | 0.0352 |
| ciciot2023 | llr | memory | 0.765 | 0.195 | 0.570 | 10 / 0 | 0.0352 |
| ciciot2023 | llr | reset | 0.765 | 0.175 | 0.590 | 10 / 0 | 0.0352 |
| ciciot2023 | ewma | memory | 0.640 | 0.155 | 0.485 | 10 / 0 | 0.0352 |
| ciciot2023 | ewma | reset | 0.640 | 0.160 | 0.480 | 10 / 0 | 0.0352 |
| cicids2017 | share | memory | 0.890 | 0.775 | 0.115 | 7 / 0 | 0.0352 |
| cicids2017 | share | reset | 0.890 | 0.415 | 0.475 | 10 / 0 | 0.0352 |
| cicids2017 | llr | memory | 0.865 | 0.775 | 0.090 | 5 / 0 | 0.0625 |
| cicids2017 | llr | reset | 0.865 | 0.490 | 0.375 | 9 / 0 | 0.0352 |
| cicids2017 | ewma | memory | 0.890 | 0.775 | 0.115 | 8 / 0 | 0.0352 |
| cicids2017 | ewma | reset | 0.890 | 0.415 | 0.475 | 10 / 0 | 0.0352 |

**A2 W0, equal test false-alarm rate 1%/h** — verdict: **not robust** (5g_nidd: best llr, beats both False; ciciot2023: best llr, beats both True; cicids2017: best share, beats both False)

| dataset | statistic | vs | carry | other | mean Δ (seed) | W / L | p (Holm) |
|---|---|---|---|---|---|---|---|
| 5g_nidd | share | memory | 0.995 | 0.795 | 0.200 | 10 / 0 | 0.0352 |
| 5g_nidd | share | reset | 0.995 | 0.440 | 0.555 | 10 / 0 | 0.0352 |
| 5g_nidd | llr | memory | 1.000 | 0.900 | 0.100 | 7 / 0 | 0.0625 |
| 5g_nidd | llr | reset | 1.000 | 0.650 | 0.350 | 10 / 0 | 0.0352 |
| 5g_nidd | ewma | memory | 0.990 | 0.765 | 0.225 | 10 / 0 | 0.0352 |
| 5g_nidd | ewma | reset | 0.990 | 0.390 | 0.600 | 10 / 0 | 0.0352 |
| ciciot2023 | share | memory | 0.830 | 0.230 | 0.600 | 10 / 0 | 0.0352 |
| ciciot2023 | share | reset | 0.830 | 0.195 | 0.635 | 10 / 0 | 0.0352 |
| ciciot2023 | llr | memory | 0.860 | 0.345 | 0.515 | 10 / 0 | 0.0352 |
| ciciot2023 | llr | reset | 0.860 | 0.245 | 0.615 | 10 / 0 | 0.0352 |
| ciciot2023 | ewma | memory | 0.845 | 0.335 | 0.510 | 10 / 0 | 0.0352 |
| ciciot2023 | ewma | reset | 0.845 | 0.215 | 0.630 | 10 / 0 | 0.0352 |
| cicids2017 | share | memory | 0.890 | 0.865 | 0.025 | 3 / 0 | 0.2500 |
| cicids2017 | share | reset | 0.890 | 0.545 | 0.345 | 10 / 0 | 0.0352 |
| cicids2017 | llr | memory | 0.865 | 0.835 | 0.030 | 4 / 0 | 0.2500 |
| cicids2017 | llr | reset | 0.865 | 0.670 | 0.195 | 8 / 0 | 0.0391 |
| cicids2017 | ewma | memory | 0.890 | 0.820 | 0.070 | 6 / 0 | 0.0938 |
| cicids2017 | ewma | reset | 0.890 | 0.465 | 0.425 | 10 / 0 | 0.0352 |

**A2 W1, equal test false-alarm rate 0.1%/h** — verdict: **not robust** (5g_nidd: best llr, beats both False; ciciot2023: best llr, beats both False; cicids2017: best llr, beats both True)

| dataset | statistic | vs | carry | other | mean Δ (seed) | W / L | p (Holm) |
|---|---|---|---|---|---|---|---|
| 5g_nidd | share | memory | 0.185 | 0.060 | 0.125 | 7 / 0 | 0.0938 |
| 5g_nidd | share | reset | 0.185 | 0.050 | 0.135 | 7 / 0 | 0.0938 |
| 5g_nidd | llr | memory | 0.715 | 0.515 | 0.200 | 9 / 0 | 0.0508 |
| 5g_nidd | llr | reset | 0.715 | 0.415 | 0.300 | 10 / 0 | 0.0352 |
| 5g_nidd | ewma | memory | 0.320 | 0.080 | 0.240 | 10 / 0 | 0.0352 |
| 5g_nidd | ewma | reset | 0.320 | 0.175 | 0.145 | 9 / 1 | 0.0703 |
| ciciot2023 | share | memory | 0.265 | 0.140 | 0.125 | 8 / 0 | 0.0859 |
| ciciot2023 | share | reset | 0.265 | 0.060 | 0.205 | 10 / 0 | 0.0352 |
| ciciot2023 | llr | memory | 0.355 | 0.180 | 0.175 | 8 / 0 | 0.0859 |
| ciciot2023 | llr | reset | 0.355 | 0.115 | 0.240 | 8 / 0 | 0.0859 |
| ciciot2023 | ewma | memory | 0.060 | 0.000 | 0.060 | 4 / 0 | 0.5000 |
| ciciot2023 | ewma | reset | 0.060 | 0.045 | 0.015 | 4 / 4 | 1.0000 |
| cicids2017 | share | memory | 0.010 | 0.015 | -0.005 | 0 / 1 | 1.0000 |
| cicids2017 | share | reset | 0.010 | 0.010 | 0.000 | 0 / 0 | 1.0000 |
| cicids2017 | llr | memory | 0.455 | 0.225 | 0.230 | 10 / 0 | 0.0352 |
| cicids2017 | llr | reset | 0.455 | 0.220 | 0.235 | 10 / 0 | 0.0352 |
| cicids2017 | ewma | memory | 0.395 | 0.150 | 0.245 | 8 / 0 | 0.0859 |
| cicids2017 | ewma | reset | 0.395 | 0.155 | 0.240 | 8 / 1 | 0.0859 |

**A2 W1, equal test false-alarm rate 1%/h** — verdict: **robust** (5g_nidd: best ewma, beats both True; ciciot2023: best llr, beats both False; cicids2017: best ewma, beats both True)

| dataset | statistic | vs | carry | other | mean Δ (seed) | W / L | p (Holm) |
|---|---|---|---|---|---|---|---|
| 5g_nidd | share | memory | 0.185 | 0.060 | 0.125 | 7 / 0 | 0.0781 |
| 5g_nidd | share | reset | 0.185 | 0.050 | 0.135 | 7 / 0 | 0.0781 |
| 5g_nidd | llr | memory | 0.790 | 0.650 | 0.140 | 9 / 0 | 0.0391 |
| 5g_nidd | llr | reset | 0.790 | 0.570 | 0.220 | 9 / 0 | 0.0391 |
| 5g_nidd | ewma | memory | 0.800 | 0.250 | 0.550 | 10 / 0 | 0.0352 |
| 5g_nidd | ewma | reset | 0.800 | 0.215 | 0.585 | 10 / 0 | 0.0352 |
| ciciot2023 | share | memory | 0.265 | 0.140 | 0.125 | 8 / 0 | 0.0547 |
| ciciot2023 | share | reset | 0.265 | 0.060 | 0.205 | 10 / 0 | 0.0352 |
| ciciot2023 | llr | memory | 0.440 | 0.310 | 0.130 | 7 / 0 | 0.0781 |
| ciciot2023 | llr | reset | 0.440 | 0.245 | 0.195 | 8 / 1 | 0.0547 |
| ciciot2023 | ewma | memory | 0.355 | 0.050 | 0.305 | 10 / 0 | 0.0352 |
| ciciot2023 | ewma | reset | 0.355 | 0.135 | 0.220 | 10 / 0 | 0.0352 |
| cicids2017 | share | memory | 0.010 | 0.015 | -0.005 | 0 / 1 | 1.0000 |
| cicids2017 | share | reset | 0.010 | 0.010 | 0.000 | 0 / 0 | 1.0000 |
| cicids2017 | llr | memory | 0.685 | 0.460 | 0.225 | 9 / 0 | 0.0391 |
| cicids2017 | llr | reset | 0.685 | 0.355 | 0.330 | 10 / 0 | 0.0352 |
| cicids2017 | ewma | memory | 0.735 | 0.255 | 0.480 | 10 / 0 | 0.0352 |
| cicids2017 | ewma | reset | 0.735 | 0.185 | 0.550 | 10 / 0 | 0.0352 |

**A2 W2, equal test false-alarm rate 0.1%/h** — verdict: **not robust** (5g_nidd: best llr, beats both False; ciciot2023: best llr, beats both False; cicids2017: best llr, beats both False)

| dataset | statistic | vs | carry | other | mean Δ (seed) | W / L | p (Holm) |
|---|---|---|---|---|---|---|---|
| 5g_nidd | share | memory | 0.105 | 0.050 | 0.055 | 5 / 0 | 0.4375 |
| 5g_nidd | share | reset | 0.105 | 0.005 | 0.100 | 6 / 0 | 0.3125 |
| 5g_nidd | llr | memory | 0.700 | 0.615 | 0.085 | 6 / 0 | 0.3125 |
| 5g_nidd | llr | reset | 0.700 | 0.470 | 0.230 | 8 / 0 | 0.1094 |
| 5g_nidd | ewma | memory | 0.095 | 0.015 | 0.080 | 7 / 0 | 0.1875 |
| 5g_nidd | ewma | reset | 0.095 | 0.135 | -0.040 | 2 / 4 | 1.0000 |
| ciciot2023 | share | memory | 0.350 | 0.270 | 0.080 | 6 / 0 | 0.3125 |
| ciciot2023 | share | reset | 0.350 | 0.070 | 0.280 | 10 / 0 | 0.0352 |
| ciciot2023 | llr | memory | 0.270 | 0.210 | 0.060 | 5 / 0 | 0.4375 |
| ciciot2023 | llr | reset | 0.270 | 0.120 | 0.150 | 7 / 0 | 0.1875 |
| ciciot2023 | ewma | memory | 0.030 | 0.015 | 0.015 | 3 / 1 | 1.0000 |
| ciciot2023 | ewma | reset | 0.030 | 0.065 | -0.035 | 2 / 6 | 1.0000 |
| cicids2017 | share | memory | 0.060 | 0.050 | 0.010 | 1 / 1 | 1.0000 |
| cicids2017 | share | reset | 0.060 | 0.020 | 0.040 | 2 / 0 | 1.0000 |
| cicids2017 | llr | memory | 0.535 | 0.400 | 0.135 | 9 / 0 | 0.0664 |
| cicids2017 | llr | reset | 0.535 | 0.300 | 0.235 | 9 / 0 | 0.0664 |
| cicids2017 | ewma | memory | 0.350 | 0.125 | 0.225 | 9 / 0 | 0.0664 |
| cicids2017 | ewma | reset | 0.350 | 0.150 | 0.200 | 8 / 0 | 0.1094 |

**A2 W2, equal test false-alarm rate 1%/h** — verdict: **not robust** (5g_nidd: best llr, beats both False; ciciot2023: best ewma, beats both False; cicids2017: best ewma, beats both True)

| dataset | statistic | vs | carry | other | mean Δ (seed) | W / L | p (Holm) |
|---|---|---|---|---|---|---|---|
| 5g_nidd | share | memory | 0.105 | 0.050 | 0.055 | 5 / 0 | 0.3125 |
| 5g_nidd | share | reset | 0.105 | 0.005 | 0.100 | 6 / 0 | 0.2500 |
| 5g_nidd | llr | memory | 0.760 | 0.730 | 0.030 | 5 / 0 | 0.3125 |
| 5g_nidd | llr | reset | 0.760 | 0.600 | 0.160 | 9 / 0 | 0.0508 |
| 5g_nidd | ewma | memory | 0.750 | 0.290 | 0.460 | 10 / 0 | 0.0352 |
| 5g_nidd | ewma | reset | 0.750 | 0.270 | 0.480 | 10 / 0 | 0.0352 |
| ciciot2023 | share | memory | 0.350 | 0.270 | 0.080 | 6 / 0 | 0.2500 |
| ciciot2023 | share | reset | 0.350 | 0.070 | 0.280 | 10 / 0 | 0.0352 |
| ciciot2023 | llr | memory | 0.340 | 0.290 | 0.050 | 5 / 1 | 0.4688 |
| ciciot2023 | llr | reset | 0.340 | 0.240 | 0.100 | 6 / 1 | 0.2500 |
| ciciot2023 | ewma | memory | 0.395 | 0.155 | 0.240 | 9 / 0 | 0.0508 |
| ciciot2023 | ewma | reset | 0.395 | 0.155 | 0.240 | 9 / 0 | 0.0508 |
| cicids2017 | share | memory | 0.060 | 0.050 | 0.010 | 1 / 1 | 1.0000 |
| cicids2017 | share | reset | 0.060 | 0.025 | 0.035 | 2 / 0 | 1.0000 |
| cicids2017 | llr | memory | 0.700 | 0.575 | 0.125 | 8 / 0 | 0.0703 |
| cicids2017 | llr | reset | 0.700 | 0.465 | 0.235 | 9 / 0 | 0.0508 |
| cicids2017 | ewma | memory | 0.745 | 0.450 | 0.295 | 10 / 0 | 0.0352 |
| cicids2017 | ewma | reset | 0.745 | 0.285 | 0.460 | 10 / 0 | 0.0352 |

**A2 W3, equal test false-alarm rate 0.1%/h** — verdict: **robust** (5g_nidd: best llr, beats both True; ciciot2023: best llr, beats both False; cicids2017: best llr, beats both True)

| dataset | statistic | vs | carry | other | mean Δ (seed) | W / L | p (Holm) |
|---|---|---|---|---|---|---|---|
| 5g_nidd | share | memory | 0.160 | 0.050 | 0.110 | 5 / 0 | 0.3125 |
| 5g_nidd | share | reset | 0.160 | 0.050 | 0.110 | 5 / 0 | 0.3125 |
| 5g_nidd | llr | memory | 0.535 | 0.275 | 0.260 | 9 / 0 | 0.0391 |
| 5g_nidd | llr | reset | 0.535 | 0.220 | 0.315 | 10 / 0 | 0.0352 |
| 5g_nidd | ewma | memory | 0.210 | 0.050 | 0.160 | 10 / 0 | 0.0352 |
| 5g_nidd | ewma | reset | 0.210 | 0.105 | 0.105 | 9 / 1 | 0.1230 |
| ciciot2023 | share | memory | 0.350 | 0.150 | 0.200 | 10 / 0 | 0.0352 |
| ciciot2023 | share | reset | 0.350 | 0.090 | 0.260 | 10 / 0 | 0.0352 |
| ciciot2023 | llr | memory | 0.325 | 0.160 | 0.165 | 7 / 0 | 0.1250 |
| ciciot2023 | llr | reset | 0.325 | 0.140 | 0.185 | 7 / 0 | 0.1250 |
| ciciot2023 | ewma | memory | 0.085 | 0.015 | 0.070 | 8 / 1 | 0.1250 |
| ciciot2023 | ewma | reset | 0.085 | 0.070 | 0.015 | 7 / 3 | 1.0000 |
| cicids2017 | share | memory | 0.005 | 0.000 | 0.005 | 1 / 0 | 1.0000 |
| cicids2017 | share | reset | 0.005 | 0.000 | 0.005 | 1 / 0 | 1.0000 |
| cicids2017 | llr | memory | 0.655 | 0.330 | 0.325 | 10 / 0 | 0.0352 |
| cicids2017 | llr | reset | 0.655 | 0.230 | 0.425 | 10 / 0 | 0.0352 |
| cicids2017 | ewma | memory | 0.450 | 0.160 | 0.290 | 10 / 0 | 0.0352 |
| cicids2017 | ewma | reset | 0.450 | 0.180 | 0.270 | 10 / 0 | 0.0352 |

**A2 W3, equal test false-alarm rate 1%/h** — verdict: **robust** (5g_nidd: best ewma, beats both True; ciciot2023: best ewma, beats both True; cicids2017: best ewma, beats both True)

| dataset | statistic | vs | carry | other | mean Δ (seed) | W / L | p (Holm) |
|---|---|---|---|---|---|---|---|
| 5g_nidd | share | memory | 0.160 | 0.050 | 0.110 | 5 / 0 | 0.2500 |
| 5g_nidd | share | reset | 0.160 | 0.050 | 0.110 | 5 / 0 | 0.2500 |
| 5g_nidd | llr | memory | 0.685 | 0.530 | 0.155 | 9 / 0 | 0.0352 |
| 5g_nidd | llr | reset | 0.685 | 0.450 | 0.235 | 10 / 0 | 0.0352 |
| 5g_nidd | ewma | memory | 0.775 | 0.240 | 0.535 | 10 / 0 | 0.0352 |
| 5g_nidd | ewma | reset | 0.775 | 0.185 | 0.590 | 10 / 0 | 0.0352 |
| ciciot2023 | share | memory | 0.350 | 0.150 | 0.200 | 10 / 0 | 0.0352 |
| ciciot2023 | share | reset | 0.350 | 0.090 | 0.260 | 10 / 0 | 0.0352 |
| ciciot2023 | llr | memory | 0.385 | 0.240 | 0.145 | 9 / 0 | 0.0352 |
| ciciot2023 | llr | reset | 0.385 | 0.185 | 0.200 | 10 / 0 | 0.0352 |
| ciciot2023 | ewma | memory | 0.415 | 0.130 | 0.285 | 10 / 0 | 0.0352 |
| ciciot2023 | ewma | reset | 0.415 | 0.140 | 0.275 | 10 / 0 | 0.0352 |
| cicids2017 | share | memory | 0.005 | 0.000 | 0.005 | 1 / 0 | 1.0000 |
| cicids2017 | share | reset | 0.005 | 0.005 | 0.000 | 1 / 1 | 1.0000 |
| cicids2017 | llr | memory | 0.730 | 0.560 | 0.170 | 10 / 0 | 0.0352 |
| cicids2017 | llr | reset | 0.730 | 0.400 | 0.330 | 10 / 0 | 0.0352 |
| cicids2017 | ewma | memory | 0.810 | 0.335 | 0.475 | 10 / 0 | 0.0352 |
| cicids2017 | ewma | reset | 0.810 | 0.190 | 0.620 | 10 / 0 | 0.0352 |

