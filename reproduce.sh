#!/usr/bin/env bash
# Reproduce the MERIT-FL v2 results from scratch. Run from the repository root with the environment of
# requirements.lock.txt (Python 3.11, torch 2.5.1). CPU only; about two days of compute on a 16-thread laptop.
# The analyses at the end of each block also run directly on the included results (results/v2), without new runs.
set -euo pipefail
PY=${PY:-python}

# 1. Data (data/README.md): 5G-NIDD from Fairdata; CICIoT2023 and CICIDS2017 downloaded manually from the CIC
$PY scripts/fetch_5g_nidd.py                       # downloads and verifies SHA-256 (data/5g_nidd/raw)
$PY scripts/preprocess.py --dataset 5g_nidd
$PY scripts/preprocess.py --dataset ciciot2023 --raw "$CICIOT2023_RAW"     # CSV folder or zip
$PY scripts/preprocess.py --dataset cicids2017 --raw "$CICIDS2017_RAW"     # MachineLearningCSV.zip
$PY scripts/leakage_audit.py --dataset 5g_nidd --out reports/leakage_audit_5g_nidd

# 2. Mobility traces (SUMO 1.27.1; SUMO_HOME must point to the installation). Included in data/v2_traces.
$PY scripts/v2_traces.py --seeds 100 101 102 42 43 44 45 46 47 48 49 50 51

# 3. Run plan and calibration on the validation split (trace seeds 100-102)
$PY scripts/v2_run.py plan
$PY scripts/v2_run.py run --groups cal_hfl --workers 2
$PY scripts/v2_freeze.py                           # configs/v2/<dataset>.yaml
$PY scripts/v2_containment.py cal                  # configs/v2/containment.yaml

# 4. Test runs (trace seeds 42-51), each configuration once
$PY scripts/v2_run.py run --groups test_hfl test_round --workers 2 --first-arm random
$PY scripts/v2_containment.py test

# 5. Pre-specified analysis
$PY scripts/v2_analyze.py                          # reports/stats/V2_STATS.md, v2_stats.json

# 6. Robustness (post hoc)
$PY scripts/v2_containment.py test --arm "residence+fw"
$PY scripts/v2_reanalysis.py                       # reports/stats/V2_REANALYSIS.md
$PY scripts/v2_robust.py cache                     # detector flag profiles (results/v2/robust/cache)
$PY scripts/v2_robust.py cal                       # validation-calibrated thresholds
for w in W0 W1 W2 W3; do $PY scripts/v2_robust.py test --worlds $w; $PY scripts/v2_robust.py curves --worlds $w; done
$PY scripts/v2_robust_analyze.py                   # reports/stats/V2_ROBUST.md
$PY scripts/v2_overhead.py                         # reports/stats/v2_overhead.json

# 6b. Type reference with the budget-derived threshold (original traces)
$PY scripts/v2_r7.py cal
for w in W0 W1 W2 W3 W4; do $PY scripts/v2_r7.py test --worlds $w; $PY scripts/v2_r7.py curves --worlds $w; done
$PY scripts/v2_r7_analyze.py                       # reports/stats/V2_R7.md

# 6c. Fresh 6-h traces, device reference, stress tests, comparators
$PY scripts/v2_traces.py --seeds 52 53 54 55 56 57 58 59 60 61 --horizon 21600
$PY scripts/v2_r8.py cal && $PY scripts/v2_r8.py calabl
$PY scripts/v2_r8_queue.py                         # main (chained, unchained), ablation, sweep, curves, rates
$PY scripts/v2_r8_analyze.py                       # reports/stats/V2_R8.md

# 6d. Real benign-flow replay (5G-NIDD per-session files) and the robust increment
$PY scripts/v2_r9.py cal                           # robust increment chosen on validation -> configs/v2/r9.yaml
$PY scripts/v2_r9.py prep && $PY scripts/v2_r9.py flags   # cross-fitted detectors, per-window flags
$PY scripts/v2_r9.py describe
$PY scripts/v2_r9.py sweep
$PY scripts/v2_r9.py replay
$PY scripts/v2_r9.py replaycurves && $PY scripts/v2_r9.py replaycurves --share 0.2
$PY scripts/v2_r9_analyze.py                       # reports/stats/V2_R9.md
