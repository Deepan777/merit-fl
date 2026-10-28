# MERIT-FL

Code and generated data for **MERIT-FL** (mobility- and resource-aware zero-trust federated intrusion detection), a
simulation study of how intrusion evidence about IoT devices should be kept when devices move between mobile edge nodes.

- **Containment.** Each edge node scores the flows a device sends and accumulates a per-device cumulative-sum (CUSUM)
  statistic of the flagged flows against a reference for the device's benign traffic.
- **Handover.** The statistic is carried across handover in shared context and in an authenticated token.
- **Comparisons.** Carry-over is compared with per-node memory and with a reset at every edge change, at equal
  false-containment budgets.
- **Training.** Edge nodes select devices by their predicted residence in the cell and forward the updates of devices
  that hand over during a round.

## Repository layout

| Path | Contents |
|---|---|
| `src/v2/` | Mobility and radio model, hierarchical federated training, containment, robustness statistics |
| `src/data`, `src/models`, `src/federated`, `src/evaluation`, `src/utils` | Preprocessing, CNN-BiLSTM detector, aggregation, metrics and statistics |
| `scripts/` | Trace generation, runs, calibration, robustness studies, real-flag replay and analyses |
| `configs/v2/` | Frozen configurations chosen on the validation split |
| `experiments/v2_registry.csv` | Registry of every training run |
| `data/v2_traces/` | SUMO mobility traces (`seed*.npz`) used by all runs |
| `results/v2/` | Run outputs and robustness results (JSON / gzip JSON / NPZ), including per-run detector weights |
| `reports/` | Statistical results (`reports/stats/`) and calibration tables |
| `tests/` | Unit tests |

## Setup

Python 3.11. The exact environment is in `requirements.lock.txt`; a minimal set is in `requirements.txt`.

```bash
python -m venv venv
venv/Scripts/pip install -r requirements.txt
python -m pytest -q tests
```

PyTorch with CUDA can be installed from the PyTorch index; all experiments also run on the CPU. Mobility traces were
generated with SUMO 1.27.1 (`SUMO_HOME` must point to the installation to regenerate them).

## Data

See `data/README.md`.
- **Included.** The generated data: mobility traces, run outputs, detector flag profiles, robustness results and the
  per-window flag counts of the real-flag replay.
- **Not redistributed.** The public intrusion-detection datasets (5G-NIDD, CICIoT2023 and CICIDS2017). The scripts
  download or read them from their original sources.

## Reproducing the results

`reproduce.sh` lists every step, from data preparation to the statistical reports.
- **Analyses.** They read the included results, for example `python scripts/v2_r9_analyze.py`, and regenerate the
  reports in `reports/stats/` without new runs.
- **Simulation runs.** Re-running them requires the processed datasets.

Every run records its resolved configuration, code version and package versions (`env.json`). The `git_commit` values
refer to the development repository; this repository contains the same code.

## Licence

MIT (see `LICENSE`). The third-party datasets remain under their own licences.
