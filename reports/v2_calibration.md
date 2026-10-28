# v2 calibration (validation split, trace seeds 100-102)

Energy-gated random participation without forwarding; holdout macro-F1, mean of the last three checkpoints; mean over seeds (min-max).

| dataset | lr 0.001, epochs 1 | lr 0.001, epochs 3 | lr 0.003, epochs 1 | lr 0.003, epochs 3 | chosen |
|---|---|---|---|---|---|
| 5g_nidd | 0.627 (0.538-0.676, n=3) | 0.701 (0.686-0.720, n=3) | 0.701 (0.692-0.717, n=3) | 0.736 (0.729-0.744, n=3) | set 3 |
| ciciot2023 | 0.345 (0.322-0.358, n=3) | 0.447 (0.403-0.476, n=3) | 0.412 (0.374-0.461, n=3) | 0.462 (0.397-0.503, n=3) | set 3 |
| cicids2017 | 0.270 (0.234-0.290, n=3) | 0.504 (0.484-0.543, n=3) | 0.392 (0.339-0.483, n=3) | 0.617 (0.572-0.672, n=3) | set 3 |
