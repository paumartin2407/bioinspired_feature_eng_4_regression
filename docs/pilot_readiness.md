# RQ1 pilot health check

Before the dataset scope was reduced to California Housing and Diabetes Progression, the reduced RQ1 pilot ran on five prespecified exploration datasets with one outer split, three search seeds and 300 optimizer evaluations per FE arm. These historical counts do not describe a run under the current two-dataset manifest. Generated artifacts are local under `results/rq1_pilot/` and are ignored by Git. No pilot predictive score was used to choose or alter a full-study configuration.

- 50 output rows: five no-FE baselines and 45 FE runs.
- 13,500 optimizer candidate evaluations and 40,545 FE model fits, including final refits.
- Zero failed candidates and zero failed final fits.
- Longest single FE arm runtime: 11.1 seconds on this machine.
- A saved California Housing construction run reproduced its final development features, test features, predictions and MAE exactly from the source file and saved artifacts.

| Arm | Selected `k` across 15 runs | Maximum arm runtime |
| --- | --- | ---: |
| Selection | `0`: 15 | 5.4 s |
| Construction | `1`: 2, `4`: 4, `8`: 9 | 11.1 s |
| Combined | `1`: 2, `2`: 3, `4`: 3, `8`: 7 | 7.3 s |

These are workflow diagnostics only. Frequent selection of `k=8` does not justify expanding the candidate grid. The full RQ1 config, candidate grid and budget remain unchanged; the pilot does not support a feature-engineering recommendation.
