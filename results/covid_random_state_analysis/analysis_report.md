# Covid3MonthDiscrete Random-State Analysis

- Input: `results\covid_loop_random_state_results.csv`
- Rows: 1001
- Unique seeds: 1001
- Seed range: 0–1000
- Dataset(s): Covid3MonthDiscrete
- Resample ID(s): 0

## Metric summary

|                   |       count |     mean |      std |      min |   median |      max |
|:------------------|------------:|---------:|---------:|---------:|---------:|---------:|
| accuracy          | 1001.000000 | 0.727600 | 0.018668 | 0.655738 | 0.721311 | 0.770492 |
| balanced_accuracy | 1001.000000 | 0.633973 | 0.025301 | 0.540237 | 0.635008 | 0.692810 |
| log_loss          | 1001.000000 | 0.745640 | 0.012250 | 0.707890 | 0.745805 | 0.784277 |
| auc               | 1001.000000 | 0.779988 | 0.009106 | 0.754452 | 0.779875 | 0.809533 |
| fit_time_s        | 1001.000000 | 0.251707 | 0.034963 | 0.202000 | 0.244000 | 0.566000 |
| predict_time_s    | 1001.000000 | 0.092162 | 0.015645 | 0.070000 | 0.091000 | 0.291000 |

## Accuracy extremes

- Best: seed 69, accuracy 0.770492
- Worst: seed 739, accuracy 0.655738
- Best log loss: seed 857, log loss 0.707890

## Validation

- Duplicate run keys: 0
- Missing metric values: 0

These experiments vary classifier random_state on the same recorded dataset/resample. They do not measure variation across different train/test resamples.