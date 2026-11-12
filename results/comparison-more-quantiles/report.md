# Benchmark Comparison: more_quantiles_d1-2 vs `quant_base` (Resample 0)

**Seeds per method:** `{'more_quantiles_d1-2': 3, 'quant_base': 3}`  
**Notes:** $\Delta = \text{method} - \text{baseline}$. Metrics are direction-aware (lower is better for log loss, Brier, ECE, and fit/predict execution times).

## Summary Results

| Method              | Metric                     |   Datasets |   Baseline Mean |   Method Mean |   Mean Δ | 95% Bootstrap CI     | Win / Draw / Loss   |   Wilcoxon p (Holm) |
|:--------------------|:---------------------------|-----------:|----------------:|--------------:|---------:|:---------------------|:--------------------|--------------------:|
| more_quantiles_d1-2 | accuracy                   |        142 |         0.84565 |       0.84628 |  0.00063 | [-0.00115, +0.00260] | 66 / 23 / 53        |             0.3168  |
| more_quantiles_d1-2 | balanced_accuracy          |        142 |         0.81825 |       0.81923 |  0.00098 | [-0.00083, +0.00303] | 71 / 15 / 56        |             0.12564 |
| more_quantiles_d1-2 | f1_macro                   |        142 |         0.81655 |       0.81761 |  0.00106 | [-0.00080, +0.00320] | 68 / 14 / 60        |             0.22518 |
| more_quantiles_d1-2 | mcc                        |        142 |         0.77191 |       0.77295 |  0.00105 | [-0.00155, +0.00378] | 67 / 14 / 61        |             0.38019 |
| more_quantiles_d1-2 | log_loss                   |        142 |         0.54531 |       0.54795 |  0.00264 | [-0.00704, +0.01471] | 84 / 0 / 58         |             0.01903 |
| more_quantiles_d1-2 | auc                        |        142 |         0.94771 |       0.94766 | -5e-05   | [-0.00077, +0.00072] | 66 / 13 / 63        |             0.61081 |
| more_quantiles_d1-2 | brier                      |        142 |         0.24667 |       0.24379 | -0.00287 | [-0.00459, -0.00144] | 87 / 0 / 55         |             0.00023 |
| more_quantiles_d1-2 | ece                        |        142 |         0.12331 |       0.11971 | -0.0036  | [-0.00578, -0.00135] | 94 / 0 / 48         |             2e-05   |
| more_quantiles_d1-2 | fit_time_s                 |        142 |         0.95179 |       0.67695 | -0.27484 | [-0.31483, -0.22695] | 133 / 0 / 9         |             0       |
| more_quantiles_d1-2 | predict_time_s             |        142 |         0.72253 |       0.2006  | -0.52193 | [-0.63775, -0.42599] | 142 / 0 / 0         |             0       |
| more_quantiles_d1-2 | predict_time_per_series_ms |        142 |         1.8321  |       0.44368 | -1.38842 | [-1.63598, -1.16703] | 142 / 0 / 0         |             0       |