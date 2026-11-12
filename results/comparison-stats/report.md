# Benchmark Comparison: quant_stats vs `quant_base` (Resample 0)

**Seeds per method:** `{'quant_base': 3, 'quant_stats': 3}`  
**Notes:** $\Delta = \text{method} - \text{baseline}$. Metrics are direction-aware (lower is better for log loss, Brier, ECE, and fit/predict execution times).

## Summary Results

| Method      | Metric                     |   Datasets |   Baseline Mean |   Method Mean |   Mean Δ | 95% Bootstrap CI     | Win / Draw / Loss   |   Wilcoxon p (Holm) |
|:------------|:---------------------------|-----------:|----------------:|--------------:|---------:|:---------------------|:--------------------|--------------------:|
| quant_stats | accuracy                   |        142 |         0.84565 |       0.84593 |  0.00028 | [-0.00159, +0.00201] | 71 / 18 / 53        |             0.05676 |
| quant_stats | balanced_accuracy          |        142 |         0.81825 |       0.81867 |  0.00042 | [-0.00145, +0.00219] | 74 / 16 / 52        |             0.05835 |
| quant_stats | f1_macro                   |        142 |         0.81655 |       0.81671 |  0.00016 | [-0.00191, +0.00209] | 75 / 13 / 54        |             0.08092 |
| quant_stats | mcc                        |        142 |         0.77191 |       0.77243 |  0.00053 | [-0.00272, +0.00349] | 74 / 13 / 55        |             0.06725 |
| quant_stats | log_loss                   |        142 |         0.54531 |       0.5439  | -0.00141 | [-0.00772, +0.00478] | 80 / 0 / 62         |             0.11428 |
| quant_stats | auc                        |        142 |         0.94771 |       0.94822 |  0.00051 | [-0.00022, +0.00125] | 72 / 11 / 59        |             0.28028 |
| quant_stats | brier                      |        142 |         0.24667 |       0.2449  | -0.00177 | [-0.00290, -0.00067] | 84 / 0 / 58         |             0.00574 |
| quant_stats | ece                        |        142 |         0.12331 |       0.12423 |  0.00092 | [-0.00118, +0.00307] | 66 / 0 / 76         |             0.5986  |
| quant_stats | fit_time_s                 |        142 |         0.95179 |       0.8201  | -0.13169 | [-0.22152, -0.02114] | 125 / 0 / 17        |             0       |
| quant_stats | predict_time_s             |        142 |         0.72253 |       0.34741 | -0.37511 | [-0.46739, -0.30062] | 140 / 0 / 2         |             0       |
| quant_stats | predict_time_per_series_ms |        142 |         1.8321  |       0.72539 | -1.1067  | [-1.31993, -0.91766] | 140 / 0 / 2         |             0       |