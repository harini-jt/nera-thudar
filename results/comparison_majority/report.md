# Benchmark Comparison: majority vs `quant_base` (Resample 0)

**Seeds per method:** `{'majority': 1, 'quant_base': 3}`  
**Notes:** $\Delta = \text{method} - \text{baseline}$. Metrics are direction-aware (lower is better for log loss, Brier, ECE, and fit/predict execution times).

## Summary Results

| Method   | Metric                     |   Datasets |   Baseline Mean |   Method Mean |   Mean Δ | 95% Bootstrap CI     | Win / Draw / Loss   |   Wilcoxon p (Holm) |
|:---------|:---------------------------|-----------:|----------------:|--------------:|---------:|:---------------------|:--------------------|--------------------:|
| majority | accuracy                   |        142 |         0.84575 |       0.34471 | -0.50105 | [-0.53801, -0.46350] | 0 / 0 / 142         |                   0 |
| majority | balanced_accuracy          |        142 |         0.81838 |       0.2882  | -0.53018 | [-0.56293, -0.49684] | 0 / 0 / 142         |                   0 |
| majority | f1_macro                   |        142 |         0.817   |       0.17564 | -0.64136 | [-0.67322, -0.60838] | 0 / 0 / 142         |                   0 |
| majority | mcc                        |        142 |         0.77261 |       0       | -0.77261 | [-0.80601, -0.73789] | 0 / 0 / 142         |                   0 |
| majority | log_loss                   |        142 |         0.54751 |       1.51279 |  0.96528 | [+0.84511, +1.09147] | 5 / 0 / 137         |                   0 |
| majority | auc                        |        142 |         0.94765 |       0.5     | -0.44765 | [-0.45978, -0.43414] | 0 / 0 / 142         |                   0 |
| majority | brier                      |        142 |         0.24666 |       0.69344 |  0.44678 | [+0.41432, +0.47888] | 0 / 0 / 142         |                   0 |
| majority | ece                        |        142 |         0.1239  |       0.03741 | -0.08648 | [-0.10465, -0.06812] | 125 / 0 / 17        |                   0 |
| majority | fit_time_s                 |        142 |         0.57738 |       0.00015 | -0.57723 | [-0.77744, -0.40606] | 142 / 0 / 0         |                   0 |
| majority | predict_time_s             |        142 |         0.20102 |       2e-05   | -0.20099 | [-0.25417, -0.15561] | 142 / 0 / 0         |                   0 |
| majority | predict_time_per_series_ms |        142 |         0.45033 |       7e-05   | -0.45026 | [-0.53451, -0.37455] | 142 / 0 / 0         |                   0 |