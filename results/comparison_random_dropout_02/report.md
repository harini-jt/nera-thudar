# Benchmark Comparison: random_dropout_0.2 vs `quant_base` (Resample 0)

**Seeds per method:** `{'quant_base': 3, 'random_dropout_0.2': 3}`  
**Notes:** $\Delta = \text{method} - \text{baseline}$. Metrics are direction-aware (lower is better for log loss, Brier, ECE, and fit/predict execution times).

## Summary Results

| Method             | Metric                     |   Datasets |   Baseline Mean |   Method Mean |   Mean Δ | 95% Bootstrap CI     | Win / Draw / Loss   |   Wilcoxon p (Holm) |
|:-------------------|:---------------------------|-----------:|----------------:|--------------:|---------:|:---------------------|:--------------------|--------------------:|
| random_dropout_0.2 | accuracy                   |        142 |         0.84575 |       0.84375 | -0.002   | [-0.00320, -0.00085] | 48 / 18 / 76        |             0.00243 |
| random_dropout_0.2 | balanced_accuracy          |        142 |         0.81838 |       0.81584 | -0.00254 | [-0.00389, -0.00122] | 45 / 15 / 82        |             0.00051 |
| random_dropout_0.2 | f1_macro                   |        142 |         0.817   |       0.81404 | -0.00297 | [-0.00454, -0.00144] | 47 / 12 / 83        |             0.00037 |
| random_dropout_0.2 | mcc                        |        142 |         0.77261 |       0.76868 | -0.00393 | [-0.00656, -0.00168] | 51 / 12 / 79        |             0.00282 |
| random_dropout_0.2 | log_loss                   |        142 |         0.54751 |       0.55655 |  0.00904 | [+0.00040, +0.01918] | 44 / 0 / 98         |             1e-05   |
| random_dropout_0.2 | auc                        |        142 |         0.94765 |       0.94734 | -0.00031 | [-0.00080, +0.00021] | 55 / 11 / 76        |             0.06677 |
| random_dropout_0.2 | brier                      |        142 |         0.24666 |       0.24859 |  0.00193 | [+0.00133, +0.00253] | 35 / 0 / 107        |             0       |
| random_dropout_0.2 | ece                        |        142 |         0.1239  |       0.1253  |  0.0014  | [-0.00032, +0.00304] | 46 / 0 / 96         |             0.00051 |
| random_dropout_0.2 | fit_time_s                 |        142 |         0.57738 |       0.45526 | -0.12212 | [-0.21445, -0.04265] | 103 / 0 / 39        |             0       |
| random_dropout_0.2 | predict_time_s             |        142 |         0.20102 |       0.21874 |  0.01773 | [+0.00190, +0.03670] | 69 / 2 / 71         |             0.65172 |
| random_dropout_0.2 | predict_time_per_series_ms |        142 |         0.45033 |       0.45585 |  0.00552 | [-0.01931, +0.03369] | 71 / 0 / 71         |             0.34783 |