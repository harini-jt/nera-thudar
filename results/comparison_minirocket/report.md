# Benchmark Comparison: minirocket vs `quant_base` (Resample 0)

**Seeds per method:** `{'minirocket': 3, 'quant_base': 3}`  
**Notes:** $\Delta = \text{method} - \text{baseline}$. Metrics are direction-aware (lower is better for log loss, Brier, ECE, and fit/predict execution times).

## Summary Results

| Method     | Metric                     |   Datasets |   Baseline Mean |   Method Mean |   Mean Δ | 95% Bootstrap CI     | Win / Draw / Loss   |   Wilcoxon p (Holm) |
|:-----------|:---------------------------|-----------:|----------------:|--------------:|---------:|:---------------------|:--------------------|--------------------:|
| minirocket | accuracy                   |        142 |         0.84575 |       0.85125 |  0.0055  | [-0.00336, +0.01591] | 70 / 13 / 59        |             0.45689 |
| minirocket | balanced_accuracy          |        142 |         0.81838 |       0.82783 |  0.00945 | [-0.00005, +0.02042] | 74 / 11 / 57        |             0.1289  |
| minirocket | f1_macro                   |        142 |         0.817   |       0.82672 |  0.00972 | [+0.00013, +0.02080] | 74 / 10 / 58        |             0.12807 |
| minirocket | mcc                        |        142 |         0.77261 |       0.77828 |  0.00567 | [-0.00614, +0.01866] | 75 / 10 / 57        |             0.56094 |
| minirocket | fit_time_s                 |        142 |         0.57738 |       1.19598 |  0.6186  | [-0.31835, +2.13636] | 128 / 0 / 14        |             0       |
| minirocket | predict_time_s             |        142 |         0.20102 |       0.16645 | -0.03456 | [-0.04611, -0.02254] | 127 / 0 / 15        |             0       |
| minirocket | predict_time_per_series_ms |        142 |         0.45033 |       0.22773 | -0.2226  | [-0.27951, -0.17193] | 127 / 0 / 15        |             0       |