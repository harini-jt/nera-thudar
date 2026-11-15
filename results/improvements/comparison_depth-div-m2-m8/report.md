# Benchmark Comparison: depth_div_m2_m8 vs `quant_base` (Resample 0)

**Seeds per method:** `{'depth_div_m2_m8': 3, 'quant_base': 3}`  
**Notes:** $\Delta = \text{method} - \text{baseline}$. Metrics are direction-aware (lower is better for log loss, Brier, ECE, and fit/predict execution times).

## Summary Results

| Method          | Metric                     |   Datasets |   Baseline Mean |   Method Mean |   Mean Δ | 95% Bootstrap CI     | Win / Draw / Loss   |   Wilcoxon p (Holm) |
|:----------------|:---------------------------|-----------:|----------------:|--------------:|---------:|:---------------------|:--------------------|--------------------:|
| depth_div_m2_m8 | accuracy                   |        142 |         0.84575 |       0.84589 |  0.00014 | [-0.00147, +0.00191] | 57 / 22 / 63        |             0.71584 |
| depth_div_m2_m8 | balanced_accuracy          |        142 |         0.81838 |       0.81846 |  8e-05   | [-0.00159, +0.00191] | 59 / 18 / 65        |             0.6392  |
| depth_div_m2_m8 | f1_macro                   |        142 |         0.817   |       0.81718 |  0.00018 | [-0.00158, +0.00207] | 63 / 14 / 65        |             0.8065  |
| depth_div_m2_m8 | mcc                        |        142 |         0.77261 |       0.77221 | -0.00041 | [-0.00304, +0.00221] | 59 / 14 / 69        |             0.65653 |
| depth_div_m2_m8 | log_loss                   |        142 |         0.54751 |       0.54151 | -0.006   | [-0.01417, +0.00183] | 93 / 0 / 49         |             0.00116 |
| depth_div_m2_m8 | auc                        |        142 |         0.94765 |       0.94748 | -0.00017 | [-0.00101, +0.00071] | 67 / 11 / 64        |             0.80939 |
| depth_div_m2_m8 | brier                      |        142 |         0.24666 |       0.24328 | -0.00337 | [-0.00556, -0.00143] | 91 / 0 / 51         |             0.00015 |
| depth_div_m2_m8 | ece                        |        142 |         0.1239  |       0.11761 | -0.00629 | [-0.00858, -0.00404] | 100 / 0 / 42        |             0       |
| depth_div_m2_m8 | fit_time_s                 |        142 |         0.57738 |       1.3856  |  0.80822 | [+0.54527, +1.11516] | 0 / 0 / 142         |             0       |
| depth_div_m2_m8 | predict_time_s             |        142 |         0.20102 |       0.36981 |  0.16879 | [+0.13010, +0.21261] | 0 / 0 / 142         |             0       |
| depth_div_m2_m8 | predict_time_per_series_ms |        142 |         0.45033 |       0.79742 |  0.34709 | [+0.29219, +0.40887] | 0 / 0 / 142         |             0       |