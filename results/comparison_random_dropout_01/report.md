# Benchmark Comparison: random_dropout_0.1 vs `quant_base` (Resample 0)

**Seeds per method:** `{'quant_base': 3, 'random_dropout_0.1': 3}`  
**Notes:** $\Delta = \text{method} - \text{baseline}$. Metrics are direction-aware (lower is better for log loss, Brier, ECE, and fit/predict execution times).

## Summary Results

| Method             | Metric                     |   Datasets |   Baseline Mean |   Method Mean |   Mean Δ | 95% Bootstrap CI     | Win / Draw / Loss   |   Wilcoxon p (Holm) |
|:-------------------|:---------------------------|-----------:|----------------:|--------------:|---------:|:---------------------|:--------------------|--------------------:|
| random_dropout_0.1 | accuracy                   |        142 |         0.84565 |       0.84492 | -0.00073 | [-0.00220, +0.00060] | 62 / 23 / 57        |             0.98519 |
| random_dropout_0.1 | balanced_accuracy          |        142 |         0.81825 |       0.81744 | -0.00082 | [-0.00231, +0.00058] | 61 / 17 / 64        |             0.92247 |
| random_dropout_0.1 | f1_macro                   |        142 |         0.81655 |       0.81565 | -0.0009  | [-0.00257, +0.00062] | 66 / 16 / 60        |             0.78602 |
| random_dropout_0.1 | mcc                        |        142 |         0.77191 |       0.77086 | -0.00105 | [-0.00358, +0.00117] | 67 / 16 / 59        |             0.75994 |
| random_dropout_0.1 | log_loss                   |        142 |         0.54531 |       0.55237 |  0.00706 | [-0.00172, +0.01871] | 45 / 0 / 97         |             0.00058 |
| random_dropout_0.1 | auc                        |        142 |         0.94771 |       0.94729 | -0.00042 | [-0.00099, +0.00006] | 58 / 12 / 72        |             0.11758 |
| random_dropout_0.1 | brier                      |        142 |         0.24667 |       0.24767 |  0.00101 | [+0.00046, +0.00156] | 44 / 0 / 98         |             4e-05   |
| random_dropout_0.1 | ece                        |        142 |         0.12331 |       0.12567 |  0.00236 | [+0.00075, +0.00416] | 51 / 0 / 91         |             0.00054 |
| random_dropout_0.1 | fit_time_s                 |        142 |         0.95179 |       0.44586 | -0.50592 | [-0.60750, -0.42626] | 142 / 0 / 0         |             0       |
| random_dropout_0.1 | predict_time_s             |        142 |         0.72253 |       0.19568 | -0.52685 | [-0.64241, -0.43060] | 142 / 0 / 0         |             0       |
| random_dropout_0.1 | predict_time_per_series_ms |        142 |         1.8321  |       0.42434 | -1.40776 | [-1.66090, -1.18050] | 142 / 0 / 0         |             0       |