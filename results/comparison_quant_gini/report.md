# Benchmark Comparison: quant_gini vs `quant_base` (Resample 0)

**Seeds per method:** `{'quant_base': 3, 'quant_gini': 3}`  
**Notes:** $\Delta = \text{method} - \text{baseline}$. Metrics are direction-aware (lower is better for log loss, Brier, ECE, and fit/predict execution times).

## Summary Results

| Method     | Metric                     |   Datasets |   Baseline Mean |   Method Mean |   Mean Δ | 95% Bootstrap CI     | Win / Draw / Loss   | Wilcoxon p (Holm)   |
|:-----------|:---------------------------|-----------:|----------------:|--------------:|---------:|:---------------------|:--------------------|:--------------------|
| quant_gini | accuracy                   |        142 |         0.84565 |       0.84565 |  0       | [+0.00000, +0.00000] | 0 / 142 / 0         | N/A                 |
| quant_gini | balanced_accuracy          |        142 |         0.81825 |       0.81825 |  0       | [+0.00000, +0.00000] | 0 / 142 / 0         | N/A                 |
| quant_gini | f1_macro                   |        142 |         0.81655 |       0.81655 |  0       | [+0.00000, +0.00000] | 0 / 142 / 0         | N/A                 |
| quant_gini | mcc                        |        142 |         0.77191 |       0.77191 |  0       | [+0.00000, +0.00000] | 0 / 142 / 0         | N/A                 |
| quant_gini | log_loss                   |        142 |         0.54531 |       0.54531 |  0       | [+0.00000, +0.00000] | 0 / 142 / 0         | N/A                 |
| quant_gini | auc                        |        142 |         0.94771 |       0.94771 |  0       | [+0.00000, +0.00000] | 1 / 141 / 0         | 0.31731             |
| quant_gini | brier                      |        142 |         0.24667 |       0.24667 |  0       | [+0.00000, +0.00000] | 0 / 142 / 0         | N/A                 |
| quant_gini | ece                        |        142 |         0.12331 |       0.12331 |  0       | [+0.00000, +0.00000] | 0 / 142 / 0         | N/A                 |
| quant_gini | fit_time_s                 |        142 |         0.95179 |       0.94523 | -0.00656 | [-0.04789, +0.02693] | 63 / 0 / 79         | 0.55140             |
| quant_gini | predict_time_s             |        142 |         0.72253 |       0.73004 |  0.00751 | [-0.00527, +0.02008] | 64 / 0 / 78         | 0.15167             |
| quant_gini | predict_time_per_series_ms |        142 |         1.8321  |       1.84688 |  0.01478 | [-0.03759, +0.06294] | 64 / 0 / 78         | 0.17501             |