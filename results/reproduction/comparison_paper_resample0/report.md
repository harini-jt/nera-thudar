# Benchmark Comparison: quant_base vs `paper` (Resample 0)

**Seeds per method:** `{'paper': 1, 'quant_base': 3}`  
**Notes:** $\Delta = \text{method} - \text{baseline}$. Metrics are direction-aware (lower is better for log loss, Brier, ECE, and fit/predict execution times).

## Summary Results

| Method     | Metric   |   Datasets |   Baseline Mean |   Method Mean |   Mean Δ | 95% Bootstrap CI     | Win / Draw / Loss   |   Wilcoxon p (Holm) |
|:-----------|:---------|-----------:|----------------:|--------------:|---------:|:---------------------|:--------------------|--------------------:|
| quant_base | accuracy |        142 |         0.84731 |       0.84565 | -0.00166 | [-0.00696, +0.00290] | 61 / 12 / 69        |             0.21423 |