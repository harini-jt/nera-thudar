# nera-thudar: Quant for Time series classification

This guide covers running deterministic QUANT feature extraction benchmarks across time-series datasets using `src.runner` and comparing variant performances against `quant_base` using `scripts.compare_methods`.

---

## 0. Pre-requisites (IG)

- Testing equivalence: `python -m pytest tests -v`
- Generating dataset stats: `python -m scripts.dataset_stats`

## 1. Running Experiments (`python -m src.runner`)

The runner executes feature extraction and tree fitting across specified datasets, seeds, and methods. Each evaluated method produces its own isolated CSV result file.

### Command-Line Arguments

| Flag         | Default                                   | Description                                                                                |
| ------------ | ----------------------------------------- | ------------------------------------------------------------------------------------------ |
| `--methods`  | `quant_base`                              | One or more registered pipeline names (e.g., `quant_base`, `dup_d1-2`, `depth_div_m2_m8`). |
| `--resample` | `0`                                       | Resample split ID to load (`0` for standard train/test split).                             |
| `--seeds`    | `7 11 42`                                 | Random seeds for `ExtraTreesClassifier` model initializations.                             |
| `--out`      | `results/{method}_resample{resample}.csv` | Output path template or file path. Supports `{method}` and `{resample}` placeholders.      |
| `--datasets` | _(All datasets)_                          | Space-separated list of dataset names. Omit to run on all valid folders in `data/`.        |
| `--trees`    | `200`                                     | Number of trees in the `ExtraTreesClassifier`.                                             |
| `--depth`    | `6`                                       | QUANT dyadic decomposition depth level limit.                                              |
| `--div`      | `4`                                       | Default quantile divisor ($m / 4$).                                                        |
| `--n-jobs`   | `-1`                                      | Core count to use for parallel tree fitting (`-1` uses all CPU threads).                   |

---

### `--out` File Path Resolution

The runner handles `--out` dynamically based on how it is specified:

1. **Template String (Recommended for multiple methods):**

```bash
--out "results/{method}_resample{resample}.csv"

```

_Automatically expands `{method}` and `{resample}` for each method run in batch._ 2. **Directory Path:**

```bash
--out results/

```

_Auto-names output files as `results/<method>_resample<resample_id>.csv`._ 3. **Explicit Single File Path:**

```bash
--out results/quant_base_resample0.csv

```

_Writes directly to the specified file (ideal when executing one method per run)._

---

### Command Examples

#### Single Method Execution

```bash
python -m src.runner \
    --methods quant_base \
    --resample 0 \
    --seeds 7 11 42 \
    --out results/quant_base_resample0.csv

```

#### Multi-Method Batch Execution

```bash
python -m src.runner \
    --methods quant_base dup_d1-2 dup_d1-3 depth_div_m2_m8 \
    --resample 0 \
    --seeds 7 11 42 \
    --out "results/{method}_resample{resample}.csv"

```

#### Specific Datasets Only

```bash
python -m src.runner \
    --methods dup_d1-2 \
    --datasets GunPoint CBF ECG200 \
    --resample 0 \
    --out results/dup_d1-2_subset.csv

```

---

## 2. Comparing Methods (`python -m scripts.compare_methods`)

The comparison script aggregates per-method CSV results, calculates dataset-level performance deltas against a designated baseline, performs statistical testing (Wilcoxon signed-rank with Holm-Bonferroni correction), and exports clean Markdown summary tables and figures.

### Command-Line Arguments

| Flag             | Default              | Description                                                                                 |
| ---------------- | -------------------- | ------------------------------------------------------------------------------------------- |
| `--results`      | _(Required)_         | Space-separated list of per-method CSV file paths to evaluate.                              |
| `--baseline`     | `quant_base`         | Method name serving as reference baseline for all delta calculations.                       |
| `--methods`      | _(Auto-detected)_    | Specific methods to compare against baseline. Defaults to all non-baseline methods in CSVs. |
| `--resample`     | `0`                  | Resample ID to slice from results.                                                          |
| `--out`          | `results/comparison` | Output directory where CSVs, figures, and `report.md` will be created.                      |
| `--metrics`      | _(11 core metrics)_  | Evaluation metrics to include in summary tables and statistical tests.                      |
| `--plot-metrics` | `accuracy log_loss`  | Metrics for which scatter, delta distribution, and dataset bar charts will be plotted.      |
| `--n-boot`       | `10000`              | Number of bootstrap iterations for computing 95% Confidence Intervals.                      |

---

### Comparison Command Example

```bash
python -m scripts.compare_methods \
    --results results/quant_base_resample0.csv results/dup_d1-2_resample0.csv results/depth_div_m2_m8_resample0.csv \
    --baseline quant_base \
    --resample 0 \
    --out results/comparison

```

---

## 3. Directory & Output Structure

Executing the pipeline produces the following output file organization (left buckets out of the picture):

```text
results/
├── quant_base_resample0.csv             # Raw per-run results for quant_base
├── dup_d1-2_resample0.csv               # Raw per-run results for dup_d1-2
├── depth_div_m2_m8_resample0.csv        # Raw per-run results for depth_div_m2_m8
└── comparison/                          # Generated comparison artifacts
    ├── report.md                        # Formatted Markdown report with summary tables
    ├── csvs/
    │   ├── summary.csv                  # Aggregated metric deltas, CIs, and p-values
    │   ├── per_dataset_comparison.csv   # Per-dataset paired differences
    │   └── seed_robustness.csv          # Per-seed breakdown of outcomes
    └── figures/
        ├── dup_d1-2_accuracy_scatter.png
        ├── dup_d1-2_accuracy_deltas.png
        ├── dup_d1-2_accuracy_per_dataset.png  # Sorted vertical bar chart (all datasets)
        └── forest_accuracy.png               # 95% Bootstrap CI forest plot across variants
        ...(same for log loss)

```

---

## 4. End-to-End Execution Flow

```bash
# Step 1: Run baseline model
python -m src.runner --methods quant_base --resample 0 --out results/quant_base_resample0.csv

# Step 2: Run experimental variants
python -m src.runner --methods dup_d1-2 --resample 0 --out results/dup_d1-2_resample0.csv
python -m src.runner --methods depth_div_m2_m8 --resample 0 --out results/depth_div_m2_m8_resample0.csv

# Step 3: Run comparative evaluation and generate report
python -m scripts.compare_methods \
    --results results/*.csv \
    --baseline quant_base \
    --out results/comparison

```
