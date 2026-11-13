# nera-thudar: reproducing and extending QUANT

A reproduction of *QUANT: A Minimalist Interval Method for Time Series Classification*
(Dempster, Schmidt and Webb, Data Mining and Knowledge Discovery, 2024;
[paper](https://doi.org/10.1007/s10618-024-01036-9), [authors' code](https://github.com/angus924/quant)),
followed by a study of seven variants of the method.

- Report source: [report/main.tex](report/main.tex). Tables and figures: [report/tables](report/tables), [report/figures](report/figures).
- Per-dataset comparison with the paper: [docs/reproduction_table.md](docs/reproduction_table.md).
- Discrepancy log: [docs/discrepancy_log.md](docs/discrepancy_log.md).

## 1. Environment setup

Python 3.13 was used. No GPU is needed.

```bash
git clone https://github.com/harini-jt/nera-thudar.git
cd nera-thudar
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests          # checks our transform against the authors' quant.py
```

Run every command from the repository root.

## 2. Data

The experiments use 142 datasets (112 from the UCR archive and the 30 added in the
bake-off redux), each with 30 train/test resamples stored as
`data/<Dataset>/<Dataset><resample>_TRAIN.ts` and `..._TEST.ts`. Resample 0 is the
original split. The full set is about 38 GB.

**Option A: download the prepared copy** (Hugging Face dataset `NeraThudar/NeraThudarDS`):

```bash
hf auth login                       # only needed while the dataset repository is private
python -m scripts.download_dataset  # writes to ./data
```

**Option B: build it from the public sources.** Download the UCR archive and the 30
additional datasets from <https://www.timeseriesclassification.com>, then follow
[reference/dataset_creation_steps.ipynb](reference/dataset_creation_steps.ipynb), which
renames the folders and generates resamples 1 to 29 with
`tsml_eval.utils.experiments.stratified_resample_data(..., random_state=resample_id)`.

To keep the data somewhere else, set `DATASET_DIR`:

```bash
export DATASET_DIR=/path/to/data
```

Check the data and regenerate the dataset statistics:

```bash
python -m scripts.verify_dataset   # writes docs/data_verification.csv (142 x 30 splits)
python -m scripts.dataset_stats    # writes docs/dataset_stats.csv
```

## 3. Runner reference (`python -m src.runner`)

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

## 4. Comparing Methods (`python -m scripts.compare_methods`)

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

## 5. Directory & Output Structure

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

## 6. End-to-End Execution Flow

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