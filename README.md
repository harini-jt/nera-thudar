# nera-thudar: Reproducing and Extending QUANT

A reproduction of _QUANT: A Minimalist Interval Method for Time Series Classification_ (Dempster, Schmidt, and Webb, _Data Mining and Knowledge Discovery_, 2024; [paper](https://doi.org/10.1007/s10618-024-01036-9), [authors' code](https://github.com/angus924/quant)), followed by an empirical study of architectural and training extensions across 142 UCR/UEA time-series classification datasets.

- **Per-dataset comparison with the paper:** [docs/reproduction_table.md](docs/reproduction_table.md)
- **Dataset audit and summary statistics:** [docs/data_verification.csv](docs/data_verification.csv), [docs/dataset_stats.csv](docs/dataset_stats.csv)

---

## 1. Repository Structure

```text
nera-thudar/
├── data/                        # 142 datasets x 30 train/test resamples (.ts format)
├── docs/                        # Verification logs, dataset stats, paper comparison tables
├── reference/                   # Authors' reference implementation & dataset creation notebook
│   ├── quant_src_code/
│   ├── quant_base.py
│   ├── dataset_creation_steps.ipynb
│   └── roughwork.ipynb
├── results/                     # Saved experiment outputs, statistical summaries, and figures
│   ├── reproduction/            # QUANT reproduction CSVs & cross-machine consistency outputs
│   ├── other_baselines/         # Majority-class and MiniRocket baseline results
│   └── improvements/            # Variant experiment CSVs and pairwise comparison reports
├── scripts/                     # Data download, verification, stats, baselines, runtime & analysis
│   ├── download_dataset.py
│   ├── verify_dataset.py
│   ├── dataset_stats.py
│   ├── run_baselines.py
│   ├── benchmark_runtime.py
|   ├── generate_reproduction_table.py
|   ├── generate_readme_table.py
│   ├── cross_machine.py
│   ├── compare_methods.py
│   └── common.py
├── src/                         # Core QUANT implementation, extensions, loader, metrics, runner
│   ├── implementations/
│   │   ├── quant.py
│   │   ├── pipelines.py
│   │   └── improvements/
│   ├── data_loader.py
│   ├── metrics.py
│   └── runner.py
├── tests/                       # Unit & equivalence tests against reference/quant_base.py
├── requirements.txt
└── README.md

```

---

## 2. Environment Setup

Python 3.13 was used. No GPU is required; all feature extraction and `ExtraTreesClassifier` runs execute on CPU.

```bash
git clone [https://github.com/harini-jt/nera-thudar.git](https://github.com/harini-jt/nera-thudar.git)
cd nera-thudar
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests          # verifies our transform matches reference/quant_base.py

```

Run all commands below from the repository root.

---

## 3. Data Download & Verification

The benchmark uses **142 univariate datasets** (112 from the UCR archive and 30 added in the bake-off redux), each with **30 stratified train/test resamples** stored as `data/<Dataset>/<Dataset><resample>_TRAIN.ts` and `..._TEST.ts` (`resample = 0` is the original split; full archive is ~38 GB).

### Option A: Download the Prepared Archive (Recommended)

Pull the pre-packaged splits from Hugging Face (`NeraThudar/NeraThudarDS`):

```bash
hf auth login                       # only required if the repository is set to private
python -m scripts.download_dataset  # downloads into ./data

```

### Option B: Build from Official UCR/UEA Sources

Download the 142 datasets from [https://www.timeseriesclassification.com](https://www.timeseriesclassification.com) and follow [reference/dataset_creation_steps.ipynb](https://www.google.com/search?q=reference/dataset_creation_steps.ipynb), which standardizes folder names and generates resamples `1..29` using `tsml_eval.utils.experiments.stratified_resample_data(..., random_state=resample_id)`.

To store the dataset directory outside the repository, set `DATASET_DIR`:

```bash
export DATASET_DIR=/path/to/data    # Windows PowerShell: $env:DATASET_DIR="D:\path\to\data"

```

### Verify Splits and Compute Dataset Metadata

```bash
python -m scripts.verify_dataset    # validates 142 x 30 splits -> docs/data_verification.csv
python -m scripts.dataset_stats     # computes size/length/balance stats -> docs/dataset_stats.csv

```

---

## 4. Results Overview (Reproduction vs. Paper)

### Table 1: QUANT Reproduction vs. Dempster et al. (2024)

| Setting / Split                            | Datasets | Mean Accuracy (Ours) | Mean Accuracy (Paper) | Mean $\Delta$ (Ours $-$ Paper) | Win / Draw / Loss vs. Paper |
| :----------------------------------------- | :------: | :------------------: | :-------------------: | :----------------------------: | :-------------------------: |
| **Resample 0** (3 seeds: `7, 11, 42`)      |   142    |        0.8457        |        0.8473         |            -0.0017             |        61 / 12 / 69         |
| **All 30 Resamples (`0..29`)** (`seed=42`) |   142    |        0.8551        |        0.8568         |            -0.0017             |         67 / 4 / 71         |
| **All 30 Resamples (`0..29`)** (unseeded)  |   142    |        0.8546        |        0.8568         |            -0.0022             |         61 / 5 / 76         |

### Table 2: Baselines & QUANT Variants on Resample 0 (`142` Datasets, Seeds `7, 11, 42`)

| Category           | Method (`--methods` key) | Description                                                 | Mean Accuracy | Mean $\Delta$ vs. `quant_base` |  95% Bootstrap CI  | Win / Draw / Loss | Wilcoxon $p$ (Holm) |
| :----------------- | :----------------------- | :---------------------------------------------------------- | :-----------: | :----------------------------: | :----------------: | :---------------: | :-----------------: |
| **Baseline**       | `quant_base`             | QUANT (`depth=6, div=4`) + 200 ExtraTrees (`entropy`)       |    0.8457     |               —                |         —          |         —         |          —          |
| **Reference**      | `majority`               | Prior-class dummy baseline                                  |    0.3447     |            -0.5009             | [-0.5380, -0.4632] |    0 / 0 / 142    |      < 0.0001       |
| **Reference**      | `minirocket`             | MiniRocket (`10k` kernels) + `RidgeClassifierCV`            |    0.8512     |            +0.0056             | [-0.0033, +0.0161] |   71 / 12 / 59    |       1.0000        |
| **Criterion**      | `quant_gini`             | Split criterion changed from `entropy` to `gini`            |    0.8457     |            +0.0000             | [+0.0000, +0.0000] |    0 / 142 / 0    |      < 0.001         |
| **Weighting**      | `dup_d1-2`               | Duplicate interval features at depths `1, 2` ($\times 2$)   |    0.8458     |            +0.0001             | [-0.0017, +0.0022] |   60 / 23 / 59    |       1.0000        |
| **Weighting**      | `dup_d1_4_d2_2`          | Duplicate depth `1` ($\times 4$) and depth `2` ($\times 2$) |    0.8454     |            -0.0002             | [-0.0024, +0.0023] |   48 / 20 / 74    |       0.6234        |
| **Resolution**     | `more_quantiles_d1-2`    | Finer quantile grid (`div=2`) at depths `1, 2`              |    0.8463     |            +0.0006             | [-0.0012, +0.0026] |   66 / 23 / 53    |       1.0000        |
| **Resolution**     | `depth_div_m2_m8`        | `div=2` at depths `1..3`, `div=8` at depths `5..6`          |    0.8459     |            +0.0002             | [-0.0014, +0.0021] |   54 / 25 / 63    |       1.0000        |
| **Features**       | `quant_stats`            | Append interval standard deviation and OLS slope            |    0.8459     |            +0.0003             | [-0.0016, +0.0020] |   71 / 18 / 53    |       0.3973        |
| **Regularization** | `random_dropout_0.1`     | Random `10%` feature column dropout before forest           |    0.8449     |            -0.0007             | [-0.0022, +0.0006] |   62 / 23 / 57    |       1.0000        |
| **Regularization** | `random_dropout_0.2`     | Random `20%` feature column dropout before forest           |    0.8437     |            -0.0019             | [-0.0032, -0.0007] |   48 / 19 / 75    |       0.0521        |

---

## 5. Commands to Reproduce All Report Results

Every runner script checkpoints incrementally to CSV—if interrupted, re-running the exact command resumes from the last completed `(dataset, resample_id, method, seed)` tuple.

### 5.1 QUANT Paper Reproduction (`results/reproduction/`)

1. **Resample 0 (default split, 3 seeds):**

```bash
python -m src.runner \
    --methods quant_base \
    --resample 0 \
    --seeds 7 11 42 \
    --out results/reproduction/quant_base_resample0.csv

```

2. **All 30 resamples (`0..29`, fixed seed `42`):**

```bash
for r in $(seq 0 29); do
    python -m src.runner \
        --methods quant_base \
        --resample "$r" \
        --seeds 42 \
        --out results/reproduction/quant_base_all-resamples_seed-42.csv
done

```

3. **Cross-machine consistency check (Windows vs. macOS on Resample 0):**

```bash
python -m scripts.cross_machine \
    --a results/reproduction/quant_base_resample0.csv \
    --b results/reproduction/quant_base_resample0_mac.csv \
    --label-a Windows \
    --label-b Mac \
    --out results/reproduction/cross_machine

```

### 5.2 Reference Baselines (`results/other_baselines/`)

Run the majority-class and MiniRocket baselines on Resample 0 and compare against `quant_base`:

```bash
python -m src.run_baselines \
    --methods majority minirocket \
    --resample 0 \
    --seeds 7 11 42 \
    --out "results/other_baselines/{method}_resample{resample}.csv"

python -m scripts.compare_methods \
    --results results/reproduction/quant_base_resample0.csv results/other_baselines/majority_resample0.csv \
    --baseline quant_base \
    --out results/other_baselines/comparison_majority

python -m scripts.compare_methods \
    --results results/reproduction/quant_base_resample0.csv results/other_baselines/minirocket_resample0.csv \
    --baseline quant_base \
    --out results/other_baselines/comparison_minirocket

```

_(Note: run `python -m scripts.run_baselines ...` for the baseline script.)_

### 5.3 QUANT Extensions & Variants (`results/improvements/`)

Run all 8 experimental variants on Resample 0:

```bash
python -m src.runner --methods quant_gini          --resample 0 --seeds 7 11 42 --out results/improvements/quant_gini_resample0.csv

python -m src.runner --methods dup_d1-2            --resample 0 --seeds 7 11 42 --out results/improvements/dup_d1-2.csv

python -m src.runner --methods dup_d1_4_d2_2       --resample 0 --seeds 7 11 42 --out results/improvements/dup_d1-4_d2-2.csv

python -m src.runner --methods more_quantiles_d1-2 --resample 0 --seeds 7 11 42 --out results/improvements/more_quantiles_d1-2.csv

python -m src.runner --methods depth_div_m2_m8     --resample 0 --seeds 7 11 42 --out results/improvements/depth-div-m2-m8.csv

python -m src.runner --methods quant_stats         --resample 0 --seeds 7 11 42 --out results/improvements/quant_stats.csv

python -m src.runner --methods random_dropout_0.1  --resample 0 --seeds 7 11 42 --out results/improvements/random_dropout_01.csv

python -m src.runner --methods random_dropout_0.2  --resample 0 --seeds 7 11 42 --out results/improvements/random_dropout_02.csv

```

Generate pairwise statistical comparisons (paired Wilcoxon signed-rank tests with Holm-Bonferroni correction, `10,000`-iteration bootstrap 95% CIs, regime bucket breakdowns, and figures) against `quant_base`:

```bash
python -m scripts.compare_methods --results results/reproduction/quant_base_resample0.csv results/improvements/quant_gini_resample0.csv   --baseline quant_base --out results/improvements/comparison_quant_gini

python -m scripts.compare_methods --results results/reproduction/quant_base_resample0.csv results/improvements/dup_d1-2.csv               --baseline quant_base --out results/improvements/comparison_dup_d1-2

python -m scripts.compare_methods --results results/reproduction/quant_base_resample0.csv results/improvements/dup_d1-4_d2-2.csv          --baseline quant_base --out results/improvements/comparison_dup_d1-4_d2-2

python -m scripts.compare_methods --results results/reproduction/quant_base_resample0.csv results/improvements/more_quantiles_d1-2.csv    --baseline quant_base --out results/improvements/comparison_more_quantiles_d1-2

python -m scripts.compare_methods --results results/reproduction/quant_base_resample0.csv results/improvements/depth-div-m2-m8.csv        --baseline quant_base --out results/improvements/comparison_depth-div-m2-m8

python -m scripts.compare_methods --results results/reproduction/quant_base_resample0.csv results/improvements/quant_stats.csv            --baseline quant_base --out results/improvements/comparison_quant_stats

python -m scripts.compare_methods --results results/reproduction/quant_base_resample0.csv results/improvements/random_dropout_01.csv      --baseline quant_base --out results/improvements/comparison_random_dropout_01

python -m scripts.compare_methods --results results/reproduction/quant_base_resample0.csv results/improvements/random_dropout_02.csv      --baseline quant_base --out results/improvements/comparison_random_dropout_02

```

Each `comparison_<method>/` output directory contains:

- `report.md` — formatted summary table with mean $\Delta$, 95% bootstrap CIs, Win/Draw/Loss counts, and Holm-adjusted $p$-values
- `csvs/summary.csv`, `csvs/per_dataset_comparison.csv`, `csvs/seed_robustness.csv`, `csvs/bucket_summary.csv`
- `figures/` — parity scatter plots, sorted delta bar charts, regime bucket bar charts, and forest plots

### 5.4 Single-Core Runtime Benchmarks

Time all four pipeline stages (`transform_train_s`, `fit_s`, `transform_test_s`, `predict_s`) with BLAS, OpenMP, PyTorch, and Numba pinned to 1 thread:

```bash
# Full 142-dataset single-core runtime check for plain QUANT
python -m scripts.benchmark_runtime \
    --methods quant_base \
    --datasets all \
    --repeats 1 \
    --threads 1 \
    --out results/runtime_1core_full.csv

# 3-repeat runtime comparison across variants and MiniRocket on the 30-dataset cost-stratified subset
python -m scripts.benchmark_runtime \
    --repeats 3 \
    --threads 1 \
    --out results/runtime_1core_subset.csv

```

---

## 6. CLI Reference (`src.runner`)

| Flag             | Default                            | Description                                                                                   |
| ---------------- | ---------------------------------- | --------------------------------------------------------------------------------------------- |
| `--methods`      | `quant_base`                       | Registered pipeline names from `src/implementations/pipelines.py`.                            |
| `--resample`     | `0`                                | Stratified resample split ID (`0` = original split, `1..29` = resamples).                     |
| `--seeds`        | `7 11 42`                          | Random seeds passed to `ExtraTreesClassifier` (and `RandomFeatureDropout`).                   |
| `--out`          | `results/quant_base_resample0.csv` | Output `.csv` path, directory, or template supporting `{method}` and `{resample}`.            |
| `--datasets`     | _(All 142 datasets)_               | Space-separated dataset names; omit to run on every valid folder in `data/`.                  |
| `--trees`        | `200`                              | Number of trees (`n_estimators`) in `ExtraTreesClassifier`.                                   |
| `--depth`        | `6`                                | Maximum dyadic interval tree depth.                                                           |
| `--div`          | `4`                                | Default quantile divisor ($1 + \lfloor (m - 1) / \text{div} \rfloor$ quantiles per interval). |
| `--max-features` | `0.1`                              | Fraction of features sampled per split in `ExtraTreesClassifier`.                             |
| `--criterion`    | `entropy`                          | Impurity criterion (`entropy` or `gini`).                                                     |
| `--n-jobs`       | `-1`                               | CPU threads for `ExtraTreesClassifier` (`-1` uses all available cores).                       |
