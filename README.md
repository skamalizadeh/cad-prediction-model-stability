# CAD prediction — analysis pipeline

Repository: https://github.com/skamalizadeh/cad-prediction-model-stability

The supplementary workbook with all result tables of the manuscript, the search grids and the configuration selected in every outer fold is in `supplementary_workbook/CAD_supplementary_workbook.xlsx`; running the pipeline rebuilds it as `outputs/CAD_supplementary_workbook.xlsx`.

Reproducibility code for the manuscript comparing a penalised
logistic-regression baseline with nine non-linear or more complex models for
predicting coronary artery disease (CAD) on the public Z-Alizadeh Sani dataset
(303 angiography-referred patients, 53 candidate predictors).

The primary analysis is nested repeated stratified cross-validation (10 repeats
x 5 outer folds, inner 5-fold grid search) with class weighting, in which
scaling, class balancing, tuning, Boruta feature selection, the classification
threshold, and recalibration are all done inside each outer training part
(steps 11-16). It includes variance-corrected and equivalence tests, ranking
stability, calibration before and after recalibration, decision curves
(also by class-balancing strategy), published Diamond-Forrester and CAD
Consortium models, SHAP for logistic regression and random forest, a
sample-size calculation, and more conservative models.

A single stratified 70/30 split (steps 01-10) is kept as a secondary analysis,
to show what a single split reports; it also provides the descriptive data
checks, MDI importance, the logistic-regression coefficient bootstrap, and an
exploratory repeated holdout with fixed hyperparameters.

## Requirements

- Python 3.11 or newer (numpy 2.4 requires Python 3.11 or newer)
- The exact package versions pinned in `requirements.txt`

Exact versions matter: XGBoost and pyGAM fits, and the grid-search choice
among near-tied hyperparameter configurations, can differ slightly across
library versions and across operating systems/CPUs. `00_run_all.py` checks
the installed versions against `requirements.txt` and warns if they differ,
and always records the full environment to `outputs/environment_report.txt`.

## Setup

**Windows (PowerShell / cmd):**

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the raw dataset at `data/CAD_raw.csv` (already included in this
package).

## Running the pipeline

Run everything with one command:

```bash
python 00_run_all.py
```

Useful options:

```bash
python 00_run_all.py --list            # show all steps and exit
python 00_run_all.py --from 05         # resume from step 05 (e.g. after a failure)
python 00_run_all.py --from 03 --to 04 # run only steps 03-04
python 00_run_all.py --only 08         # run a single step
```

Each numbered script can also be run on its own (e.g. `python
03_train_models.py`), as long as the scripts before it have already been run
so their outputs exist.

## Pipeline steps

| Step | Script | Description |
|-----:|--------|-------------|
| 01  | `01_data_preprocessing.py`               | Data preprocessing, baseline table, leakage screen |
| 02  | `02_split_and_feature_selection.py`      | Train/test split, over-sampling, Boruta feature selection |
| 03  | `03_train_models.py`                     | Hyperparameter tuning and training of all models |
| 04  | `04_evaluate_models.py`                  | Evaluation: AUC, DeLong tests, calibration, threshold metrics |
| 05  | `05_shap_analysis.py`                    | SHAP analysis |
| 06  | `06_workflow_diagram_and_final_tables.py`| Workflow figure and final tables |
| 07  | `07_boruta_sensitivity_analysis.py`      | Boruta-7 sensitivity analysis |
| 08  | `08_repeated_holdout_robustness.py`      | Repeated holdout (full 53-feature set, Boruta-7 set, summary) |
| 08a | `08a_paired_tests.py`                    | Repeated holdout: differences from logistic regression with corrected resampled 95% CIs and Cohen's d_z |
| 09  | `09_decision_curve_and_lr_bootstrap.py`  | Decision-curve analysis and LR coefficient bootstrap |
| 10  | `10_build_results_workbook.py`           | Excel workbook collecting all result tables |
| 11  | `11_nested_cv.py`                        | Nested repeated stratified CV of all ten models; class weighting (primary), over-sampling and no balancing |
| 12  | `12_nested_cv_summary.py`                | Nested-CV summary: corrected resampled t-tests, Holm, TOST, ranking stability, calibration, thresholds, decision curves (raw and recalibrated predictions) |
| 13  | `13_clinical_reference_models.py`        | Published Diamond–Forrester and CAD Consortium models; nested logistic models (basic → +clinical → +ECG/lab → +echo), ΔAUC |
| 14  | `14_boruta_nested.py`                    | Boruta repeated inside each outer fold; feature-selection frequency; reduced vs full models |
| 15  | `15_shap_lr_vs_rf.py`                    | SHAP (LinearExplainer) for logistic regression; agreement with random-forest SHAP on the primary split and across outer folds |
| 16  | `16_regularization_and_sample_size.py`   | Riley et al. minimum sample size; elastic-net LR, PCA + LR, restricted RF and LR with an extended C grid, with apparent vs test AUC |
| 16a | `16a_performance_by_sex.py`             | AUC of logistic regression, CatBoost and random forest in men and in women, from the pooled nested-CV predictions |
| 17  | `17_results_tables_figures.py`           | Results tables as LaTeX fragments (`tables/latex/`) and CSV, and the workflow, nested-CV, calibration, decision-curve (including by balancing strategy) and feature-agreement figures |
| 18  | `18_build_supplementary_workbook.py`     | Supplementary workbook (`outputs/CAD_supplementary_workbook.xlsx`): search grids, configuration selected in every outer fold, all result tables |

Running time: step 11 (nested cross-validation, all three class-balancing strategies) takes about 2 to 3 hours on a two-core machine, most of it for the GAM; steps 13 to 16 take about one hour together. Steps 11 and 14 can be stopped and resumed (`--max-minutes`).

`utils_stats.py` holds shared statistical utilities (DeLong test, logistic
recalibration, Youden-threshold metrics) used by several steps and is not
run directly.

## Outputs

- `tables/` — CSV result tables written by each step
- `figures/` — PNG figures
- `models/` — fitted model objects
- `data/` — processed data and predictions (`CAD_processed.csv`,
  `train_raw.csv`, `test_raw.csv`, `predictions_*.csv`)
- `outputs/CAD_prediction_results_tables.xlsx` — single-split result tables
  (built by step 10)
- `outputs/CAD_supplementary_workbook.xlsx` — supplementary workbook with all
  result tables of the manuscript (built by step 18)
- `figures/manuscript/` — the figures under their manuscript names
  (Fig1-Fig6, FigS1-FigS3)
- `outputs/environment_report.txt` — Python/OS/package versions for the run
- `logs/run_all_<timestamp>.log` — full console output of the run

None of these folders need to be created by hand; `00_run_all.py` creates
them as needed.

## Notes on reproducibility

- Every random step uses a fixed seed, so re-running on the same machine and
  library versions reproduces the same results: `random_state=42` for the
  single split, the models, over-sampling and Boruta; seed 2026 for the outer
  folds of the nested cross-validation and 42 + k for the inner folds of outer
  fold k (k = 0 to 49); seeds 1000 to 1049 for the 50 repeated-holdout splits.
- Hyperparameters are chosen by 5-fold grid-search cross-validation
  (Section 3 of the manuscript; single-split values in
  `tables/best_hyperparameters.csv`, the configuration selected in every outer
  fold in `tables/nested_cv_<mode>_folds.csv`).
  When two or more configurations are effectively tied on cross-validated
  AUC, which one is selected can depend on floating-point tie-breaking, and
  this has been observed to differ between operating systems/CPUs for
  XGBoost specifically, changing its test AUC by a few thousandths — see the
  Limitations section of the manuscript. If you need to reproduce the exact
  hyperparameters reported in the paper, compare your
  `tables/best_hyperparameters.csv` against Table 1 of the manuscript.
- In the nested cross-validation (steps 11-16) the GAM's smoothing parameter
  is tuned by the same inner 5-fold grid search (AUC) as every other model.
  Only in the secondary single split (step 03) is it chosen by pyGAM's
  generalised cross-validation, as noted in Table 1 of the manuscript.
