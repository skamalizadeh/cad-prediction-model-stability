"""
08a_paired_tests.py
===================
Comparison of Logistic Regression with every other model across the
repeated-holdout splits, for the full 53-feature and the Boruta-7 feature
sets. Repeats in which either model failed to converge are excluded (see
n_paired_repeats).

Because the training and test sets of different splits overlap, a naive
paired t-test is anti-conservative. The variance of the paired differences
is therefore inflated by (1/J + n_test/n_train) (corrected resampled t-test,
Nadeau and Bengio 2003), and the mean difference is reported with its
corrected 95% confidence interval and Cohen's d_z (descriptive). This
analysis is exploratory: the hyperparameters were selected on a partition
that overlaps with later test sets.

Nadeau C, Bengio Y. Inference for the generalization error.
Machine Learning 2003;52(3):239-281.

Input : tables/table_repeated_holdout_full53.csv
        tables/table_repeated_holdout_boruta7.csv
Output: tables/table_repeated_holdout_paired_tests.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

TABLE_DIR = "tables"
N_TEST, N_TRAIN = 91, 212   # stratified 70/30 split of 303 patients

rows = []
for fset, fname in [("Full (53 features)", "table_repeated_holdout_full53.csv"),
                     ("Boruta-confirmed (7 features)", "table_repeated_holdout_boruta7.csv")]:
    df = pd.read_csv(f"{TABLE_DIR}/{fname}")
    ref = "Logistic Regression"
    for m in df.columns:
        if m == ref:
            continue
        paired = df[[ref, m]].dropna()
        diff = paired[m] - paired[ref]          # model minus logistic regression
        J = len(diff)
        se = diff.std(ddof=1) * np.sqrt(1 / J + N_TEST / N_TRAIN)
        t95 = stats.t.ppf(0.975, J - 1)
        rows.append({
            "feature_set": fset, "model": m, "n_paired_repeats": J,
            "mean_AUC_LR": paired[ref].mean(), "mean_AUC_model": paired[m].mean(),
            "mean_diff_model_minus_LR": diff.mean(), "sd_diff": diff.std(ddof=1),
            "corrected_ci95_lo": diff.mean() - t95 * se, "corrected_ci95_hi": diff.mean() + t95 * se,
            "cohens_dz": diff.mean() / diff.std(ddof=1),
        })

out = pd.DataFrame(rows)
out.to_csv(f"{TABLE_DIR}/table_repeated_holdout_paired_tests.csv", index=False)
pd.set_option("display.width", 160)
print(out.round(4).to_string(index=False))
