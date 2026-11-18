"""
12_nested_cv_summary.py
=======================
Summarises the nested repeated cross-validation of 11_nested_cv.py.

For every class-balancing mode found in tables/nested_cv_<mode>_folds.csv:
  1. Discrimination: mean and SD of outer-fold AUC.
  2. Comparison with logistic regression (reference):
       - mean AUC difference with the corrected repeated k-fold t-test
         (Nadeau & Bengio 2003; Bouckaert & Frank 2004): the variance of
         the fold differences is inflated by (1/(k*r) + n_test/n_train);
       - Holm adjustment over all comparisons with the reference;
       - Cohen's d_z of the paired fold differences;
       - two one-sided tests (TOST) for equivalence at the margin given by
         --margin, and the smallest margin at which equivalence would hold
         (the larger absolute limit of the corrected 90% CI).
  3. Ranking stability: share of outer folds in which each model ranks first,
     its mean rank, and Kendall's coefficient of concordance W across folds.
  4. Calibration and overall accuracy from the pooled outer-fold predictions
     of each repeat (calibration intercept and slope, Brier score), before
     and after the recalibration learned inside the inner folds.
  5. Threshold metrics at the Youden threshold chosen on the inner OOF
     predictions, and at the fixed threshold 0.5.
  6. Net benefit (decision-curve analysis) from the pooled outer-fold
     predictions, averaged over repeats.

Usage:
    python 12_nested_cv_summary.py --margin 0.02

Output: tables/nested_cv_summary_<mode>.csv, tables/nested_cv_comparison_<mode>.csv,
        tables/nested_cv_ranks_<mode>.csv, tables/nested_cv_calibration_<mode>.csv,
        tables/nested_cv_thresholds_<mode>.csv, tables/nested_cv_decision_curve_<mode>.csv,
        tables/nested_cv_modes_overview.csv
"""
import argparse
import os
import warnings

warnings.filterwarnings("ignore")  # overflow notes from extreme naive-Bayes probabilities

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
import statsmodels.api as sm

TABLE_DIR = "tables"
REFERENCE = "Logistic Regression"
MODES = ["class_weight", "oversample", "none"]


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj


def calibration_intercept_slope(y, p):
    # Probabilities are clipped to [0.001, 0.999] so that models returning exact
    # 0/1 predictions (decision tree, naive Bayes) give finite logits.
    z = logit(np.clip(p, 1e-3, 1 - 1e-3))
    slope = LogisticRegression(C=1e6, max_iter=5000).fit(z.reshape(-1, 1), y).coef_[0, 0]
    # calibration-in-the-large: intercept with the slope fixed at 1 (offset model)
    citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=z).fit().params[0]
    if abs(citl) > 50:  # not estimable (near-deterministic predictions, e.g. naive Bayes)
        citl = np.nan
    return citl, slope


def net_benefit(y, p, t):
    n = len(y)
    pred = p >= t
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
    return tp / n - fp / n * t / (1 - t)


def summarise_mode(mode, margin):
    fold_path = f"{TABLE_DIR}/nested_cv_{mode}_folds.csv"
    pred_path = f"{TABLE_DIR}/nested_cv_{mode}_predictions.csv"
    if not os.path.exists(fold_path):
        return None
    f = pd.read_csv(fold_path)
    f = f.drop_duplicates(["model", "repeat", "fold"], keep="last")
    k = f["fold"].max() + 1
    r = f["repeat"].max() + 1
    n_tr, n_te = f["n_train"].mean(), f["n_test"].mean()
    ok = f[f["converged"] == True]

    # 1. discrimination
    summ = (ok.groupby("model")["auc"].agg(["mean", "std", "min", "max", "count"])
              .rename(columns={"mean": "mean_auc", "std": "sd_auc", "count": "n_folds_converged"}))
    summ["n_folds_total"] = f.groupby("model").size()
    summ = summ.sort_values("mean_auc", ascending=False)
    summ.to_csv(f"{TABLE_DIR}/nested_cv_summary_{mode}.csv")

    # 2. comparison with the reference
    wide = ok.pivot_table(index=["repeat", "fold"], columns="model", values="auc")
    rows = []
    for m in wide.columns:
        if m == REFERENCE:
            continue
        d = (wide[REFERENCE] - wide[m]).dropna()
        J = len(d)
        mu, sd = d.mean(), d.std(ddof=1)
        se = sd * np.sqrt(1 / J + n_te / n_tr)
        df_ = J - 1
        p = 2 * stats.t.sf(abs(mu / se), df_)
        t95, t90 = stats.t.ppf(0.975, df_), stats.t.ppf(0.95, df_)
        lo90, hi90 = mu - t90 * se, mu + t90 * se
        p_tost = max(stats.t.sf((mu + margin) / se, df_), stats.t.cdf((mu - margin) / se, df_))
        rows.append({"model": m, "n_paired_folds": J, "mean_diff_ref_minus_model": mu,
                     "ci95_lo": mu - t95 * se, "ci95_hi": mu + t95 * se, "p_corrected": p,
                     "cohens_dz": mu / sd if sd > 0 else np.nan,
                     "ci90_lo": lo90, "ci90_hi": hi90, "tost_margin": margin, "p_tost": p_tost,
                     "equivalent_at_margin": p_tost < 0.05,
                     "smallest_equivalence_margin": max(abs(lo90), abs(hi90))})
    comp = pd.DataFrame(rows)
    comp["p_holm"] = holm(comp["p_corrected"])
    comp = comp.sort_values("mean_diff_ref_minus_model")
    comp.to_csv(f"{TABLE_DIR}/nested_cv_comparison_{mode}.csv", index=False)

    # 3. ranking stability (folds in which every model converged)
    # AUCs are rounded to 9 decimals first, so that two models with the same AUC are
    # treated as tied even if floating-point summation differs in the last digits
    complete = wide.dropna().round(9)
    ranks = complete.rank(axis=1, ascending=False, method="average")
    n_raters, n_items = ranks.shape
    R = ranks.sum(axis=0)
    W = 12 * np.sum((R - R.mean()) ** 2) / (n_raters ** 2 * (n_items ** 3 - n_items))
    # a fold in which several models share the highest AUC counts as "first" for each
    # of them, so the shares can sum to more than 100%; the number of such folds is recorded
    first = ranks == ranks.min(axis=1).values[:, None]
    rk = pd.DataFrame({"share_rank_1": first.mean(),
                       "mean_rank": ranks.mean(), "sd_rank": ranks.std()})
    rk = rk.sort_values("mean_rank")
    rk["kendall_W"] = W
    rk["n_folds_used"] = n_raters
    rk["n_folds_tied_first"] = int((first.sum(axis=1) > 1).sum())
    rk.to_csv(f"{TABLE_DIR}/nested_cv_ranks_{mode}.csv")

    # 4-6. pooled predictions per repeat
    cal_rows, dca_rows = [], []
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    if os.path.exists(pred_path):
        pr = pd.read_csv(pred_path).drop_duplicates(["model", "repeat", "patient"], keep="last")
        for (m, rep), g in pr.groupby(["model", "repeat"]):
            if len(g) < g["patient"].nunique() or len(g) < 0.99 * pr["patient"].nunique():
                continue  # repeat incomplete (a fold failed)
            yy = g["y"].values
            for kind, col in (("raw", "p"), ("recalibrated", "p_recal")):
                citl, slope = calibration_intercept_slope(yy, g[col].values)
                cal_rows.append({"model": m, "repeat": rep, "predictions": kind, "citl": citl,
                                 "slope": slope, "brier": brier_score_loss(yy, g[col].values)})
            # net benefit of the recalibrated predictions (main analysis) and of the
            # raw predictions (to show how class balancing itself affects the curves)
            for kind, col in (("recalibrated", "p_recal"), ("raw", "p")):
                for t in thresholds:
                    dca_rows.append({"model": m, "predictions": kind, "repeat": rep, "threshold": t,
                                     "net_benefit": net_benefit(yy, g[col].values, t)})
            if m == REFERENCE:
                prev = yy.mean()
                for kind in ("recalibrated", "raw"):
                    for t in thresholds:
                        dca_rows.append({"model": "Treat all", "predictions": kind, "repeat": rep, "threshold": t,
                                         "net_benefit": prev - (1 - prev) * t / (1 - t)})
    cal = pd.DataFrame(cal_rows)
    if len(cal):
        cal.groupby(["model", "predictions"])[["citl", "slope", "brier"]].agg(["mean", "std"]) \
           .to_csv(f"{TABLE_DIR}/nested_cv_calibration_{mode}.csv")
    dca = pd.DataFrame(dca_rows)
    if len(dca):
        dca.groupby(["model", "predictions", "threshold"])["net_benefit"].agg(["mean", "std"]).reset_index() \
           .to_csv(f"{TABLE_DIR}/nested_cv_decision_curve_{mode}.csv", index=False)

    thr_cols = ["threshold", "sensitivity", "specificity", "ppv", "npv", "balanced_accuracy",
                "fixed05_sensitivity", "fixed05_specificity", "fixed05_balanced_accuracy"]
    ok.groupby("model")[thr_cols].mean().loc[summ.index] \
      .to_csv(f"{TABLE_DIR}/nested_cv_thresholds_{mode}.csv")

    print(f"\n=== Mode: {mode} ({r} repeats x {k} folds) ===")
    print(summ[["mean_auc", "sd_auc", "n_folds_converged"]].round(3).to_string())
    print("\nComparison with logistic regression (corrected t-test, Holm, TOST):")
    print(comp[["model", "mean_diff_ref_minus_model", "ci95_lo", "ci95_hi", "p_corrected", "p_holm",
                "cohens_dz", "smallest_equivalence_margin", "equivalent_at_margin"]]
          .round(4).to_string(index=False))
    print(f"\nRanking stability (Kendall's W = {W:.3f}, {n_raters} folds):")
    print(rk[["share_rank_1", "mean_rank", "sd_rank"]].round(2).to_string())
    if len(cal):
        print("\nCalibration of pooled predictions (mean over repeats):")
        print(cal.groupby(["model", "predictions"])[["citl", "slope", "brier"]].mean().round(3).to_string())
    return summ["mean_auc"].rename(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.02,
                    help="pre-specified equivalence margin (AUC units) for TOST")
    args = ap.parse_args()
    overview = [s for s in (summarise_mode(m, args.margin) for m in MODES) if s is not None]
    if overview:
        ov = pd.concat(overview, axis=1)
        ov.to_csv(f"{TABLE_DIR}/nested_cv_modes_overview.csv")
        print("\nMean outer-fold AUC by balancing mode:")
        print(ov.round(3).to_string())


if __name__ == "__main__":
    main()
