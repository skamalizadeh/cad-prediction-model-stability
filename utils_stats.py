"""
utils_stats.py
================
Shared statistical utilities used across the modelling / evaluation
scripts:

  * Fast DeLong algorithm (Sun & Xu, 2014) for AUC variance estimation
    and for the paired DeLong test comparing two correlated ROC AUCs
    on the same test set -- mirrors 'pROC::roc.test(..., method="delong")'
    used in the reference paper.
  * Logistic (Cox) recalibration -> calibration intercept & slope with
    95% CI, calibration-in-the-large, and Brier score.
  * Youden-index optimal threshold + full confusion-matrix metric set
    (sensitivity, specificity, PPV, NPV, accuracy, F1).
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import roc_curve, brier_score_loss, confusion_matrix


# ---------------------------------------------------------------
# Fast DeLong (Sun & Xu, 2014, IEEE Signal Processing Letters)
# ---------------------------------------------------------------
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_T, m):
    n = preds_sorted_T.shape[1] - m
    pos = preds_sorted_T[:, :m]
    neg = preds_sorted_T[:, m:]
    k = preds_sorted_T.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_T[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    sx = np.atleast_2d(sx); sy = np.atleast_2d(sy)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_auc_ci(y_true, y_score, alpha=0.95):
    """Single-model AUC with DeLong 95% CI."""
    y_true = np.asarray(y_true); y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_true, kind="mergesort")
    y_true_s = y_true[order]; y_score_s = y_score[order]
    m = int(y_true_s.sum())
    preds_sorted_T = y_score_s.reshape(1, -1)
    aucs, delongcov = _fast_delong(preds_sorted_T, m)
    auc = aucs[0]
    var = delongcov[0, 0]
    se = np.sqrt(max(var, 0))
    z = stats.norm.ppf(1 - (1 - alpha) / 2)
    lo, hi = auc - z * se, auc + z * se
    return auc, max(0, lo), min(1, hi), se


def delong_two_model_test(y_true, y_score_1, y_score_2):
    """Paired DeLong test for H0: AUC_1 == AUC_2 on the SAME test set."""
    y_true = np.asarray(y_true)
    y_score_1 = np.asarray(y_score_1, dtype=float)
    y_score_2 = np.asarray(y_score_2, dtype=float)
    order = np.argsort(-y_true, kind="mergesort")
    y_true_s = y_true[order]
    s1 = y_score_1[order]; s2 = y_score_2[order]
    m = int(y_true_s.sum())
    preds_sorted_T = np.vstack([s1, s2])
    aucs, delongcov = _fast_delong(preds_sorted_T, m)
    diff = aucs[0] - aucs[1]
    var_diff = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    se = np.sqrt(max(var_diff, 1e-12))
    z = diff / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"auc_1": aucs[0], "auc_2": aucs[1], "diff": diff, "z": z, "p_value": p}


# ---------------------------------------------------------------
# Calibration: logistic (Cox) recalibration
# ---------------------------------------------------------------
def _robust_logit_fit(endog, exog, offset=None):
    """Newton's method first; fall back to BFGS if the Hessian is singular
    (common when predictions from a flexible ML model nearly perfectly
    separate a small test set)."""
    try:
        return sm.Logit(endog, exog, offset=offset).fit(disp=0)
    except np.linalg.LinAlgError:
        return sm.Logit(endog, exog, offset=offset).fit(disp=0, method="bfgs",
                                                          maxiter=500)


def calibration_metrics(y_true, y_prob, eps=1e-3):
    y_true = np.asarray(y_true, dtype=float)
    # a wider epsilon than machine precision is used deliberately: flexible
    # ML models routinely output near-0/near-1 probabilities on a small test
    # set, and logit() of an extreme probability blows up the Cox-model
    # Hessian; 1e-3 keeps the recalibration numerically stable without
    # materially changing calibration slope/intercept estimates
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    logit_p = np.log(p / (1 - p))

    # calibration slope + intercept (Cox): y ~ 1 + logit(p)
    X = sm.add_constant(logit_p)
    model = _robust_logit_fit(y_true, X)
    intercept, slope = model.params
    ci = model.conf_int(alpha=0.05)
    intercept_ci = tuple(ci[0]); slope_ci = tuple(ci[1])

    # calibration-in-the-large: intercept-only model with slope fixed at 1
    # (logit_p enters as a fixed offset, not as a fitted covariate)
    intercept_only = np.ones((len(y_true), 1))
    citl_model = _robust_logit_fit(y_true, intercept_only, offset=logit_p)
    citl = citl_model.params[0]
    citl_ci = tuple(citl_model.conf_int(alpha=0.05)[0])

    brier = brier_score_loss(y_true, p)
    return {
        "calibration_intercept": intercept, "intercept_ci_lo": intercept_ci[0], "intercept_ci_hi": intercept_ci[1],
        "calibration_slope": slope, "slope_ci_lo": slope_ci[0], "slope_ci_hi": slope_ci[1],
        "calibration_in_the_large": citl, "citl_ci_lo": citl_ci[0], "citl_ci_hi": citl_ci[1],
        "brier_score": brier,
    }


# ---------------------------------------------------------------
# Youden-index threshold + confusion-matrix metrics
# ---------------------------------------------------------------
def youden_threshold_metrics(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    ix = np.argmax(youden)
    best_thr = thr[ix]
    y_pred = (y_prob >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv and sens and not np.isnan(ppv) and not np.isnan(sens) and (ppv + sens) > 0) else np.nan

    # Wilson CI helper for proportions
    def wilson_ci(k, n, z=1.96):
        if n == 0:
            return (np.nan, np.nan)
        phat = k / n
        denom = 1 + z ** 2 / n
        center = (phat + z ** 2 / (2 * n)) / denom
        half = (z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))) / denom
        return (max(0, center - half), min(1, center + half))

    sens_ci = wilson_ci(tp, tp + fn)
    spec_ci = wilson_ci(tn, tn + fp)
    acc_ci = wilson_ci(tp + tn, tp + tn + fp + fn)

    return {
        "threshold": best_thr, "sensitivity": sens, "sens_ci_lo": sens_ci[0], "sens_ci_hi": sens_ci[1],
        "specificity": spec, "spec_ci_lo": spec_ci[0], "spec_ci_hi": spec_ci[1],
        "ppv": ppv, "npv": npv, "accuracy": acc, "acc_ci_lo": acc_ci[0], "acc_ci_hi": acc_ci[1],
        "f1_score": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
