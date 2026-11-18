"""
04_evaluate_models.py
=====================
Evaluation of all ten models on the training and test sets:

  * AUC with DeLong 95% confidence intervals (train and test)
  * DeLong pairwise test of the best test-AUC model against every other model
  * Calibration on the test set: intercept, slope and Brier score
  * Sensitivity, specificity, PPV, NPV, accuracy and F1 at the
    Youden-index threshold (test set)
  * ROC curves and calibration plot

Input : data/predictions_train.csv, data/predictions_test.csv
Output: tables/table_auc_train_test.csv
        tables/table_delong_pairwise_test.csv
        tables/table_calibration.csv
        tables/table2_performance_metrics.csv
        figures/fig3_roc_curves.png
        figures/fig3_calibration.png
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils_stats import (delong_auc_ci, delong_two_model_test,
                          calibration_metrics, youden_threshold_metrics)

ptr = pd.read_csv("data/predictions_train.csv")
pte = pd.read_csv("data/predictions_test.csv")
MODELS = [c for c in ptr.columns if c != "y_true"]

# ---------------------------------------------------------------
# AUC (train & test) with DeLong 95% CI
# ---------------------------------------------------------------
auc_rows = []
for m in MODELS:
    auc_tr, lo_tr, hi_tr, _ = delong_auc_ci(ptr["y_true"], ptr[m])
    auc_te, lo_te, hi_te, _ = delong_auc_ci(pte["y_true"], pte[m])
    auc_rows.append({
        "model": m,
        "AUC_train": auc_tr, "AUC_train_CI_lo": lo_tr, "AUC_train_CI_hi": hi_tr,
        "AUC_test": auc_te, "AUC_test_CI_lo": lo_te, "AUC_test_CI_hi": hi_te,
    })
auc_table = pd.DataFrame(auc_rows).sort_values("AUC_test", ascending=False)
auc_table.to_csv("tables/table_auc_train_test.csv", index=False)
print("=== AUC (train / test) with DeLong 95% CI ===")
for _, r in auc_table.iterrows():
    print(f"{r['model']:28s}  Train={r.AUC_train:.3f} ({r.AUC_train_CI_lo:.3f}-{r.AUC_train_CI_hi:.3f})   "
          f"Test={r.AUC_test:.3f} ({r.AUC_test_CI_lo:.3f}-{r.AUC_test_CI_hi:.3f})")

# ---------------------------------------------------------------
# Pairwise DeLong test on TEST set: best model vs every other model
# ---------------------------------------------------------------
best_model = auc_table.iloc[0]["model"]
pairwise_rows = []
for m in MODELS:
    if m == best_model:
        continue
    res = delong_two_model_test(pte["y_true"], pte[best_model], pte[m])
    pairwise_rows.append({"model_A (best)": best_model, "model_B": m,
                           "AUC_A": res["auc_1"], "AUC_B": res["auc_2"],
                           "diff": res["diff"], "z": res["z"], "p_value": res["p_value"]})
pairwise_df = pd.DataFrame(pairwise_rows)
pairwise_df.to_csv("tables/table_delong_pairwise_test.csv", index=False)
print(f"\n=== DeLong pairwise test (best model = {best_model}) on TEST set ===")
print(pairwise_df.to_string(index=False))

# ---------------------------------------------------------------
# Calibration (test set) for every model
# ---------------------------------------------------------------
cal_rows = []
for m in MODELS:
    p = pte[m].clip(1e-6, 1 - 1e-6)
    cal = calibration_metrics(pte["y_true"], p)
    cal["model"] = m
    cal_rows.append(cal)
cal_table = pd.DataFrame(cal_rows).set_index("model").reset_index()
cal_table.to_csv("tables/table_calibration.csv", index=False)
print("\n=== Calibration (test set) ===")
print(cal_table[["model", "calibration_intercept", "calibration_slope", "brier_score"]]
      .to_string(index=False))

# ---------------------------------------------------------------
# Youden-index threshold metrics (test set) for every model
# ---------------------------------------------------------------
yth_rows = []
for m in MODELS:
    yth = youden_threshold_metrics(pte["y_true"], pte[m])
    yth["model"] = m
    yth_rows.append(yth)
yth_table = pd.DataFrame(yth_rows).set_index("model").reset_index()
col_order = ["model", "sensitivity", "sens_ci_lo", "sens_ci_hi", "specificity",
             "spec_ci_lo", "spec_ci_hi", "ppv", "npv", "accuracy", "acc_ci_lo",
             "acc_ci_hi", "f1_score", "threshold", "tp", "fp", "tn", "fn"]
yth_table = yth_table[col_order]
yth_table.to_csv("tables/table2_performance_metrics.csv", index=False)
print("\n=== Performance metrics at Youden index threshold (test set) ===")
print(yth_table[["model", "sensitivity", "specificity", "ppv", "npv", "accuracy",
                  "f1_score", "threshold"]].round(3).to_string(index=False))

# =================================================================
# FIGURE: ROC curves, train (A) & test (B) panels  -> Fig 3
# =================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))

for ax, (df_, title) in zip(axes, [(ptr, "A) AUC – Training Dataset (n = 212)"),
                                     (pte, "B) AUC – Testing Dataset (n = 91)")]):
    for m, c in zip(MODELS, colors):
        fpr, tpr, _ = roc_curve(df_["y_true"], df_[m])
        auc_val = auc_table.set_index("model").loc[m, "AUC_train" if "Training" in title else "AUC_test"]
        ax.plot(fpr * 100, tpr * 100, label=f"{m} = {auc_val:.2f}", color=c, lw=1.6)
    ax.plot([0, 100], [0, 100], "k--", lw=1)
    ax.set_xlabel("False Positive Rate (%) (100 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity in %)")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")

plt.tight_layout()
plt.savefig("figures/fig3_roc_curves.png", dpi=200)
print("\nFig 3 (ROC curves) saved.")

# =================================================================
# FIGURE: Calibration curves for the two best test-AUC models -> Fig 3C/D
# =================================================================
top2 = auc_table.head(2)["model"].tolist()
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, m in zip(axes, top2):
    p = pte[m].clip(1e-6, 1 - 1e-6).values
    y = pte["y_true"].values
    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]
    # flexible calibration curve via LOWESS (locally weighted scatterplot
    # smoothing) -- the standard non-parametric smoother for calibration
    # plots, playing the role of the reference paper's restricted-cubic-
    # spline (RCS) flexible calibration curve
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smoothed = lowess(y_sorted.astype(float), p_sorted, frac=0.5, it=0, return_sorted=True)
    ax.plot([0, 1], [0, 1], color="blue", lw=1, label="Ideal")
    ax.plot(smoothed[:, 0], smoothed[:, 1], color="black", lw=1.8, label="Flexible calibration (LOWESS)")
    ax.plot(p_sorted, y_sorted, "|", color="gray", alpha=0.5, ms=8, label="Observed events (0/1)")
    cal = cal_table.set_index("model").loc[m]
    ax.set_title(f"{m} calibration\nintercept={cal.calibration_intercept:.2f}, "
                  f"slope={cal.calibration_slope:.2f}")
    ax.set_xlabel("Predicted Probability of CAD")
    ax.set_ylabel("Observed Risk of CAD")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig3_calibration.png", dpi=200)
print("Calibration figure saved for:", top2)

print("\nBest model on TEST AUC:", best_model)
