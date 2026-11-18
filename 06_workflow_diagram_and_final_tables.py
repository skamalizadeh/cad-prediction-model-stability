"""
06_workflow_diagram_and_final_tables.py
=======================================
Draws the study workflow diagram (Figure 1) and assembles the final
tables (discrimination and calibration; performance metrics at the
Youden-index threshold) from the tables produced by steps 03 and 04.

Input : tables/best_hyperparameters.csv, tables/table_auc_train_test.csv,
        tables/table_calibration.csv, tables/table2_performance_metrics.csv
Output: figures/fig1_workflow.png
        tables/table_discrimination_calibration_FINAL.csv
        tables/table2_FINAL.csv
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

# =================================================================
# FIG 1: Model development workflow (mirrors the reference paper's Fig. 1)
# =================================================================
fig, ax = plt.subplots(figsize=(13, 4.2))
ax.set_xlim(0, 4.15); ax.set_ylim(0, 1.3)
ax.axis("off")

boxes = [
    ("Data processing", ["Cleaning data (0 missing)", "Variable coding",
                          "Data-leakage screen", "Boruta feature\nimportance selection"]),
    ("Train algorithms (70%, n=212)", ["Random over-sampling\n(class imbalance)",
                                         "Feature importance\n(RF impurity, SHAP)",
                                         "5-fold grid-search\nhyperparameter tuning"]),
    ("Algorithm evaluation", ["Discrimination: AUC\n(DeLong 95% CI + test)",
                                "Calibration: intercept,\nslope, Brier score",
                                "Accuracy, sensitivity,\nspecificity, PPV, NPV, F1"]),
    ("Test algorithm (30%, n=91)", ["Internal validation on\nheld-out imbalanced set"]),
]

box_w = 0.85
xs = [0.15, 1.15, 2.15, 3.15]
for x, (title, items) in zip(xs, boxes):
    rect = mpatches.FancyBboxPatch((x, 0.15), box_w, 1.0, boxstyle="round,pad=0.02",
                                     linewidth=1.3, edgecolor="black", facecolor="#eef1fb")
    ax.add_patch(rect)
    ax.text(x + box_w / 2, 1.07, title, ha="center", va="top", fontsize=9.5, fontweight="bold", wrap=True)
    body = "\n\n".join(items)
    ax.text(x + box_w / 2, 0.62, body, ha="center", va="center", fontsize=7.8)

for x1 in xs[:-1]:
    ax.annotate("", xy=(x1 + box_w + 0.13, 0.65), xytext=(x1 + box_w, 0.65),
                arrowprops=dict(arrowstyle="->", lw=1.5))

plt.suptitle("Model development workflow", y=0.98, fontsize=12)
plt.tight_layout()
plt.savefig("figures/fig1_workflow.png", dpi=200, bbox_inches="tight")
print("Fig 1 saved.")

# =================================================================
# Consolidated final tables
# =================================================================
auc_t = pd.read_csv("tables/table_auc_train_test.csv")
cal_t = pd.read_csv("tables/table_calibration.csv")
perf_t = pd.read_csv("tables/table2_performance_metrics.csv")
hp_t = pd.read_csv("tables/best_hyperparameters.csv")

# Table: discrimination + calibration combined (mirrors ref. paper narrative table)
disc_cal = auc_t.merge(cal_t, on="model").sort_values("AUC_test", ascending=False)
disc_cal["AUC_test_fmt"] = disc_cal.apply(
    lambda r: f"{r.AUC_test:.2f} (95% CI {r.AUC_test_CI_lo:.2f}-{r.AUC_test_CI_hi:.2f})", axis=1)
disc_cal["AUC_train_fmt"] = disc_cal.apply(
    lambda r: f"{r.AUC_train:.2f} (95% CI {r.AUC_train_CI_lo:.2f}-{r.AUC_train_CI_hi:.2f})", axis=1)
disc_cal_out = disc_cal[["model", "AUC_train_fmt", "AUC_test_fmt", "calibration_intercept",
                          "calibration_slope", "brier_score"]]
disc_cal_out.columns = ["Model", "AUC Training (95% CI)", "AUC Testing (95% CI)",
                         "Calibration intercept", "Calibration slope", "Brier score"]
disc_cal_out.to_csv("tables/table_discrimination_calibration_FINAL.csv", index=False)
print("\n=== Discrimination + Calibration (final) ===")
print(disc_cal_out.round(3).to_string(index=False))

# Supplementary Table S1 formatting (Sensitivity/Specificity/PPV/NPV/Acc/F1/threshold)
perf_final = perf_t.copy()
ci_prefix = {"sensitivity": "sens", "specificity": "spec", "accuracy": "acc"}
for c, pfx in ci_prefix.items():
    perf_final[c + "_fmt"] = perf_final.apply(
        lambda r, c=c, pfx=pfx: f"{r[c]:.2f} ({r[pfx+'_ci_lo']:.2f}-{r[pfx+'_ci_hi']:.2f})", axis=1)
perf_final_out = perf_final[["model", "sensitivity_fmt", "specificity_fmt", "ppv", "npv",
                               "accuracy_fmt", "f1_score", "threshold"]]
perf_final_out.columns = ["Model", "Sensitivity (95% CI)", "Specificity (95% CI)", "PPV", "NPV",
                            "Accuracy (95% CI)", "F1-Score", "Youden index threshold"]
perf_final_out = perf_final_out.merge(auc_t[["model", "AUC_test"]].rename(columns={"model": "Model"}), on="Model")
perf_final_out = perf_final_out.sort_values("AUC_test", ascending=False).drop(columns="AUC_test")
perf_final_out.to_csv("tables/table2_FINAL.csv", index=False)
print("\n=== Supplementary Table S1 (sorted by test AUC) ===")
print(perf_final_out.round(3).to_string(index=False))

print("\nHyperparameters used:")
print(hp_t.to_string(index=False))
