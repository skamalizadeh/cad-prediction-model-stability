"""
05_shap_analysis.py
=====================
SHAP (SHapley Additive exPlanations) analysis of the Random Forest model.

Model choice rationale: Logistic Regression had the single highest test
AUC, but Random Forest remained in the top tier of discrimination (test
AUC within ~0.02-0.03 of the leader, overlapping 95% CI) and is directly
supported by SHAP's TreeExplainer (fast, exact tree-based SHAP values,
unlike the sampling-based KernelExplainer needed for logistic regression
or the neural network). It is used here for explainability, not because
of its calibration, which is discussed separately and is not the best
of the ten models on this split.

Output: tables/shap_values_test.csv, tables/shap_feature_ranking.csv
        figures/fig4_shap_summary.png
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd

test_raw = pd.read_csv("data/test_raw.csv")
FEATURES = [c for c in test_raw.columns if c != "CAD"]
yte_raw = test_raw["CAD"]

# Random_Forest.joblib is a fitted imblearn Pipeline (scaler -> sampler ->
# clf; see 03_train_models.py). Pull out the fitted scaler
# and the raw RandomForestClassifier for SHAP's TreeExplainer, which needs
# the bare tree model, not the full pipeline.
rf_pipeline = joblib.load("models/Random_Forest.joblib")
scaler = rf_pipeline.named_steps["scaler"]
rf = rf_pipeline.named_steps["clf"]
Xte_raw_s = pd.DataFrame(scaler.transform(test_raw[FEATURES]), columns=FEATURES)

explainer = shap.TreeExplainer(rf)
shap_exp = explainer(Xte_raw_s)

# shap_exp.values shape for binary RF via TreeExplainer: (n, p, 2) classes
if shap_exp.values.ndim == 3:
    sv = shap_exp.values[:, :, 1]  # class 1 = CAD present
    base_value = shap_exp.base_values[0, 1] if np.ndim(shap_exp.base_values) > 1 else shap_exp.base_values[0]
else:
    sv = shap_exp.values
    base_value = shap_exp.base_values[0] if hasattr(shap_exp.base_values, "__len__") else shap_exp.base_values

sv_df = pd.DataFrame(sv, columns=FEATURES)
sv_df.insert(0, "row_id", range(len(sv_df)))
sv_df.to_csv("tables/shap_values_test.csv", index=False)

mean_abs_shap = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": np.abs(sv).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)
mean_abs_shap.to_csv("tables/shap_feature_ranking.csv", index=False)
print("=== Top 15 features by mean |SHAP value| (Random Forest) ===")
print(mean_abs_shap.head(15).to_string(index=False))

# ---------------------------------------------------------------
# Beeswarm-style summary plot (mirrors Fig. 4 of the reference paper)
# ---------------------------------------------------------------
top_n = 18
top_feats = mean_abs_shap.head(top_n)["feature"].tolist()
idx = [FEATURES.index(f) for f in top_feats]

explanation_subset = shap.Explanation(
    values=sv[:, idx],
    base_values=np.full(sv.shape[0], base_value),
    data=Xte_raw_s.values[:, idx],
    feature_names=top_feats,
)

plt.figure(figsize=(9, 8))
shap.plots.beeswarm(explanation_subset, max_display=top_n, show=False)
plt.title("Random Forest SHAP summary plot: feature contribution to CAD vs no-CAD prediction")
plt.tight_layout()
plt.savefig("figures/fig4_shap_summary.png", dpi=200, bbox_inches="tight")
print("\nFig 4 (SHAP summary) saved.")
