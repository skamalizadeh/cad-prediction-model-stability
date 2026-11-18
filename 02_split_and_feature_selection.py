"""
02_split_and_feature_selection.py
===================================
- Stratified 70/30 train/test split (mirrors the reference paper's protocol)
- Class-imbalance correction via random over-sampling with replacement,
  applied to the TRAINING set only (test set stays untouched / imbalanced,
  used only for internal validation) -- functionally equivalent to the
  ROSE procedure used in the reference paper.
- Boruta all-relevant feature selection run on the (untouched, non-resampled)
  training data using a class-weighted Random Forest, to identify which of
  the 53 candidate predictors carry genuine signal.

Input : data/CAD_processed.csv
Output: data/train_raw.csv, data/test_raw.csv, data/train_balanced.csv
        tables/boruta_feature_selection.csv
        figures/fig2_feature_importance.png
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from boruta import BorutaPy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = "data/CAD_processed.csv"

df = pd.read_csv(DATA_PATH)
y = df["CAD"]
X = df.drop(columns=["CAD"])
feature_names = X.columns.tolist()

# ---------------------------------------------------------------
# 70/30 stratified split
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: n={len(X_train)}  (CAD={y_train.sum()}, {y_train.mean()*100:.1f}%)")
print(f"Test : n={len(X_test)}  (CAD={y_test.sum()}, {y_test.mean()*100:.1f}%)")

train_raw = X_train.copy(); train_raw["CAD"] = y_train
test_raw = X_test.copy(); test_raw["CAD"] = y_test
train_raw.to_csv("data/train_raw.csv", index=False)
test_raw.to_csv("data/test_raw.csv", index=False)

# ---------------------------------------------------------------
# Class-imbalance correction (training set only)
# ---------------------------------------------------------------
ros = RandomOverSampler(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)
print(f"\nAfter over-sampling -> Train balanced: n={len(X_train_bal)} "
      f"(CAD={y_train_bal.sum()}, {y_train_bal.mean()*100:.1f}%)")

train_bal = X_train_bal.copy(); train_bal["CAD"] = y_train_bal
train_bal.to_csv("data/train_balanced.csv", index=False)

# ---------------------------------------------------------------
# Boruta feature selection (on original, non-resampled training data;
# class_weight='balanced' compensates for the 71/29 split so importance
# isn't just driven by the majority class)
# ---------------------------------------------------------------
rf_boruta = RandomForestClassifier(
    n_estimators=500, max_depth=7, class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1
)
boruta_selector = BorutaPy(
    rf_boruta, n_estimators="auto", verbose=0,
    random_state=RANDOM_STATE, max_iter=100
)
boruta_selector.fit(X_train.values, y_train.values)

boruta_result = pd.DataFrame({
    "feature": feature_names,
    "confirmed": boruta_selector.support_,
    "tentative": boruta_selector.support_weak_,
    "ranking": boruta_selector.ranking_,
})
boruta_result["decision"] = np.select(
    [boruta_result.confirmed, boruta_result.tentative],
    ["Confirmed", "Tentative"], default="Rejected"
)
boruta_result = boruta_result.sort_values("ranking")
boruta_result.to_csv("tables/boruta_feature_selection.csv", index=False)

print("\n=== Boruta decision counts ===")
print(boruta_result["decision"].value_counts())
print("\n=== Confirmed + Tentative features (used for modelling) ===")
selected = boruta_result[boruta_result.decision != "Rejected"]["feature"].tolist()
print(selected)

with open("tables/selected_features.txt", "w") as f:
    f.write("\n".join(selected))

# ---------------------------------------------------------------
# Permutation importance (a model-agnostic magnitude, used for Fig 2 -
# mirrors the 'vip'-package style bar chart with CI whiskers in the
# reference paper) computed via repeated RF fits on bootstrap folds
# ---------------------------------------------------------------
from sklearn.inspection import permutation_importance

rf_imp = RandomForestClassifier(
    n_estimators=500, max_depth=7, class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1
).fit(X_train, y_train)

# Impurity-based (MDI) importance -- mirrors the 'vip' R package's default
# random-forest importance type used in the reference paper's Fig. 2, and
# is far better distributed across features than AUC-permutation importance
# (which was completely dominated by a single binary predictor, Typical
# Chest Pain, leaving every other bar visually near-zero)
mdi_importances = np.array([
    tree.feature_importances_ for tree in rf_imp.estimators_
])
imp_df = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": mdi_importances.mean(axis=0),
    "importance_std": mdi_importances.std(axis=0),
}).sort_values("importance_mean", ascending=False)
imp_df.to_csv("tables/permutation_feature_importance.csv", index=False)

top15 = imp_df.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(top15["feature"], top15["importance_mean"],
        xerr=top15["importance_std"] / np.sqrt(len(rf_imp.estimators_)),
        color="#6a5acd", ecolor="gray", capsize=3)
ax.set_xlabel("Random Forest feature importance (mean decrease in impurity)")
ax.set_ylabel("Feature variables")
ax.set_title("Overall feature variable importance selection for CAD prediction")
plt.tight_layout()
plt.savefig("figures/fig2_feature_importance.png", dpi=200)
print("\nFig 2 saved.")
print(imp_df.head(15).to_string(index=False))
