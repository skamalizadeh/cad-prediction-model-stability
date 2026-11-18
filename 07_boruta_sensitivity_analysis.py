"""
07_boruta_sensitivity_analysis.py
=================================
Sensitivity analysis: retrains all 10 models using ONLY the 7
Boruta-confirmed features (Age, Typical Chest Pain, Atypical, FBS,
Region RWMA, ESR, TG), on the exact same train/test split as the primary
(53-feature) analysis, so the two feature sets are directly comparable.

Hyperparameters are tuned as in 03_train_models.py: GridSearchCV on the
RAW (non-resampled) 7-feature training data with an imblearn Pipeline
(scaler -> sampler -> clf), so over-sampling is nested inside each CV fold.

Output: models/*_boruta7.joblib
        data/predictions_train_boruta7.csv, data/predictions_test_boruta7.csv
        tables/table_boruta7_vs_full_comparison.csv
"""
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pygam import LogisticGAM, s

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

from utils_stats import delong_auc_ci, calibration_metrics

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

DATA_DIR = "data"
TABLE_DIR = "tables"
MODEL_DIR = "models"

BORUTA7 = ["Age", "Typical Chest Pain", "Atypical", "FBS", "Region RWMA", "ESR", "TG"]

train_raw_full = pd.read_csv(f"{DATA_DIR}/train_raw.csv")
test_raw_full = pd.read_csv(f"{DATA_DIR}/test_raw.csv")

Xtr_raw, ytr_raw = train_raw_full[BORUTA7], train_raw_full["CAD"]
Xte_raw, yte_raw = test_raw_full[BORUTA7], test_raw_full["CAD"]

predictions_train = pd.DataFrame({"y_true": ytr_raw.values})
predictions_test = pd.DataFrame({"y_true": yte_raw.values})

model_grid = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        {"clf__C": [0.01, 0.1, 1, 10], "clf__solver": ["lbfgs"]}),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 5, 10], "clf__min_samples_leaf": [1, 3, 5]}),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.01, 0.1], "clf__max_depth": [2, 3]}),
    "XGBoost": (
        XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1),
        {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.01, 0.1],
         "clf__max_depth": [3, 4], "clf__subsample": [0.8, 1.0]}),
    "SVM": (
        SVC(probability=True, random_state=RANDOM_STATE),
        {"clf__C": [0.1, 1, 10], "clf__kernel": ["rbf", "linear"], "clf__gamma": ["scale"]}),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"clf__max_depth": [3, 5, 7, None], "clf__min_samples_leaf": [1, 5, 10]}),
    "CatBoost": (
        CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, thread_count=1),
        {"clf__iterations": [200, 400], "clf__depth": [4, 6], "clf__learning_rate": [0.03, 0.1]}),
    "Naive Bayes": (
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]}),
    "Neural Network": (
        MLPClassifier(max_iter=2000, random_state=RANDOM_STATE, early_stopping=True),
        {"clf__hidden_layer_sizes": [(4,), (8,), (8, 4)], "clf__alpha": [0.0001, 0.001, 0.01]}),
}

best_params_boruta7 = {}
for name, (estimator, grid) in model_grid.items():
    t0 = time.time()
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
        ("clf", estimator),
    ])
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=CV, n_jobs=-1, refit=True)
    gs.fit(Xtr_raw, ytr_raw)
    best = gs.best_estimator_
    joblib.dump(best, f"{MODEL_DIR}/{name.replace(' ', '_')}_boruta7.joblib")
    best_params_boruta7[name] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    predictions_train[name] = best.predict_proba(Xtr_raw)[:, 1]
    predictions_test[name] = best.predict_proba(Xte_raw)[:, 1]
    print(f"[Boruta7 | {name}] best CV-AUC={gs.best_score_:.3f}  ({time.time()-t0:.1f}s)")

# GAM (7 features -> 7 spline terms). Smoothing parameter selected by
# pyGAM's own GCV grid search on the scaled+over-sampled full training
# set, for the same reason given in 03_train_models.py.
t0 = time.time()
scaler_gam7 = StandardScaler().fit(Xtr_raw)
ros_gam7 = RandomOverSampler(random_state=RANDOM_STATE)
Xtr_gam7_bal, ytr_gam7_bal = ros_gam7.fit_resample(scaler_gam7.transform(Xtr_raw), ytr_raw)

terms = s(0, n_splines=6)
for i in range(1, len(BORUTA7)):
    terms += s(i, n_splines=6)
gam = LogisticGAM(terms)
gam.gridsearch(Xtr_gam7_bal, ytr_gam7_bal.values, lam=np.logspace(-2, 2, 6), progress=False)
joblib.dump(gam, f"{MODEL_DIR}/GAM_boruta7.joblib")
joblib.dump(scaler_gam7, f"{MODEL_DIR}/scaler_gam_boruta7.joblib")
predictions_train["GAM"] = gam.predict_mu(scaler_gam7.transform(Xtr_raw))
predictions_test["GAM"] = gam.predict_mu(scaler_gam7.transform(Xte_raw))
print(f"[Boruta7 | GAM] fitted ({time.time()-t0:.1f}s)")

predictions_train.to_csv(f"{DATA_DIR}/predictions_train_boruta7.csv", index=False)
predictions_test.to_csv(f"{DATA_DIR}/predictions_test_boruta7.csv", index=False)

# =================================================================
# Comparison: full 53-feature model vs Boruta-7 model
# =================================================================
ptr_full = pd.read_csv(f"{DATA_DIR}/predictions_train.csv")
pte_full = pd.read_csv(f"{DATA_DIR}/predictions_test.csv")
MODELS = [c for c in predictions_test.columns if c != "y_true"]

rows = []
for m in MODELS:
    auc_tr_full, *_ = delong_auc_ci(ptr_full["y_true"], ptr_full[m])
    auc_te_full, lo_f, hi_f, _ = delong_auc_ci(pte_full["y_true"], pte_full[m])
    auc_tr_b7, *_ = delong_auc_ci(predictions_train["y_true"], predictions_train[m])
    auc_te_b7, lo_b, hi_b, _ = delong_auc_ci(predictions_test["y_true"], predictions_test[m])

    cal_full = calibration_metrics(pte_full["y_true"], pte_full[m].clip(1e-3, 1 - 1e-3))
    cal_b7 = calibration_metrics(predictions_test["y_true"], predictions_test[m].clip(1e-3, 1 - 1e-3))

    rows.append({
        "model": m,
        "AUC_train_full53": auc_tr_full, "AUC_test_full53": auc_te_full,
        "train_test_gap_full53": auc_tr_full - auc_te_full,
        "Brier_full53": cal_full["brier_score"],
        "AUC_train_boruta7": auc_tr_b7, "AUC_test_boruta7": auc_te_b7,
        "AUC_test_boruta7_CI_lo": lo_b, "AUC_test_boruta7_CI_hi": hi_b,
        "train_test_gap_boruta7": auc_tr_b7 - auc_te_b7,
        "Brier_boruta7": cal_b7["brier_score"],
        "AUC_test_delta (boruta7 - full53)": auc_te_b7 - auc_te_full,
    })

comp = pd.DataFrame(rows).sort_values("AUC_test_boruta7", ascending=False)
comp.to_csv(f"{TABLE_DIR}/table_boruta7_vs_full_comparison.csv", index=False)
print("\n=== Boruta-7 vs full-53-feature comparison (test set) ===")
print(comp[["model", "AUC_test_full53", "AUC_test_boruta7", "AUC_test_delta (boruta7 - full53)",
            "train_test_gap_full53", "train_test_gap_boruta7", "Brier_full53", "Brier_boruta7"]]
      .round(3).to_string(index=False))

print("\nMean train-test AUC gap -- full 53 features:", round(comp["train_test_gap_full53"].mean(), 4))
print("Mean train-test AUC gap -- Boruta 7 features:", round(comp["train_test_gap_boruta7"].mean(), 4))
print("Mean Brier -- full 53 features:", round(comp["Brier_full53"].mean(), 4))
print("Mean Brier -- Boruta 7 features:", round(comp["Brier_boruta7"].mean(), 4))
