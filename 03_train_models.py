"""
03_train_models.py
==================
Trains and hyperparameter-tunes ten models on the CAD dataset.

Hyperparameters are selected by 5-fold stratified GridSearchCV (scoring:
ROC AUC) on the RAW (non-resampled) training set. Every estimator sits
inside an imblearn Pipeline (StandardScaler -> RandomOverSampler ->
estimator), so scaling and over-sampling are applied only to the training
portion of each CV fold, and each validation fold keeps its natural class
prevalence; duplicated minority-class rows therefore never appear on both
sides of a fold. The final refit on the full training set scales and
over-samples the whole raw training set.

Input : data/train_raw.csv, data/test_raw.csv
Output: models/*.joblib (fitted pipelines + GAM model)
        data/predictions_train.csv, data/predictions_test.csv
        tables/best_hyperparameters.csv
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

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

DATA_DIR = "data"
MODEL_DIR = "models"
TABLE_DIR = "tables"

# ---------------------------------------------------------------
# Load RAW (non-resampled) data only -- over-sampling happens
# inside the pipeline, per CV fold, not once up front.
# ---------------------------------------------------------------
train_raw = pd.read_csv(f"{DATA_DIR}/train_raw.csv")
test_raw = pd.read_csv(f"{DATA_DIR}/test_raw.csv")

FEATURES = [c for c in train_raw.columns if c != "CAD"]
Xtr_raw, ytr_raw = train_raw[FEATURES], train_raw["CAD"]
Xte_raw, yte_raw = test_raw[FEATURES], test_raw["CAD"]

predictions_train = pd.DataFrame({"y_true": ytr_raw.values})
predictions_test = pd.DataFrame({"y_true": yte_raw.values})
hparam_rows = []

# ---------------------------------------------------------------
# Model + hyperparameter-grid registry. Every grid key is prefixed
# "clf__" because each estimator sits inside an imblearn Pipeline
# with named steps: scaler -> sampler -> clf.
# ---------------------------------------------------------------
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
        {"clf__hidden_layer_sizes": [(16,), (32,), (32, 16)], "clf__alpha": [0.0001, 0.001, 0.01]}),
}

fitted_models = {}
for name, (estimator, grid) in model_grid.items():
    t0 = time.time()
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
        ("clf", estimator),
    ])
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=CV, n_jobs=-1, refit=True)
    gs.fit(Xtr_raw, ytr_raw)  # <-- RAW data; pipeline over-samples inside each fold
    best = gs.best_estimator_  # fitted pipeline: scaler+sampler fit on full raw training set, then clf
    fitted_models[name] = best
    joblib.dump(best, f"{MODEL_DIR}/{name.replace(' ', '_')}.joblib")

    predictions_train[name] = best.predict_proba(Xtr_raw)[:, 1]
    predictions_test[name] = best.predict_proba(Xte_raw)[:, 1]

    best_params_clean = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    hparam_rows.append({"model": name, "best_params": str(best_params_clean),
                         "best_cv_auc": gs.best_score_, "fit_time_sec": round(time.time() - t0, 1)})
    print(f"[{name}] best CV-AUC={gs.best_score_:.3f}  params={best_params_clean}  ({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------
# Logistic GAM (pyGAM): one penalised spline per feature, i.e. smooth,
# additive non-linear effects that remain directly interpretable. The smoothing parameter lam
# is selected by pyGAM's own built-in gridsearch, which minimises the
# generalized cross-validation (GCV) score -- an analytic, in-sample
# criterion, not an explicit k-fold split -- so it does not involve
# the fold-boundary duplication issue that over-sampling before k-fold CV would cause. It
# is fit (both for lam-selection and the final model) on the scaled +
# over-sampled full training set, consistent with the final refit of
# every sklearn pipeline above. The 7-feature GAM in
# 07_boruta_sensitivity_analysis.py is treated in the same way.
# ---------------------------------------------------------------
t0 = time.time()
n_features = Xtr_raw.shape[1]

scaler_gam = StandardScaler().fit(Xtr_raw)
ros_gam = RandomOverSampler(random_state=RANDOM_STATE)
Xtr_gam_bal, ytr_gam_bal = ros_gam.fit_resample(scaler_gam.transform(Xtr_raw), ytr_raw)

terms = s(0, n_splines=5)
for i in range(1, n_features):
    terms += s(i, n_splines=5)
gam = LogisticGAM(terms)
gam.gridsearch(Xtr_gam_bal, ytr_gam_bal.values, lam=np.logspace(-2, 2, 6), progress=False)

fitted_models["GAM"] = gam
joblib.dump(gam, f"{MODEL_DIR}/GAM.joblib")
joblib.dump(scaler_gam, f"{MODEL_DIR}/scaler_gam.joblib")

predictions_train["GAM"] = gam.predict_mu(scaler_gam.transform(Xtr_raw))
predictions_test["GAM"] = gam.predict_mu(scaler_gam.transform(Xte_raw))
hparam_rows.append({"model": "GAM",
                     "best_params": f"lam={gam.lam[0][0]:.4g} (uniform), n_splines=5 (GCV-selected, not k-fold CV)",
                     "best_cv_auc": np.nan, "fit_time_sec": round(time.time() - t0, 1)})
print(f"[GAM] fitted in {time.time()-t0:.1f}s, lam={gam.lam[0][0]:.4g}")

# ---------------------------------------------------------------
# Save
# ---------------------------------------------------------------
predictions_train.to_csv(f"{DATA_DIR}/predictions_train.csv", index=False)
predictions_test.to_csv(f"{DATA_DIR}/predictions_test.csv", index=False)
pd.DataFrame(hparam_rows).to_csv(f"{TABLE_DIR}/best_hyperparameters.csv", index=False)

print("\nAll 10 models trained with over-sampling nested inside CV folds.")
print("Predictions + hyperparameters saved.")
print("Models:", list(fitted_models.keys()))
