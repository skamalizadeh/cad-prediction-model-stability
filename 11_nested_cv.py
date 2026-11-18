"""
11_nested_cv.py
===============
Nested repeated stratified cross-validation of all ten algorithms.

Outer loop : RepeatedStratifiedKFold (5 folds x R repeats; R = 10 for the primary
             class_weight mode and 5 for each sensitivity mode). Each outer test
             fold is used only once, for evaluation.
Inner loop : StratifiedKFold (5 folds) on the outer training part only.
             Scaling, class balancing and hyperparameter tuning (grid search,
             AUC) are all repeated inside every outer training part.

After tuning, the best configuration is refitted (if its refit fails, which
happens only for the GAM, the next-best configuration is used and its rank is
recorded). The selected configuration is also refitted inside the inner folds
to obtain out-of-fold (OOF) predictions of the outer training part. From
these OOF predictions, and never from the outer test fold, we derive
  * the Youden-index threshold, applied unchanged to the outer test fold;
  * a logistic recalibration (intercept a, slope b on the logit scale),
    applied unchanged to the outer test fold.

Three class-balancing strategies are available (--mode):
  class_weight : balanced sample weights n / (2 * n_class), no resampling
  oversample   : random over-sampling of the minority class (imblearn)
  none         : no balancing

The GAM is tuned with the same inner 5-fold grid search (AUC) as every
other model, so the tuning procedure is identical for all ten algorithms.

Results are checkpointed after every (model, outer fold), so the script can
be stopped and resumed; use --max-minutes to run it in time-limited chunks.

Usage:
    python 11_nested_cv.py                       # all three modes (10, 5 and 5 repeats)
    python 11_nested_cv.py --mode class_weight
    python 11_nested_cv.py --mode oversample --models "Logistic Regression,Random Forest"
    python 11_nested_cv.py --mode none --repeats 10 --max-minutes 9

Output (per mode):
    tables/nested_cv_<mode>_folds.csv        one row per model x outer fold
    tables/nested_cv_<mode>_predictions.csv  outer-fold predictions per patient
"""
import argparse
import contextlib
import io
import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pygam import LogisticGAM, s

RANDOM_STATE = 42
OUTER_SEED = 2026
DATA_PATH = "data/CAD_processed.csv"
TABLE_DIR = "tables"
DEFAULT_REPEATS = {"class_weight": 10, "oversample": 5, "none": 5}
FOLD_COLUMNS = ["model", "repeat", "fold", "mode", "n_train", "n_test", "converged", "auc", "brier",
                "brier_recal", "inner_cv_auc", "config_rank_used", "threshold", "recal_a", "recal_b", "best_params",
                "tp", "fn", "tn", "fp", "sensitivity", "specificity", "ppv", "npv", "balanced_accuracy",
                "fixed05_sensitivity", "fixed05_specificity", "fixed05_balanced_accuracy",
                "error", "fit_seconds"]


# ---------------------------------------------------------------
# Estimator wrappers
# ---------------------------------------------------------------
class BalancedWeights(ClassifierMixin, BaseEstimator):
    """Fits the wrapped estimator with balanced sample weights.

    Weight of a sample of class c = n / (2 * n_c), i.e. the same weighting
    as class_weight='balanced'. Using sample weights lets every algorithm
    (including gradient boosting, naive Bayes, the neural network, XGBoost,
    CatBoost and the GAM) be balanced in exactly the same way.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        w = compute_sample_weight("balanced", y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, sample_weight=w)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)


class GAMClassifier(ClassifierMixin, BaseEstimator):
    """scikit-learn interface to pyGAM's LogisticGAM (one spline per feature)."""

    def __init__(self, lam=1.0, n_splines=5):
        self.lam = lam
        self.n_splines = n_splines

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        terms = s(0, n_splines=self.n_splines)
        for i in range(1, X.shape[1]):
            terms += s(i, n_splines=self.n_splines)
        self.gam_ = LogisticGAM(terms, lam=self.lam)
        with contextlib.redirect_stdout(io.StringIO()):  # pyGAM prints convergence notes
            self.gam_.fit(X, y, weights=sample_weight)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = self.gam_.predict_mu(np.asarray(X, dtype=float))
        if not np.all(np.isfinite(p)):
            raise ValueError("non-finite GAM predictions")
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------
# Model registry (same grids as the primary analysis, 03_train_models.py)
# ---------------------------------------------------------------
def model_registry():
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
            {"C": [0.01, 0.1, 1, 10]}),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {"n_estimators": [200, 400], "max_depth": [None, 5, 10], "min_samples_leaf": [1, 3, 5]}),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "max_depth": [2, 3]}),
        "XGBoost": (
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1),
            {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1],
             "max_depth": [3, 4], "subsample": [0.8, 1.0]}),
        "SVM": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale"]}),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"max_depth": [3, 5, 7, None], "min_samples_leaf": [1, 5, 10]}),
        "CatBoost": (
            CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, thread_count=1),
            {"iterations": [200, 400], "depth": [4, 6], "learning_rate": [0.03, 0.1]}),
        "Naive Bayes": (
            GaussianNB(),
            {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]}),
        "Neural Network": (
            MLPClassifier(max_iter=2000, random_state=RANDOM_STATE, early_stopping=True),
            {"hidden_layer_sizes": [(16,), (32,), (32, 16)], "alpha": [0.0001, 0.001, 0.01]}),
        "GAM": (
            GAMClassifier(n_splines=5),
            {"lam": list(np.logspace(-2, 2, 6))}),
    }


def build_pipeline(estimator, grid, mode):
    """Returns (pipeline, grid with step prefixes) for the chosen balancing mode."""
    if mode == "oversample":
        pipe = ImbPipeline([("scaler", StandardScaler()),
                            ("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
                            ("clf", estimator)])
        prefix = "clf__"
    elif mode == "class_weight":
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", BalancedWeights(estimator))])
        prefix = "clf__estimator__"
    elif mode == "none":
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
        prefix = "clf__"
    else:
        raise ValueError(f"unknown mode: {mode}")
    return pipe, {prefix + k: v for k, v in grid.items()}


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(min(thr[np.argmax(j)], 1.0))


def fit_recalibration(y, p):
    """Logistic recalibration y ~ a + b * logit(p), fitted on OOF predictions."""
    z = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(z, y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def apply_recalibration(p, a, b):
    return expit(a + b * logit(np.clip(p, 1e-6, 1 - 1e-6)))


def threshold_metrics(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp, "sensitivity": sens, "specificity": spec,
            "ppv": tp / (tp + fp) if tp + fp else np.nan,
            "npv": tn / (tn + fn) if tn + fn else np.nan,
            "balanced_accuracy": (sens + spec) / 2}


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["all", "class_weight", "oversample", "none"],
                    help="'all' runs class_weight (primary) and the two sensitivity modes")
    ap.add_argument("--models", default="all", help="comma-separated model names, or 'all'")
    ap.add_argument("--repeats", type=int, default=0,
                    help="outer repeats (default: 10 for class_weight, 5 for the sensitivity modes)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-minutes", type=float, default=0, help="stop after this many minutes (0 = no limit)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    modes = ["class_weight", "oversample", "none"] if args.mode == "all" else [args.mode]
    t_start = time.time()
    for mode in modes:
        repeats = args.repeats or DEFAULT_REPEATS[mode]
        if not run_mode(mode, repeats, args, t_start):
            return


def run_mode(mode, repeats, args, t_start):
    """Runs (or resumes) one balancing mode; returns False if the time limit was reached."""
    os.makedirs(TABLE_DIR, exist_ok=True)
    fold_path = f"{TABLE_DIR}/nested_cv_{mode}_folds.csv"
    pred_path = f"{TABLE_DIR}/nested_cv_{mode}_predictions.csv"

    df = pd.read_csv(DATA_PATH)
    y = df["CAD"].astype(int).values
    X = df.drop(columns=["CAD"])
    features = X.columns.tolist()
    X = X.values.astype(float)

    registry = model_registry()
    names = list(registry) if args.models == "all" else [m.strip() for m in args.models.split(",")]

    done = set()
    if os.path.exists(fold_path):
        prev = pd.read_csv(fold_path)
        done = set(zip(prev["model"], prev["repeat"], prev["fold"]))

    outer = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=repeats, random_state=OUTER_SEED)
    splits = list(outer.split(X, y))
    print(f"Mode: {mode} | models: {len(names)} | outer folds: {len(splits)} | "
          f"{len(features)} features, n = {len(y)}")

    for name in names:
        estimator, grid = registry[name]
        for k, (tr, te) in enumerate(splits):
            rep, fold = k // args.folds, k % args.folds
            if (name, rep, fold) in done:
                continue
            if args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes:
                print("Time limit reached; re-run the same command to resume.")
                return False
            t0 = time.time()
            Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + k)
            pipe, pgrid = build_pipeline(clone(estimator), grid, mode)
            row = {"model": name, "repeat": rep, "fold": fold, "mode": mode,
                   "n_train": len(tr), "n_test": len(te)}
            try:
                gs = GridSearchCV(pipe, pgrid, scoring="roc_auc", cv=inner,
                                  n_jobs=args.n_jobs, refit=False, error_score=np.nan)
                gs.fit(Xtr, ytr)
                scores = np.asarray(gs.cv_results_["mean_test_score"], dtype=float)
                order = [i for i in np.argsort(-np.nan_to_num(scores, nan=-np.inf)) if np.isfinite(scores[i])]
                if not order:
                    raise ValueError("all configurations failed")
                # Use the best configuration whose refit, OOF fits and test predictions
                # all succeed. Only the GAM ever needs a lower-ranked configuration
                # (its fit occasionally diverges); the rank used is recorded.
                last_err = None
                for rank_used, ci in enumerate(order, 1):
                    params = gs.cv_results_["params"][ci]
                    try:
                        final = clone(pipe).set_params(**params).fit(Xtr, ytr)
                        oof = np.full(len(tr), np.nan)
                        for itr, iva in inner.split(Xtr, ytr):
                            m = clone(pipe).set_params(**params).fit(Xtr[itr], ytr[itr])
                            oof[iva] = m.predict_proba(Xtr[iva])[:, 1]
                        p = final.predict_proba(Xte)[:, 1]
                        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(oof))):
                            raise ValueError("non-finite predictions")
                        break
                    except Exception as e:
                        last_err = e
                else:
                    raise ValueError(f"no configuration could be refitted: {last_err}")
                thr = youden_threshold(ytr, oof)
                a, b = fit_recalibration(ytr, oof)
                p_rec = apply_recalibration(p, a, b)
                row.update({"converged": True, "auc": roc_auc_score(yte, p),
                            "brier": brier_score_loss(yte, p), "brier_recal": brier_score_loss(yte, p_rec),
                            "inner_cv_auc": scores[ci], "config_rank_used": rank_used, "threshold": thr,
                            "recal_a": a, "recal_b": b,
                            "best_params": str({kk.split("__")[-1]: vv for kk, vv in params.items()})})
                row.update(threshold_metrics(yte, p, thr))
                row.update({f"fixed05_{kk}": vv for kk, vv in threshold_metrics(yte, p, 0.5).items()
                            if kk in ("sensitivity", "specificity", "balanced_accuracy")})
                preds = pd.DataFrame({"model": name, "repeat": rep, "fold": fold, "patient": te,
                                      "y": yte, "p": p, "p_recal": p_rec})
            except Exception as e:  # e.g. GAM failing to converge on a fold
                row.update({"converged": False, "error": str(e)[:200]})
                preds = None
            row["fit_seconds"] = round(time.time() - t0, 1)

            pd.DataFrame([row]).reindex(columns=FOLD_COLUMNS).to_csv(fold_path, mode="a", header=not os.path.exists(fold_path), index=False)
            if preds is not None:
                preds.to_csv(pred_path, mode="a", header=not os.path.exists(pred_path), index=False)
            auc_txt = f"{row['auc']:.3f}" if row.get("converged") else "failed"
            print(f"[{mode}] {name:20s} repeat {rep} fold {fold}: AUC {auc_txt} ({row['fit_seconds']}s)")

    print(f"[{mode}] finished; elapsed {(time.time() - t_start) / 60:.1f} min.")
    return True


if __name__ == "__main__":
    main()
