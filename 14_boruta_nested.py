"""
14_boruta_nested.py
===================
Boruta feature selection repeated inside every outer training part of the
nested repeated cross-validation (same outer splits as 11_nested_cv.py).

Boruta is an all-relevant method: it keeps every feature whose importance
exceeds that of random "shadow" features. It is not designed to find the
smallest subset with the best predictive performance (minimal-optimal), so
the reduced-feature comparison below is exploratory.

For every outer fold:
  1. Boruta (same settings as 02_split_and_feature_selection.py) on the
     outer training part only;
  2. logistic regression, random forest and the neural network are tuned
     (inner 5-fold grid search, class weighting) and refitted on the
     confirmed features only, then evaluated on the outer test fold.

Reports:
  * selection frequency of each feature (confirmed / tentative) across folds;
  * the number of confirmed features per fold;
  * AUC of the reduced models and the paired difference from the same model
    on all 53 features (tables/nested_cv_class_weight_folds.csv), with the
    corrected repeated k-fold t-test.

Usage:
    python 14_boruta_nested.py [--max-minutes 9]   (resumable)

Output:
    tables/boruta_nested_folds.csv
    tables/boruta_nested_selection_frequency.csv
    tables/boruta_nested_comparison.csv
"""
import argparse
import importlib.util
import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from boruta import BorutaPy
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

spec = importlib.util.spec_from_file_location("nested_cv", "11_nested_cv.py")
ncv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ncv)

TABLE_DIR = "tables"
REPEATS, FOLDS = 10, 5
MODELS = ["Logistic Regression", "Random Forest", "Neural Network"]
FOLD_PATH = f"{TABLE_DIR}/boruta_nested_folds.csv"


def run_boruta(X, y):
    rf = RandomForestClassifier(n_estimators=500, max_depth=7, class_weight="balanced",
                                random_state=ncv.RANDOM_STATE, n_jobs=-1)
    b = BorutaPy(rf, n_estimators="auto", verbose=0, random_state=ncv.RANDOM_STATE, max_iter=100)
    b.fit(X, y)
    return b.support_, b.support_weak_


def summarise(features):
    f = pd.read_csv(FOLD_PATH)
    n_folds = f[["repeat", "fold"]].drop_duplicates().shape[0]
    sel = f.drop_duplicates(["repeat", "fold"])
    conf = pd.Series(0, index=features, dtype=float)
    tent = pd.Series(0, index=features, dtype=float)
    for _, r in sel.iterrows():
        for c in str(r["confirmed"]).split("|"):
            if c in conf.index:
                conf[c] += 1
        for c in str(r["tentative"]).split("|"):
            if c in tent.index:
                tent[c] += 1
    freq = pd.DataFrame({"confirmed_share": conf / n_folds, "tentative_share": tent / n_folds})
    freq = freq.sort_values(["confirmed_share", "tentative_share"], ascending=False)
    freq.to_csv(f"{TABLE_DIR}/boruta_nested_selection_frequency.csv")

    full = pd.read_csv(f"{TABLE_DIR}/nested_cv_class_weight_folds.csv")
    full = full[full["converged"] == True]
    rows = []
    for m in MODELS:
        a = f[f["model"] == m][["repeat", "fold", "auc", "n_train", "n_test"]]
        b = full[full["model"] == m][["repeat", "fold", "auc"]]
        mm = a.merge(b, on=["repeat", "fold"], suffixes=("_boruta", "_all53"))
        d = mm["auc_boruta"] - mm["auc_all53"]
        J = len(d)
        se = d.std(ddof=1) * np.sqrt(1 / J + mm["n_test"].mean() / mm["n_train"].mean())
        t95 = stats.t.ppf(0.975, J - 1)
        rows.append({"model": m, "n_folds": J,
                     "mean_auc_boruta": mm["auc_boruta"].mean(), "sd_auc_boruta": mm["auc_boruta"].std(),
                     "mean_auc_all53": mm["auc_all53"].mean(), "sd_auc_all53": mm["auc_all53"].std(),
                     "delta_auc": d.mean(), "ci95_lo": d.mean() - t95 * se, "ci95_hi": d.mean() + t95 * se,
                     "p_corrected": 2 * stats.t.sf(abs(d.mean() / se), J - 1)})
    comp = pd.DataFrame(rows)
    comp.to_csv(f"{TABLE_DIR}/boruta_nested_comparison.csv", index=False)

    nsel = sel["n_confirmed"]
    print(f"\nBoruta inside {n_folds} outer folds: confirmed features per fold "
          f"median {nsel.median():.0f} (range {nsel.min()}-{nsel.max()})")
    print("\nSelection frequency (share of folds; features confirmed in at least one fold):")
    print(freq[freq["confirmed_share"] > 0].round(2).to_string())
    print("\nReduced (Boruta) vs all 53 features, same outer folds:")
    print(comp.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=0)
    args = ap.parse_args()
    os.makedirs(TABLE_DIR, exist_ok=True)

    df = pd.read_csv(ncv.DATA_PATH)
    y = df["CAD"].astype(int).values
    Xdf = df.drop(columns=["CAD"])
    features = Xdf.columns.tolist()
    X = Xdf.values.astype(float)

    done = set()
    if os.path.exists(FOLD_PATH):
        prev = pd.read_csv(FOLD_PATH)
        done = set(zip(prev["repeat"], prev["fold"]))

    outer = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=ncv.OUTER_SEED)
    registry = ncv.model_registry()
    t_start = time.time()
    for k, (tr, te) in enumerate(outer.split(X, y)):
        rep, fold = k // FOLDS, k % FOLDS
        if (rep, fold) in done:
            continue
        if args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes:
            print("Time limit reached; re-run the same command to resume.")
            return
        t0 = time.time()
        confirmed, tentative = run_boruta(X[tr], y[tr])
        cols = np.where(confirmed)[0]
        rows = []
        for m in MODELS:
            est, grid = registry[m]
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=ncv.RANDOM_STATE + k)
            pipe, pgrid = ncv.build_pipeline(clone(est), grid, "class_weight")
            gs = GridSearchCV(pipe, pgrid, scoring="roc_auc", cv=inner, n_jobs=-1)
            gs.fit(X[tr][:, cols], y[tr])
            p = gs.predict_proba(X[te][:, cols])[:, 1]
            rows.append({"repeat": rep, "fold": fold, "model": m, "n_train": len(tr), "n_test": len(te),
                         "n_confirmed": len(cols), "auc": roc_auc_score(y[te], p),
                         "best_params": str({kk.split("__")[-1]: vv for kk, vv in gs.best_params_.items()}),
                         "confirmed": "|".join(features[i] for i in cols),
                         "tentative": "|".join(features[i] for i in np.where(tentative)[0])})
        pd.DataFrame(rows).to_csv(FOLD_PATH, mode="a", header=not os.path.exists(FOLD_PATH), index=False)
        aucs = ", ".join(f"{r['model']} {r['auc']:.3f}" for r in rows)
        print(f"Repeat {rep} fold {fold}: {len(cols)} confirmed | {aucs} ({time.time() - t0:.0f}s)")

    summarise(features)


if __name__ == "__main__":
    main()
