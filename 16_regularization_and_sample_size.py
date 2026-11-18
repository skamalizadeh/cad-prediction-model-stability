"""
16_regularization_and_sample_size.py
====================================
A. Minimum sample size for a prediction model with 53 candidate parameters
   (Riley et al. 2020, criteria i-iii for a binary outcome):
     (i)   expected uniform shrinkage >= 0.9;
     (ii)  optimism in Nagelkerke R2 <= 0.05;
     (iii) outcome prevalence estimated within +/- 0.05.
   The anticipated Cox-Snell R2 is derived from an anticipated C statistic by
   simulation (Riley, Van Calster & Collins 2021), and also taken as the
   conservative default of 15% of its maximum.
   References:
   Riley RD, Ensor J, Snell KIE, et al. Calculating the sample size required
   for developing a clinical prediction model. BMJ 2020;368:m441.
   Riley RD, Van Calster B, Collins GS. A note on estimating the Cox-Snell R2
   from a reported C statistic (AUROC) to inform sample size calculations for
   developing a prediction model with a binary outcome. Statistics in Medicine
   2021;40(4):859-864.

B. More conservative models, evaluated with the same nested repeated CV
   (outer splits, inner 5-fold grid search, class weighting) as
   11_nested_cv.py:
     * elastic-net logistic regression (C and L1 ratio tuned);
     * principal components + logistic regression (number of components tuned);
     * restricted random forest (shallow trees, large leaves);
     * logistic regression with an extended grid of the regularisation
       strength (C from 0.0003 to 10), because the primary grid's smallest
       value (C = 0.01) was selected in most outer folds.
   For these and for the random forest of the primary grid (refitted with the
   configuration selected in each fold), the apparent AUC on the outer
   training part is reported next to the outer-test AUC, to show optimism.

Output:
    tables/sample_size_riley.csv
    tables/regularization_folds.csv
    tables/regularization_summary.csv
"""
import ast
import importlib.util
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

spec = importlib.util.spec_from_file_location("nested_cv", "11_nested_cv.py")
ncv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ncv)

TABLE_DIR = "tables"
N_PARAMETERS = 53


# ---------------------------------------------------------------
# A. Sample size (Riley et al. 2020)
# ---------------------------------------------------------------
def r2cs_from_c(c, prevalence, n_sim=1_000_000, seed=1):
    """Cox-Snell R2 implied by a C statistic: normal linear predictor with equal
    variance in events and non-events (Riley, Van Calster & Collins 2021)."""
    rng = np.random.default_rng(seed)
    yy = rng.random(n_sim) < prevalence
    mu = np.sqrt(2) * norm.ppf(c)
    x = rng.normal(0, 1, n_sim) + mu * yy
    fit = LogisticRegression(C=1e8, max_iter=1000).fit(x.reshape(-1, 1), yy)
    p = fit.predict_proba(x.reshape(-1, 1))[:, 1]
    ll_model = np.sum(yy * np.log(p) + (~yy) * np.log(1 - p))
    ll_null = n_sim * (prevalence * np.log(prevalence) + (1 - prevalence) * np.log(1 - prevalence))
    return 1 - np.exp(-2 * (ll_model - ll_null) / n_sim)


def riley_sample_size(r2cs, prevalence, p=N_PARAMETERS, shrinkage=0.9, delta=0.05, margin=0.05):
    max_r2 = 1 - (prevalence ** prevalence * (1 - prevalence) ** (1 - prevalence)) ** 2
    n1 = p / ((shrinkage - 1) * np.log(1 - r2cs / shrinkage))
    s2 = r2cs / (r2cs + delta * max_r2)
    n2 = p / ((s2 - 1) * np.log(1 - r2cs / s2))
    n3 = (1.96 / margin) ** 2 * prevalence * (1 - prevalence)
    n = int(np.ceil(max(n1, n2, n3)))
    return {"anticipated_r2cs": r2cs, "max_r2cs": max_r2, "n_criterion_i": np.ceil(n1),
            "n_criterion_ii": np.ceil(n2), "n_criterion_iii": np.ceil(n3), "n_required": n,
            "events_required": int(np.ceil(n * prevalence)),
            "minority_class_required": int(np.ceil(n * min(prevalence, 1 - prevalence))),
            "minority_per_parameter": n * min(prevalence, 1 - prevalence) / p}


def part_a(prevalence, n_available):
    rows = []
    max_r2 = 1 - (prevalence ** prevalence * (1 - prevalence) ** (1 - prevalence)) ** 2
    for label, r2 in [("conservative default (15% of max R2cs)", 0.15 * max_r2),
                      ("C = 0.86 (published Diamond-Forrester in this cohort)", r2cs_from_c(0.86, prevalence)),
                      ("C = 0.93 (nested-CV logistic regression)", r2cs_from_c(0.93, prevalence))]:
        rows.append({"assumption": label, **riley_sample_size(r2, prevalence)})
    t = pd.DataFrame(rows)
    t["n_available"] = n_available
    t.to_csv(f"{TABLE_DIR}/sample_size_riley.csv", index=False)
    print(f"A. Minimum sample size for {N_PARAMETERS} parameters, prevalence {prevalence:.3f} "
          f"(available: n = {n_available}, minority class = {int(round(n_available * (1 - prevalence)))})")
    print(t[["assumption", "anticipated_r2cs", "n_criterion_i", "n_criterion_ii", "n_criterion_iii",
             "n_required", "minority_class_required"]].round(3).to_string(index=False))


# ---------------------------------------------------------------
# B. Conservative models
# ---------------------------------------------------------------
def conservative_models():
    bw = ncv.BalancedWeights
    return {
        "Elastic-net LR": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", bw(LogisticRegression(solver="saga", max_iter=20000,
                                                    random_state=ncv.RANDOM_STATE)))]),
            {"clf__estimator__C": [0.01, 0.1, 1, 10], "clf__estimator__l1_ratio": [0.25, 0.5, 0.75, 1.0]}),
        "PCA + LR": (
            Pipeline([("scaler", StandardScaler()), ("pca", PCA(random_state=ncv.RANDOM_STATE)),
                      ("clf", bw(LogisticRegression(max_iter=5000, random_state=ncv.RANDOM_STATE)))]),
            {"pca__n_components": [3, 5, 10, 15, 20], "clf__estimator__C": [0.01, 0.1, 1, 10]}),
        "Restricted RF": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", bw(RandomForestClassifier(n_estimators=300, random_state=ncv.RANDOM_STATE,
                                                        n_jobs=1)))]),
            {"clf__estimator__max_depth": [3, 5], "clf__estimator__min_samples_leaf": [5, 10, 20]}),
        "LR (extended C grid)": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", bw(LogisticRegression(max_iter=5000, random_state=ncv.RANDOM_STATE)))]),
            {"clf__estimator__C": [0.0003, 0.001, 0.003, 0.01, 0.1, 1, 10]}),
    }


def part_b(only=None):
    """Runs the conservative models; with `only`, runs just those models and
    replaces their rows in an existing regularization_folds.csv."""
    df = pd.read_csv(ncv.DATA_PATH)
    y = df["CAD"].astype(int).values
    X = df.drop(columns=["CAD"]).values.astype(float)
    primary = pd.read_csv(f"{TABLE_DIR}/nested_cv_class_weight_folds.csv")
    reps = int(primary["repeat"].max()) + 1
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=ncv.OUTER_SEED)
    rows = []
    for k, (tr, te) in enumerate(outer.split(X, y)):
        rep, fold = k // 5, k % 5
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=ncv.RANDOM_STATE + k)
        for name, (pipe, grid) in conservative_models().items():
            if only and name not in only:
                continue
            gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=-1).fit(X[tr], y[tr])
            rows.append({"model": name, "repeat": rep, "fold": fold, "n_train": len(tr), "n_test": len(te),
                         "auc": roc_auc_score(y[te], gs.predict_proba(X[te])[:, 1]),
                         "apparent_auc": roc_auc_score(y[tr], gs.predict_proba(X[tr])[:, 1]),
                         "best_params": str({kk.split("__")[-1]: vv for kk, vv in gs.best_params_.items()})})
        # primary-grid random forest and logistic regression: apparent AUC with the selected configuration
        for m in (() if only else ("Random Forest", "Logistic Regression")):
            r = primary[(primary.model == m) & (primary.repeat == rep) & (primary.fold == fold)].iloc[0]
            est, _ = ncv.model_registry()[m]
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("clf", ncv.BalancedWeights(clone(est).set_params(**ast.literal_eval(r["best_params"]))))])
            pipe.fit(X[tr], y[tr])
            rows.append({"model": f"{m} (primary grid)", "repeat": rep, "fold": fold, "n_train": len(tr),
                         "n_test": len(te), "auc": r["auc"],
                         "apparent_auc": roc_auc_score(y[tr], pipe.predict_proba(X[tr])[:, 1]),
                         "best_params": r["best_params"]})
        if fold == 4:
            print(f"   repeat {rep + 1}/{reps} done")
    f = pd.DataFrame(rows)
    path = f"{TABLE_DIR}/regularization_folds.csv"
    if only and os.path.exists(path):
        prev = pd.read_csv(path)
        f = pd.concat([prev[~prev.model.isin(only)], f], ignore_index=True)
    f.to_csv(path, index=False)

    ref = f[f.model == "Logistic Regression (primary grid)"].set_index(["repeat", "fold"])["auc"]
    out = []
    for m, g in f.groupby("model"):
        d = (ref - g.set_index(["repeat", "fold"])["auc"]).dropna()
        J = len(d)
        se = d.std(ddof=1) * np.sqrt(1 / J + g["n_test"].mean() / g["n_train"].mean()) if d.std() > 0 else np.nan
        out.append({"model": m, "mean_auc": g["auc"].mean(), "sd_auc": g["auc"].std(),
                    "mean_apparent_auc": g["apparent_auc"].mean(),
                    "optimism": g["apparent_auc"].mean() - g["auc"].mean(),
                    "diff_vs_LR": d.mean(),
                    "p_corrected_vs_LR": 2 * stats.t.sf(abs(d.mean() / se), J - 1) if se == se else np.nan})
    s = pd.DataFrame(out).sort_values("mean_auc", ascending=False)
    s.to_csv(f"{TABLE_DIR}/regularization_summary.csv", index=False)
    print("\nB. Conservative models (nested CV, class weighting):")
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated conservative models to (re)run; default: all")
    args = ap.parse_args()
    os.makedirs(TABLE_DIR, exist_ok=True)
    d = pd.read_csv(ncv.DATA_PATH)
    part_a(d["CAD"].mean(), len(d))
    part_b([m.strip() for m in args.only.split(",") if m.strip()] or None)
