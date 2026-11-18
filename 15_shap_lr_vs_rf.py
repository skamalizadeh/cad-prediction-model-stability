"""
15_shap_lr_vs_rf.py
===================
SHAP values for logistic regression and their agreement with the random
forest, to check whether feature rankings depend on the model.

A. Primary split (same models and test set as 05_shap_analysis.py):
   LinearExplainer for the fitted logistic regression (exact for linear
   models; background = scaled training data), compared with the random-forest
   ranking in tables/shap_feature_ranking.csv.
B. Nested CV (class_weight mode, same outer folds as 11_nested_cv.py):
   in every outer fold, logistic regression and random forest are refitted on
   the outer training part with the configuration selected for that fold
   (tables/nested_cv_class_weight_folds.csv), and mean |SHAP| is computed on
   the outer test fold. Reported: Spearman correlation between the two
   rankings in each fold, and how often each feature is in the top 10.

Requires: models/Logistic_Regression.joblib, models/Random_Forest.joblib
(03_train_models.py), tables/shap_feature_ranking.csv (05_shap_analysis.py),
tables/nested_cv_class_weight_folds.csv (11_nested_cv.py).

Output:
    tables/shap_lr_feature_ranking.csv          (A) LR mean |SHAP|, test set
    tables/shap_lr_vs_rf_primary.csv            (A) both rankings side by side
    tables/shap_lr_vs_rf_nested_folds.csv       (B) Spearman rho per fold
    tables/shap_lr_vs_rf_nested_top10.csv       (B) top-10 frequency per feature
"""
import ast
import importlib.util
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

spec = importlib.util.spec_from_file_location("nested_cv", "11_nested_cv.py")
ncv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ncv)

TABLE_DIR = "tables"
TOP = 10


def mean_abs_shap_lr(clf, Xbg, Xev):
    explainer = shap.LinearExplainer(clf, shap.maskers.Independent(Xbg, max_samples=len(Xbg)))
    return np.abs(explainer(Xev).values).mean(axis=0)


def mean_abs_shap_rf(clf, Xev):
    v = shap.TreeExplainer(clf)(Xev).values
    if v.ndim == 3:
        v = v[:, :, 1]
    return np.abs(v).mean(axis=0)


def top_overlap(a, b, features, k=TOP):
    ta = set(pd.Series(a, index=features).nlargest(k).index)
    tb = set(pd.Series(b, index=features).nlargest(k).index)
    return len(ta & tb)


def part_a():
    train = pd.read_csv("data/train_raw.csv")
    test = pd.read_csv("data/test_raw.csv")
    feats = [c for c in test.columns if c != "CAD"]
    pipe = joblib.load("models/Logistic_Regression.joblib")
    scaler, clf = pipe.named_steps["scaler"], pipe.named_steps["clf"]
    Xtr = scaler.transform(train[feats])
    Xte = scaler.transform(test[feats])
    lr_imp = mean_abs_shap_lr(clf, Xtr, Xte)
    lr_rank = pd.DataFrame({"feature": feats, "mean_abs_shap": lr_imp,
                            "coefficient": clf.coef_[0]}).sort_values("mean_abs_shap", ascending=False)
    lr_rank.to_csv(f"{TABLE_DIR}/shap_lr_feature_ranking.csv", index=False)

    rf_rank = pd.read_csv(f"{TABLE_DIR}/shap_feature_ranking.csv")
    both = lr_rank[["feature", "mean_abs_shap"]].rename(columns={"mean_abs_shap": "lr_mean_abs_shap"}) \
        .merge(rf_rank.rename(columns={"mean_abs_shap": "rf_mean_abs_shap"}), on="feature")
    both["lr_rank"] = both["lr_mean_abs_shap"].rank(ascending=False).astype(int)
    both["rf_rank"] = both["rf_mean_abs_shap"].rank(ascending=False).astype(int)
    both = both.sort_values("lr_rank")
    both.to_csv(f"{TABLE_DIR}/shap_lr_vs_rf_primary.csv", index=False)
    rho = spearmanr(both["lr_mean_abs_shap"], both["rf_mean_abs_shap"]).correlation
    ov = top_overlap(both["lr_mean_abs_shap"].values, both["rf_mean_abs_shap"].values,
                     both["feature"].values)
    print("A. Primary split (test set)")
    print(f"   Spearman rho, LR vs RF mean |SHAP| over 53 features: {rho:.3f}")
    print(f"   Features shared by the two top-{TOP} lists: {ov}/{TOP}")
    print(both.head(15)[["feature", "lr_rank", "rf_rank"]].to_string(index=False))


def part_b():
    df = pd.read_csv(ncv.DATA_PATH)
    y = df["CAD"].astype(int).values
    feats = [c for c in df.columns if c != "CAD"]
    X = df[feats].values.astype(float)
    folds = pd.read_csv(f"{TABLE_DIR}/nested_cv_class_weight_folds.csv")
    reps = int(folds["repeat"].max()) + 1
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=ncv.OUTER_SEED)
    rows, top_lr, top_rf = [], pd.Series(0, index=feats), pd.Series(0, index=feats)
    for k, (tr, te) in enumerate(outer.split(X, y)):
        rep, fold = k // 5, k % 5
        par = {m: ast.literal_eval(folds[(folds.model == m) & (folds.repeat == rep) & (folds.fold == fold)]
                                   ["best_params"].iloc[0])
               for m in ("Logistic Regression", "Random Forest")}
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        w = compute_sample_weight("balanced", y[tr])
        lr = LogisticRegression(max_iter=5000, random_state=ncv.RANDOM_STATE, **par["Logistic Regression"]) \
            .fit(Xtr, y[tr], sample_weight=w)
        rf = RandomForestClassifier(random_state=ncv.RANDOM_STATE, n_jobs=-1, **par["Random Forest"]) \
            .fit(Xtr, y[tr], sample_weight=w)
        a, b = mean_abs_shap_lr(lr, Xtr, Xte), mean_abs_shap_rf(rf, Xte)
        rows.append({"repeat": rep, "fold": fold, "spearman_rho": spearmanr(a, b).correlation,
                     f"top{TOP}_overlap": top_overlap(a, b, feats)})
        top_lr[pd.Series(a, index=feats).nlargest(TOP).index] += 1
        top_rf[pd.Series(b, index=feats).nlargest(TOP).index] += 1
    r = pd.DataFrame(rows)
    r.to_csv(f"{TABLE_DIR}/shap_lr_vs_rf_nested_folds.csv", index=False)
    n = len(r)
    t = pd.DataFrame({"lr_top10_share": top_lr / n, "rf_top10_share": top_rf / n})
    t = t[(t > 0).any(axis=1)].sort_values(["lr_top10_share", "rf_top10_share"], ascending=False)
    t.to_csv(f"{TABLE_DIR}/shap_lr_vs_rf_nested_top10.csv")
    print(f"\nB. Nested CV ({n} outer folds)")
    print(f"   Spearman rho LR vs RF: median {r['spearman_rho'].median():.3f} "
          f"(IQR {r['spearman_rho'].quantile(.25):.3f}-{r['spearman_rho'].quantile(.75):.3f})")
    print(f"   Top-{TOP} overlap: median {r[f'top{TOP}_overlap'].median():.0f}/{TOP}")
    print("   Share of folds in which a feature is in the top 10:")
    print(t.head(15).round(2).to_string())


if __name__ == "__main__":
    part_a()
    part_b()
