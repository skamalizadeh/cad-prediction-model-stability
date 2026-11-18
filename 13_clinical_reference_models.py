"""
13_clinical_reference_models.py
===============================
Incremental value of routinely collected variables over established clinical
pre-test probability models, using the same nested repeated cross-validation
(outer splits, inner grid search, class weighting) as 11_nested_cv.py.

A. Published models applied as they are (no fitting; evaluated on every outer
   test fold):
   * Diamond-Forrester, approximated logistic form reported by Genders et al.
     (2011), Table A1: age 0.09/year, male 1.35, typical chest pain 3.77,
     atypical chest pain 1.70; intercept -7.52.
   * CAD Consortium updated model (Genders et al. 2011, Table 3):
     age 0.04/year, male 1.34, typical chest pain 1.91, atypical chest pain
     0.64; intercept -4.37.
   Reference category: non-anginal chest pain (patients with no chest-pain
   category recorded are also scored in the reference category).
   Genders TSS, Steyerberg EW, Alkadhi H, et al. A clinical prediction rule for
   the diagnosis of coronary artery disease: validation, updating, and
   extension. European Heart Journal 2011;32(11):1316-1330.
   The intercepts only shift all predictions equally, so they do not affect
   the AUC; they matter only for calibration.

B. Nested logistic-regression models refitted on this cohort:
   M1 basic     : age, sex, typical and atypical chest pain
                  (the variables of the models above)
   M2 clinical  : M1 + risk factors, other symptoms and physical examination
   M3 ECG + lab : M2 + resting ECG and laboratory variables
   M4 full      : M3 + echocardiography (all 53 features)
   Each is tuned and evaluated exactly like the logistic regression of
   11_nested_cv.py (class_weight mode).

Usage:
    python 13_clinical_reference_models.py

Output:
    tables/clinical_models_folds.csv        AUC per model and outer fold
    tables/clinical_models_predictions.csv  outer-fold predictions
    tables/clinical_models_summary.csv      mean AUC, calibration of pooled predictions
    tables/clinical_models_increments.csv   delta AUC between steps, corrected t-test
"""
import importlib.util
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

spec = importlib.util.spec_from_file_location("nested_cv", "11_nested_cv.py")
ncv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ncv)

summary_spec = importlib.util.spec_from_file_location("ncv_summary", "12_nested_cv_summary.py")
ncs = importlib.util.module_from_spec(summary_spec)
summary_spec.loader.exec_module(ncs)

TABLE_DIR = "tables"
REPEATS, FOLDS = 10, 5

BASIC = ["Age", "Sex_Male", "Typical Chest Pain", "Atypical"]
CLINICAL = ["Weight", "Length", "BMI", "DM", "HTN", "Current Smoker", "EX-Smoker", "FH", "Obesity",
            "CRF", "CVA", "Airway disease", "Thyroid Disease", "CHF", "DLP", "BP", "PR", "Edema",
            "Weak Peripheral Pulse", "Lung rales", "Systolic Murmur", "Diastolic Murmur", "Dyspnea",
            "Function Class", "Nonanginal", "LowTH Ang"]
ECG_LAB = ["Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH", "Poor R Progression",
           "FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT"]
ECHO = ["EF-TTE", "Region RWMA", "VHD_ordinal"]

NESTED = {
    "M1 basic (age, sex, chest pain)": BASIC,
    "M2 + clinical history and examination": BASIC + CLINICAL,
    "M3 + ECG and laboratory": BASIC + CLINICAL + ECG_LAB,
    "M4 + echocardiography (all 53 features)": BASIC + CLINICAL + ECG_LAB + ECHO,
}
PUBLISHED = {
    # name: (intercept, age, male, typical, atypical)
    "Diamond-Forrester (published)": (-7.52, 0.09, 1.35, 3.77, 1.70),
    "CAD Consortium (published)": (-4.37, 0.04, 1.34, 1.91, 0.64),
}


def published_probability(df, coef):
    b0, b_age, b_male, b_typ, b_atyp = coef
    lp = (b0 + b_age * df["Age"] + b_male * df["Sex_Male"]
          + b_typ * df["Typical Chest Pain"] + b_atyp * df["Atypical"])
    return expit(lp.values)


def corrected_ttest(d, n_te, n_tr):
    J = len(d)
    mu, sd = d.mean(), d.std(ddof=1)
    se = sd * np.sqrt(1 / J + n_te / n_tr)
    t95 = stats.t.ppf(0.975, J - 1)
    return mu, mu - t95 * se, mu + t95 * se, 2 * stats.t.sf(abs(mu / se), J - 1)


def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    df = pd.read_csv(ncv.DATA_PATH)
    y = df["CAD"].astype(int).values
    all_feats = [c for c in df.columns if c != "CAD"]
    assert sorted(NESTED["M4 + echocardiography (all 53 features)"]) == sorted(all_feats), \
        "feature groups must cover all features exactly once"
    print(f"Feature groups: basic {len(BASIC)}, clinical {len(CLINICAL)}, "
          f"ECG+lab {len(ECG_LAB)}, echo {len(ECHO)}")

    outer = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=ncv.OUTER_SEED)
    splits = list(outer.split(df, y))
    est, grid = ncv.model_registry()["Logistic Regression"]
    pub_p = {name: published_probability(df, c) for name, c in PUBLISHED.items()}

    rows, preds = [], []
    for k, (tr, te) in enumerate(splits):
        rep, fold = k // FOLDS, k % FOLDS
        for name, p_all in pub_p.items():
            rows.append({"model": name, "repeat": rep, "fold": fold, "n_train": len(tr), "n_test": len(te),
                         "auc": roc_auc_score(y[te], p_all[te])})
            preds.append(pd.DataFrame({"model": name, "repeat": rep, "patient": te, "y": y[te], "p": p_all[te]}))
        for name, feats in NESTED.items():
            X = df[feats].values.astype(float)
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=ncv.RANDOM_STATE + k)
            pipe, pgrid = ncv.build_pipeline(clone(est), grid, "class_weight")
            gs = GridSearchCV(pipe, pgrid, scoring="roc_auc", cv=inner, n_jobs=-1).fit(X[tr], y[tr])
            p = gs.predict_proba(X[te])[:, 1]
            rows.append({"model": name, "repeat": rep, "fold": fold, "n_train": len(tr), "n_test": len(te),
                         "auc": roc_auc_score(y[te], p),
                         "best_C": gs.best_params_["clf__estimator__C"]})
            preds.append(pd.DataFrame({"model": name, "repeat": rep, "patient": te, "y": y[te], "p": p}))
        if fold == FOLDS - 1:
            print(f"Repeat {rep + 1}/{REPEATS} done")

    f = pd.DataFrame(rows)
    f.to_csv(f"{TABLE_DIR}/clinical_models_folds.csv", index=False)
    pr = pd.concat(preds, ignore_index=True)
    pr.to_csv(f"{TABLE_DIR}/clinical_models_predictions.csv", index=False)

    # Add the two strongest non-linear models of the primary analysis (same outer splits)
    nested_path = f"{TABLE_DIR}/nested_cv_class_weight_folds.csv"
    if os.path.exists(nested_path):
        nf = pd.read_csv(nested_path)
        nf = nf[nf["model"].isin(["CatBoost", "Random Forest"]) & (nf["converged"] == True)]
        nf = nf.assign(model=nf["model"] + " (all 53 features)")[["model", "repeat", "fold", "n_train", "n_test", "auc"]]
        f = pd.concat([f, nf], ignore_index=True)

    # Summary: discrimination and calibration of pooled predictions per repeat
    cal = []
    for (m, rep), g in pr.groupby(["model", "repeat"]):
        citl, slope = ncs.calibration_intercept_slope(g["y"].values, g["p"].values)
        cal.append({"model": m, "repeat": rep, "citl": citl, "slope": slope,
                    "mean_predicted": g["p"].mean(), "observed": g["y"].mean()})
    cal = pd.DataFrame(cal).groupby("model").mean(numeric_only=True).drop(columns="repeat")
    summ = f.groupby("model")["auc"].agg(mean_auc="mean", sd_auc="std").join(cal, how="left")
    extra = ["CatBoost (all 53 features)", "Random Forest (all 53 features)"]
    order = list(PUBLISHED) + list(NESTED) + extra
    summ = summ.loc[[m for m in order if m in summ.index]]
    summ.to_csv(f"{TABLE_DIR}/clinical_models_summary.csv")

    # Increments: each step vs the previous one, and key contrasts
    wide = f.pivot_table(index=["repeat", "fold"], columns="model", values="auc")
    n_tr, n_te = f["n_train"].mean(), f["n_test"].mean()
    names = list(NESTED)
    contrasts = [(names[i + 1], names[i]) for i in range(len(names) - 1)]
    contrasts += [(names[0], "CAD Consortium (published)"), (names[0], "Diamond-Forrester (published)"),
                  (names[-1], names[0]), (names[-1], "CAD Consortium (published)"),
                  (names[-1], "Diamond-Forrester (published)")]
    contrasts += [(m, names[0]) for m in extra if m in wide.columns]
    contrasts += [(names[-1], m) for m in extra if m in wide.columns]
    inc = []
    for a, b in contrasts:
        mu, lo, hi, p = corrected_ttest((wide[a] - wide[b]).dropna(), n_te, n_tr)
        inc.append({"model": a, "compared_with": b, "delta_auc": mu, "ci95_lo": lo, "ci95_hi": hi, "p_corrected": p})
    inc = pd.DataFrame(inc)
    inc["p_holm"] = ncs.holm(inc["p_corrected"])
    inc.to_csv(f"{TABLE_DIR}/clinical_models_increments.csv", index=False)

    print("\nDiscrimination and calibration (outer folds, 10 repeats x 5 folds):")
    print(summ.round(3).to_string())
    print("\nIncremental value (delta AUC, corrected repeated k-fold t-test, Holm):")
    print(inc.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
