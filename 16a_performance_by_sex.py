"""
16a_performance_by_sex.py
=========================
Discrimination of the three leading models by sex (Section 4.7 of the manuscript).

Uses the pooled outer-fold predictions of the primary nested cross-validation
(class weighting, written by 11_nested_cv.py). Within every repeat, the
predictions of the five outer folds cover each patient once; the AUC is
computed separately for men and for women in each repeat and then averaged
over the repeats. This is a descriptive analysis: the number of patients
without CAD in each group is small, so no test is reported.

Input : tables/nested_cv_class_weight_predictions.csv, data/CAD_processed.csv
Output: tables/performance_by_sex.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TABLE_DIR = "tables"
PRED = f"{TABLE_DIR}/nested_cv_class_weight_predictions.csv"
DATA = "data/CAD_processed.csv"
MODELS = ["Logistic Regression", "CatBoost", "Random Forest"]


def main():
    if not os.path.exists(PRED):
        raise SystemExit(f"{PRED} not found: run 11_nested_cv.py first")
    pred = pd.read_csv(PRED)
    data = pd.read_csv(DATA)
    # 'patient' is the row index of the patient in data/CAD_processed.csv
    pred["male"] = data["Sex_Male"].to_numpy()[pred["patient"].to_numpy()]

    rows = []
    for model in MODELS:
        p = pred[pred.model == model]
        for male, label in ((1, "Men"), (0, "Women")):
            g = p[p.male == male]
            aucs = [roc_auc_score(r.y, r.p) for _, r in g.groupby("repeat")]
            patients = data[data["Sex_Male"] == male]
            n_cad = int(patients["CAD"].sum())
            rows.append({"model": model, "sex": label, "n_patients": len(patients),
                         "n_cad": n_cad, "n_normal": len(patients) - n_cad,
                         "mean_auc": np.mean(aucs), "sd_auc": np.std(aucs, ddof=1),
                         "n_repeats": len(aucs)})
    out = pd.DataFrame(rows)
    out["cad_prevalence"] = out["n_cad"] / out["n_patients"]
    out.to_csv(f"{TABLE_DIR}/performance_by_sex.csv", index=False)
    print(out.round(3).to_string(index=False))
    print(f"Wrote {TABLE_DIR}/performance_by_sex.csv")


if __name__ == "__main__":
    main()
