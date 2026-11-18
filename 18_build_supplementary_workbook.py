"""
18_build_supplementary_workbook.py
==================================
Collects every result table into one Excel workbook for submission:
the search grids, the configuration selected in every outer fold, all
nested-cross-validation summaries (three class-balancing strategies), the
clinical reference models, Boruta and SHAP analyses, the sample-size
calculation, the conservative models, and the single-split tables built by
10_build_results_workbook.py.

Output: outputs/CAD_supplementary_workbook.xlsx
"""
import importlib.util
import os

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

T = "tables"
OUT = "outputs/CAD_supplementary_workbook.xlsx"
SINGLE = "outputs/CAD_prediction_results_tables.xlsx"
SKIP_SINGLE = {"Baseline_MeanSD_check"}

spec = importlib.util.spec_from_file_location("nested_cv", "11_nested_cv.py")
ncv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ncv)


def grids():
    rows = []
    for name, (_, grid) in ncv.model_registry().items():
        for k, v in grid.items():
            rows.append({"model": name, "hyperparameter": k,
                         "values": ", ".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in v)})
    return pd.DataFrame(rows)


def flat_cal(name):
    d = pd.read_csv(f"{T}/{name}", header=[0, 1], index_col=[0, 1])
    d.columns = [f"{a}_{b}" for a, b in d.columns]
    d.index.names = ["model", "predictions"]
    return d.reset_index()


def csv(name, **kw):
    path = f"{T}/{name}"
    return pd.read_csv(path, **kw) if os.path.exists(path) else None


def dca_recalibrated(mode):
    d = csv(f"nested_cv_decision_curve_{mode}.csv")
    if d is not None and "predictions" in d.columns:
        d = d[d.predictions == "recalibrated"].drop(columns="predictions")
    return d


DCA_MODELS = ["Logistic Regression", "CatBoost", "Random Forest", "Treat all"]


def dca_by_balancing():
    parts = []
    for mode in ("class_weight", "oversample", "none"):
        d = csv(f"nested_cv_decision_curve_{mode}.csv")
        if d is None or "predictions" not in d.columns:
            return None
        d = d[d.model.isin(DCA_MODELS) & (d.threshold <= 0.85 + 1e-9)].copy()
        d.insert(0, "balancing", mode)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


SHEETS = [
    ("Search_grids", "Hyperparameter grids (identical in all analyses)", grids),
    ("Nested_CW_folds", "Nested CV, class weighting: every model x outer fold (selected configuration, AUC, threshold metrics)",
     lambda: csv("nested_cv_class_weight_folds.csv")),
    ("Nested_CW_summary", "Nested CV, class weighting: mean outer-fold AUC", lambda: csv("nested_cv_summary_class_weight.csv")),
    ("Nested_CW_vs_LR", "Nested CV, class weighting: comparison with logistic regression (corrected t-test, Holm, Cohen's d_z, TOST)",
     lambda: csv("nested_cv_comparison_class_weight.csv")),
    ("Nested_CW_ranks", "Nested CV, class weighting: ranking stability", lambda: csv("nested_cv_ranks_class_weight.csv")),
    ("Nested_CW_calibration", "Nested CV, class weighting: calibration before/after recalibration",
     lambda: flat_cal("nested_cv_calibration_class_weight.csv")),
    ("Nested_CW_thresholds", "Nested CV, class weighting: threshold metrics (Youden from training part; fixed 0.5)",
     lambda: csv("nested_cv_thresholds_class_weight.csv")),
    ("Nested_CW_DCA", "Nested CV, class weighting: net benefit of recalibrated predictions by threshold (Fig. 4)",
     lambda: dca_recalibrated("class_weight")),
    ("Nested_OS_folds", "Nested CV, over-sampling: every model x outer fold", lambda: csv("nested_cv_oversample_folds.csv")),
    ("Nested_OS_vs_LR", "Nested CV, over-sampling: comparison with logistic regression", lambda: csv("nested_cv_comparison_oversample.csv")),
    ("Nested_NONE_folds", "Nested CV, no balancing: every model x outer fold", lambda: csv("nested_cv_none_folds.csv")),
    ("Nested_NONE_vs_LR", "Nested CV, no balancing: comparison with logistic regression", lambda: csv("nested_cv_comparison_none.csv")),
    ("DCA_by_balancing", "Net benefit of logistic regression, CatBoost and random forest by class-balancing strategy, raw and recalibrated predictions (Supplementary Fig. S3)",
     dca_by_balancing),
    ("Nested_modes_AUC", "Mean outer-fold AUC by class-balancing strategy", lambda: csv("nested_cv_modes_overview.csv")),
    ("Clinical_folds", "Clinical reference and nested logistic models: AUC per outer fold", lambda: csv("clinical_models_folds.csv")),
    ("Clinical_summary", "Clinical reference and nested logistic models: summary", lambda: csv("clinical_models_summary.csv")),
    ("Clinical_contrasts", "Clinical models: differences in AUC (Holm within families)", lambda: csv("clinical_models_table_contrasts.csv")),
    ("Boruta_nested_folds", "Boruta inside each outer fold: confirmed features and reduced-model AUC", lambda: csv("boruta_nested_folds.csv")),
    ("Boruta_nested_frequency", "Boruta inside each outer fold: selection frequency", lambda: csv("boruta_nested_selection_frequency.csv")),
    ("Boruta_nested_vs_all", "Boruta-reduced vs all-feature models", lambda: csv("boruta_nested_comparison.csv")),
    ("SHAP_LR_ranking", "Single split: logistic-regression SHAP ranking and coefficients", lambda: csv("shap_lr_feature_ranking.csv")),
    ("SHAP_LR_vs_RF", "Single split: LR vs RF SHAP ranks", lambda: csv("shap_lr_vs_rf_primary.csv")),
    ("SHAP_folds", "Nested CV: LR vs RF SHAP agreement per fold", lambda: csv("shap_lr_vs_rf_nested_folds.csv")),
    ("SHAP_top10", "Nested CV: share of folds in the SHAP top 10", lambda: csv("shap_lr_vs_rf_nested_top10.csv")),
    ("Sample_size", "Minimum sample size (Riley et al. 2020)", lambda: csv("sample_size_riley.csv")),
    ("Performance_by_sex", "Nested CV, class weighting: AUC of the leading models in men and in women (Section 4.7)",
     lambda: csv("performance_by_sex.csv")),
    ("Conservative_folds", "Conservative models: AUC and apparent AUC per outer fold", lambda: csv("regularization_folds.csv")),
    ("Conservative_summary", "Conservative models: summary", lambda: csv("regularization_summary.csv")),
    ("Baseline_continuous", "Baseline characteristics, continuous variables: median (IQR), Mann-Whitney test (Table 2)",
     lambda: csv("baseline_continuous.csv")),
    ("Baseline_categorical", "Baseline characteristics, binary and ordinal variables: n (%) (Table 3)",
     lambda: csv("baseline_categorical.csv", keep_default_na=False)),
    ("Single_DeLong_Holm", "Single split: DeLong tests vs logistic regression with Holm adjustment", lambda: csv("single_split_delong_holm.csv")),
]


def main():
    os.makedirs("outputs", exist_ok=True)
    index = []
    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        pd.DataFrame().to_excel(w, sheet_name="Contents")
        for name, title, fn in SHEETS:
            df = fn()
            if df is None:
                print(f"Skipped {name}: source table not found")
                continue
            df.to_excel(w, sheet_name=name, index=False)
            index.append((name, title))
        if os.path.exists(SINGLE):
            # Sheets of the single-split workbook start with a title row and a blank
            # row; the title goes to the Contents sheet and the table starts at row 1.
            # The mean (SD) baseline table of that workbook is skipped; the Baseline_* sheets give median (IQR).
            for sh, df in pd.read_excel(SINGLE, sheet_name=None, header=None).items():
                if sh in SKIP_SINGLE:
                    continue
                df = df.dropna(how="all").reset_index(drop=True)
                title = str(df.iloc[0, 0]) if df.iloc[0].notna().sum() == 1 else sh
                if df.iloc[0].notna().sum() == 1:
                    df = df.iloc[1:]
                nm = ("Single_" + sh)[:31]
                df.to_excel(w, sheet_name=nm, index=False, header=False)
                index.append((nm, f"Single-split analysis: {title}"))
    wb = load_workbook(OUT)
    ws = wb["Contents"]
    ws.delete_rows(1, ws.max_row)
    ws.append(["Sheet", "Content"])
    for r in index:
        ws.append(list(r))
    for sheet in wb.worksheets:
        for row in sheet.iter_rows():
            for c in row:
                c.font = Font(name="Arial", size=10, bold=c.font.bold)
        if True:  # every sheet now has its column names in row 1
            for c in sheet[1]:
                c.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
                c.fill = PatternFill("solid", fgColor="1F4E78")
            sheet.freeze_panes = "A2"
        for i in range(1, sheet.max_column + 1):
            vals = [len(str(sheet.cell(row=r, column=i).value or "")) for r in range(1, min(sheet.max_row, 60) + 1)]
            sheet.column_dimensions[get_column_letter(i)].width = min(60, max(9, max(vals) + 2))
    ws.column_dimensions["B"].width = 100
    for row in ws.iter_rows(min_row=2):
        row[1].alignment = Alignment(wrap_text=True)
    wb.save(OUT)
    print(f"Wrote {OUT} with {len(index)} sheets")


if __name__ == "__main__":
    main()
