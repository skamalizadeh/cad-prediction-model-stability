"""
10_build_results_workbook.py
===========================
Consolidates every output table produced by the analysis pipeline into
a single, formatted, multi-sheet Excel workbook for delivery alongside
the manuscript and the individual chart images.
"""
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

TABLES = "tables"
OUT = "outputs/CAD_prediction_results_tables.xlsx"

FONT_NAME = "Arial"
HEADER_FILL = PatternFill(start_color="2F3E9E", end_color="2F3E9E", fill_type="solid")
HEADER_FONT = Font(name=FONT_NAME, bold=True, color="FFFFFF", size=10)
BODY_FONT = Font(name=FONT_NAME, size=10)
TITLE_FONT = Font(name=FONT_NAME, bold=True, size=13, color="2F3E9E")
NOTE_FONT = Font(name=FONT_NAME, italic=True, size=9, color="595959")
THIN = Side(style="thin", color="D0D0D0")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def write_sheet(wb, name, title, df, note=None, round_map=None, col_widths=None):
    ws = wb.create_sheet(name[:31])
    ws["A1"] = title
    ws["A1"].font = TITLE_FONT
    start_row = 3
    if round_map:
        df = df.copy()
        for col, nd in round_map.items():
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(nd)
    for j, col in enumerate(df.columns, start=1):
        c = ws.cell(row=start_row, column=j, value=str(col))
        c.font = HEADER_FONT
        c.fill = HEADER_FILL
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = BORDER
    for i, (_, r) in enumerate(df.iterrows(), start=start_row + 1):
        for j, col in enumerate(df.columns, start=1):
            val = r[col]
            if pd.isna(val):
                val = ""
            c = ws.cell(row=i, column=j, value=val)
            c.font = BODY_FONT
            c.border = BORDER
            c.alignment = Alignment(horizontal="center", vertical="center")
    ws.freeze_panes = ws.cell(row=start_row + 1, column=1)
    widths = col_widths or [max(12, min(38, len(str(c)) + 4)) for c in df.columns]
    for j, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(j)].width = w
    if note:
        note_row = start_row + len(df) + 2
        ws.cell(row=note_row, column=1, value=note).font = NOTE_FONT
    return ws


wb = Workbook()
wb.remove(wb.active)

# 1. Baseline characteristics, mean (SD): descriptive check
t1 = pd.read_csv(f"{TABLES}/table1_baseline_characteristics.csv")
write_sheet(wb, "Baseline_MeanSD_check", "Baseline characteristics, mean (SD) and t-test (descriptive check only; the manuscript reports median (IQR) in Tables 2-3)", t1,
            note="Continuous variables: mean ± SD, Welch t-test. Categorical/binary: n (%), chi-square test. "
                 "Multi-level ordinal (Function Class, Region RWMA, VHD_ordinal): group mean, Mann-Whitney U test.",
            col_widths=[22, 16, 18, 18, 12])

# 2. Data quality / leakage screen
lk = pd.read_csv(f"{TABLES}/data_quality_and_leakage_screen.csv")
write_sheet(wb, "DataQuality_LeakageScreen", "Data quality and univariate data-leakage screen", lk,
            note="flag_possible_leakage = TRUE if univariate AUC > 0.97 (pre-specified threshold). None flagged.",
            round_map={"univariate_AUC": 4})

# 3. Boruta feature selection
bo = pd.read_csv(f"{TABLES}/boruta_feature_selection.csv")
write_sheet(wb, "Boruta_FeatureSelection", "Boruta all-relevant feature selection (training set)", bo,
            note="confirmed/tentative = boolean flags from BorutaPy; ranking: 1 = confirmed relevant.")

# 4. RF impurity feature importance
fi = pd.read_csv(f"{TABLES}/permutation_feature_importance.csv")
write_sheet(wb, "RF_FeatureImportance", "Random Forest feature importance (mean decrease in impurity)", fi,
            round_map={"importance_mean": 5, "importance_std": 5})

# 5. AUC train/test with DeLong CI
auc = pd.read_csv(f"{TABLES}/table_auc_train_test.csv")
write_sheet(wb, "AUC_Discrimination", "Discrimination: AUC (train & test) with DeLong 95% CI", auc,
            round_map={c: 4 for c in auc.columns if c != "model"})

# 6. DeLong pairwise test
dl = pd.read_csv(f"{TABLES}/table_delong_pairwise_test.csv")
write_sheet(wb, "DeLong_PairwiseTests", "DeLong pairwise test: best model (Logistic Regression) vs. others", dl,
            round_map={c: 4 for c in dl.columns if "model" not in c})

# 7. Calibration
cal = pd.read_csv(f"{TABLES}/table_calibration.csv")
write_sheet(wb, "Calibration", "Calibration: intercept, slope, calibration-in-the-large, Brier score", cal,
            round_map={c: 4 for c in cal.columns if c != "model"}, col_widths=[18]+[13]*10)

# 8. Supplementary Table S1: performance at the test-set Youden threshold
t2 = pd.read_csv(f"{TABLES}/table2_FINAL.csv")
write_sheet(wb, "TableS1_YoudenTestSet", "Supplementary Table S1. Performance at the Youden-index threshold chosen on the test set (apparent)", t2,
            col_widths=[18, 16, 16, 8, 8, 16, 8, 12])

# 9. SHAP feature ranking
sh = pd.read_csv(f"{TABLES}/shap_feature_ranking.csv")
write_sheet(wb, "SHAP_FeatureRanking", "SHAP feature ranking (Random Forest, mean |SHAP value|, test set)", sh,
            round_map={"mean_abs_shap": 5})

# 10. Hyperparameters
hp = pd.read_csv(f"{TABLES}/best_hyperparameters.csv")
write_sheet(wb, "Best_Hyperparameters", "Selected hyperparameters per model (5-fold grid-search CV)", hp,
            round_map={"best_cv_auc": 4}, col_widths=[18, 55, 12, 12])

# 11. Boruta-7 vs full-53 comparison (sensitivity analysis)
b7 = pd.read_csv(f"{TABLES}/table_boruta7_vs_full_comparison.csv")
write_sheet(wb, "Sensitivity_Boruta7_vs_Full", "Sensitivity analysis: Boruta-7-feature vs full-53-feature models", b7,
            note="train_test_gap = AUC_train - AUC_test (overfitting proxy). Same 70/30 split and over-sampling as primary analysis.",
            round_map={c: 4 for c in b7.columns if c != "model"}, col_widths=[18]+[14]*11)

# 12. Repeated-holdout summary (mean/SD/percentiles, both feature sets)
rh_sum = pd.read_csv(f"{TABLES}/table_repeated_holdout_summary.csv")
write_sheet(wb, "RepeatedHoldout_Summary", "Monte Carlo repeated-holdout (B=50): AUC mean/SD/percentiles", rh_sum,
            note="50 independent stratified 70/30 splits; fixed pre-tuned hyperparameters refit on each resample.",
            round_map={c: 4 for c in rh_sum.columns if c not in ("model", "feature_set", "n_valid_repeats")})

# 13. Repeated-holdout stability comparison (SD full53 vs boruta7)
stab = pd.read_csv(f"{TABLES}/table_stability_comparison.csv")
write_sheet(wb, "RepeatedHoldout_Stability", "Stability comparison: SD of test AUC across 50 repeats, full-53 vs Boruta-7", stab,
            round_map={c: 4 for c in stab.columns if c != "model"})

# 14. Paired significance tests (Logistic Regression vs other top models, across repeats)
paired = pd.read_csv(f"{TABLES}/table_repeated_holdout_paired_tests.csv")
write_sheet(wb, "RepeatedHoldout_PairedTests", "Repeated holdout (exploratory): difference from Logistic Regression with corrected resampled 95% CI (Nadeau-Bengio) and Cohen's d_z", paired,
            round_map={c: 4 for c in paired.columns if c != "comparison"}, col_widths=[36, 14, 14, 14, 14])

# 15. Raw repeated-holdout AUC values (full transparency / re-analysis)
rh_full_raw = pd.read_csv(f"{TABLES}/table_repeated_holdout_full53.csv")
write_sheet(wb, "RepeatedHoldout_Raw_Full53", "Raw test AUC per repeat, full 53-feature models (50 rows)", rh_full_raw,
            round_map={c: 4 for c in rh_full_raw.columns})
rh_b7_raw = pd.read_csv(f"{TABLES}/table_repeated_holdout_boruta7.csv")
write_sheet(wb, "RepeatedHoldout_Raw_Boruta7", "Raw test AUC per repeat, Boruta-7-feature models (50 rows)", rh_b7_raw,
            round_map={c: 4 for c in rh_b7_raw.columns})

# 16. Decision-curve analysis (net benefit vs threshold probability)
dca = pd.read_csv(f"{TABLES}/table_decision_curve.csv")
write_sheet(wb, "DecisionCurve_NetBenefit", "Net benefit vs threshold probability, top-4 models vs treat-all/treat-none (test set)", dca,
            note="Vickers & Elkin 2006. Threshold probabilities 5%-95% in 1% steps.",
            round_map={c: 4 for c in dca.columns})

# 17. Logistic-regression coefficient bootstrap (B=1000)
lrb = pd.read_csv(f"{TABLES}/table_lr_coef_bootstrap.csv")
write_sheet(wb, "LR_Coefficient_Bootstrap", "Standardised logistic-regression coefficients, 95% bootstrap CI (B=1000)", lrb,
            note="Stratified bootstrap of the raw training set; over-sampling and scaling re-applied per resample; C fixed at 0.01 (not re-tuned).",
            round_map={c: 4 for c in lrb.columns if c not in ("feature", "excludes_zero")})

wb.save(OUT)
print("Workbook saved:", OUT)
print("Sheets:", wb.sheetnames)
