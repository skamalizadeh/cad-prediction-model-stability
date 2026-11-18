"""
01_data_preprocessing.py
=========================
Loads the raw Z-Alizadeh Sani coronary artery disease (CAD) dataset,
performs data cleaning / variable encoding, runs a data-quality &
leakage screen, and builds the baseline characteristics table
(a descriptive check, stratified by angiographically-confirmed CAD status;
the manuscript's Tables 2-3 are built by step 17).

Input : data/CAD_raw.csv
Output: data/CAD_processed.csv
        tables/table1_baseline_characteristics.csv
        tables/data_quality_and_leakage_screen.csv
"""

import pandas as pd
import numpy as np
from scipy import stats

RAW_PATH = "data/CAD_raw.csv"
OUT_DATA = "data/CAD_processed.csv"
OUT_TABLE1 = "tables/table1_baseline_characteristics.csv"
OUT_QUALITY = "tables/data_quality_and_leakage_screen.csv"

# ---------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------
df = pd.read_csv(RAW_PATH)
print(f"Raw shape: {df.shape}")
print(f"Missing values total: {df.isnull().sum().sum()}")

# ---------------------------------------------------------------
# 2. Drop zero-variance / degenerate columns
# ---------------------------------------------------------------
nunique = df.nunique()
constant_cols = nunique[nunique == 1].index.tolist()
print(f"Constant (zero-variance) columns dropped: {constant_cols}")
df = df.drop(columns=constant_cols)

# ---------------------------------------------------------------
# 3. Standardise binary Yes/No text columns -> 1/0
# ---------------------------------------------------------------
binary_yn_cols = [c for c in df.columns
                   if pd.api.types.is_string_dtype(df[c]) and set(df[c].unique()) <= {"Y", "N"}]
for c in binary_yn_cols:
    df[c] = (df[c] == "Y").astype(int)

# ---------------------------------------------------------------
# 4. Sex: Male / Fmale (dataset's original spelling of "Female")
# ---------------------------------------------------------------
df["Sex_Male"] = (df["Sex"] == "Male").astype(int)
df = df.drop(columns=["Sex"])

# ---------------------------------------------------------------
# 5. VHD (valvular heart disease): ordinal N < mild < Moderate < Severe
# ---------------------------------------------------------------
vhd_map = {"N": 0, "mild": 1, "Moderate": 2, "Severe": 3}
df["VHD_ordinal"] = df["VHD"].map(vhd_map)
assert df["VHD_ordinal"].isnull().sum() == 0
df = df.drop(columns=["VHD"])

# ---------------------------------------------------------------
# 6. Target variable: Cath -> Cad = 1 (CAD present), Normal = 0
# ---------------------------------------------------------------
df["CAD"] = (df["Cath"] == "Cad").astype(int)
df = df.drop(columns=["Cath"])

print(f"\nProcessed shape: {df.shape}")
print(f"Target distribution:\n{df['CAD'].value_counts()} "
      f"({df['CAD'].mean()*100:.2f}% CAD-positive)")

df.to_csv(OUT_DATA, index=False)

# =================================================================
# DATA QUALITY / LEAKAGE SCREEN
# For every candidate predictor, measure univariate association with
# the target. A feature that in isolation achieves near-perfect
# separation (AUC > 0.97 for continuous/ordinal, or an empty cell in
# a 2x2 table) is flagged for manual review before modelling -- this
# is standard practice to catch deterministic/leaked variables before
# they contaminate a prediction model.
# =================================================================
from sklearn.metrics import roc_auc_score

feature_cols = [c for c in df.columns if c != "CAD"]
screen_rows = []
for c in feature_cols:
    x = df[c].values
    y = df["CAD"].values
    try:
        auc = roc_auc_score(y, x)
        auc = max(auc, 1 - auc)  # direction-agnostic
    except Exception:
        auc = np.nan
    # point-biserial / chi-square style flag for near-deterministic split
    screen_rows.append({"feature": c, "univariate_AUC": round(auc, 4),
                         "n_unique": df[c].nunique()})

screen_df = pd.DataFrame(screen_rows).sort_values("univariate_AUC", ascending=False)
screen_df["flag_possible_leakage"] = screen_df["univariate_AUC"] > 0.97
screen_df.to_csv(OUT_QUALITY, index=False)

print("\n=== Top 10 univariate AUC (leakage screen) ===")
print(screen_df.head(10).to_string(index=False))
n_flagged = screen_df["flag_possible_leakage"].sum()
print(f"\nFeatures flagged for possible leakage (univariate AUC > 0.97): {n_flagged}")
if n_flagged > 0:
    print(screen_df[screen_df.flag_possible_leakage])

# =================================================================
# TABLE 1: BASELINE CHARACTERISTICS BY CAD STATUS
# =================================================================
continuous_vars = ["Age", "Weight", "Length", "BMI", "BP", "PR", "FBS", "CR", "TG",
                    "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", "WBC", "Lymph",
                    "Neut", "PLT", "EF-TTE"]
ordinal_vars = ["Function Class", "Region RWMA", "VHD_ordinal"]
binary_vars = [c for c in feature_cols if c not in continuous_vars + ordinal_vars]

rows = []
grp1 = df[df["CAD"] == 1]  # Cad
grp0 = df[df["CAD"] == 0]  # Normal

for v in continuous_vars:
    m1, s1 = grp1[v].mean(), grp1[v].std()
    m0, s0 = grp0[v].mean(), grp0[v].std()
    tstat, p = stats.ttest_ind(grp1[v], grp0[v], equal_var=False)
    rows.append({"Variable": v, "Type": "continuous",
                 "CAD (n=%d)" % len(grp1): f"{m1:.2f} ± {s1:.2f}",
                 "Normal (n=%d)" % len(grp0): f"{m0:.2f} ± {s0:.2f}",
                 "p-value": f"{p:.3f}" if p >= 0.001 else "<0.001"})

for v in ordinal_vars + binary_vars:
    n1 = grp1[v].sum(); pct1 = n1 / len(grp1) * 100
    n0 = grp0[v].sum(); pct0 = n0 / len(grp0) * 100
    # chi-square on 2x2 (binary) or Mann-Whitney (ordinal with >2 levels)
    if df[v].nunique() <= 2:
        table = pd.crosstab(df[v], df["CAD"])
        chi2, p, _, _ = stats.chi2_contingency(table)
    else:
        _, p = stats.mannwhitneyu(grp1[v], grp0[v])
        n1 = grp1[v].mean(); n0 = grp0[v].mean()
    label = "mean" if df[v].nunique() > 2 else "n (%)"
    val1 = f"{n1:.2f} (mean)" if df[v].nunique() > 2 else f"{int(n1)} ({pct1:.1f}%)"
    val0 = f"{n0:.2f} (mean)" if df[v].nunique() > 2 else f"{int(n0)} ({pct0:.1f}%)"
    rows.append({"Variable": v, "Type": "categorical/ordinal",
                 "CAD (n=%d)" % len(grp1): val1,
                 "Normal (n=%d)" % len(grp0): val0,
                 "p-value": f"{p:.3f}" if p >= 0.001 else "<0.001"})

table1 = pd.DataFrame(rows)
table1.to_csv(OUT_TABLE1, index=False)
print(f"\nBaseline check table saved -> {OUT_TABLE1}")
print(table1.to_string(index=False))
