"""
17_results_tables_figures.py
=============================
Builds the manuscript tables (LaTeX fragments) and figures for the Results
section from the outputs of scripts 01-16.

Tables (tables/latex/):
    tab_baseline_continuous.tex   continuous variables, median (IQR), Mann-Whitney U
    tab_baseline_categorical.tex  binary variables n (%), chi-square; ordinal
                                  variables as category distributions, permutation chi-square
    tab_nested.tex                nested CV: AUC, difference from LR, Holm p, equivalence, ranks
    tab_nested_calibration.tex    calibration before/after recalibration, threshold metrics
    tab_clinical.tex              published and nested clinical models, incremental AUC
    tab_balancing.tex             mean AUC under three class-balancing strategies
    tab_conservative.tex          conservative models and Boruta-reduced models
    tab_features.tex              Boruta selection frequency and SHAP top-10 frequency
Also tables/single_split_delong_holm.csv (DeLong p-values with Holm adjustment).

Figures (figures/):
    fig_workflow.png, fig_nested_auc.png, fig_calibration_nested.png,
    fig_decision_curve_nested.png, fig_feature_agreement.png
and copies all manuscript figures, with their final names (Fig1-Fig6,
FigS1-FigS3), to figures/manuscript/.
"""
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats

T, F, L = "tables", "figures", "tables/latex"
os.makedirs(L, exist_ok=True)
os.makedirs(F, exist_ok=True)

# Validated categorical palette (slots 1-4) and text inks
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#a8a7a2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})

ORDER = ["Logistic Regression", "CatBoost", "Random Forest", "Gradient Boosting", "XGBoost", "SVM",
         "Naive Bayes", "GAM", "Decision Tree", "Neural Network"]
LABEL = {"Logistic Regression": "Logistic regression", "CatBoost": "CatBoost", "Random Forest": "Random forest",
         "Gradient Boosting": "Gradient boosting", "XGBoost": "XGBoost", "SVM": "SVM",
         "Naive Bayes": "Naive Bayes", "GAM": "GAM", "Decision Tree": "Decision tree",
         "Neural Network": "Neural network"}
FEAT = {"Age": "Age", "Typical Chest Pain": "Typical chest pain", "Atypical": "Atypical chest pain",
        "Region RWMA": "RWMA", "TG": "Triglycerides", "FBS": "FBS", "EF-TTE": "EF-TTE", "ESR": "ESR",
        "DM": "Diabetes mellitus", "HTN": "Hypertension", "Tinversion": "T-wave inversion",
        "Nonanginal": "Non-anginal chest pain", "BP": "Blood pressure", "K": "Potassium",
        "Dyspnea": "Dyspnoea", "PR": "Pulse rate", "Lymph": "Lymphocytes", "Neut": "Neutrophils",
        "St Depression": "ST depression", "Q Wave": "Q wave", "St Elevation": "ST elevation",
        "VHD_ordinal": "VHD", "Diastolic Murmur": "Diastolic murmur", "Sex_Male": "Male sex",
        "Function Class": "Functional class", "BMI": "BMI", "Weight": "Weight", "CR": "Creatinine",
        "BUN": "BUN", "HDL": "HDL", "LDL": "LDL", "PLT": "Platelets", "WBC": "WBC", "HB": "Haemoglobin",
        "Na": "Sodium", "Length": "Height", "Poor R Progression": "Poor R-wave progression"}


def fp(p):
    """Formats a p-value for the tables."""
    if pd.isna(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def ci(v):
    """Formats a confidence limit; limits that would round to -0.000 or +0.000 keep
    enough decimals to show their sign."""
    return f"{v:+.3f}" if abs(v) >= 0.0005 else f"{v:+.5f}"


def tex_plain(x):
    """Removes the LaTeX markup used in table labels (for the CSV copies)."""
    for a, b in (("$^2$", "2"), ("$\\mu$", "u"), ("$10^3/\\mu$L", "10^3/uL"), ("\\%", "%"), ("--", "-")):
        x = x.replace(a, b)
    return x


def write(name, body):
    with open(f"{L}/{name}", "w") as fh:
        fh.write(body)
    print(f"Wrote {L}/{name}")


# ---------------------------------------------------------------
# Baseline characteristics
# ---------------------------------------------------------------
def baseline_tables():
    d = pd.read_csv("data/CAD_processed.csv")
    cad, nor = d[d.CAD == 1], d[d.CAD == 0]
    cont = [("Age", "Age (years)"), ("Weight", "Weight (kg)"), ("Length", "Height (cm)"), ("BMI", "BMI (kg/m$^2$)"),
            ("BP", "Blood pressure (mmHg)"), ("PR", "Pulse rate (bpm)"), ("FBS", "FBS (mg/dL)"),
            ("CR", "Creatinine (mg/dL)"), ("TG", "Triglycerides (mg/dL)"), ("LDL", "LDL (mg/dL)"),
            ("HDL", "HDL (mg/dL)"), ("BUN", "BUN (mg/dL)"), ("ESR", "ESR (mm/h)"), ("HB", "Haemoglobin (g/dL)"),
            ("K", "Potassium (mEq/L)"), ("Na", "Sodium (mEq/L)"), ("WBC", "WBC (cells/$\\mu$L)"),
            ("Lymph", "Lymphocytes (\\%)"), ("Neut", "Neutrophils (\\%)"), ("PLT", "Platelets ($10^3/\\mu$L)"),
            ("EF-TTE", "EF-TTE (\\%)")]

    def miqr(x):
        q1, q2, q3 = np.percentile(x, [25, 50, 75])
        dec = 0 if q2 >= 100 else (2 if q2 < 10 else 1)
        return f"{q2:.{dec}f} ({q1:.{dec}f}--{q3:.{dec}f})"

    rows, csv_rows = [], []
    for v, lab in cont:
        p = stats.mannwhitneyu(cad[v], nor[v]).pvalue
        rows.append(f"{lab} & {miqr(cad[v])} & {miqr(nor[v])} & {fp(p)} \\\\")
        csv_rows.append({"variable": tex_plain(lab), f"CAD (n={len(cad)}), median (IQR)": tex_plain(miqr(cad[v])),
                         f"Normal (n={len(nor)}), median (IQR)": tex_plain(miqr(nor[v])), "p (Mann-Whitney)": p})
    write("tab_baseline_continuous.tex", "\n".join(rows) + "\n")
    pd.DataFrame(csv_rows).to_csv(f"{T}/baseline_continuous.csv", index=False)

    binary = [("Sex_Male", "Male sex"), ("DM", "Diabetes mellitus"), ("HTN", "Hypertension"),
              ("Current Smoker", "Current smoker"), ("EX-Smoker", "Ex-smoker"), ("FH", "Family history"),
              ("Obesity", "Obesity (as coded)"), ("CRF", "Chronic renal failure"),
              ("CVA", "Cerebrovascular accident"), ("Airway disease", "Airway disease"),
              ("Thyroid Disease", "Thyroid disease"), ("CHF", "Congestive heart failure"),
              ("DLP", "Dyslipidaemia"), ("Edema", "Oedema"), ("Weak Peripheral Pulse", "Weak peripheral pulse"),
              ("Lung rales", "Lung rales"), ("Systolic Murmur", "Systolic murmur"),
              ("Diastolic Murmur", "Diastolic murmur"), ("Typical Chest Pain", "Typical chest pain"),
              ("Atypical", "Atypical chest pain"), ("Nonanginal", "Non-anginal chest pain"),
              ("LowTH Ang", "Low-threshold angina"), ("Dyspnea", "Dyspnoea"), ("Q Wave", "Q wave"),
              ("St Elevation", "ST elevation"), ("St Depression", "ST depression"),
              ("Tinversion", "T-wave inversion"), ("LVH", "LVH"), ("Poor R Progression", "Poor R-wave progression")]
    rows, csv_rows = [], []
    for v, lab in binary:
        tab = pd.crosstab(d[v], d.CAD).reindex(index=[0, 1], columns=[1, 0], fill_value=0)
        p = stats.chi2_contingency(tab.values)[1]
        a, b = tab.loc[1, 1], tab.loc[1, 0]
        rows.append(f"{lab} & {a} ({100 * a / len(cad):.1f}) & {b} ({100 * b / len(nor):.1f}) & {fp(p)} \\\\")
        csv_rows.append({"variable": lab, "level": "yes", "CAD, n (%)": f"{a} ({100 * a / len(cad):.1f})",
                         "Normal, n (%)": f"{b} ({100 * b / len(nor):.1f})", "p": p,
                         "test": "chi-squared with continuity correction"})

    rng = np.random.default_rng(2026)

    def perm_chi2(x, y, n_perm=10000):
        obs = stats.chi2_contingency(pd.crosstab(x, y).values, correction=False)[0]
        yy = y.values.copy()
        cnt = 0
        for _ in range(n_perm):
            rng.shuffle(yy)
            cnt += stats.chi2_contingency(pd.crosstab(x.values, yy).values, correction=False)[0] >= obs
        return (cnt + 1) / (n_perm + 1)

    ordinal = [("Function Class", "Functional class", {0: "None", 1: "I", 2: "II", 3: "III"}),
               ("Region RWMA", "Regions with RWMA", {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}),
               ("VHD_ordinal", "VHD", {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"})]
    for v, lab, levels in ordinal:
        p = perm_chi2(d[v], d.CAD)
        rows.append(f"{lab}$^{{a}}$ & & & {fp(p)} \\\\")
        for k, name in levels.items():
            a, b = int((cad[v] == k).sum()), int((nor[v] == k).sum())
            rows.append(f"\\quad {name} & {a} ({100 * a / len(cad):.1f}) & {b} ({100 * b / len(nor):.1f}) & \\\\")
            csv_rows.append({"variable": lab, "level": name, "CAD, n (%)": f"{a} ({100 * a / len(cad):.1f})",
                             "Normal, n (%)": f"{b} ({100 * b / len(nor):.1f})", "p": p,
                             "test": "permutation chi-squared (10000 permutations)"})
    write("tab_baseline_categorical.tex", "\n".join(rows) + "\n")
    pd.DataFrame(csv_rows).to_csv(f"{T}/baseline_categorical.csv", index=False)


# ---------------------------------------------------------------
# Nested CV tables
# ---------------------------------------------------------------
def nested_tables():
    s = pd.read_csv(f"{T}/nested_cv_summary_class_weight.csv", index_col=0)
    c = pd.read_csv(f"{T}/nested_cv_comparison_class_weight.csv").set_index("model")
    r = pd.read_csv(f"{T}/nested_cv_ranks_class_weight.csv", index_col=0)
    rows = []
    for m in ORDER:
        auc = f"{s.loc[m, 'mean_auc']:.3f} ({s.loc[m, 'sd_auc']:.3f})"
        if m == "Logistic Regression":
            diff, dz, ph, eq = "Reference", "--", "--", "--"
        else:
            # difference expressed as model minus logistic regression
            d0, lo, hi = -c.loc[m, "mean_diff_ref_minus_model"], -c.loc[m, "ci95_hi"], -c.loc[m, "ci95_lo"]
            diff = f"{d0:+.3f} ({lo:+.3f} to {hi:+.3f})"
            dz = f"{-c.loc[m, 'cohens_dz']:.2f}"   # sign as for the difference (model minus LR)
            ph = fp(c.loc[m, "p_holm"])
            eq = f"{c.loc[m, 'smallest_equivalence_margin']:.3f}"
        rk = f"{100 * r.loc[m, 'share_rank_1']:.0f} & {r.loc[m, 'mean_rank']:.1f}"
        nconv = int(s.loc[m, "n_folds_converged"])
        lab = LABEL[m] + ("$^{b}$" if nconv < 50 else "")
        rows.append(f"{lab} & {auc} & {diff} & {dz} & {ph} & {eq} & {rk} \\\\")
    w = r["kendall_W"].iloc[0]
    ties = int(r["n_folds_tied_first"].iloc[0]) if "n_folds_tied_first" in r.columns else -1
    write("tab_nested.tex", "\n".join(rows) + f"\n% Kendall W = {w:.3f}; folds used for ranks = {int(r['n_folds_used'].iloc[0])}"
          f"; folds with a tie for first place = {ties}\n")

    cal = pd.read_csv(f"{T}/nested_cv_calibration_class_weight.csv", header=[0, 1], index_col=[0, 1])
    thr = pd.read_csv(f"{T}/nested_cv_thresholds_class_weight.csv", index_col=0)
    rows = []
    for m in ORDER:
        def cv(kind, col):
            v = cal.loc[(m, kind), (col, "mean")]
            return "n.e." if pd.isna(v) else f"{v:.2f}"
        br = cal.loc[(m, "raw"), ("brier", "mean")]
        bc = cal.loc[(m, "recalibrated"), ("brier", "mean")]
        rows.append(f"{LABEL[m]} & {cv('raw', 'citl')} & {cv('raw', 'slope')} & {br:.3f} & "
                    f"{cv('recalibrated', 'citl')} & {cv('recalibrated', 'slope')} & {bc:.3f} & "
                    f"{thr.loc[m, 'sensitivity']:.2f} & {thr.loc[m, 'specificity']:.2f} & "
                    f"{thr.loc[m, 'balanced_accuracy']:.2f} \\\\")
    write("tab_nested_calibration.tex", "\n".join(rows) + "\n")

    ov = pd.read_csv(f"{T}/nested_cv_modes_overview.csv", index_col=0)
    rows = [f"{LABEL[m]} & {ov.loc[m, 'class_weight']:.3f} & {ov.loc[m, 'oversample']:.3f} & {ov.loc[m, 'none']:.3f} \\\\"
            for m in ORDER]
    write("tab_balancing.tex", "\n".join(rows) + "\n")


def corrected_diff(wide, a, b, n_te, n_tr):
    d = (wide[a] - wide[b]).dropna()
    J = len(d)
    se = d.std(ddof=1) * np.sqrt(1 / J + n_te / n_tr)
    t95 = stats.t.ppf(0.975, J - 1)
    return d.mean(), d.mean() - t95 * se, d.mean() + t95 * se, 2 * stats.t.sf(abs(d.mean() / se), J - 1)


def holm(p):
    p = np.asarray(p, dtype=float); order = np.argsort(p); adj = np.empty(len(p)); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1, (len(p) - k) * p[i])); adj[i] = run
    return adj


def clinical_table():
    """Two families of comparisons, each Holm-adjusted: successive steps (M2-M1, M3-M2, M4-M3)
    and every model against M1."""
    s = pd.read_csv(f"{T}/clinical_models_summary.csv", index_col=0)
    f = pd.read_csv(f"{T}/clinical_models_folds.csv")
    nf = pd.read_csv(f"{T}/nested_cv_class_weight_folds.csv")
    nf = nf[nf.model.isin(["CatBoost", "Random Forest"]) & (nf.converged == True)]
    nf = nf.assign(model=nf.model + " (all 53 features)")
    f = pd.concat([f, nf[["model", "repeat", "fold", "n_train", "n_test", "auc"]]], ignore_index=True)
    wide = f.pivot_table(index=["repeat", "fold"], columns="model", values="auc")
    n_tr, n_te = f.n_train.mean(), f.n_test.mean()
    names = {"Diamond-Forrester (published)": "Diamond--Forrester, published",
             "CAD Consortium (published)": "CAD Consortium, published",
             "M1 basic (age, sex, chest pain)": "M1: age, sex, chest-pain type",
             "M2 + clinical history and examination": "M2: M1 + history and examination",
             "M3 + ECG and laboratory": "M3: M2 + ECG and laboratory",
             "M4 + echocardiography (all 53 features)": "M4: M3 + echocardiography (all 53)",
             "CatBoost (all 53 features)": "CatBoost, all 53 features",
             "Random Forest (all 53 features)": "Random forest, all 53 features"}
    m1 = "M1 basic (age, sex, chest pain)"
    steps = [("M2 + clinical history and examination", m1),
             ("M3 + ECG and laboratory", "M2 + clinical history and examination"),
             ("M4 + echocardiography (all 53 features)", "M3 + ECG and laboratory")]
    st = {a: corrected_diff(wide, a, b, n_te, n_tr) for a, b in steps}
    st_adj = dict(zip(st, holm([v[3] for v in st.values()])))
    others = [m for m in names if m != m1]
    v1 = {m: corrected_diff(wide, m, m1, n_te, n_tr) for m in others}
    v1_adj = dict(zip(v1, holm([v[3] for v in v1.values()])))
    out = []
    rows = []
    for m in names:
        a = f"{s.loc[m, 'mean_auc']:.3f} ({s.loc[m, 'sd_auc']:.3f})"
        if m in st:
            d0, lo, hi, _ = st[m]
            step = f"{d0:+.3f} ({ci(lo)} to {ci(hi)}) & {fp(st_adj[m])}"
        else:
            step = "-- & --"
        if m == m1:
            vs1 = "Reference & --"
        else:
            d0, lo, hi, _ = v1[m]
            vs1 = f"{d0:+.3f} ({ci(lo)} to {ci(hi)}) & {fp(v1_adj[m])}"
            out.append({"model": m, "vs": "M1", "delta": d0, "lo": lo, "hi": hi, "p_holm": v1_adj[m]})
        mp = f"{s.loc[m, 'mean_predicted']:.2f}" if m in s.index and pd.notna(s.loc[m].get("mean_predicted")) else "--"
        rows.append(f"{names[m]} & {a} & {mp} & {step} & {vs1} \\\\")
    for m in st:
        out.append({"model": m, "vs": "previous step", "delta": st[m][0], "lo": st[m][1], "hi": st[m][2], "p_holm": st_adj[m]})
    pd.DataFrame(out).to_csv(f"{T}/clinical_models_table_contrasts.csv", index=False)
    write("tab_clinical.tex", "\n".join(rows) + "\n")


def conservative_table():
    s = pd.read_csv(f"{T}/regularization_summary.csv").set_index("model")
    b = pd.read_csv(f"{T}/boruta_nested_comparison.csv").set_index("model")
    rows = []
    for m, lab in [("Logistic Regression (primary grid)", "Logistic regression (main grid)"),
                   ("LR (extended C grid)", "Logistic regression (extended $C$ grid)"),
                   ("Elastic-net LR", "Elastic-net logistic regression"), ("PCA + LR", "PCA + logistic regression"),
                   ("Random Forest (primary grid)", "Random forest (main grid)"),
                   ("Restricted RF", "Restricted random forest")]:
        if m not in s.index:
            continue
        q = s.loc[m]
        rows.append(f"{lab} & {q.mean_auc:.3f} ({q.sd_auc:.3f}) & {q.mean_apparent_auc:.3f} & {q.optimism:.3f} \\\\")
    rows.append("\\midrule")
    rows.append("\\multicolumn{4}{l}{\\emph{Boruta-confirmed features only (selected in each outer fold)}} \\\\")
    for m in ["Logistic Regression", "Random Forest", "Neural Network"]:
        q = b.loc[m]
        rows.append(f"{LABEL[m]} & {q.mean_auc_boruta:.3f} ({q.sd_auc_boruta:.3f}) & "
                    f"\\multicolumn{{2}}{{l}}{{{q.delta_auc:+.3f} ({ci(q.ci95_lo)} to {ci(q.ci95_hi)}) vs.\\ 53 features}} \\\\")
    write("tab_conservative.tex", "\n".join(rows) + "\n")


def feature_table():
    bf = pd.read_csv(f"{T}/boruta_nested_selection_frequency.csv", index_col=0)
    tp = pd.read_csv(f"{T}/shap_lr_vs_rf_nested_top10.csv", index_col=0)
    pr = pd.read_csv(f"{T}/shap_lr_vs_rf_primary.csv").set_index("feature")
    df = bf.join(tp, how="outer").fillna(0)
    keep = df[(df.confirmed_share >= 0.2) | (df.lr_top10_share >= 0.5) | (df.rf_top10_share >= 0.5)]
    keep = keep.sort_values(["confirmed_share", "rf_top10_share", "lr_top10_share"], ascending=False)
    rows = [f"{FEAT.get(f, f)} & {100 * q.confirmed_share:.0f} & {100 * q.lr_top10_share:.0f} & "
            f"{100 * q.rf_top10_share:.0f} & {int(pr.loc[f, 'lr_rank'])} & {int(pr.loc[f, 'rf_rank'])} \\\\"
            for f, q in keep.iterrows()]
    write("tab_features.tex", "\n".join(rows) + "\n")


def single_split_holm():
    d = pd.read_csv(f"{T}/table_delong_pairwise_test.csv")
    p = d["p_value"].values
    order = np.argsort(p)
    adj = np.empty(len(p)); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1, (len(p) - k) * p[i])); adj[i] = run
    d["p_holm"] = adj
    d.to_csv(f"{T}/single_split_delong_holm.csv", index=False)
    print("Single-split DeLong vs LR with Holm:")
    print(d[["model_B", "diff", "p_value", "p_holm"]].round(4).to_string(index=False))


# ---------------------------------------------------------------
# Figures
# ---------------------------------------------------------------
def fig_workflow():
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.set_xlim(0, 100); ax.set_ylim(-1, 47); ax.axis("off")

    def box(x, y, w, h, text, fc="#f0efec", ec=GREY, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.0, color=INK,
                fontweight="bold" if bold else "normal", linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7, shrinkA=0, shrinkB=0))

    box(0.5, 15, 15, 12, "Z-Alizadeh Sani\n303 patients\n53 features", bold=True)
    box(21, 22, 32, 19, "Primary analysis\nnested repeated cross-validation\nouter: 5 folds x 10 repeats (50 folds)\n"
        "inner: 5-fold grid search (AUC)\nclass weighting", fc="#dceafa", ec=BLUE)
    box(21, 0, 32, 10, "Secondary analysis\nsingle stratified 70/30 split\n(train 212, test 91)")
    right = [(38.5, "Inside each outer training part:\nscaling, balancing, tuning, Boruta,\nthreshold, recalibration"),
             (29, "Outer test folds: AUC, corrected t-test with Holm,\nTOST, ranks (Kendall's W), calibration,\nthreshold metrics, decision curves"),
             (19.5, "Clinical reference models (Diamond-Forrester,\nCAD Consortium) and nested models M1-M4"),
             (10, "Sensitivity: over-sampling, no balancing,\nconservative and Boruta-reduced models,\nSHAP (LR vs RF), sample size (Riley)"),
             (0, "Single-split outputs: test AUC (DeLong, Holm),\nMDI and SHAP importance, LR coefficients")]
    for y, text in right:
        box(59, y, 40, 8, text)
    arrow(15.8, 23, 20.7, 31); arrow(15.8, 19, 20.7, 5)
    for y, _ in right[:4]:
        arrow(53.3, 31.5, 58.7, y + 4)
    arrow(53.3, 5, 58.7, 4)
    fig.savefig(f"{F}/fig_workflow.png", bbox_inches="tight")
    plt.close(fig)


def fig_nested_auc():
    f = pd.read_csv(f"{T}/nested_cv_class_weight_folds.csv")
    f = f[f.converged == True]
    r = pd.read_csv(f"{T}/nested_cv_ranks_class_weight.csv", index_col=0)
    wide = f.pivot_table(index=["repeat", "fold"], columns="model", values="auc").dropna()
    ranks = wide.round(9).rank(axis=1, ascending=False, method="min").astype(int)
    order = list(f.groupby("model")["auc"].mean().sort_values(ascending=False).index)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
    for i, m in enumerate(order):
        y = len(order) - 1 - i
        v = f.loc[f.model == m, "auc"].values
        col = BLUE if m == "Logistic Regression" else GREY
        bp = a1.boxplot(v, positions=[y], vert=False, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=INK2, lw=0.8),
                        capprops=dict(color=INK2, lw=0.8), boxprops=dict(lw=0.8, ec=INK2))
        bp["boxes"][0].set_facecolor(col if m == "Logistic Regression" else "#e4e3df")
        jit = np.random.default_rng(i).uniform(-0.18, 0.18, len(v))
        a1.scatter(v, y + jit, s=5, color=col, alpha=0.6, lw=0, zorder=3)
        a1.text(1.005, y, f"{v.mean():.3f}", va="center", fontsize=7, color=INK2, transform=a1.get_yaxis_transform())
    a1.set_yticks(range(len(order)))
    a1.set_yticklabels([LABEL[m] for m in order][::-1])
    a1.set_xlabel("AUC in outer test fold")
    a1.set_xlim(0.45, 1.0)
    a1.grid(axis="x", color=GRID, lw=0.6); a1.set_axisbelow(True)
    a1.set_title("A  AUC across 50 outer folds (mean at right)", loc="left", fontsize=8.5, color=INK)

    mat = np.array([[np.mean(ranks[m] == k) for k in range(1, 11)] for m in order])
    im = a2.imshow(mat, cmap="Blues", vmin=0, vmax=max(0.5, mat.max()), aspect="auto")
    for i in range(mat.shape[0]):
        for k in range(mat.shape[1]):
            if mat[i, k] >= 0.1:
                a2.text(k, i, f"{100 * mat[i, k]:.0f}", ha="center", va="center", fontsize=6.3,
                        color="white" if mat[i, k] > 0.3 else INK)
    a2.set_xticks(range(10)); a2.set_xticklabels(range(1, 11))
    a2.set_yticks(range(len(order))); a2.set_yticklabels([LABEL[m] for m in order])
    a2.set_xlabel("Rank in outer fold (1 = highest AUC)")
    a2.set_title(f"B  Share of folds at each rank (%); Kendall's W = {r['kendall_W'].iloc[0]:.2f}",
                 loc="left", fontsize=8.5, color=INK)
    for sp in a2.spines.values():
        sp.set_visible(False)
    a2.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(f"{F}/fig_nested_auc.png", bbox_inches="tight")
    plt.close(fig)


SHOW = [("Logistic Regression", BLUE, "-"), ("CatBoost", ORANGE, "--"), ("Random Forest", AQUA, "-."),
        ("Gradient Boosting", YELLOW, ":")]


def fig_calibration():
    pr = pd.read_csv(f"{T}/nested_cv_class_weight_predictions.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for ax, col, title in ((axs[0], "p", "A  Before recalibration"), (axs[1], "p_recal", "B  After recalibration (learned in inner folds)")):
        ax.plot([0, 1], [0, 1], color=GREY, lw=0.9, ls="--", zorder=1)
        for m, c, ls in SHOW[:3]:
            g = pr[pr.model == m]
            q = pd.qcut(g[col], 10, duplicates="drop")
            b = g.groupby(q, observed=True).agg(p=(col, "mean"), y=("y", "mean"))
            ax.plot(b.p, b.y, color=c, ls=ls, lw=2, marker="o", ms=4, mec="white", mew=0.8, label=LABEL[m], zorder=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability (deciles)")
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=8.5, color=INK)
    axs[0].set_ylabel("Observed proportion with CAD")
    axs[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{F}/fig_calibration_nested.png", bbox_inches="tight")
    plt.close(fig)


def fig_dca():
    d = pd.read_csv(f"{T}/nested_cv_decision_curve_class_weight.csv")
    if "predictions" in d.columns:
        d = d[d.predictions == "recalibrated"]
    d = d[(d.threshold >= 0.05) & (d.threshold <= 0.85 + 1e-9)]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ta = d[d.model == "Treat all"]
    ax.plot(ta.threshold, ta["mean"], color=GREY, lw=1.6, label="Angiography for all")
    ax.axhline(0, color=INK, lw=0.9, label="Angiography for none")
    for m, c, ls in SHOW:
        g = d[d.model == m]
        ax.plot(g.threshold, g["mean"], color=c, ls=ls, lw=2, label=LABEL[m])
    ax.set_xlim(0.05, 0.85); ax.set_ylim(-0.05, 0.75)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.3, loc="lower left", bbox_to_anchor=(0.0, 0.1))
    fig.tight_layout()
    fig.savefig(f"{F}/fig_decision_curve_nested.png", bbox_inches="tight")
    plt.close(fig)


def fig_dca_balancing():
    """Supplementary figure: decision curves of logistic regression, CatBoost and
    random forest under the three class-balancing strategies, for raw and for
    recalibrated predictions (pooled outer-fold predictions, mean over repeats)."""
    modes = [("class_weight", "Class weighting", BLUE, "-"), ("oversample", "Over-sampling", ORANGE, "--"),
             ("none", "No balancing", AQUA, ":")]
    models = ["Logistic Regression", "CatBoost", "Random Forest"]
    data = {}
    for mode, *_ in modes:
        path = f"{T}/nested_cv_decision_curve_{mode}.csv"
        if not os.path.exists(path):
            print(f"fig_dca_balancing skipped: {path} not found")
            return
        d = pd.read_csv(path)
        if "predictions" not in d.columns:
            print("fig_dca_balancing skipped: re-run 12_nested_cv_summary.py first")
            return
        data[mode] = d[(d.threshold >= 0.05) & (d.threshold <= 0.85 + 1e-9)]
    fig, axs = plt.subplots(2, 3, figsize=(8.6, 5.0), sharex=True, sharey=True)
    for i, kind in enumerate(["raw", "recalibrated"]):
        for j, m in enumerate(models):
            ax = axs[i, j]
            ta = data["class_weight"]
            ta = ta[(ta.model == "Treat all") & (ta.predictions == kind)]
            ax.plot(ta.threshold, ta["mean"], color=GREY, lw=1.4, label="Angiography for all")
            ax.axhline(0, color=INK, lw=0.8)
            for mode, lab, c, ls in modes:
                g = data[mode]
                g = g[(g.model == m) & (g.predictions == kind)]
                ax.plot(g.threshold, g["mean"], color=c, ls=ls, lw=1.8, label=lab)
            ax.set_xlim(0.05, 0.85); ax.set_ylim(-0.05, 0.75)
            ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
            ttl = f"{LABEL[m]}, {'before' if kind == 'raw' else 'after'} recalibration"
            ax.set_title(f"{'ABCDEF'[3 * i + j]}  {ttl}", loc="left", fontsize=8, color=INK)
            if i == 1:
                ax.set_xlabel("Threshold probability")
            if j == 0:
                ax.set_ylabel("Net benefit")
    axs[0, 0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{F}/fig_decision_curve_balancing.png", bbox_inches="tight")
    plt.close(fig)
    rows = []
    for mode, lab, *_ in modes:
        d = data[mode]
        for m in models:
            for kind in ("raw", "recalibrated"):
                g = d[(d.model == m) & (d.predictions == kind)].set_index("threshold")["mean"]
                rows.append({"strategy": lab, "model": LABEL[m], "predictions": kind,
                             **{f"nb_{t:.2f}": g.get(round(t, 2), np.nan) for t in (0.2, 0.5, 0.7, 0.85)}})
    pd.DataFrame(rows).to_csv(f"{T}/decision_curve_by_balancing.csv", index=False)


def fig_features():
    bf = pd.read_csv(f"{T}/boruta_nested_selection_frequency.csv", index_col=0)
    tp = pd.read_csv(f"{T}/shap_lr_vs_rf_nested_top10.csv", index_col=0)
    top = bf[bf.confirmed_share > 0].head(14).iloc[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.6))
    a1.barh([FEAT.get(f, f) for f in top.index], 100 * top.confirmed_share, color=BLUE, height=0.62)
    for y, v in enumerate(100 * top.confirmed_share):
        a1.text(v + 1.5, y, f"{v:.0f}", va="center", fontsize=6.8, color=INK2)
    a1.set_xlim(0, 112); a1.set_xlabel("Folds in which Boruta confirmed the feature (%)")
    a1.set_title("A  Boruta inside 50 outer folds", loc="left", fontsize=8.5, color=INK)
    a1.grid(axis="x", color=GRID, lw=0.6); a1.set_axisbelow(True)

    t = tp[(tp.lr_top10_share >= 0.4) | (tp.rf_top10_share >= 0.4)].copy()
    t["mx"] = t[["lr_top10_share", "rf_top10_share"]].max(axis=1)
    t = t.sort_values("mx")
    ys = np.arange(len(t))
    for y, (_, q) in zip(ys, t.iterrows()):
        a2.plot([100 * q.lr_top10_share, 100 * q.rf_top10_share], [y + 0.17, y - 0.17], color=GRID, lw=2, zorder=1)
    a2.scatter(100 * t.lr_top10_share, ys + 0.17, s=34, color=BLUE, marker="o", ec="white", lw=0.8, zorder=3, label="Logistic regression")
    a2.scatter(100 * t.rf_top10_share, ys - 0.17, s=34, color=ORANGE, marker="s", ec="white", lw=0.8, zorder=3, label="Random forest")
    a2.set_yticks(ys); a2.set_yticklabels([FEAT.get(f, f) for f in t.index])
    a2.set_xlim(-3, 103); a2.set_xlabel("Folds with the feature in the SHAP top 10 (%)", labelpad=2)
    a2.set_title("B  SHAP top-10 membership by model", loc="left", fontsize=8.5, color=INK)
    a2.grid(axis="x", color=GRID, lw=0.6); a2.set_axisbelow(True)
    a2.legend(frameon=False, fontsize=7.2, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2)
    fig.tight_layout()
    fig.savefig(f"{F}/fig_feature_agreement.png", bbox_inches="tight")
    plt.close(fig)


# Final file names used in the manuscript and supplement
MANUSCRIPT_FIGURES = {"fig_workflow.png": "Fig1_workflow.png", "fig_nested_auc.png": "Fig2_nested_cv_auc_ranks.png",
                      "fig_calibration_nested.png": "Fig3_calibration.png",
                      "fig_decision_curve_nested.png": "Fig4_decision_curves.png",
                      "fig3_roc_curves.png": "Fig5_single_split_roc.png",
                      "fig_feature_agreement.png": "Fig6_feature_agreement.png",
                      "fig2_feature_importance.png": "FigS1_mdi_importance.png",
                      "fig4_shap_summary.png": "FigS2_rf_shap_summary.png",
                      "fig_decision_curve_balancing.png": "FigS3_decision_curves_balancing.png"}


def copy_manuscript_figures():
    import shutil
    os.makedirs(f"{F}/manuscript", exist_ok=True)
    for src, dst in MANUSCRIPT_FIGURES.items():
        if os.path.exists(f"{F}/{src}"):
            shutil.copy(f"{F}/{src}", f"{F}/manuscript/{dst}")
    print(f"Copied manuscript figures to {F}/manuscript/")


if __name__ == "__main__":
    baseline_tables()
    nested_tables()
    clinical_table()
    conservative_table()
    feature_table()
    single_split_holm()
    fig_workflow()
    fig_nested_auc()
    fig_calibration()
    fig_dca()
    fig_dca_balancing()
    fig_features()
    copy_manuscript_figures()
    print("Done.")
