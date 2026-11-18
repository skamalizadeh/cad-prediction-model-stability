"""
08_repeated_holdout_robustness.py
=================================
Monte Carlo repeated-holdout robustness analysis. A single 70/30 split of
n=303 patients (test n=91) gives an unstable AUC estimate; this script
quantifies that instability directly: B=50 independent stratified 70/30
splits (a different random seed for each), refitting every model at the
hyperparameters selected in the primary grid search
(03_train_models.py and 07_boruta_sensitivity_analysis.py), on freshly
over-sampled training data in each repeat, then evaluating test AUC.
This is a repeated-holdout robustness check, not nested cross-validation
(hyperparameters are not re-tuned in each repeat).

The analysis is run for both the full 53-feature and the Boruta-7-feature
models, which also allows the stability (not only the mean) of the two
feature sets to be compared across resamples.

Usage:
    python 08_repeated_holdout_robustness.py
        runs full53, boruta7 and the summary in one go (default)
    python 08_repeated_holdout_robustness.py full53 [SEED_START SEED_END]
    python 08_repeated_holdout_robustness.py boruta7 [SEED_START SEED_END]
    python 08_repeated_holdout_robustness.py summary
        single stages, optionally in chunks. A stage that starts at
        SEED_START=0 overwrites its CSV, so re-running is always safe;
        later chunks are appended.

Output: tables/table_repeated_holdout_full53.csv
        tables/table_repeated_holdout_boruta7.csv
        tables/table_repeated_holdout_summary.csv
        tables/table_stability_comparison.csv
        figures/fig5_repeated_holdout_stability.png
"""
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score
from pygam import LogisticGAM, s
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42
B = 50  # total number of repeated holdout splits (across all chunks)
DATA_DIR = "data"
MODEL_DIR = "models"
TABLE_DIR = "tables"
FIG_DIR = "figures"

BORUTA7 = ["Age", "Typical Chest Pain", "Atypical", "FBS", "Region RWMA", "ESR", "TG"]

full_data = pd.read_csv(f"{DATA_DIR}/CAD_processed.csv")
y_all = full_data["CAD"]
X_all_full = full_data.drop(columns=["CAD"])
X_all_b7 = full_data[BORUTA7]

sklearn_model_names = ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost",
                        "SVM", "Decision Tree", "CatBoost", "Naive Bayes", "Neural Network"]


def load_template_models(suffix=""):
    templates = {}
    for name in sklearn_model_names:
        path = f"{MODEL_DIR}/{name.replace(' ', '_')}{suffix}.joblib"
        templates[name] = joblib.load(path)
    return templates


def run_repeated_holdout(X_all, feature_names, suffix, gam_lam, n_splines, seed_start, seed_end):
    # Templates are fitted imblearn Pipelines (scaler -> sampler -> classifier),
    # with over-sampling nested inside the pipeline. clone() resets a Pipeline
    # to its unfitted hyperparameters, including its own scaler and sampler
    # steps, so each repeat below fits the pipeline directly on the RAW
    # training portion of that repeat's split; no separate manual
    # over-sampling or scaling is needed here.
    templates = load_template_models(suffix)
    results = {name: [] for name in sklearn_model_names + ["GAM"]}

    for b in range(seed_start, seed_end):
        seed = 1000 + b
        Xtr, Xte, ytr, yte = train_test_split(
            X_all, y_all, test_size=0.30, stratify=y_all, random_state=seed
        )

        for name, tmpl in templates.items():
            model = clone(tmpl)
            model.fit(Xtr, ytr)               # pipeline over-samples internally, training portion only
            p = model.predict_proba(Xte)[:, 1]  # pipeline scales Xte; sampler is skipped at inference
            results[name].append(roc_auc_score(yte, p))

        # GAM: fixed lambda (no per-repeat gridsearch, for tractability), but
        # scaling + over-sampling are (re)fit on this repeat's raw training
        # portion only, exactly as for the sklearn pipelines above -- the
        # test portion of this repeat is never touched by either step.
        # Occasionally fails to converge on a particular resample; recorded
        # as NaN for that repeat rather than aborting the whole run.
        try:
            scaler = StandardScaler().fit(Xtr)
            ros = RandomOverSampler(random_state=seed)
            Xtr_bal_s, ytr_bal = ros.fit_resample(scaler.transform(Xtr), ytr)
            Xte_s = scaler.transform(Xte)

            terms = s(0, n_splines=n_splines)
            for i in range(1, len(feature_names)):
                terms += s(i, n_splines=n_splines)
            gam = LogisticGAM(terms, lam=gam_lam)
            gam.fit(Xtr_bal_s, ytr_bal.values)
            p_gam = gam.predict_mu(Xte_s)
            if np.any(~np.isfinite(p_gam)):
                raise ValueError("non-finite GAM predictions")
            results["GAM"].append(roc_auc_score(yte, p_gam))
        except Exception:
            results["GAM"].append(np.nan)

        if (b + 1) % 10 == 0:
            print(f"  [{suffix or 'full53'}] completed {b+1}/{B} repeats")

    return pd.DataFrame(results)


import sys, os
VALID_STAGES = ("all", "full53", "boruta7", "summary")
RUN_WHICH = sys.argv[1] if len(sys.argv) > 1 else "all"
if RUN_WHICH not in VALID_STAGES:
    sys.exit(f"Unknown stage '{RUN_WHICH}'. Use one of: {', '.join(VALID_STAGES)}")
SEED_START = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SEED_END = int(sys.argv[3]) if len(sys.argv) > 3 else B


def write_chunk(path, new_df, seed_start):
    """A chunk starting at seed 0 overwrites the file (re-running never duplicates
    rows); later chunks are appended to it."""
    if seed_start > 0 and os.path.exists(path):
        combined = pd.concat([pd.read_csv(path), new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(path, index=False)
    return combined


t0 = time.time()
if RUN_WHICH in ("all", "full53"):
    print(f"Running repeated holdout: FULL 53-feature models, seeds {SEED_START}-{SEED_END-1}...")
    chunk = run_repeated_holdout(X_all_full, X_all_full.columns.tolist(), "", gam_lam=15.848931924611142, n_splines=5,
                                  seed_start=SEED_START, seed_end=SEED_END)
    combined = write_chunk(f"{TABLE_DIR}/table_repeated_holdout_full53.csv", chunk, SEED_START)
    print(f"Done in {time.time()-t0:.0f}s. Total rows so far: {len(combined)}\n")

if RUN_WHICH in ("all", "boruta7"):
    t0 = time.time()
    print(f"Running repeated holdout: Boruta-7-feature models, seeds {SEED_START}-{SEED_END-1}...")
    chunk = run_repeated_holdout(X_all_b7, BORUTA7, "_boruta7", gam_lam=0.3981071705534973, n_splines=6,
                                  seed_start=SEED_START, seed_end=SEED_END)
    combined = write_chunk(f"{TABLE_DIR}/table_repeated_holdout_boruta7.csv", chunk, SEED_START)
    print(f"Done in {time.time()-t0:.0f}s. Total rows so far: {len(combined)}\n")

if RUN_WHICH in ("full53", "boruta7"):
    sys.exit(0)  # single-stage run: the summary is produced by the "summary" (or default) run

# ---------------------------------------------------------------
# Summary: mean, SD, 2.5-97.5 percentile AUC across B repeats,
# for both feature sets (loaded from disk so this works whether
# full53/boruta7 were run in this process or a separate one)
# ---------------------------------------------------------------
rh_full = pd.read_csv(f"{TABLE_DIR}/table_repeated_holdout_full53.csv")
rh_b7 = pd.read_csv(f"{TABLE_DIR}/table_repeated_holdout_boruta7.csv")
for _label, _df in (("full53", rh_full), ("boruta7", rh_b7)):
    if len(_df) != B:
        sys.exit(f"table_repeated_holdout_{_label}.csv has {len(_df)} rows, expected {B}. "
                 f"Re-run the '{_label}' stage from seed 0.")

def summarize(df, label):
    rows = []
    for col in df.columns:
        vals = df[col].values
        valid = vals[np.isfinite(vals)]
        rows.append({
            "model": col, "feature_set": label, "n_valid_repeats": len(valid),
            "mean_AUC": np.nanmean(vals), "sd_AUC": np.nanstd(valid, ddof=1),
            "p2.5": np.nanpercentile(valid, 2.5), "p97.5": np.nanpercentile(valid, 97.5),
            "min": np.nanmin(valid), "max": np.nanmax(valid),
        })
    return pd.DataFrame(rows)

summary = pd.concat([summarize(rh_full, "Full (53 features)"),
                      summarize(rh_b7, "Boruta-confirmed (7 features)")], ignore_index=True)
summary.to_csv(f"{TABLE_DIR}/table_repeated_holdout_summary.csv", index=False)
print(summary.round(4).to_string(index=False))
_incomplete = summary[summary["n_valid_repeats"] < B]
if len(_incomplete):
    print("\nNOTE: some fits did not converge and are excluded from the statistics above:")
    print(_incomplete[["model", "feature_set", "n_valid_repeats"]].to_string(index=False))

# stability comparison: does the 7-feature set have lower SD (more stable) per model?
pivot_sd = summary.pivot(index="model", columns="feature_set", values="sd_AUC")
pivot_sd["SD_reduction_with_boruta7"] = pivot_sd["Full (53 features)"] - pivot_sd["Boruta-confirmed (7 features)"]
pivot_sd.to_csv(f"{TABLE_DIR}/table_stability_comparison.csv")
print("\n=== SD of test AUC across 50 repeats: full-53 vs Boruta-7 ===")
print(pivot_sd.round(4).to_string())
print(f"\nModels more STABLE (lower SD) with Boruta-7: "
      f"{(pivot_sd['SD_reduction_with_boruta7'] > 0).sum()} of {len(pivot_sd)}")

# =================================================================
# FIGURE: boxplot of repeated-holdout test AUC, full-53 vs Boruta-7
# =================================================================
models_order = summary.groupby("model")["mean_AUC"].mean().sort_values(ascending=False).index.tolist()
fig, ax = plt.subplots(figsize=(11, 6))
positions_full = np.arange(len(models_order)) * 2.2
positions_b7 = positions_full + 0.9

def _clean(s):
    return s[np.isfinite(s)]

bp1 = ax.boxplot([_clean(rh_full[m].values) for m in models_order], positions=positions_full, widths=0.75,
                  patch_artist=True, showfliers=False)
bp2 = ax.boxplot([_clean(rh_b7[m].values) for m in models_order], positions=positions_b7, widths=0.75,
                  patch_artist=True, showfliers=False)
for box in bp1["boxes"]:
    box.set_facecolor("#6a5acd"); box.set_alpha(0.7)
for box in bp2["boxes"]:
    box.set_facecolor("#e07b39"); box.set_alpha(0.7)

ax.set_xticks(positions_full + 0.45)
ax.set_xticklabels(models_order, rotation=35, ha="right")
ax.set_ylabel("Test AUC across 50 repeated 70/30 holdout splits")
ax.set_title("Repeated-holdout robustness: full 53-feature vs Boruta-confirmed 7-feature models")
ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Full (53 features)", "Boruta-confirmed (7 features)"],
          loc="lower right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig5_repeated_holdout_stability.png", dpi=200)
print("\nFig 5 (repeated-holdout stability) saved.")
