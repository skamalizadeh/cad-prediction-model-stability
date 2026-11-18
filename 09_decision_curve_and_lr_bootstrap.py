"""
09_decision_curve_and_lr_bootstrap.py
=====================================
Two supplementary analyses:

(a) Decision-curve analysis (Vickers & Elkin 2006) on the held-out test
    set for the top-tier models, compared against "treat all" and
    "treat none" strategies, across a clinically plausible range of
    threshold probabilities.

(b) Bootstrap 95% confidence intervals for the logistic-regression
    standardised coefficients (percentile method, B=1000), since the
    model is L2-penalised and has no closed-form CI. Over-sampling is
    re-applied inside each bootstrap resample (not just once), so the
    resampling respects the same nested design used for tuning.

Input : data/train_raw.csv, data/test_raw.csv, data/predictions_test.csv
Output: tables/table_decision_curve.csv
        tables/table_lr_coef_bootstrap.csv
        figures/fig7_decision_curve.png
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

DATA_DIR = "data"
TABLE_DIR = "tables"
FIG_DIR = "figures"

# ---------------------------------------------------------------
# (a) Decision-curve analysis on the test set
# ---------------------------------------------------------------
pred_test = pd.read_csv(f"{DATA_DIR}/predictions_test.csv")
y = pred_test["y_true"].values
n = len(y)
prevalence = y.mean()

models_to_plot = ["Logistic Regression", "Neural Network", "Gradient Boosting", "XGBoost"]
pt_grid = np.linspace(0.05, 0.95, 91)  # threshold probabilities 5%-95%

rows = []
for pt in pt_grid:
    row = {"threshold_prob": pt,
           "net_benefit_treat_all": prevalence - (1 - prevalence) * (pt / (1 - pt)),
           "net_benefit_treat_none": 0.0}
    for m in models_to_plot:
        p = pred_test[m].values
        predicted_positive = p >= pt
        tp = np.sum(predicted_positive & (y == 1))
        fp = np.sum(predicted_positive & (y == 0))
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        row[f"net_benefit_{m}"] = nb
    rows.append(row)

dca = pd.DataFrame(rows)
dca.to_csv(f"{TABLE_DIR}/table_decision_curve.csv", index=False)

plt.figure(figsize=(7, 5.5))
colors = {"Logistic Regression": "#1f77b4", "Neural Network": "#bcbd22",
          "Gradient Boosting": "#2ca02c", "XGBoost": "#d62728"}
for m in models_to_plot:
    plt.plot(dca["threshold_prob"], dca[f"net_benefit_{m}"], label=m, color=colors[m], lw=2)
plt.plot(dca["threshold_prob"], dca["net_benefit_treat_all"], "k--", lw=1.3, label="Treat all")
plt.plot(dca["threshold_prob"], dca["net_benefit_treat_none"], "k:", lw=1.3, label="Treat none")
plt.ylim(-0.15, max(0.8, dca["net_benefit_treat_all"].max() + 0.05))
plt.xlabel("Threshold probability")
plt.ylabel("Net benefit")
plt.title("Decision-curve analysis, held-out test set (n = 91)")
plt.legend(fontsize=8, loc="upper right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig7_decision_curve.png", dpi=200)
plt.close()
print("Decision-curve table + figure saved.")

# ---------------------------------------------------------------
# (b) Bootstrap CIs for logistic-regression standardised coefficients
# ---------------------------------------------------------------
train_raw = pd.read_csv(f"{DATA_DIR}/train_raw.csv")
FEATURES = [c for c in train_raw.columns if c != "CAD"]
X = train_raw[FEATURES].values
yv = train_raw["CAD"].values
n_train = len(yv)
C_SELECTED = 0.01  # from the nested grid search (Table "Selected hyperparameters")

B = 1000
coef_samples = np.zeros((B, len(FEATURES)))
idx_pos = np.where(yv == 1)[0]
idx_neg = np.where(yv == 0)[0]

for b in range(B):
    # stratified bootstrap resample of the RAW training set (with replacement)
    samp_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
    samp_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
    samp_idx = np.concatenate([samp_pos, samp_neg])
    Xb, yb = X[samp_idx], yv[samp_idx]

    scaler_b = StandardScaler().fit(Xb)
    Xb_s = scaler_b.transform(Xb)
    ros_b = RandomOverSampler(random_state=int(rng.integers(0, 1_000_000)))
    Xb_bal, yb_bal = ros_b.fit_resample(Xb_s, yb)

    clf_b = LogisticRegression(max_iter=2000, C=C_SELECTED, solver="lbfgs",
                                random_state=RANDOM_STATE)
    clf_b.fit(Xb_bal, yb_bal)
    coef_samples[b, :] = clf_b.coef_[0]

ci_lo = np.percentile(coef_samples, 2.5, axis=0)
ci_hi = np.percentile(coef_samples, 97.5, axis=0)
coef_mean = coef_samples.mean(axis=0)

boot_df = pd.DataFrame({
    "feature": FEATURES,
    "coef_std_bootstrap_mean": coef_mean,
    "ci_lo_2.5": ci_lo,
    "ci_hi_97.5": ci_hi,
})
boot_df["excludes_zero"] = (boot_df["ci_lo_2.5"] > 0) | (boot_df["ci_hi_97.5"] < 0)
boot_df["abs_mean"] = boot_df["coef_std_bootstrap_mean"].abs()
boot_df = boot_df.sort_values("abs_mean", ascending=False).drop(columns="abs_mean")
boot_df.to_csv(f"{TABLE_DIR}/table_lr_coef_bootstrap.csv", index=False)

print(f"\nBootstrap done, B={B}. Top 12 by |mean standardised coefficient|:")
print(boot_df.head(12).to_string(index=False))
print(f"\n{boot_df['excludes_zero'].sum()} of {len(FEATURES)} 95% bootstrap CIs exclude zero.")
