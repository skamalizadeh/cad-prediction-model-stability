"""
00_run_all.py
=============
Runs the whole analysis pipeline in the correct order with one command:

    python 00_run_all.py                 # all steps
    python 00_run_all.py --list          # show the steps and exit
    python 00_run_all.py --from 05       # resume from step 05 (after a failure)
    python 00_run_all.py --from 03 --to 04
    python 00_run_all.py --only 08       # a single step

Each step is an ordinary script that can also be run on its own from this
folder. The steps are executed one after another; if a step fails, the run
stops and prints the command needed to resume.

What it does besides running the scripts:
  * checks that the required packages are installed and warns when their
    versions differ from requirements.txt;
  * saves the software environment to outputs/environment_report.txt;
  * writes the complete console output to logs/run_all_<timestamp>.log.

Results are written by the scripts themselves to tables/, figures/,
models/, data/ and outputs/CAD_prediction_results_tables.xlsx.
The input data file must be present at data/CAD_raw.csv.
"""
import argparse
import importlib.metadata as md
import importlib.util
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (id, script, description) in execution order.
STEPS = [
    ("01", "01_data_preprocessing.py",               "Data preprocessing, baseline table, leakage screen"),
    ("02", "02_split_and_feature_selection.py",      "Train/test split, over-sampling, Boruta feature selection"),
    ("03", "03_train_models.py",                     "Hyperparameter tuning and training of all models"),
    ("04", "04_evaluate_models.py",                  "Evaluation: AUC, DeLong tests, calibration, threshold metrics"),
    ("05", "05_shap_analysis.py",                    "SHAP analysis"),
    ("06", "06_workflow_diagram_and_final_tables.py", "Workflow figure and final tables"),
    ("07", "07_boruta_sensitivity_analysis.py",      "Boruta-7 sensitivity analysis"),
    ("08", "08_repeated_holdout_robustness.py",      "Repeated holdout (full53, boruta7, summary)"),
    ("08a", "08a_paired_tests.py",                   "Paired tests on repeated-holdout results"),
    ("09", "09_decision_curve_and_lr_bootstrap.py",  "Decision curve analysis and LR coefficient bootstrap"),
    ("10", "10_build_results_workbook.py",           "Excel workbook of all result tables"),
    ("11", "11_nested_cv.py",                        "Nested repeated cross-validation (three class-balancing modes)"),
    ("12", "12_nested_cv_summary.py",                "Summary of nested CV: corrected tests, Holm, TOST, ranks, calibration"),
    ("13", "13_clinical_reference_models.py",        "Diamond-Forrester / CAD Consortium and nested clinical models"),
    ("14", "14_boruta_nested.py",                    "Boruta inside the outer folds: selection frequency, reduced models"),
    ("15", "15_shap_lr_vs_rf.py",                    "SHAP for logistic regression and agreement with random forest"),
    ("16", "16_regularization_and_sample_size.py",   "Riley sample size; elastic-net, PCA and restricted-RF models"),
    ("16a", "16a_performance_by_sex.py",             "Discrimination of the leading models by sex"),
    ("17", "17_results_tables_figures.py",          "Manuscript tables (LaTeX fragments) and figures for the Results"),
    ("18", "18_build_supplementary_workbook.py",     "Supplementary workbook with all result tables"),
]

# pip distribution name -> import name (only where they differ or are needed for the check)
REQUIRED = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "scikit-learn": "sklearn",
    "imbalanced-learn": "imblearn", "xgboost": "xgboost", "catboost": "catboost",
    "pygam": "pygam", "shap": "shap", "Boruta": "boruta", "statsmodels": "statsmodels",
    "matplotlib": "matplotlib", "joblib": "joblib", "openpyxl": "openpyxl",
}


def pinned_versions():
    pins = {}
    req = ROOT / "requirements.txt"
    if req.exists():
        for line in req.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "==" in line:
                name, ver = line.split("==", 1)
                pins[name.strip().lower()] = ver.strip()
    return pins


def check_environment():
    """Stop if a package is missing; warn if a version differs from requirements.txt."""
    pins = pinned_versions()
    missing, differing = [], []
    for dist, module in REQUIRED.items():
        if importlib.util.find_spec(module) is None:
            missing.append(dist)
            continue
        try:
            installed = md.version(dist)
        except md.PackageNotFoundError:
            continue
        want = pins.get(dist.lower())
        if want and installed != want:
            differing.append((dist, installed, want))
    if missing:
        print("ERROR: missing packages: " + ", ".join(missing))
        print("Install them with:  python -m pip install -r requirements.txt")
        sys.exit(1)
    if differing:
        print("WARNING: installed versions differ from requirements.txt "
              "(numerical results may differ slightly):")
        for dist, installed, want in differing:
            print(f"   {dist}: installed {installed}, expected {want}")
        print()


def save_environment_report():
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    lines = [f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"Python: {sys.version.split()[0]} ({platform.python_implementation()})",
             f"Platform: {platform.platform()}",
             f"Processor: {platform.processor() or 'unknown'}",
             f"CPU cores: {os.cpu_count()}", "", "Key packages:"]
    for dist in REQUIRED:
        try:
            lines.append(f"  {dist}=={md.version(dist)}")
        except md.PackageNotFoundError:
            lines.append(f"  {dist}: not found")
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                text=True, timeout=120).stdout.strip()
        if freeze:
            lines += ["", "Full environment (pip freeze):", freeze]
    except Exception:
        pass
    (out / "environment_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_step(step_id, script, description, log):
    """Run one script, echo its output live and copy it to the log file."""
    header = f"\n{'=' * 78}\nSTEP {step_id}: {script}\n{description}\n{'=' * 78}"
    print(header)
    log.write(header + "\n")
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, script], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in proc.stdout:
        print(line, end="")
        log.write(line)
    proc.wait()
    return proc.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="Run the full analysis pipeline.")
    ap.add_argument("--list", action="store_true", help="show the steps and exit")
    ap.add_argument("--from", dest="first", default=None, help="first step id to run (e.g. 05)")
    ap.add_argument("--to", dest="last", default=None, help="last step id to run (e.g. 08a)")
    ap.add_argument("--only", default=None, help="run a single step id")
    args = ap.parse_args()

    ids = [s[0] for s in STEPS]
    if args.list:
        for sid, script, desc in STEPS:
            print(f"  {sid:>3}  {script:<42} {desc}")
        return 0
    for value in (args.first, args.last, args.only):
        if value is not None and value not in ids:
            sys.exit(f"Unknown step '{value}'. Valid ids: {', '.join(ids)}")

    if args.only:
        selected = [s for s in STEPS if s[0] == args.only]
    else:
        i0 = ids.index(args.first) if args.first else 0
        i1 = ids.index(args.last) if args.last else len(STEPS) - 1
        selected = STEPS[i0:i1 + 1]

    for folder in ("data", "tables", "models", "figures", "outputs", "logs"):
        (ROOT / folder).mkdir(exist_ok=True)
    for _, script, _ in selected:
        if not (ROOT / script).exists():
            sys.exit(f"Script not found: {script}")
    if selected[0][0] == "01" and not (ROOT / "data" / "CAD_raw.csv").exists():
        sys.exit("Input data not found: copy the raw dataset to data/CAD_raw.csv and run again.")

    check_environment()
    save_environment_report()

    log_path = ROOT / "logs" / f"run_all_{datetime.now():%Y%m%d_%H%M%S}.log"
    results, t_start = [], time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        for sid, script, desc in selected:
            code, secs = run_step(sid, script, desc, log)
            results.append((sid, script, code, secs))
            if code != 0:
                msg = (f"\nSTEP {sid} FAILED (exit code {code}). The run was stopped.\n"
                       f"Fix the problem, then resume with:  python 00_run_all.py --from {sid}")
                print(msg)
                log.write(msg + "\n")
                break

        lines = ["", "=" * 78, "SUMMARY", "=" * 78]
        for sid, script, code, secs in results:
            lines.append(f"  {sid:>3}  {script:<42} {'OK' if code == 0 else 'FAILED':<7} {secs:7.1f} s")
        lines.append(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")
        ok = all(r[2] == 0 for r in results)
        if ok:
            lines += ["", "All steps finished successfully.",
                      "Results:  tables/   figures/   outputs/CAD_prediction_results_tables.xlsx",
                      "Environment:  outputs/environment_report.txt",
                      f"Full log:  {log_path.relative_to(ROOT)}"]
        text = "\n".join(lines)
        print(text)
        log.write(text + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
