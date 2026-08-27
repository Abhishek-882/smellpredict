"""
SmellPredict — Phase 5 & 6: Paper Tables, Figures & SHAP Explainability Suite
=============================================================================
Generates:
  - Table I: 50-Repository Dataset Summary (LaTeX + CSV)
  - Table II: Temporal Cross-Validation Results (LaTeX + CSV)
  - Table III: Leave-One-Project-Out (LOPO) Results (LaTeX + CSV)
  - Table IV: Paired Significance Tests (LaTeX + CSV)
  - Figure 1: SHAP Summary Beeswarm Plot (PNG)
  - Figure 2: Empirical Calibration Curve (PNG)
  - SHAP Feature Importance Bar Chart (PNG)
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from pathlib import Path
from loguru import logger

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from smellpredict.explainability.explain import (
    compute_shap_values,
    plot_shap_summary,
    plot_calibration_curve,
    compute_calibration_metrics,
)


def generate_all_artifacts():
    tables_dir = Path("reports/tables")
    figures_dir = Path("reports/figures")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATING EMPIRICAL PAPER TABLES & FIGURES (Phases 5 & 6)")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Table I: Dataset Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1] Generating Table I: Dataset Summary...")
    raw_dir = Path("data/raw")
    repo_rows = []
    for p in sorted(raw_dir.glob("*.parquet")):
        if p.name in ["all_repos_merged.parquet", "sample_500.parquet"]:
            continue
        try:
            df_r = pd.read_parquet(p)
            n_rows = len(df_r)
            n_bugs = int(df_r["future_bug_fix"].sum()) if "future_bug_fix" in df_r.columns else 0
            bug_pct = (n_bugs / max(n_rows, 1)) * 100
            loc_mean = df_r["code_loc"].mean() if "code_loc" in df_r.columns else 0
            smells_mean = df_r["total_smells"].mean() if "total_smells" in df_r.columns else 0

            repo_rows.append({
                "Repository": p.stem,
                "Snapshots": n_rows,
                "Bug-Fix Snapshots": n_bugs,
                "Bug-Fix (%)": round(bug_pct, 1),
                "Mean LOC": round(loc_mean, 1),
                "Mean Smells": round(smells_mean, 1),
            })
        except Exception as e:
            logger.warning(f"Error reading {p.name}: {e}")

    df_table_i = pd.DataFrame(repo_rows)
    df_table_i.to_csv(tables_dir / "table_i_dataset_summary.csv", index=False)
    
    # Save LaTeX format
    with open(tables_dir / "table_i_dataset_summary.tex", "w") as f:
        f.write(df_table_i.to_latex(index=False, caption="Summary of 50 Mined Open-Source Repositories", label="tab:dataset_summary"))
    print(f"  Saved Table I ({len(df_table_i)} repos) to {tables_dir / 'table_i_dataset_summary.csv'}")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Table II & Table III: Model Evaluation Results
    # ─────────────────────────────────────────────────────────────────────────
    exp_file = Path("data/processed/experiment_results.csv")
    if exp_file.exists():
        df_exp = pd.read_csv(exp_file)
        print(f"\n[2] Processing {len(df_exp)} experiment evaluations from {exp_file}...")

        # Table II: Temporal CV
        temp_df = df_exp[df_exp["regime"] == "temporal_cv"]
        if not temp_df.empty:
            table_ii = (
                temp_df.groupby(["model", "feature_group"])
                .agg(
                    Mean_PR_AUC=("pr_auc", "mean"),
                    Std_PR_AUC=("pr_auc", "std"),
                    Mean_ROC_AUC=("roc_auc", "mean"),
                    Mean_F1=("f1", "mean"),
                    Mean_Brier=("brier_score", "mean"),
                    Folds=("pr_auc", "count"),
                )
                .reset_index()
                .round(4)
            )
            table_ii.to_csv(tables_dir / "table_ii_temporal_cv.csv", index=False)
            with open(tables_dir / "table_ii_temporal_cv.tex", "w") as f:
                f.write(table_ii.to_latex(index=False, caption="Temporal Walk-Forward Cross-Validation Results across Feature Groups", label="tab:temporal_cv"))
            print(f"  Saved Table II (Temporal CV) to {tables_dir / 'table_ii_temporal_cv.csv'}")

        # Table III: LOPO
        lopo_df = df_exp[df_exp["regime"] == "lopo"]
        if not lopo_df.empty:
            table_iii = (
                lopo_df.groupby(["model", "feature_group"])
                .agg(
                    Mean_PR_AUC=("pr_auc", "mean"),
                    Std_PR_AUC=("pr_auc", "std"),
                    Mean_ROC_AUC=("roc_auc", "mean"),
                    Mean_F1=("f1", "mean"),
                    Mean_Brier=("brier_score", "mean"),
                    Evaluated_Repos=("pr_auc", "count"),
                )
                .reset_index()
                .round(4)
            )
            table_iii.to_csv(tables_dir / "table_iii_lopo.csv", index=False)
            with open(tables_dir / "table_iii_lopo.tex", "w") as f:
                f.write(table_iii.to_latex(index=False, caption="Leave-One-Project-Out (LOPO) Cross-Validation Results across 50 Repositories", label="tab:lopo"))
            print(f"  Saved Table III (LOPO) to {tables_dir / 'table_iii_lopo.csv'}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Table IV: Significance Tests
    # ─────────────────────────────────────────────────────────────────────────
    sig_file = Path("data/processed/significance_tests.csv")
    if sig_file.exists():
        df_sig = pd.read_csv(sig_file)
        df_sig.to_csv(tables_dir / "table_iv_significance.csv", index=False)
        with open(tables_dir / "table_iv_significance.tex", "w") as f:
            f.write(df_sig.to_latex(index=False, caption="Paired Non-Parametric Bootstrap Significance Tests (FG_C vs. Baseline Groups)", label="tab:significance"))
        print(f"  Saved Table IV (Significance Tests) to {tables_dir / 'table_iv_significance.csv'}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Phase 5: SHAP Explainability & Beeswarm Plot
    # ─────────────────────────────────────────────────────────────────────────
    best_model_path = Path("models/best_model.pkl")
    merged_path = Path("data/raw/all_repos_merged.parquet")
    screened_meta_path = Path("data/processed/screened_features.json")

    if best_model_path.exists() and merged_path.exists():
        print("\n[4] Generating SHAP Explainability Artifacts...")
        with open(best_model_path, "rb") as f:
            calibrated_model = pickle.load(f)

        df_all = pd.read_parquet(merged_path)
        
        # Extract underlying estimator from CalibratedClassifierCV
        underlying_pipeline = calibrated_model.calibrated_classifiers_[0].estimator
        preprocessor = underlying_pipeline.named_steps.get("preprocessor")
        core_model = underlying_pipeline.named_steps.get("model")

        from smellpredict.models.trainer import ExperimentRunner
        runner = ExperimentRunner(df=df_all)
        
        # Check best model config
        exp_file = Path("data/processed/experiment_results.csv")
        best_fg = "B"
        if exp_file.exists():
            df_exp = pd.read_csv(exp_file)
            best_config = (
                df_exp.groupby(["model", "feature_group"])["pr_auc"]
                .mean()
                .reset_index()
                .sort_values("pr_auc", ascending=False)
                .iloc[0]
            )
            best_fg = best_config["feature_group"]

        feature_cols = runner._get_feature_cols(best_fg)
        print(f"  Using {len(feature_cols)} feature columns for Feature Group {best_fg}...")
        X_sample = df_all[feature_cols].fillna(0).sample(n=min(1000, len(df_all)), random_state=42)
        y_sample = df_all.loc[X_sample.index, "future_bug_fix"].values

        X_sample_proc = preprocessor.transform(X_sample.values) if preprocessor else X_sample.values

        # Compute SHAP
        if HAS_SHAP:
            print("  Running TreeExplainer...")
            try:
                explainer = shap.TreeExplainer(core_model)
                shap_vals = explainer.shap_values(X_sample_proc)
                if isinstance(shap_vals, list) and len(shap_vals) == 2:
                    shap_vals = shap_vals[1]

                # Figure 1: SHAP Beeswarm
                plt.figure(figsize=(12, 10))
                shap.summary_plot(
                    shap_vals,
                    features=X_sample.values,
                    feature_names=feature_cols,
                    max_display=20,
                    show=False,
                )
                plt.title("Figure 1: SHAP Summary Beeswarm (Impact on Defect Prediction)", fontsize=14, pad=15)
                plt.tight_layout()
                plt.savefig(figures_dir / "shap_beeswarm.png", dpi=200)
                plt.close()
                print(f"  Saved Figure 1 (SHAP Beeswarm) to {figures_dir / 'shap_beeswarm.png'}")

                # Feature Importance Bar
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    shap_vals,
                    features=X_sample.values,
                    feature_names=feature_cols,
                    plot_type="bar",
                    max_display=20,
                    show=False,
                )
                plt.title("Mean |SHAP Value| (Global Feature Importance)", fontsize=14, pad=15)
                plt.tight_layout()
                plt.savefig(figures_dir / "shap_feature_importance.png", dpi=200)
                plt.close()
                print(f"  Saved Feature Importance Bar to {figures_dir / 'shap_feature_importance.png'}")

            except Exception as e:
                logger.warning(f"TreeExplainer note: {e}")

        # Figure 2: Calibration Curve
        print("  Generating Calibration Curve...")
        y_prob = calibrated_model.predict_proba(X_sample.values)[:, 1]
        cal_metrics = compute_calibration_metrics(y_sample, y_prob)
        ece = cal_metrics.get("ece", 0.0)
        brier = cal_metrics.get("brier_score", 0.0)
        plot_calibration_curve(
            y_sample,
            {"Calibrated Random Forest": y_prob},
            output_path=figures_dir / "calibration_curve.png",
        )
        print(f"  Saved Figure 2 (Calibration Curve, ECE={ece:.4f}, Brier={brier:.4f}) to {figures_dir / 'calibration_curve.png'}")

    print("\n" + "=" * 70)
    print("ALL EMPIRICAL ARTIFACTS GENERATED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    generate_all_artifacts()
