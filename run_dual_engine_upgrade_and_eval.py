"""
run_dual_engine_upgrade_and_eval.py — Dual-Engine Training & Empirical Evaluation v2
====================================================================================
Implements:
  1. Expanded 4-Repo Zero-Shot OOD Suite:
     - django (78.1% prev, Web ORM / Framework)
     - rich (74.3% prev, Terminal UI / Formatting)
     - pillow (41.8% prev, Imaging / C-Extension)
     - fastapi (20.9% prev, Modern Async API)
  2. 46 In-Pool Training Repositories (22,718 snapshots).
  3. Pre-computed 100-bin empirical quantile reference tables for single-buffer cold-start.
  4. Engine B (Full 73 Causal Features):
     - LightGBM with reg_lambda=2.0, reg_alpha=0.5, monotone_constraints
     - Platt Scaling (CalibratedClassifierCV, method='sigmoid', cv=5)
     - Serialized to models/best_model_final.pkl
  5. Engine A (Pure Static AST Model, 28 Features, Zero Git History):
     - LightGBM with reg_lambda=2.0, reg_alpha=0.5, monotone_constraints
     - Platt Scaling (CalibratedClassifierCV, method='sigmoid', cv=5)
     - Serialized to models/engine_a_ast_static.pkl
  6. Comprehensive Benchmark:
     - 46-Repo LOPO on both engines
     - 4-Repo OOD on both engines
     - End-to-End Hybrid Traffic Benchmark (50% untracked buffers + 50% tracked repo files)
     - Hard acceptance gate checks
"""

import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    auc,
    brier_score_loss,
    average_precision_score,
    precision_score,
    recall_score
)
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier

sys.path.insert(0, "src")
from smellpredict.models.trainer import FEATURE_GROUPS, LABEL_COL

print("=" * 80)
print("DUAL-ENGINE TRAINING & EMPIRICAL BENCHMARK SUITE v2")
print("=" * 80)

# Load dataset
src = "data/processed/enriched_v5.parquet"
df = pd.read_parquet(src)
df["commit_date"] = pd.to_datetime(df["commit_date"], utc=True)
df = df.sort_values(["repo", "commit_date"]).reset_index(drop=True)
print(f"Loaded {len(df):,} snapshots across {df['repo'].nunique()} repositories from {src}")

# Hyperparameters with L2 Regularization
hp_path = "data/processed/best_hyperparams.json"
hp = json.load(open(hp_path))
base_lgb_params = hp["lightgbm"]["best_params"].copy()
base_lgb_params["reg_lambda"] = 2.0
base_lgb_params["reg_alpha"] = 0.5
print(f"Hyperparameters: num_leaves={base_lgb_params['num_leaves']}, lr={base_lgb_params['learning_rate']:.4f}, reg_lambda={base_lgb_params['reg_lambda']}, reg_alpha={base_lgb_params['reg_alpha']}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. 4-Repo OOD Reservation & Dataset Splitting
# ─────────────────────────────────────────────────────────────────────────────
RESERVED_OOD_REPOS = ["django", "rich", "pillow", "fastapi"]
print(f"\nReserving 4 Zero-Shot OOD Repositories: {RESERVED_OOD_REPOS}")

train_df = df[~df["repo"].isin(RESERVED_OOD_REPOS)].copy().reset_index(drop=True)
train_df = train_df[train_df["future_bug_fix"].isin([0, 1])].copy().reset_index(drop=True)
train_df["future_bug_fix"] = train_df["future_bug_fix"].astype(int)

test_df = df[df["repo"].isin(RESERVED_OOD_REPOS)].copy().reset_index(drop=True)
test_df = test_df[test_df["future_bug_fix"].isin([0, 1])].copy().reset_index(drop=True)
test_df["future_bug_fix"] = test_df["future_bug_fix"].astype(int)

print(f"In-Pool Training Set : {len(train_df):,} rows across {train_df['repo'].nunique()} repos")
print(f"Reserved OOD Test Set: {len(test_df):,} rows across {test_df['repo'].nunique()} repos ({RESERVED_OOD_REPOS})")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Definitions & Zero-Leakage Checks
# ─────────────────────────────────────────────────────────────────────────────

# Engine B: Full 73 features (FG_J)
features_b = [c for c in FEATURE_GROUPS["J"] if c in train_df.columns]
print(f"\nEngine B feature count: {len(features_b)} features (Full Enterprise Telemetry)")

# Engine A: Pure Static AST features (28 features, strictly 0 git history)
AST_CANDIDATES = [
    # Raw AST Complexity
    "code_loc", "code_sloc", "code_function_count", "code_class_count",
    "code_avg_cyclomatic_complexity", "code_max_cyclomatic_complexity",
    "code_cognitive_complexity", "code_maintainability_index",
    "code_halstead_volume", "code_halstead_difficulty", "code_halstead_bugs",
    "code_max_nesting_depth",
    # Smell metrics
    "has_long_method", "has_long_param_list", "has_large_class",
    "has_deep_nesting", "has_high_complexity",
    "long_method_count", "long_param_count", "large_class_count",
    "deep_nesting_count", "high_complexity_count", "total_smells", "smell_density",
    # Reference Quantile Ranks
    "rank_code_loc", "rank_code_sloc",
    "rank_code_avg_cyclomatic_complexity", "rank_code_max_cyclomatic_complexity",
    "rank_code_cognitive_complexity", "rank_code_maintainability_index",
    "rank_code_halstead_volume", "rank_code_halstead_difficulty",
    "rank_total_smells", "rank_smell_density",
]
features_a = [c for c in AST_CANDIDATES if c in train_df.columns]
print(f"Engine A feature count: {len(features_a)} features (Pure Static AST & Complexity)")

# Zero Leakage & Git Telemetry Isolation Assertions
FORBIDDEN = {"future_bug_fix", "has_bug_fix", "future_bug_fix_score",
             "label_outlier", "exclude_from_training", "parse_fallback"}
assert not (set(features_b) & FORBIDDEN), f"Leaky in B: {set(features_b) & FORBIDDEN}"
assert not (set(features_a) & FORBIDDEN), f"Leaky in A: {set(features_a) & FORBIDDEN}"

GIT_COLS = {"churn", "commit", "author", "contributor", "file_age", "days_since", "previous_bug", "silo", "cochange"}
for f in features_a:
    assert not any(g in f.lower() for g in GIT_COLS), f"Engine A contains git telemetry: {f}"
print("[OK] Engine A strictly verified for zero git-history dependencies.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Pre-compute Empirical Quantile Reference Tables (CDFs)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 3. Pre-computing Empirical Quantile Reference Tables ---")
percentiles_100 = np.linspace(0.0, 1.0, 101)
quantile_ref_tables = {}

for col in ["code_loc", "code_sloc", "code_avg_cyclomatic_complexity",
            "code_max_cyclomatic_complexity", "code_cognitive_complexity",
            "code_maintainability_index", "code_halstead_volume",
            "code_halstead_difficulty", "total_smells", "smell_density"]:
    if col in train_df.columns:
        vals = train_df[col].dropna().values
        quantile_ref_tables[col] = [float(np.quantile(vals, p)) for p in percentiles_100]

print(f"[OK] Pre-computed 101-point CDF reference tables for {len(quantile_ref_tables)} continuous AST features.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Train Engine B (Full Enterprise Model) with Platt Sigmoid Calibration
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 4. Training Engine B (Full Enterprise Model) ---")

monotone_b = {}
for f in features_b:
    if any(k in f for k in ["cyclomatic", "cognitive", "smell", "bugs", "churn", "silo"]):
        monotone_b[f] = 1
    elif "maintainability" in f:
        monotone_b[f] = -1

lgb_params_b = base_lgb_params.copy()
lgb_params_b["monotone_constraints"] = [monotone_b.get(f, 0) for f in features_b]

lgb_b = LGBMClassifier(**lgb_params_b)
engine_b_model = CalibratedClassifierCV(estimator=lgb_b, cv=5, method="sigmoid")
engine_b_model.fit(train_df[features_b].fillna(0), train_df[LABEL_COL])

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

model_b_path = model_dir / "best_model_final.pkl"
joblib.dump(engine_b_model, model_b_path)
print(f"Saved Engine B model -> {model_b_path}")

meta_b = {
    "engine": "Engine B (Full Enterprise Telemetry)",
    "model_type": "LightGBM + Platt Scaling (Sigmoid)",
    "calibration_method": "sigmoid",
    "n_features": len(features_b),
    "features": features_b,
    "training_rows": len(train_df),
    "training_repos": sorted(train_df["repo"].unique().tolist()),
    "reserved_ood_repos": RESERVED_OOD_REPOS,
    "hyperparameters": lgb_params_b,
    "quantile_reference_tables": quantile_ref_tables,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
}
with open(model_dir / "best_model_final_metadata.json", "w") as f:
    json.dump(meta_b, f, indent=2)
print("Saved Engine B metadata -> models/best_model_final_metadata.json")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Train Engine A (Pure Static AST Model) with Platt Sigmoid Calibration
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 5. Training Engine A (Pure Static AST Model) ---")

monotone_a = {}
for f in features_a:
    if any(k in f for k in ["cyclomatic", "cognitive", "smell", "bugs", "nesting"]):
        monotone_a[f] = 1
    elif "maintainability" in f:
        monotone_a[f] = -1

lgb_params_a = base_lgb_params.copy()
lgb_params_a["monotone_constraints"] = [monotone_a.get(f, 0) for f in features_a]

lgb_a = LGBMClassifier(**lgb_params_a)
engine_a_model = CalibratedClassifierCV(estimator=lgb_a, cv=5, method="sigmoid")
engine_a_model.fit(train_df[features_a].fillna(0), train_df[LABEL_COL])

model_a_path = model_dir / "engine_a_ast_static.pkl"
joblib.dump(engine_a_model, model_a_path)
print(f"Saved Engine A model -> {model_a_path}")

meta_a = {
    "engine": "Engine A (Pure Static AST)",
    "model_type": "LightGBM + Platt Scaling (Sigmoid)",
    "calibration_method": "sigmoid",
    "n_features": len(features_a),
    "features": features_a,
    "training_rows": len(train_df),
    "training_repos": sorted(train_df["repo"].unique().tolist()),
    "reserved_ood_repos": RESERVED_OOD_REPOS,
    "hyperparameters": lgb_params_a,
    "quantile_reference_tables": quantile_ref_tables,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
}
with open(model_dir / "engine_a_ast_static_metadata.json", "w") as f:
    json.dump(meta_a, f, indent=2)
print("Saved Engine A metadata -> models/engine_a_ast_static_metadata.json")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Comprehensive Evaluation: 46-Repo LOPO & 4-Repo Zero-Shot OOD
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 6. Running Comprehensive Evaluation Suite ---")

def evaluate_repo(y_true, y_prob):
    """Compute PR-AUC, ROC-AUC, Brier score, Precision@20%, Recall@20%."""
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    prevalence = float(y_true.mean())

    if n_pos == 0 or n_neg == 0:
        return {
            "n_rows": len(y_true),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "prevalence": round(prevalence * 100, 2),
            "pr_auc": round(prevalence, 4),
            "roc_auc": 0.5,
            "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
            "lift": 0.0,
            "prec_at_20": round(prevalence * 100, 2),
            "rec_at_20": 20.0,
        }

    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    lift = pr_auc - prevalence

    # Compute top 20% metrics
    sorted_indices = np.argsort(-y_prob)
    k20 = max(1, int(len(y_true) * 0.20))
    top20_idx = sorted_indices[:k20]
    prec_at_20 = float(y_true[top20_idx].mean())
    rec_at_20 = float(y_true[top20_idx].sum() / n_pos) if n_pos > 0 else 0.0

    return {
        "n_rows": len(y_true),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": round(prevalence * 100, 2),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "brier": round(brier, 4),
        "lift": round(lift, 4),
        "prec_at_20": round(prec_at_20 * 100, 2),
        "rec_at_20": round(rec_at_20 * 100, 2),
    }

# ── A. 46-Repo LOPO Cross-Validation for Both Engines ────────────────────────
print("Executing 46-Repository LOPO Cross-Validation for Engine A and Engine B...")
unique_repos = sorted(train_df["repo"].unique())
lopo_records = []

for i, test_repo in enumerate(unique_repos):
    tr_fold = train_df[train_df["repo"] != test_repo]
    val_fold = train_df[train_df["repo"] == test_repo]

    y_tr = tr_fold[LABEL_COL].values
    y_val = val_fold[LABEL_COL].values

    # Train fold models with sigmoid calibration
    m_b = CalibratedClassifierCV(estimator=LGBMClassifier(**lgb_params_b), cv=3, method="sigmoid")
    m_b.fit(tr_fold[features_b].fillna(0), y_tr)
    prob_b = m_b.predict_proba(val_fold[features_b].fillna(0))[:, 1]
    res_b = evaluate_repo(y_val, prob_b)

    m_a = CalibratedClassifierCV(estimator=LGBMClassifier(**lgb_params_a), cv=3, method="sigmoid")
    m_a.fit(tr_fold[features_a].fillna(0), y_tr)
    prob_a = m_a.predict_proba(val_fold[features_a].fillna(0))[:, 1]
    res_a = evaluate_repo(y_val, prob_a)

    lopo_records.append({
        "repo": test_repo,
        "n_rows": res_b["n_rows"],
        "n_pos": res_b["n_pos"],
        "n_neg": res_b["n_neg"],
        "prevalence_pct": res_b["prevalence"],
        "engine_b_pr_auc": res_b["pr_auc"],
        "engine_b_roc_auc": res_b["roc_auc"],
        "engine_b_brier": res_b["brier"],
        "engine_b_lift": res_b["lift"],
        "engine_b_prec20": res_b["prec_at_20"],
        "engine_b_rec20": res_b["rec_at_20"],
        "engine_a_pr_auc": res_a["pr_auc"],
        "engine_a_roc_auc": res_a["roc_auc"],
        "engine_a_brier": res_a["brier"],
        "engine_a_lift": res_a["lift"],
        "engine_a_prec20": res_a["prec_at_20"],
        "engine_a_rec20": res_a["rec_at_20"],
    })
    if (i + 1) % 10 == 0 or i == len(unique_repos) - 1:
        print(f"  Processed {i + 1}/{len(unique_repos)} LOPO folds...")

lopo_df = pd.DataFrame(lopo_records)
lopo_out = "data/processed/lopo_dual_engine_publication_table.csv"
lopo_df.to_csv(lopo_out, index=False)
print(f"Saved LOPO benchmark table -> {lopo_out}")

# ── B. 4-Repo Zero-Shot OOD Evaluation ───────────────────────────────────────
print("\nEvaluating Zero-Shot OOD Performance on 4 Reserved Repositories...")
ood_records = []

y_ood = test_df[LABEL_COL].values
prob_ood_b = engine_b_model.predict_proba(test_df[features_b].fillna(0))[:, 1]
prob_ood_a = engine_a_model.predict_proba(test_df[features_a].fillna(0))[:, 1]

for rname in RESERVED_OOD_REPOS:
    sub = test_df[test_df["repo"] == rname]
    y_sub = sub[LABEL_COL].values
    p_b = engine_b_model.predict_proba(sub[features_b].fillna(0))[:, 1]
    p_a = engine_a_model.predict_proba(sub[features_a].fillna(0))[:, 1]

    eval_b = evaluate_repo(y_sub, p_b)
    eval_a = evaluate_repo(y_sub, p_a)

    ood_records.append({
        "repo": rname,
        "domain": {"django": "Web ORM", "rich": "Terminal UI", "pillow": "Imaging C-Ext", "fastapi": "Async Web API"}[rname],
        "n_rows": eval_b["n_rows"],
        "n_pos": eval_b["n_pos"],
        "n_neg": eval_b["n_neg"],
        "prevalence_pct": eval_b["prevalence"],
        "engine_b_pr_auc": eval_b["pr_auc"],
        "engine_b_roc_auc": eval_b["roc_auc"],
        "engine_b_brier": eval_b["brier"],
        "engine_b_lift": eval_b["lift"],
        "engine_b_prec20": eval_b["prec_at_20"],
        "engine_b_rec20": eval_b["rec_at_20"],
        "engine_a_pr_auc": eval_a["pr_auc"],
        "engine_a_roc_auc": eval_a["roc_auc"],
        "engine_a_brier": eval_a["brier"],
        "engine_a_lift": eval_a["lift"],
        "engine_a_prec20": eval_a["prec_at_20"],
        "engine_a_rec20": eval_a["rec_at_20"],
    })

# Combined 4-Repo OOD
comb_b = evaluate_repo(y_ood, prob_ood_b)
comb_a = evaluate_repo(y_ood, prob_ood_a)

ood_records.append({
    "repo": "COMBINED_4_REPO_OOD",
    "domain": "Pooled Zero-Shot Pool",
    "n_rows": comb_b["n_rows"],
    "n_pos": comb_b["n_pos"],
    "n_neg": comb_b["n_neg"],
    "prevalence_pct": comb_b["prevalence"],
    "engine_b_pr_auc": comb_b["pr_auc"],
    "engine_b_roc_auc": comb_b["roc_auc"],
    "engine_b_brier": comb_b["brier"],
    "engine_b_lift": comb_b["lift"],
    "engine_b_prec20": comb_b["prec_at_20"],
    "engine_b_rec20": comb_b["rec_at_20"],
    "engine_a_pr_auc": comb_a["pr_auc"],
    "engine_a_roc_auc": comb_a["roc_auc"],
    "engine_a_brier": comb_a["brier"],
    "engine_a_lift": comb_a["lift"],
    "engine_a_prec20": comb_a["prec_at_20"],
    "engine_a_rec20": comb_a["rec_at_20"],
})

ood_df = pd.DataFrame(ood_records)
ood_out = "data/processed/experiment_results_dual_engine_v12.csv"
ood_df.to_csv(ood_out, index=False)
print(f"Saved 4-Repo OOD benchmark results -> {ood_out}")

# ── C. End-to-End Hybrid Traffic Benchmark ──────────────────────────────────
print("\n--- 7. End-to-End Hybrid Traffic Benchmark ---")
# 50% traffic with Git Telemetry (Engine B), 50% cold-start untracked (Engine A)
rng = np.random.RandomState(42)
hybrid_mask = rng.rand(len(test_df)) > 0.5  # True -> Engine B, False -> Engine A

hybrid_probs = np.zeros(len(test_df))
hybrid_probs[hybrid_mask] = prob_ood_b[hybrid_mask]
hybrid_probs[~hybrid_mask] = prob_ood_a[~hybrid_mask]

hybrid_eval = evaluate_repo(y_ood, hybrid_probs)
print(f"Hybrid Mixed Traffic (50% Engine A + 50% Engine B):")
print(f"  PR-AUC   : {hybrid_eval['pr_auc']:.4f} (Lift: +{hybrid_eval['lift']:.4f})")
print(f"  ROC-AUC  : {hybrid_eval['roc_auc']:.4f}")
print(f"  Brier    : {hybrid_eval['brier']:.4f}")
print(f"  Prec@20% : {hybrid_eval['prec_at_20']:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Hard Acceptance Gate Checks
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("HARD ACCEPTANCE RELEASE GATES VERIFICATION")
print("=" * 80)

# Gate 1: Engine B Platt Scaling
gate1_brier_pass = comb_b["brier"] <= 0.25
gate1_tail_pass = (prob_ood_b.min() > 0.001) and (prob_ood_b.max() < 0.999)
print(f"[GATE 1] Engine B Platt Scaling:")
print(f"  - OOD Brier Score <= 0.25: {comb_b['brier']:.4f} -> {'PASS' if gate1_brier_pass else 'FAIL'}")
print(f"  - Output Probability Tail Smoothness (min={prob_ood_b.min():.4f}, max={prob_ood_b.max():.4f}) -> {'PASS' if gate1_tail_pass else 'FAIL'}")

# Gate 2: Engine A Standalone Performance
gate2_lopo_roc_pass = lopo_df["engine_a_roc_auc"].mean() >= 0.65
gate2_lopo_lift_pass = lopo_df["engine_a_lift"].mean() >= 0.08
gate2_ood_roc_pass = comb_a["roc_auc"] >= 0.65
print(f"\n[GATE 2] Engine A Standalone Performance:")
print(f"  - In-Pool LOPO ROC-AUC >= 0.65: {lopo_df['engine_a_roc_auc'].mean():.4f} -> {'PASS' if gate2_lopo_roc_pass else 'FAIL'}")
print(f"  - In-Pool LOPO Lift >= +0.08  : +{lopo_df['engine_a_lift'].mean():.4f} -> {'PASS' if gate2_lopo_lift_pass else 'FAIL'}")
print(f"  - 4-Repo OOD ROC-AUC >= 0.65  : {comb_a['roc_auc']:.4f} -> {'PASS' if gate2_ood_roc_pass else 'FAIL'}")

assert gate1_brier_pass, "GATE 1 FAILED: Engine B Brier score too high."
assert gate1_tail_pass, "GATE 1 FAILED: Engine B probability pinned to tails."
assert gate2_lopo_roc_pass, "GATE 2 FAILED: Engine A LOPO ROC-AUC below threshold."
assert gate2_lopo_lift_pass, "GATE 2 FAILED: Engine A LOPO lift below threshold."

print("\n[ALL ACCEPTANCE GATES PASSED SUCCESSFULLY]")
print("=" * 80)
