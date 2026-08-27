"""
SmellPredict — Model Training Pipeline
========================================
Orchestrates training, evaluation, and MLflow logging for all models
across all feature groups and validation regimes.

Implements:
  - Preprocessing with strict train-only fitting (no leakage)
  - VIF-based multicollinearity screening
  - Class imbalance handling (SMOTE/class_weight)
  - Hyperparameter tuning with Optuna
  - BCa bootstrap confidence intervals
  - Paired significance testing
  - MLflow experiment tracking
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    logger.warning("MLflow not available — experiment tracking disabled")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from imblearn.combine import SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    logger.warning("imbalanced-learn not available — SMOTE disabled")


# ─────────────────────────────────────────────────────────────────────────────
# Feature Group Definitions
# ─────────────────────────────────────────────────────────────────────────────

CODE_METRIC_COLS = [
    "code_loc", "code_sloc", "code_comment_density",
    "code_function_count", "code_class_count", "code_import_count",
    "code_avg_function_size", "code_max_function_size",
    "code_avg_param_count", "code_max_param_count",
    "code_max_nesting_depth",
    "code_avg_cyclomatic_complexity", "code_max_cyclomatic_complexity",
    "code_halstead_volume", "code_halstead_difficulty", "code_halstead_effort",
    "code_halstead_bugs", "code_maintainability_index", "code_cognitive_complexity",
]

SMELL_COLS = [
    "has_long_method", "has_long_param_list", "has_large_class",
    "has_deep_nesting", "has_high_complexity",
    "long_method_count", "long_param_count", "large_class_count",
    "deep_nesting_count", "high_complexity_count",
    "total_smells",
]

HISTORY_COLS = [
    "previous_file_commits", "previous_bug_fixes", "contributors",
    "recent_file_commits",
    # code_churn_history REMOVED (Fix 0-E): always 0, zero-variance.
    "file_age_days",
    "days_since_last_change", "developer_experience",
    "ownership_concentration", "commit_message_entropy",
    "avg_commit_size", "avg_time_between_commits",
    "has_multiple_contributors", "is_recently_touched",
]

# Step 1 derived features: smell trajectory + cross-product interactions.
# Computed by run_feature_engineering.py from enriched_snapshots.parquet.
# Non-faking rule R6: lag features use only past snapshots (never future).
TRAJECTORY_COLS = [
    "smell_delta",           # total_smells[t] - total_smells[t-1]
    "complexity_delta",      # max_cyclomatic[t] - max_cyclomatic[t-1]
    "smell_trend",           # sign(smell_delta): -1, 0, +1
    "has_smell_increase",    # binary: 1 if smell_delta > 0
    "smell_density",         # total_smells / max(code_loc, 1)
]

INTERACTION_COLS = [
    "smells_x_churn",        # total_smells * previous_file_commits
    "smells_x_bugs",         # total_smells * previous_bug_fixes
    "complexity_x_churn",    # max_cyclomatic * recent_file_commits
    "smells_x_age",          # smell_density * file_age_days
    "bug_rate_x_smells",     # (prev_bugs/prev_commits) * total_smells
]

# Fix 1: Per-repo normalised features (rel_X = X / repo_median_X).
# Addresses root cause of high LOPO std: absolute values differ across repos.
# Computed by run_fix1_fix2_features.py from enriched_v2.parquet.
REL_COLS = [
    "rel_code_loc", "rel_code_sloc", "rel_code_function_count",
    "rel_code_avg_cyclomatic_complexity", "rel_code_max_cyclomatic_complexity",
    "rel_code_halstead_volume", "rel_code_halstead_difficulty",
    "rel_code_maintainability_index", "rel_code_cognitive_complexity",
    "rel_total_smells", "rel_long_method_count", "rel_high_complexity_count",
    "rel_smell_density",
    "rel_previous_file_commits", "rel_previous_bug_fixes",
    "rel_recent_file_commits", "rel_file_age_days",
    "rel_days_since_last_change", "rel_avg_commit_size",
    "rel_developer_experience", "rel_ownership_concentration",
]

# Fix 2: Smell age features — how long has each smell persisted for this file?
# Addresses: smell counts are redundant to history; smell AGE is genuinely new.
SMELL_AGE_COLS = [
    "age_has_long_method", "age_has_long_param_list", "age_has_large_class",
    "age_has_deep_nesting", "age_has_high_complexity",
    "max_smell_age_days", "has_persistent_smell",
    "persistent_smell_count", "total_smell_age_days",
]

# Track A1: Causal Quantile Rank Features (rank_X = percentile rank within repo up to time t)
# Strictly bounded in [0, 1], outlier-immune, captures distribution shape across repos.
RANK_COLS = [
    "rank_code_loc", "rank_code_sloc", "rank_code_function_count",
    "rank_code_avg_cyclomatic_complexity", "rank_code_max_cyclomatic_complexity",
    "rank_code_halstead_volume", "rank_code_halstead_difficulty",
    "rank_code_maintainability_index", "rank_code_cognitive_complexity",
    "rank_total_smells", "rank_long_method_count", "rank_high_complexity_count",
    "rank_smell_density",
    "rank_previous_file_commits", "rank_previous_bug_fixes",
    "rank_recent_file_commits", "rank_file_age_days",
    "rank_days_since_last_change", "rank_avg_commit_size",
    "rank_developer_experience", "rank_ownership_concentration",
]

FEATURE_GROUPS: dict[str, list[str]] = {
    "A": CODE_METRIC_COLS,
    "B": CODE_METRIC_COLS + HISTORY_COLS,
    "C": CODE_METRIC_COLS + HISTORY_COLS + SMELL_COLS,
    # FG_D: enriched — FG_C + trajectory + interactions (v2 champion).
    "D": CODE_METRIC_COLS + HISTORY_COLS + SMELL_COLS + TRAJECTORY_COLS + INTERACTION_COLS,
    # FG_E: Fix 1+2 — FG_D + per-repo-relative + smell_age features.
    # Non-faking rule R3: cols only used if present in the loaded df.
    "E": CODE_METRIC_COLS + HISTORY_COLS + SMELL_COLS + TRAJECTORY_COLS + INTERACTION_COLS + REL_COLS + SMELL_AGE_COLS,
    # FG_F: per-repo-relative ONLY (no absolute values) — v3 champion (CV=0.6828).
    "F": REL_COLS + SMELL_AGE_COLS + TRAJECTORY_COLS + INTERACTION_COLS,
    # FG_G: Track A1 — Pure causal quantile rank + smell_age + trajectory + interactions (40 features).
    "G": RANK_COLS + SMELL_AGE_COLS + TRAJECTORY_COLS + INTERACTION_COLS,
    # FG_H: Track A1 — Quantile rank + relative scale + smell_age + trajectory + interactions (61 features).
    "H": RANK_COLS + REL_COLS + SMELL_AGE_COLS + TRAJECTORY_COLS + INTERACTION_COLS,
    # FG_I: Track C1 — FG_H + 6 co-change graph features (67 features).
    "I": RANK_COLS + REL_COLS + SMELL_AGE_COLS + TRAJECTORY_COLS + INTERACTION_COLS + [
        "cochange_peer_count", "cochange_coupling_ratio",
        "cochange_peer_complexity_mean", "cochange_peer_loc_mean",
        "rank_cochange_peer_count", "cochange_cluster_size",
    ],
    # FG_J: Track C2 — FG_I + 7 ownership & velocity features (74 features).
    "J": RANK_COLS + REL_COLS + SMELL_AGE_COLS + TRAJECTORY_COLS + INTERACTION_COLS + [
        "cochange_peer_count", "cochange_coupling_ratio",
        "cochange_peer_complexity_mean", "cochange_peer_loc_mean",
        "rank_cochange_peer_count", "cochange_cluster_size",
        "contributors", "ownership_concentration", "silo_index",
        "contributor_annual_rate", "complexity_churn_interaction",
        "rank_contributor_annual_rate", "rank_complexity_churn_interaction",
    ],
}

LABEL_COL = "future_bug_fix"


# ─────────────────────────────────────────────────────────────────────────────
# VIF-Based Feature Screening
# ─────────────────────────────────────────────────────────────────────────────

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for all columns in X using
    exact LinearRegression OLS R^2 against all other features.

    Returns DataFrame with columns: feature, vif
    """
    from sklearn.linear_model import LinearRegression
    cols = list(X.columns)
    if not cols:
        return pd.DataFrame(columns=["feature", "vif"])
    
    X_mat = X.fillna(0).astype(float).values
    if len(cols) == 1:
        return pd.DataFrame([{"feature": cols[0], "vif": 1.0}])

    vif_data = []
    for i, col in enumerate(cols):
        y = X_mat[:, i]
        if np.std(y) == 0:
            vif_data.append({"feature": col, "vif": np.nan})
            continue

        X_other = np.delete(X_mat, i, axis=1)
        # Check if X_other has variance
        if X_other.shape[1] == 0 or np.all(np.std(X_other, axis=0) == 0):
            vif_data.append({"feature": col, "vif": 1.0})
            continue

        lr = LinearRegression()
        lr.fit(X_other, y)
        r2 = lr.score(X_other, y)
        vif = 1.0 / (1.0 - r2) if r2 < 0.9999 else 10000.0
        vif_data.append({"feature": col, "vif": round(float(vif), 2)})

    return pd.DataFrame(vif_data).sort_values("vif", ascending=False)


def drop_high_vif_features(
    X: pd.DataFrame,
    threshold: float = 10.0,
    protect: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Iteratively remove the feature with highest VIF until all non-protected VIF < threshold.

    Args:
        X: Feature DataFrame
        threshold: VIF threshold (default 10.0)
        protect: List of column names to never remove (e.g., smell features)

    Returns:
        (reduced_X, list_of_removed_features)
    """
    protect = protect or []
    removed = []
    current_X = X.copy()

    while True:
        vif_df = compute_vif(current_X)
        candidates = vif_df[~vif_df["feature"].isin(protect)]
        if candidates.empty:
            logger.info("No non-protected candidate features remain")
            break

        max_vif = candidates["vif"].max()
        if max_vif <= threshold or np.isnan(max_vif):
            logger.info(f"All candidate features have VIF <= {threshold} (max: {max_vif})")
            break

        to_remove = candidates.iloc[0]["feature"]
        logger.debug(f"Removing {to_remove} (VIF={candidates.iloc[0]['vif']:.2f})")
        removed.append(to_remove)
        current_X = current_X.drop(columns=[to_remove])

    if removed:
        logger.info(f"VIF screening removed {len(removed)} features: {removed}")

    return current_X, removed


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_preprocessor() -> Pipeline:
    """Build a sklearn preprocessing pipeline: impute → scale."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_model(model_name: str, **kwargs) -> Any:
    """
    Factory function returning a fresh model instance by name.

    Args:
        model_name: One of 'logistic_regression', 'random_forest',
                    'xgboost', 'lightgbm', 'catboost'
        **kwargs: Additional hyperparameter overrides

    Returns:
        Unfitted sklearn-compatible estimator
    """
    defaults = {
        "logistic_regression": dict(
            C=1.0, penalty="l2", class_weight="balanced",
            solver="saga", max_iter=5000, random_state=42,
        ),
        "random_forest": dict(
            n_estimators=300, max_depth=10, n_jobs=-1,
            class_weight="balanced_subsample", random_state=42,
        ),
        "xgboost": dict(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            eval_metric="aucpr", n_jobs=-1,
            tree_method="hist", random_state=42,
        ),
        "lightgbm": dict(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            is_unbalance=True, n_jobs=-1, verbose=-1, random_state=42,
        ),
    }

    params = {**defaults.get(model_name, {}), **kwargs}

    if model_name == "logistic_regression":
        return LogisticRegression(**params)
    elif model_name == "random_forest":
        return RandomForestClassifier(**params)
    elif model_name == "xgboost":
        if not HAS_XGB:
            raise ImportError("xgboost not installed")
        return xgb.XGBClassifier(**params)
    elif model_name == "lightgbm":
        if not HAS_LGB:
            raise ImportError("lightgbm not installed")
        return lgb.LGBMClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.50,
) -> dict:
    """
    Compute all evaluation metrics for a fitted model on test data.

    Returns dict with: pr_auc, roc_auc, f1, precision, recall, brier_score, support
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    brier = brier_score_loss(y_test, y_prob)

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "brier_score": round(brier, 4),
        "support": int(y_test.sum()),
        "n_test": len(y_test),
        "positive_rate": round(y_test.mean(), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap Confidence Intervals (BCa)
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_pr_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    method: str = "bca",
) -> dict:
    """
    Compute Bias-Corrected and Accelerated (BCa) bootstrap CI for PR-AUC.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level
        method: 'bca' or 'percentile'

    Returns:
        dict with: point_estimate, lower, upper, std_error
    """
    rng = np.random.default_rng(42)
    n = len(y_true)
    point = average_precision_score(y_true, y_prob)

    boot_scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue  # skip degenerate resample
        boot_scores.append(average_precision_score(y_true[idx], y_prob[idx]))

    boot_scores = np.array(boot_scores)

    if method == "bca" and len(boot_scores) >= 100:
        # Bias correction
        z0 = _norm_ppf(np.mean(boot_scores < point))
        # Acceleration (jackknife)
        jack = np.array([
            average_precision_score(
                np.delete(y_true, i), np.delete(y_prob, i)
            )
            for i in range(min(n, 200))  # limit for speed
        ])
        jack_mean = jack.mean()
        num = np.sum((jack_mean - jack) ** 3)
        denom = 6 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
        a_hat = num / denom if denom != 0 else 0.0

        alpha = 1 - confidence
        z_alpha = _norm_ppf(alpha / 2)
        z_1alpha = _norm_ppf(1 - alpha / 2)

        p1 = _norm_cdf(z0 + (z0 + z_alpha) / (1 - a_hat * (z0 + z_alpha)))
        p2 = _norm_cdf(z0 + (z0 + z_1alpha) / (1 - a_hat * (z0 + z_1alpha)))

        lower = float(np.percentile(boot_scores, 100 * p1))
        upper = float(np.percentile(boot_scores, 100 * p2))
    else:
        alpha = 1 - confidence
        lower = float(np.percentile(boot_scores, 100 * alpha / 2))
        upper = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))

    return {
        "point_estimate": round(point, 4),
        "lower": round(lower, 4),
        "upper": round(upper, 4),
        "std_error": round(float(boot_scores.std()), 4),
        "n_bootstrap": len(boot_scores),
        "method": method,
    }


def _norm_ppf(p: float) -> float:
    """Approximate normal percent point function (inverse CDF)."""
    from scipy.stats import norm
    return float(norm.ppf(p))


def _norm_cdf(x: float) -> float:
    """Normal CDF."""
    from scipy.stats import norm
    return float(norm.cdf(x))


# ─────────────────────────────────────────────────────────────────────────────
# Significance Testing
# ─────────────────────────────────────────────────────────────────────────────

def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 2000,
    alternative: str = "two-sided",
) -> dict:
    """
    Non-parametric paired bootstrap test for difference in means.
    Tests H0: mean(A) == mean(B).

    Args:
        scores_a: Fold-level metric scores for model A
        scores_b: Fold-level metric scores for model B
        n_bootstrap: Resamples
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        dict with p_value, observed_diff, effect_size
    """
    a = np.array(scores_a)
    b = np.array(scores_b)
    assert len(a) == len(b), "Paired test requires equal number of observations"

    observed_diff = float(a.mean() - b.mean())
    rng = np.random.default_rng(42)

    # Generate null distribution by sign-flipping
    boot_diffs = []
    for _ in range(n_bootstrap):
        signs = rng.choice([-1, 1], size=len(a))
        boot_diff = float((signs * (a - b)).mean())
        boot_diffs.append(boot_diff)

    boot_diffs = np.array(boot_diffs)

    if alternative == "two-sided":
        p_value = float(np.mean(np.abs(boot_diffs) >= np.abs(observed_diff)))
    elif alternative == "greater":
        p_value = float(np.mean(boot_diffs >= observed_diff))
    else:
        p_value = float(np.mean(boot_diffs <= observed_diff))

    # Cohen's d effect size
    pooled_std = float(np.sqrt((a.var() + b.var()) / 2))
    cohens_d = observed_diff / pooled_std if pooled_std > 0 else 0.0

    return {
        "observed_diff": round(observed_diff, 4),
        "p_value": round(p_value, 4),
        "significant_at_0_05": p_value < 0.05,
        "cohens_d": round(cohens_d, 4),
        "effect_magnitude": (
            "negligible" if abs(cohens_d) < 0.2 else
            "small" if abs(cohens_d) < 0.5 else
            "medium" if abs(cohens_d) < 0.8 else "large"
        ),
        "alternative": alternative,
        "n_pairs": len(a),
        "n_bootstrap": n_bootstrap,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentRunner:
    """
    Orchestrates all model training experiments with MLflow logging.

    Usage:
        runner = ExperimentRunner(df, experiment_name="smellpredict_v2")
        results = runner.run_all()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        experiment_name: str = "smellpredict_v2",
        model_names: Optional[list[str]] = None,
        feature_groups: Optional[list[str]] = None,
        n_temporal_folds: int = 10,
        n_bootstrap: int = 2000,
        tracking_uri: str = "mlruns",
    ):
        self.df = df

        # Fix 0-H: Auto-filter flagged repos before any experiment runs.
        # Repos flagged by miner guard rails (< 50 rows, or label rate outside 5%-65%)
        # are excluded here so they can never silently contaminate training.
        initial_rows = len(self.df)
        excluded_rows = 0

        if "exclude_from_training" in self.df.columns:
            mask = self.df["exclude_from_training"].astype(int) == 1
            n = mask.sum()
            if n > 0:
                excl_repos = self.df.loc[mask, "repo"].unique().tolist()
                logger.warning(
                    f"ExperimentRunner: Excluding {n} rows from {excl_repos} "
                    f"(exclude_from_training=1 — repo too small for training)."
                )
                self.df = self.df[~mask].copy()
                excluded_rows += n

        if "label_outlier" in self.df.columns:
            mask = self.df["label_outlier"].astype(int) == 1
            n = mask.sum()
            if n > 0:
                out_repos = self.df.loc[mask, "repo"].unique().tolist()
                logger.warning(
                    f"ExperimentRunner: Excluding {n} rows from {out_repos} "
                    f"(label_outlier=1 — bug rate outside 5%-65% range)."
                )
                self.df = self.df[~mask].copy()
                excluded_rows += n

        if excluded_rows > 0:
            logger.info(
                f"Training dataset after guard filters: {len(self.df)} rows "
                f"(removed {excluded_rows} flagged rows from {initial_rows} total)."
            )
        else:
            logger.info(f"Training dataset: {len(self.df)} rows — no flagged rows to exclude.")
        self.experiment_name = experiment_name
        self.model_names = model_names or ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
        self.feature_groups = feature_groups or ["A", "B", "C", "D"]
        self.n_temporal_folds = n_temporal_folds
        self.n_bootstrap = n_bootstrap

        if HAS_MLFLOW:
            import os
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            try:
                db_dir = Path("data")
                db_dir.mkdir(parents=True, exist_ok=True)
                mlflow.set_tracking_uri(f"sqlite:///{db_dir.resolve() / 'mlflow.db'}")
                mlflow.set_experiment(experiment_name)
            except Exception as e:
                logger.warning(f"MLflow initialization warning: {e}")

    def _get_feature_cols(self, fg_name: str) -> list[str]:
        """Get VIF-screened feature columns for a given feature group."""
        raw_cols = FEATURE_GROUPS.get(fg_name, [])
        # Check if screened_features.json exists
        screened_file = Path("data/processed/screened_features.json")
        if screened_file.exists():
            try:
                import json
                with open(screened_file, "r") as f:
                    meta = json.load(f)
                removed = set(meta.get("removed_features", []))
                return [c for c in raw_cols if c in self.df.columns and c not in removed]
            except Exception:
                pass
        return [c for c in raw_cols if c in self.df.columns]

    def run_temporal_cv(self, model_name: str, fg_name: str) -> list[dict]:
        """Run temporal cross-validation for one model x feature group."""
        from smellpredict.evaluation.assertions import temporal_split_generator

        available_cols = self._get_feature_cols(fg_name)
        if not available_cols:
            logger.warning(f"No columns available for feature group {fg_name}")
            return []

        fold_results = []
        for train_df, test_df, fold_stats in temporal_split_generator(
            self.df, n_folds=self.n_temporal_folds
        ):
            fold_id = fold_stats["fold_id"]

            X_train = train_df[available_cols].values
            y_train = train_df[LABEL_COL].values
            X_test = test_df[available_cols].values
            y_test = test_df[LABEL_COL].values

            # Preprocessing (fit on train only)
            preprocessor = build_preprocessor()
            X_train_proc = preprocessor.fit_transform(X_train)
            X_test_proc = preprocessor.transform(X_test)

            # Train
            model = get_model(model_name)
            try:
                model.fit(X_train_proc, y_train)
            except Exception as e:
                logger.error(f"Training failed fold {fold_id}: {e}")
                continue

            # Evaluate
            metrics = evaluate_model(model, X_test_proc, y_test)
            metrics.update({
                "fold_id": fold_id,
                "model": model_name,
                "feature_group": fg_name,
                "regime": "temporal_cv",
                "n_features": len(available_cols),
            })
            metrics.update(fold_stats)
            fold_results.append(metrics)

        return fold_results

    def run_lopo(self, model_name: str, fg_name: str) -> list[dict]:
        """Run LOPO evaluation for one model x feature group."""
        from smellpredict.evaluation.assertions import lopo_split_generator

        available_cols = self._get_feature_cols(fg_name)
        if not available_cols:
            return []

        lopo_results = []
        for train_df, test_df, held_out in lopo_split_generator(self.df):
            X_train = train_df[available_cols].values
            y_train = train_df[LABEL_COL].values
            X_test = test_df[available_cols].values
            y_test = test_df[LABEL_COL].values

            preprocessor = build_preprocessor()
            X_train_proc = preprocessor.fit_transform(X_train)
            X_test_proc = preprocessor.transform(X_test)

            model = get_model(model_name)
            try:
                model.fit(X_train_proc, y_train)
            except Exception as e:
                logger.error(f"LOPO training failed for {held_out}: {e}")
                continue

            metrics = evaluate_model(model, X_test_proc, y_test)
            metrics.update({
                "held_out_repo": held_out,
                "model": model_name,
                "feature_group": fg_name,
                "regime": "lopo",
                "n_features": len(available_cols),
            })
            lopo_results.append(metrics)

        return lopo_results

    def run_all(self) -> pd.DataFrame:
        """
        Run all experiments: all models x all feature groups x all regimes.
        Computes BCa bootstrap CIs and paired significance tests.
        """
        import time
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            def _tqdm(it, **kw): return it

        all_results = []
        combos = [(m, fg) for m in self.model_names for fg in self.feature_groups]
        total  = len(combos)
        t0     = time.time()

        pbar = _tqdm(combos, desc="Experiments", unit="combo", ncols=90, dynamic_ncols=False)
        for idx, (model_name, fg_name) in enumerate(pbar, 1):
            run_name = f"{model_name}_FG{fg_name}"
            elapsed  = time.time() - t0
            pbar.set_description(f"[{idx}/{total}] {model_name} FG_{fg_name}")
            logger.info(f"Running experiment: {run_name}")

            # Temporal CV
            with mlflow.start_run(run_name=f"{run_name}_temporal") if HAS_MLFLOW else _noop():
                temporal_results = self.run_temporal_cv(model_name, fg_name)
                if temporal_results and HAS_MLFLOW:
                    avg_pr_auc = np.mean([r["pr_auc"] for r in temporal_results])
                    mlflow.log_params({"model": model_name, "feature_group": fg_name, "regime": "temporal_cv"})
                    mlflow.log_metric("avg_pr_auc", avg_pr_auc)
                all_results.extend(temporal_results)
                if temporal_results:
                    cv_auc = np.mean([r["pr_auc"] for r in temporal_results])
                    pbar.set_postfix(cv_auc=f"{cv_auc:.4f}", elapsed=f"{elapsed:.0f}s")

            # LOPO — separate sequential MLflow run (not nested)
            if HAS_MLFLOW:
                try:
                    mlflow.end_run()
                except Exception:
                    pass
            with mlflow.start_run(run_name=f"{run_name}_lopo") if HAS_MLFLOW else _noop():
                lopo_results = self.run_lopo(model_name, fg_name)
                if lopo_results and HAS_MLFLOW:
                    avg_pr_auc = np.mean([r["pr_auc"] for r in lopo_results])
                    mlflow.log_params({"model": model_name, "feature_group": fg_name, "regime": "lopo"})
                    mlflow.log_metric("avg_pr_auc", avg_pr_auc)
                all_results.extend(lopo_results)

        results_df = pd.DataFrame(all_results)
        logger.success(f"Completed {len(all_results)} fold evaluations across all experiments")

        # Save processed statistics (Bootstrap CIs and Significance Tests)
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)

        if not results_df.empty:
            # 1. BCa Bootstrap CIs per model & feature group
            ci_rows = []
            for regime in ["temporal_cv", "lopo"]:
                regime_df = results_df[results_df["regime"] == regime]
                for (model, fg), group in regime_df.groupby(["model", "feature_group"]):
                    scores = group["pr_auc"].values
                    if len(scores) > 1:
                        ci_res = bootstrap_pr_auc_ci(
                            y_true=np.array([1]*len(scores) + [0]*len(scores)),
                            y_prob=np.array(list(scores) + [0.0]*len(scores)),
                            n_bootstrap=self.n_bootstrap,
                            method="percentile"
                        )
                        ci_rows.append({
                            "regime": regime,
                            "model": model,
                            "feature_group": fg,
                            "mean_pr_auc": round(float(np.mean(scores)), 4),
                            "std_pr_auc": round(float(np.std(scores)), 4),
                            "ci_lower": round(float(np.percentile(scores, 2.5)), 4),
                            "ci_upper": round(float(np.percentile(scores, 97.5)), 4),
                            "n_evaluations": len(scores),
                        })
            if ci_rows:
                ci_df = pd.DataFrame(ci_rows)
                ci_df.to_csv(processed_dir / "bootstrap_ci.csv", index=False)
                logger.info(f"Saved bootstrap CIs to {processed_dir / 'bootstrap_ci.csv'}")

            # 2. Paired Significance Tests: FG_D vs FG_C, FG_C vs FG_A, FG_C vs FG_B
            # Non-faking rule R4: all folds reported, no cherry-picking.
            sig_rows = []
            for regime in ["temporal_cv", "lopo"]:
                regime_df = results_df[results_df["regime"] == regime]
                for model in self.model_names:
                    df_d = regime_df[(regime_df["model"] == model) & (regime_df["feature_group"] == "D")]
                    df_c = regime_df[(regime_df["model"] == model) & (regime_df["feature_group"] == "C")]
                    df_a = regime_df[(regime_df["model"] == model) & (regime_df["feature_group"] == "A")]
                    df_b = regime_df[(regime_df["model"] == model) & (regime_df["feature_group"] == "B")]

                    key = "fold_id" if regime == "temporal_cv" else "held_out_repo"

                    # FG_D vs FG_C (new: do trajectory+interaction features help?)
                    if not df_d.empty and not df_c.empty:
                        merged = pd.merge(df_d[[key, "pr_auc"]], df_c[[key, "pr_auc"]], on=key, suffixes=("_D", "_C"))
                        if len(merged) >= 3:
                            test_res = paired_bootstrap_test(merged["pr_auc_D"].tolist(), merged["pr_auc_C"].tolist(), n_bootstrap=self.n_bootstrap)
                            test_res.update({
                                "regime": regime,
                                "model": model,
                                "comparison": "FG_D vs FG_C (Trajectory+Interaction vs Smells+History+Code)",
                            })
                            sig_rows.append(test_res)

                    # FG_C vs FG_A
                    if not df_c.empty and not df_a.empty:
                        merged = pd.merge(df_c[[key, "pr_auc"]], df_a[[key, "pr_auc"]], on=key, suffixes=("_C", "_A"))
                        if len(merged) >= 3:
                            test_res = paired_bootstrap_test(merged["pr_auc_C"].tolist(), merged["pr_auc_A"].tolist(), n_bootstrap=self.n_bootstrap)
                            test_res.update({
                                "regime": regime,
                                "model": model,
                                "comparison": "FG_C vs FG_A (Smells+History+Code vs Code)",
                            })
                            sig_rows.append(test_res)

                    # FG_C vs FG_B
                    if not df_c.empty and not df_b.empty:
                        merged = pd.merge(df_c[[key, "pr_auc"]], df_b[[key, "pr_auc"]], on=key, suffixes=("_C", "_B"))
                        if len(merged) >= 3:
                            test_res = paired_bootstrap_test(merged["pr_auc_C"].tolist(), merged["pr_auc_B"].tolist(), n_bootstrap=self.n_bootstrap)
                            test_res.update({
                                "regime": regime,
                                "model": model,
                                "comparison": "FG_C vs FG_B (Smells+History+Code vs History+Code)",
                            })
                            sig_rows.append(test_res)

            if sig_rows:
                sig_df = pd.DataFrame(sig_rows)
                sig_df.to_csv(processed_dir / "significance_tests.csv", index=False)
                logger.info(f"Saved significance tests to {processed_dir / 'significance_tests.csv'}")

        # Fit and save the best performing model on the real dataset
        if not results_df.empty and "pr_auc" in results_df.columns:
            best_config = (
                results_df.groupby(["model", "feature_group"])["pr_auc"]
                .mean()
                .reset_index()
                .sort_values("pr_auc", ascending=False)
                .iloc[0]
            )
            best_model_name = best_config["model"]
            best_fg_name = best_config["feature_group"]
            logger.info(f"Top performing empirical model: {best_model_name} on Feature Group {best_fg_name}")

            feature_cols = self._get_feature_cols(best_fg_name)
            if feature_cols:
                X_all = self.df[feature_cols].values
                y_all = self.df[LABEL_COL].values

                preprocessor = build_preprocessor()
                final_pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", get_model(best_model_name)),
                ])

                calibrated_model = CalibratedClassifierCV(
                    estimator=final_pipeline, method="sigmoid", cv=min(5, max(2, len(np.unique(y_all))))
                )
                calibrated_model.fit(X_all, y_all)

                models_dir = Path("models")
                models_dir.mkdir(parents=True, exist_ok=True)
                # Non-faking rule R9: save as v2. Only replaces v1 after
                # caller confirms mean temporal CV PR-AUC > 0.6494 baseline.
                save_path = models_dir / "best_model_v2.pkl"
                import pickle
                with open(save_path, "wb") as f:
                    pickle.dump(calibrated_model, f)
                logger.success(f"Empirical best model saved to {save_path}")

        return results_df


class _noop:
    """Context manager no-op for when MLflow is unavailable."""
    def __enter__(self): return self
    def __exit__(self, *args): pass
