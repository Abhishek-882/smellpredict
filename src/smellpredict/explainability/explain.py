"""
SmellPredict — Explainability Module
======================================
Provides SHAP and LIME-based explanations for model predictions.
Covers:
  - Global feature importance (SHAP summary + bar plots)
  - Local per-file explanations (SHAP waterfall + LIME)
  - Probability calibration (Platt Scaling + Isotonic Regression)
  - Expected Calibration Error (ECE) + calibration curve
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.debug("SHAP not installed — explainability features disabled")

try:
    from lime.lime_tabular import LimeTabularExplainer
    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    logger.debug("LIME not installed — local explanations disabled")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Global Explainability
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: list[str],
    model_type: str = "tree",
    n_background: int = 500,
) -> tuple:
    """
    Compute SHAP values for the given model and input data.

    Args:
        model: Fitted sklearn-compatible model
        X: Feature array (test set)
        feature_names: List of feature names
        model_type: 'tree' for RF/XGBoost, 'linear' for LogReg, 'kernel' fallback
        n_background: Background samples for KernelExplainer

    Returns:
        (shap_values, explainer) tuple
    """
    if not HAS_SHAP:
        raise ImportError("SHAP is required. pip install shap")

    X_df = pd.DataFrame(X, columns=feature_names)

    try:
        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)
            # For binary classifiers, shap_values is a list [neg_class, pos_class]
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]  # use positive class
        elif model_type == "linear":
            explainer = shap.LinearExplainer(model, X_df)
            shap_values = explainer.shap_values(X_df)
        else:
            # Kernel SHAP — model-agnostic but slower
            background = shap.sample(X_df, min(n_background, len(X_df)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_df[:100])[:, :, 1]
    except Exception as e:
        logger.warning(f"SHAP tree/linear explainer failed, falling back to Kernel: {e}")
        background = shap.sample(X_df, min(100, len(X_df)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_df[:50])

    return shap_values, explainer


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    output_path: str | Path,
    plot_type: str = "dot",
    max_display: int = 20,
) -> None:
    """
    Save SHAP summary (beeswarm) plot to file.

    Args:
        shap_values: SHAP values array (n_samples × n_features)
        X: Feature array
        feature_names: Feature names
        output_path: Where to save the PNG
        plot_type: 'dot' (beeswarm) or 'bar'
        max_display: Max features to show
    """
    if not HAS_MPL:
        return

    X_df = pd.DataFrame(X, columns=feature_names)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_df,
        plot_type=plot_type,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary plot saved: {output_path}")


def get_global_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Compute global feature importance as mean absolute SHAP value.

    Returns:
        DataFrame sorted by importance (descending)
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).head(top_n)

    importance_df["rank"] = range(1, len(importance_df) + 1)
    return importance_df.reset_index(drop=True)


def plot_shap_waterfall(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    instance_idx: int,
    expected_value: float,
    output_path: str | Path,
    title: str = "SHAP Waterfall — File Risk Explanation",
) -> None:
    """
    Plot SHAP waterfall chart for a single file instance.

    Args:
        shap_values: Full SHAP values array
        X: Feature array
        feature_names: Feature names
        instance_idx: Which row/file to explain
        expected_value: SHAP base value (explainer.expected_value)
        output_path: PNG save path
        title: Chart title
    """
    if not HAS_MPL:
        return

    X_df = pd.DataFrame(X, columns=feature_names)
    explanation = shap.Explanation(
        values=shap_values[instance_idx],
        base_values=expected_value,
        data=X_df.iloc[instance_idx].values,
        feature_names=feature_names,
    )
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, max_display=15, show=False)
    plt.title(title)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP waterfall saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# LIME Local Explainability
# ─────────────────────────────────────────────────────────────────────────────

def explain_instance_lime(
    model,
    X_train: np.ndarray,
    X_instance: np.ndarray,
    feature_names: list[str],
    n_features: int = 10,
    n_samples: int = 1000,
) -> Optional[dict]:
    """
    Generate LIME explanation for a single file instance.

    Args:
        model: Fitted classifier
        X_train: Training data (for distribution estimation)
        X_instance: Single row to explain (1D array)
        feature_names: Feature names
        n_features: Number of features to show
        n_samples: LIME neighbourhood samples

    Returns:
        dict with top features and their contributions, or None if LIME unavailable
    """
    if not HAS_LIME:
        logger.warning("LIME not available")
        return None

    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=["Not Bug Fix", "Bug Fix"],
        mode="classification",
        random_state=42,
    )

    exp = explainer.explain_instance(
        X_instance,
        model.predict_proba,
        num_features=n_features,
        num_samples=n_samples,
    )

    return {
        "local_explanation": exp.as_list(),
        "prediction_probabilities": {
            "no_bug_fix": exp.local_pred[0] if hasattr(exp, "local_pred") else None,
            "bug_fix": exp.local_pred[1] if hasattr(exp, "local_pred") else None,
        },
        "intercept": exp.intercept[1] if hasattr(exp, "intercept") else None,
        "score": exp.score if hasattr(exp, "score") else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Probability Calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_model(
    model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str = "sigmoid",
) -> tuple:
    """
    Apply post-hoc calibration to a fitted model.
    Uses a held-out calibration set (NOT the test set).

    Args:
        model: Already-fitted classifier
        X_cal: Calibration features (disjoint from train and test)
        y_cal: Calibration labels
        method: 'sigmoid' (Platt) or 'isotonic'

    Returns:
        (calibrated_model, calibration_metrics) tuple
    """
    from sklearn.calibration import CalibratedClassifierCV

    cal_model = CalibratedClassifierCV(model, method=method, cv="prefit")
    cal_model.fit(X_cal, y_cal)

    # Evaluate calibration
    y_prob_cal = cal_model.predict_proba(X_cal)[:, 1]
    metrics = compute_calibration_metrics(y_cal, y_prob_cal)
    metrics["method"] = method

    logger.info(
        f"Calibration ({method}): Brier={metrics['brier_score']:.4f}, "
        f"ECE={metrics['ece']:.4f}"
    )

    return cal_model, metrics


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute Expected Calibration Error (ECE) and Brier score.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for ECE computation

    Returns:
        dict with brier_score, ece, mean_calibration_error_per_bin
    """
    from sklearn.metrics import brier_score_loss
    from sklearn.calibration import calibration_curve

    brier = brier_score_loss(y_true, y_prob)

    try:
        fraction_of_positives, mean_predicted = calibration_curve(
            y_true, y_prob, n_bins=n_bins
        )
        # ECE = weighted avg |predicted - actual| across bins
        bin_sizes = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
        ece = np.sum(
            np.abs(fraction_of_positives - mean_predicted) * bin_sizes
        ) / len(y_true)
    except Exception:
        ece = np.nan
        fraction_of_positives = np.array([])
        mean_predicted = np.array([])

    return {
        "brier_score": round(float(brier), 4),
        "ece": round(float(ece), 4),
        "calibration_curve_y": fraction_of_positives.tolist() if len(fraction_of_positives) else [],
        "calibration_curve_x": mean_predicted.tolist() if len(mean_predicted) else [],
    }


def plot_calibration_curve(
    y_true: np.ndarray,
    probs_dict: dict[str, np.ndarray],
    output_path: str | Path,
    n_bins: int = 10,
) -> None:
    """
    Plot calibration curves for multiple models/methods.

    Args:
        y_true: Ground truth labels
        probs_dict: Dict of {label: predicted_probabilities}
        output_path: PNG save path
        n_bins: Number of calibration bins
    """
    if not HAS_MPL:
        return

    from sklearn.calibration import calibration_curve

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.5)

    for label, y_prob in probs_dict.items():
        try:
            frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
            brier = brier_score_loss_safe(y_true, y_prob)
            plt.plot(mean_pred, frac_pos, marker="o", label=f"{label} (Brier={brier:.3f})")
        except Exception as e:
            logger.debug(f"Calibration curve failed for {label}: {e}")

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curves — SmellPredict Models")
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Calibration curves saved: {output_path}")


def brier_score_loss_safe(y_true, y_prob):
    """Safe wrapper for brier_score_loss."""
    try:
        from sklearn.metrics import brier_score_loss
        return round(float(brier_score_loss(y_true, y_prob)), 4)
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Risk Tier Classification
# ─────────────────────────────────────────────────────────────────────────────

RISK_TIERS = [
    (0.00, 0.20, "Low", "🟢", "Standard code review"),
    (0.20, 0.50, "Medium", "🟡", "Enhanced review + smell audit"),
    (0.50, 0.75, "High", "🟠", "Priority review + automated checks"),
    (0.75, 1.00, "Critical", "🔴", "Mandatory peer review + refactoring"),
]


def classify_risk(probability: float) -> dict:
    """
    Classify a calibrated probability into an actionable risk tier.

    Args:
        probability: Calibrated bug-fix probability [0, 1]

    Returns:
        dict with tier, icon, probability, recommendation
    """
    for low, high, tier, icon, recommendation in RISK_TIERS:
        if low <= probability < high or (probability >= 0.75 and tier == "Critical"):
            return {
                "probability": round(probability, 3),
                "tier": tier,
                "icon": icon,
                "recommendation": recommendation,
            }
    return {
        "probability": round(probability, 3),
        "tier": "Unknown",
        "icon": "⚪",
        "recommendation": "Manual review required",
    }
