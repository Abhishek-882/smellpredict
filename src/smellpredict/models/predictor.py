"""
SmellPredict — Unified Prediction & Analysis Engine (Dual-Engine v2)
===================================================================
Provides unified file-level and repository-level risk prediction,
feature extraction, architecture guardrails, and refactoring analysis.

Supports Dual-Engine Routing:
  - Engine A (Pure Static AST): 28 features, zero git history, deterministic
    single-buffer quantile interpolation against empirical reference CDFs.
  - Engine B (Full Enterprise Telemetry): 73 features (AST + Churn + Ownership + Co-Change).
"""

from __future__ import annotations

import os
import re
import json
import joblib
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("smellpredict.predictor")

from smellpredict.features.extractor import (
    CodeMetrics,
    SmellFeatures,
    extract_file_features,
)
from smellpredict.features.refactor import RefactoringAdvice, analyze_refactorings


@dataclass
class GitTelemetry:
    """Git development history & process metrics for repository-level inference."""
    is_tracked_in_git: bool = False
    file_age_days: float = 0.0
    days_since_last_change: float = 0.0
    previous_file_commits: int = 0
    previous_bug_fixes: int = 0
    recent_file_commits: int = 0
    avg_commit_size: float = 0.0
    developer_experience: float = 1.0
    contributors: int = 1
    ownership_concentration: float = 1.0
    silo_index: float = 0.0
    contributor_annual_rate: float = 0.0
    cochange_peer_count: int = 0
    cochange_coupling_ratio: float = 0.0
    cochange_peer_complexity_mean: float = 0.0
    cochange_peer_loc_mean: float = 0.0


_MODEL_CACHE: Dict[str, Any] = {}


def get_engine_a_model() -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """Lazy-load Engine A (Pure Static AST Model)."""
    if "model_a" not in _MODEL_CACHE:
        paths = [
            Path("models/engine_a_ast_static.pkl"),
            Path(__file__).parent.parent.parent.parent / "models" / "engine_a_ast_static.pkl",
        ]
        loaded, meta = None, None
        for p in paths:
            if p.exists():
                try:
                    loaded = joblib.load(p)
                    meta_path = p.with_name("engine_a_ast_static_metadata.json")
                    if meta_path.exists():
                        with open(meta_path, "r", encoding="utf-8") as mf:
                            meta = json.load(mf)
                    logger.info(f"Loaded Engine A (Pure Static AST) from {p}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load Engine A from {p}: {e}")

        _MODEL_CACHE["model_a"] = loaded
        _MODEL_CACHE["meta_a"] = meta

    return _MODEL_CACHE.get("model_a"), _MODEL_CACHE.get("meta_a")


def get_engine_b_model() -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """Lazy-load Engine B (Full Enterprise Telemetry Model)."""
    if "model_b" not in _MODEL_CACHE:
        paths = [
            Path(os.environ.get("SMELLPREDICT_MODEL_PATH", "models/best_model_final.pkl")),
            Path("models/best_model_final.pkl"),
            Path(__file__).parent.parent.parent.parent / "models" / "best_model_final.pkl",
        ]
        loaded, meta = None, None
        for p in paths:
            if p.exists():
                try:
                    loaded = joblib.load(p)
                    meta_path = p.with_name("best_model_final_metadata.json")
                    if meta_path.exists():
                        with open(meta_path, "r", encoding="utf-8") as mf:
                            meta = json.load(mf)
                    logger.info(f"Loaded Engine B (Full Enterprise) from {p}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load Engine B from {p}: {e}")

        _MODEL_CACHE["model_b"] = loaded
        _MODEL_CACHE["meta_b"] = meta

    return _MODEL_CACHE.get("model_b"), _MODEL_CACHE.get("meta_b")


def get_trained_model() -> Optional[Any]:
    """Backward compatibility wrapper returning Engine B champion model."""
    m_b, _ = get_engine_b_model()
    if m_b is not None:
        return m_b
    m_a, _ = get_engine_a_model()
    return m_a


def classify_risk_tier(probability: float) -> Dict[str, str]:
    """Classify a calibrated probability [0, 1] into an advisory risk tier."""
    if probability <= 0.35:
        return {
            "tier": "Low",
            "icon": "🟢",
            "recommendation": "Code looks healthy. Standard peer review recommended.",
            "color": "#00d4aa",
        }
    elif probability <= 0.65:
        return {
            "tier": "Medium",
            "icon": "🟡",
            "recommendation": "Moderate complexity indicators. Focus testing on complex branches.",
            "color": "#f59e0b",
        }
    elif probability <= 0.80:
        return {
            "tier": "High",
            "icon": "🟠",
            "recommendation": "High defect risk prior. Prioritize code review for nested logic and smells.",
            "color": "#f97316",
        }
    else:
        return {
            "tier": "Critical",
            "icon": "🔴",
            "recommendation": "Critical defect risk prior. Recommend architectural decomposition and refactoring.",
            "color": "#ef4444",
        }


def check_architecture_guardrail(
    source_code: str,
    file_path: str = "snippet.py",
    repo_context: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """Detects FFI-backed native cores (e.g. Rust/C) and flat functional paradigms."""
    # 1. Repository-level polyglot / FFI check
    if repo_context and "repo_files" in repo_context:
        files = repo_context["repo_files"]
        if any(f.endswith(".rs") or "Cargo.toml" in f for f in files):
            return True, "Polyglot Codebase: Primary core contains Rust components. Python AST metrics have reduced fidelity."
        if any(f.endswith((".c", ".cpp", ".pyx")) for f in files):
            return True, "Native C/C++ Extensions: Defect causes may reside in compiled extensions."

    # 2. File-level FFI import signatures
    ffi_patterns = [r"\bimport\s+ctypes\b", r"\bfrom\s+cffi\b", r"\bimport\s+pyo3\b", r"\bimport\s+cython\b"]
    if any(re.search(pat, source_code) for pat in ffi_patterns):
        return True, "Native FFI Bindings: File interfaces with external C/Rust native libraries."

    return False, None


def compute_quantile_rank(val: float, ref_table: Optional[List[float]]) -> float:
    """Compute empirical percentile [0.01, 0.99] using pre-computed 101-point reference table."""
    if not ref_table or len(ref_table) < 2:
        return 0.5
    percentiles = np.linspace(0.01, 0.99, len(ref_table))
    return float(np.clip(np.interp(val, ref_table, percentiles), 0.01, 0.99))


def compute_heuristic_risk(code_metrics: CodeMetrics, smell_feats: SmellFeatures) -> float:
    """Deterministic fallback risk score for when no trained model is present."""
    complexity_score = min(code_metrics.max_cyclomatic_complexity / 30.0, 1.0)
    nesting_score = min(code_metrics.max_nesting_depth / 8.0, 1.0)
    smell_score = min(smell_feats.total_smells / 10.0, 1.0)
    loc_score = min(code_metrics.loc / 1000.0, 1.0)
    mi_score = max(0.0, (100.0 - code_metrics.maintainability_index) / 100.0)

    prob = (
        0.25 * complexity_score
        + 0.25 * smell_score
        + 0.20 * nesting_score
        + 0.15 * loc_score
        + 0.15 * mi_score
    )
    return round(float(np.clip(prob, 0.01, 0.99)), 4)


def extract_engine_a_features(
    code_metrics: CodeMetrics,
    smell_feats: SmellFeatures,
    meta_a: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    """Build exact feature DataFrame for Engine A with empirical quantile interpolation."""
    m_dict = asdict(code_metrics)
    s_dict = asdict(smell_feats)
    smell_dens = float(smell_feats.total_smells / max(1, code_metrics.loc) * 1000.0)
    ref_tables = meta_a.get("quantile_reference_tables", {}) if meta_a else {}
    features = meta_a.get("features", []) if meta_a else []

    row = {}
    for feat in features:
        if feat.startswith("rank_"):
            base_col = feat.replace("rank_", "")
            if base_col == "total_smells":
                val = float(smell_feats.total_smells)
            elif base_col == "smell_density":
                val = smell_dens
            elif base_col.startswith("code_"):
                val = float(m_dict.get(base_col.replace("code_", ""), 0.0))
            else:
                val = float(s_dict.get(base_col, 0.0))
            row[feat] = compute_quantile_rank(val, ref_tables.get(base_col))
        elif feat.startswith("rel_"):
            base_col = feat.replace("rel_", "")
            if base_col == "smell_density":
                row[feat] = smell_dens
            elif base_col.startswith("code_"):
                row[feat] = float(m_dict.get(base_col.replace("code_", ""), 0.0))
            else:
                row[feat] = float(s_dict.get(base_col, 0.0))
        elif feat == "smell_density":
            row[feat] = smell_dens
        elif feat.startswith("code_"):
            field_name = feat.replace("code_", "")
            row[feat] = float(m_dict.get(field_name, 0.0))
        elif feat in s_dict:
            row[feat] = float(s_dict.get(feat, 0.0))
        else:
            row[feat] = 0.0

    return pd.DataFrame([row])


def analyze_source_code(
    source_code: str,
    file_path: str = "snippet.py",
    git_telemetry: Optional[GitTelemetry] = None,
    repo_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Unified Dual-Engine Analysis Pipeline:
      - Extracts CodeMetrics + SmellFeatures via AST parser.
      - Checks architecture guardrails (FFI / Polyglot detection).
      - Automatically routes to Engine A (Cold-Start AST) or Engine B (Full Git Telemetry).
      - Returns calibrated defect probability and refactoring advice.
    """
    code_metrics, smell_feats = extract_file_features(source_code)
    heuristic_prob = compute_heuristic_risk(code_metrics, smell_feats)
    is_advisory, warning_msg = check_architecture_guardrail(source_code, file_path, repo_context)

    routing_mode = os.environ.get("SMELLPREDICT_ROUTING_MODE", "auto").lower()

    # Determine Engine
    use_engine_b = False
    if routing_mode == "engine_b_only":
        use_engine_b = True
    elif routing_mode == "engine_a_only":
        use_engine_b = False
    else:  # auto
        use_engine_b = (git_telemetry is not None and git_telemetry.is_tracked_in_git)

    model_loaded = False
    prob = heuristic_prob
    engine_name = "heuristic_fallback"
    engine_desc = "Heuristic AST Complexity Fallback"

    if use_engine_b:
        model_b, meta_b = get_engine_b_model()
        if model_b is not None and meta_b:
            try:
                features_b = meta_b.get("features", [])
                ref_tables = meta_b.get("quantile_reference_tables", {})
                m_dict = asdict(code_metrics)
                s_dict = asdict(smell_feats)
                g_dict = asdict(git_telemetry) if git_telemetry else {}

                row = {}
                for feat in features_b:
                    if feat.startswith("rank_"):
                        base_col = feat.replace("rank_", "")
                        val = m_dict.get(base_col.replace("code_", ""), 0.0) if base_col.startswith("code_") else s_dict.get(base_col, g_dict.get(base_col, 0.0))
                        row[feat] = compute_quantile_rank(val, ref_tables.get(base_col))
                    elif feat.startswith("code_"):
                        row[feat] = float(m_dict.get(feat.replace("code_", ""), 0.0))
                    elif feat in s_dict:
                        row[feat] = float(s_dict.get(feat, 0.0))
                    elif feat in g_dict:
                        row[feat] = float(g_dict.get(feat, 0.0))
                    elif feat == "complexity_x_churn":
                        row[feat] = float(code_metrics.max_cyclomatic_complexity * g_dict.get("previous_file_commits", 0))
                    elif feat == "smells_x_churn":
                        row[feat] = float(smell_feats.total_smells * g_dict.get("previous_file_commits", 0))
                    elif feat == "bug_rate_x_smells":
                        row[feat] = float(g_dict.get("previous_bug_fixes", 0) * smell_feats.total_smells)
                    else:
                        row[feat] = 0.0

                df_feat = pd.DataFrame([row])
                ml_prob = float(model_b.predict_proba(df_feat)[0, 1])
                prob = ml_prob
                engine_name = "engine_b_full_telemetry"
                engine_desc = "Engine B (Full Telemetry: AST + Process Churn + Ownership)"
                model_loaded = True
            except Exception as e:
                logger.warning(f"Engine B inference exception: {e}")
                use_engine_b = False

    if not use_engine_b:
        model_a, meta_a = get_engine_a_model()
        if model_a is not None and meta_a:
            try:
                df_feat = extract_engine_a_features(code_metrics, smell_feats, meta_a)
                ml_prob = float(model_a.predict_proba(df_feat)[0, 1])
                prob = ml_prob
                engine_name = "engine_a_ast_static"
                engine_desc = "Engine A (Pure Static AST, Zero Git History)"
                model_loaded = True
            except Exception as e:
                logger.warning(f"Engine A inference exception: {e}")
                prob = heuristic_prob

    prob = round(float(np.clip(prob, 0.01, 0.99)), 4)
    tier_info = classify_risk_tier(prob)
    refactor_advice = analyze_refactorings(source_code)

    return {
        "file_path": file_path,
        "risk_probability": prob,
        "risk_tier": tier_info["tier"],
        "risk_icon": tier_info["icon"],
        "risk_color": tier_info["color"],
        "recommendation": tier_info["recommendation"],
        "engine_used": engine_name,
        "engine_desc": engine_desc,
        "guardrail_status": "advisory" if is_advisory else "normal",
        "confidence_warning": warning_msg,
        "code_metrics": asdict(code_metrics),
        "smells": asdict(smell_feats),
        "refactoring_advice": refactor_advice,
        "model_loaded": model_loaded,
    }
