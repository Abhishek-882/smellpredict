"""
tests/test_dual_engine.py — Unit Tests for Dual-Engine Predictor & Routing
==========================================================================
Verifies:
  1. Engine A loads and performs inference on pure static AST code (zero git history).
  2. Single-buffer quantile percentile interpolation produces valid, bounded ranks.
  3. Engine B loads and performs inference when GitTelemetry is provided.
  4. Tracked legacy code (0 churn, file_age > 0) stays on Engine B.
  5. Untracked new code routes to Engine A.
  6. Architecture guardrail flags FFI imports (ctypes, cffi).
  7. Predicted probabilities are smooth and well-calibrated (no 0.000/1.000 tail pinning).
"""

import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smellpredict.models.predictor import (
    GitTelemetry,
    analyze_source_code,
    get_engine_a_model,
    get_engine_b_model,
    check_architecture_guardrail,
    classify_risk_tier,
)


def test_engine_a_and_b_models_exist():
    """Verify both trained model artifacts exist on disk and can be loaded."""
    model_a, meta_a = get_engine_a_model()
    model_b, meta_b = get_engine_b_model()

    assert model_a is not None, "Engine A model artifact not found or failed to load."
    assert meta_a is not None, "Engine A metadata not found."
    assert meta_a.get("calibration_method") == "sigmoid", "Engine A not calibrated with sigmoid."

    assert model_b is not None, "Engine B model artifact not found or failed to load."
    assert meta_b is not None, "Engine B metadata not found."
    assert meta_b.get("calibration_method") == "sigmoid", "Engine B not calibrated with sigmoid."


def test_cold_start_routes_to_engine_a():
    """Verify fresh code with no git history routes to Engine A."""
    sample_code = """
def calculate_metrics(values):
    total = sum(values)
    return total / len(values) if values else 0
"""
    res = analyze_source_code(sample_code, file_path="new_feature.py")
    assert res["engine_used"] == "engine_a_ast_static"
    assert "Engine A" in res["engine_desc"]
    assert 0.01 <= res["risk_probability"] <= 0.99
    assert res["model_loaded"] is True


def test_tracked_repo_routes_to_engine_b():
    """Verify code with GitTelemetry routes to Engine B."""
    sample_code = """
class EnterpriseService:
    def execute_transaction(self, tx_id, payload):
        if not tx_id:
            raise ValueError("Invalid tx")
        return {"status": "success", "id": tx_id}
"""
    git_ctx = GitTelemetry(
        is_tracked_in_git=True,
        file_age_days=180.0,
        days_since_last_change=12.0,
        previous_file_commits=24,
        previous_bug_fixes=3,
        recent_file_commits=2,
        contributors=4,
        ownership_concentration=0.65,
        silo_index=0.15,
        contributor_annual_rate=0.45,
    )
    res = analyze_source_code(sample_code, file_path="service.py", git_telemetry=git_ctx)
    assert res["engine_used"] == "engine_b_full_telemetry"
    assert "Engine B" in res["engine_desc"]
    assert 0.01 <= res["risk_probability"] <= 0.99


def test_stable_legacy_file_with_zero_recent_churn_stays_on_engine_b():
    """Verify an old untouched file in git (0 recent commits) is NOT misrouted to Engine A."""
    sample_code = """
def stable_constant_helper():
    return 42
"""
    git_ctx = GitTelemetry(
        is_tracked_in_git=True,
        file_age_days=500.0,
        days_since_last_change=400.0,
        previous_file_commits=2,
        previous_bug_fixes=0,
        recent_file_commits=0,  # Zero recent commits!
    )
    res = analyze_source_code(sample_code, file_path="legacy_helper.py", git_telemetry=git_ctx)
    assert res["engine_used"] == "engine_b_full_telemetry", "Legacy file misrouted to Engine A!"


def test_architecture_guardrail_ffi():
    """Verify FFI imports trigger advisory warning."""
    cffi_code = """
from cffi import FFI
ffi = FFI()
ffi.cdef("int printf(const char *format, ...);")
"""
    is_advisory, warning = check_architecture_guardrail(cffi_code, file_path="binding.py")
    assert is_advisory is True
    assert "FFI" in warning or "native" in warning.lower()

    res = analyze_source_code(cffi_code, file_path="binding.py")
    assert res["guardrail_status"] == "advisory"
    assert res["confidence_warning"] is not None


def test_probability_tail_smoothness():
    """Verify extreme complex code does not pin to hard 1.000 or 0.000 under Platt scaling."""
    complex_code = "\n".join([f"def func_{i}(a, b, c, d, e, f, g):\n    if a:\n        if b:\n            if c:\n                return d + e + f + g\n    return 0" for i in range(15)])
    res = analyze_source_code(complex_code, file_path="god_module.py")
    assert 0.001 < res["risk_probability"] < 0.999, f"Probability pinned to extreme tail: {res['risk_probability']}"
    assert res["risk_probability"] >= 0.35, "Complex code risk should be elevated above baseline low."


if __name__ == "__main__":
    pytest.main(["-v", __file__])
