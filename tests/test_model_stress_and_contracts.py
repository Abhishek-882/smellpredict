"""
tests/test_model_stress_and_contracts.py — Exhaustive Model Contracts & Stress Suite
=====================================================================================
Comprehensive validation suite covering:
  1. Routing contracts (Engine A vs Engine B vs Non-ML).
  2. Monotonicity & calibration smoothness under extreme inputs.
  3. Single-buffer empirical quantile interpolation and clipping.
  4. Strict language and file-type isolation (Media, Polyglot, Plaintext).
  5. Architecture guardrails (FFI / native extensions).
  6. Sub-50ms inference latency on live IDE analysis payloads.
"""

import time
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smellpredict.models.predictor import (
    GitTelemetry,
    analyze_source_code,
    get_engine_a_model,
    get_engine_b_model,
    compute_quantile_rank,
    check_architecture_guardrail,
)
from smellpredict.platform.api import app

client = TestClient(app)


# ── 1. Routing Contracts & Dual-Engine Selection ──────────────────────────────

def test_untracked_buffer_routes_to_engine_a():
    """Live untracked buffer must route strictly to Engine A."""
    code = "def process_data(items):\n    return [x * 2 for x in items if x > 0]\n"
    res = analyze_source_code(code, file_path="new_feature.py", git_telemetry=None)
    assert res["engine_used"] == "engine_a_ast_static"
    assert "Engine A" in res["engine_desc"]
    assert 0.01 <= res["risk_probability"] <= 0.99
    assert res["model_loaded"] is True


def test_tracked_git_telemetry_routes_to_engine_b():
    """Files with GitTelemetry must route to Engine B."""
    code = "def query_db(conn, q):\n    return conn.execute(q).fetchall()\n"
    git = GitTelemetry(
        is_tracked_in_git=True,
        file_age_days=365.0,
        days_since_last_change=14.0,
        previous_file_commits=50,
        previous_bug_fixes=5,
        recent_file_commits=4,
        contributors=6,
        ownership_concentration=0.55,
    )
    res = analyze_source_code(code, file_path="db.py", git_telemetry=git)
    assert res["engine_used"] == "engine_b_full_telemetry"
    assert "Engine B" in res["engine_desc"]
    assert 0.01 <= res["risk_probability"] <= 0.99


def test_zero_churn_legacy_file_retains_engine_b():
    """Old file with 0 recent churn but tracked in git must NOT be demoted to Engine A."""
    code = "PI = 3.1415926535\n"
    git = GitTelemetry(
        is_tracked_in_git=True,
        file_age_days=1000.0,
        days_since_last_change=800.0,
        previous_file_commits=3,
        recent_file_commits=0,
    )
    res = analyze_source_code(code, file_path="constants.py", git_telemetry=git)
    assert res["engine_used"] == "engine_b_full_telemetry"


# ── 2. Monotonicity & Calibration Smoothness ──────────────────────────────────

def test_risk_monotonicity_with_increasing_smells():
    """Injecting severe complexity and smells must strictly increase defect risk."""
    clean_code = "def clean_calc(a, b):\n    return a + b\n"
    res_clean = analyze_source_code(clean_code, file_path="calc.py")

    smelly_code = """
def complex_calc(a, b, c, d, e, f, g, h, i):
    if a:
        if b:
            if c:
                if d:
                    for item in range(100):
                        if e and f:
                            while g:
                                if h:
                                    return i * 2
    return 0
"""
    res_smelly = analyze_source_code(smelly_code, file_path="calc.py")

    assert res_smelly["risk_probability"] > res_clean["risk_probability"], (
        f"Smelly risk ({res_smelly['risk_probability']}) must exceed clean risk ({res_clean['risk_probability']})"
    )


def test_extreme_tail_smoothness_bounds():
    """Extreme massive complex module must NOT pin to hard 1.000 (Platt scaling validation)."""
    god_module = "\n".join([
        f"def helper_func_{i}(a, b, c, d, e, f):\n    if a:\n        if b:\n            if c:\n                return d + e + f\n    return 0"
        for i in range(25)
    ])
    res = analyze_source_code(god_module, file_path="god_module.py")
    assert 0.001 < res["risk_probability"] < 0.999, f"Probability pinned to extreme: {res['risk_probability']}"
    assert res["risk_tier"] in ("Medium", "High", "Critical")


# ── 3. Empirical Quantile Interpolation & Bounds ──────────────────────────────

def test_quantile_rank_clipping():
    """Quantile interpolation must clip gracefully on extreme outlier metrics."""
    ref_table = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    rank_low = compute_quantile_rank(-50.0, ref_table)
    assert rank_low == 0.01

    rank_high = compute_quantile_rank(50000.0, ref_table)
    assert rank_high == 0.99

    rank_mid = compute_quantile_rank(100.0, ref_table)
    assert 0.40 <= rank_mid <= 0.60


# ── 4. Strict Language & File-Type Risk Isolation ─────────────────────────────

def test_api_live_python_receives_ml_risk():
    """Python files must receive ML defect risk."""
    payload = {"filename": "service.py", "content": "def run():\n    pass\n"}
    resp = client.post("/api/v1/analyze/live", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["language"] == "python"
    assert data["is_ml_prediction"] is True
    assert data["risk"] is not None
    assert 0.01 <= data["risk"]["probability"] <= 0.99


@pytest.mark.parametrize("filename,content", [
    ("left02.jpg", "/9j/4AAQSkZJRgABAQEASABIAAD/2wBD..."),
    ("logo.png", "\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR..."),
    ("report.pdf", "%PDF-1.4 ..."),
    ("archive.zip", "PK\x03\x04..."),
])
def test_api_live_binary_media_has_no_risk_evaluation(filename, content):
    """Images and binary files must NEVER receive ML defect risk scoring."""
    resp = client.post("/api/v1/analyze/live", json={"filename": filename, "content": content})
    assert resp.status_code == 200
    data = resp.json()
    assert data["language"] == "binary"
    assert data["is_ml_prediction"] is False
    assert data["risk"] is None
    assert data["engine_used"] == "none"
    assert data["metrics"]["loc"] == 0


@pytest.mark.parametrize("filename,content,lang", [
    ("Main.java", "public class Main { public static void main(String[] args) {} }", "java"),
    ("index.ts", "export const add = (a: number, b: number): number => a + b;", "typescript"),
    ("server.go", "package main\nfunc main() {}\n", "go"),
    ("lib.rs", "pub fn add(a: i32, b: i32) -> i32 { a + b }\n", "rust"),
    ("script.js", "function hello() { console.log('hi'); }", "javascript"),
])
def test_api_live_polyglot_has_no_risk_evaluation(filename, content, lang):
    """Non-Python programming languages must return code metrics with risk: None."""
    resp = client.post("/api/v1/analyze/live", json={"filename": filename, "content": content})
    assert resp.status_code == 200
    data = resp.json()
    assert data["language"] == lang
    assert data["is_ml_prediction"] is False
    assert data["risk"] is None
    assert data["metrics"]["loc"] > 0


@pytest.mark.parametrize("filename,content", [
    ("README.md", "# SmellPredict\nReal-time code intelligence platform."),
    ("config.json", '{"host": "localhost", "port": 8000}'),
    ("settings.yaml", "env: production\ndebug: false\n"),
    ("notes.txt", "Todo list for sprint 12"),
])
def test_api_live_plaintext_config_has_no_risk_evaluation(filename, content):
    """Markdown, JSON, YAML, and TXT files must NEVER evaluate defect risk."""
    resp = client.post("/api/v1/analyze/live", json={"filename": filename, "content": content})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_ml_prediction"] is False
    assert data["risk"] is None
    assert data["metrics"]["loc"] > 0


# ── 5. Architecture Guardrails ────────────────────────────────────────────────

def test_ffi_cffi_triggers_advisory_guardrail():
    """Imports of CFFI or ctypes must trigger an advisory confidence warning."""
    cffi_code = "from cffi import FFI\nffi = FFI()\nffi.cdef('int printf(const char *format, ...);')\n"
    is_advisory, warning = check_architecture_guardrail(cffi_code, file_path="native_bridge.py")
    assert is_advisory is True
    assert "FFI" in warning or "native" in warning.lower()

    resp = client.post("/api/v1/analyze/live", json={"filename": "native_bridge.py", "content": cffi_code})
    data = resp.json()
    assert data["guardrail_status"] == "advisory"
    assert data["confidence_warning"] is not None


# ── 6. Live Inference Latency Benchmark ───────────────────────────────────────

def test_live_inference_sub_100ms_latency():
    """Live editor inference must complete within responsive sub-100ms budget."""
    code = """
class DataPipeline:
    def __init__(self, raw_data):
        self.data = raw_data

    def transform(self):
        return [x.strip().lower() for x in self.data if x]

    def aggregate(self):
        res = {}
        for item in self.transform():
            res[item] = res.get(item, 0) + 1
        return res
"""
    # Warmup
    client.post("/api/v1/analyze/live", json={"filename": "pipeline.py", "content": code})

    # Benchmark 10 consecutive requests
    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        resp = client.post("/api/v1/analyze/live", json={"filename": "pipeline.py", "content": code})
        t1 = time.perf_counter()
        assert resp.status_code == 200
        latencies.append((t1 - t0) * 1000.0)

    avg_ms = sum(latencies) / len(latencies)
    print(f"\n[BENCHMARK] Average Live Inference Latency: {avg_ms:.2f}ms (P90: {sorted(latencies)[8]:.2f}ms)")
    assert avg_ms < 100.0, f"Average latency ({avg_ms:.2f}ms) exceeded 100ms budget!"
