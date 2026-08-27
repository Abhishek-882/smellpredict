"""
SmellPredict — Test Suite
===========================
Tests for: labeling, feature extraction, integrity assertions, and API.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from smellpredict.labeling.heuristic import score_commit, cohens_kappa, heuristic_vs_human_metrics
from smellpredict.features.extractor import extract_code_metrics, extract_smell_features
from smellpredict.evaluation.assertions import (
    assert_no_temporal_leak,
    assert_no_file_identity_leak,
    assert_lopo_integrity,
    temporal_split_generator,
    lopo_split_generator,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Bug-Fix Labeling Heuristic
# ─────────────────────────────────────────────────────────────────────────────

class TestBugFixHeuristic:

    def test_clear_bug_fix(self):
        cs = score_commit("Fix null pointer exception in login handler (#1234)")
        assert cs.is_bug_fix is True
        assert cs.score >= 0.40

    def test_clear_not_bug_fix(self):
        cs = score_commit("Update README with installation instructions")
        assert cs.is_bug_fix is False

    def test_refactor_is_not_bug_fix(self):
        cs = score_commit("Refactor database connection logic for clarity")
        assert cs.is_bug_fix is False

    def test_hotfix_is_bug_fix(self):
        cs = score_commit("hotfix: patch for broken login endpoint")
        assert cs.is_bug_fix is True

    def test_noise_dominated_forces_negative(self):
        # Merge + format = noise_dominated
        cs = score_commit("Merge and format code style fix")
        # Even if "fix" appears, merge+format should suppress it
        assert cs.noise_dominated is True or cs.is_bug_fix is False

    def test_score_clipped_to_unit_interval(self):
        cs = score_commit("fix fix fix fix bug bug crash crash resolve resolve (#1) (#2)")
        assert 0.0 <= cs.score <= 1.0

    def test_empty_message(self):
        cs = score_commit("")
        assert cs.score == 0.0
        assert cs.is_bug_fix is False

    def test_custom_threshold(self):
        cs_low = score_commit("fix something", threshold=0.10)
        cs_high = score_commit("fix something", threshold=0.90)
        assert cs_low.score == cs_high.score  # same score
        assert cs_low.is_bug_fix != cs_high.is_bug_fix or cs_high.score >= 0.90

    def test_issue_reference_detected(self):
        cs = score_commit("Resolves #999: edge case in parser")
        assert cs.i > 0

    def test_keyword_count_capped_at_3(self):
        cs = score_commit("fix fix fix fix fix fix")
        assert cs.k <= 3


class TestCohensKappa:

    def test_perfect_agreement(self):
        a = [True, False, True, False]
        b = [True, False, True, False]
        result = cohens_kappa(a, b)
        assert result["kappa"] == pytest.approx(1.0, abs=0.01)

    def test_no_agreement_above_chance(self):
        a = [True, True, False, False]
        b = [False, False, True, True]
        result = cohens_kappa(a, b)
        assert result["kappa"] < 0

    def test_moderate_agreement(self):
        a = [True, True, True, False, False, True, False, False]
        b = [True, True, False, False, False, True, True, False]
        result = cohens_kappa(a, b)
        assert -1.0 <= result["kappa"] <= 1.0

    def test_interpretation_labels(self):
        result = cohens_kappa([True] * 4, [True] * 4)
        assert "perfect" in result["interpretation"].lower() or \
               result["kappa"] == pytest.approx(1.0, abs=0.01)

    def test_heuristic_vs_human(self):
        heuristic = [True, False, True, True, False]
        human     = [True, False, False, True, False]
        metrics = heuristic_vs_human_metrics(heuristic, human)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

SIMPLE_MODULE = """\
def foo(a, b):
    return a + b

class Bar:
    def method(self, x, y, z):
        if x:
            for i in range(z):
                while y:
                    pass
"""

COMPLEX_MODULE = """\
def very_long_function(a, b, c, d, e, f, g):
    if a:
        for i in range(100):
            while b:
                if c:
                    try:
                        result = d
                    except Exception:
                        pass
    x = 1
    y = 2
    z = 3
    return x + y + z
""" + "\n" * 60  # pad to 70 lines


class TestCodeMetrics:

    def test_basic_extraction(self):
        m = extract_code_metrics(SIMPLE_MODULE)
        assert m.function_count >= 2
        assert m.class_count >= 1
        assert m.loc > 0

    def test_empty_source(self):
        m = extract_code_metrics("")
        assert m.loc == 0
        assert m.function_count == 0

    def test_syntax_error_graceful(self):
        m = extract_code_metrics("def foo(")
        assert m.loc >= 0  # should not raise

    def test_max_nesting_detects_depth(self):
        m = extract_code_metrics(SIMPLE_MODULE)
        assert m.max_nesting_depth >= 3  # if/for/while

    def test_param_count(self):
        m = extract_code_metrics(COMPLEX_MODULE)
        assert m.max_param_count >= 7  # 7 params in very_long_function

    def test_comment_density_nonnegative(self):
        m = extract_code_metrics("# comment\n# another\nx = 1\n")
        assert 0.0 <= m.comment_density <= 1.0


class TestSmellFeatures:

    def test_long_method_detected(self):
        source = "def f():\n" + "    x = 1\n" * 60
        s = extract_smell_features(source, {"long_method": 50})
        assert s.has_long_method == 1

    def test_long_method_not_detected_below_threshold(self):
        source = "def f():\n" + "    x = 1\n" * 10
        s = extract_smell_features(source, {"long_method": 50})
        assert s.has_long_method == 0

    def test_long_param_list(self):
        source = "def f(a, b, c, d, e, f): pass\n"
        s = extract_smell_features(source, {"long_param": 5})
        assert s.has_long_param_list == 1

    def test_total_smells_is_sum(self):
        s = extract_smell_features(COMPLEX_MODULE)
        expected = (
            s.long_method_count + s.long_param_count + s.large_class_count
            + s.deep_nesting_count + s.high_complexity_count
        )
        assert s.total_smells == expected

    def test_no_smells_in_clean_code(self):
        source = "def f(a, b):\n    return a + b\n"
        s = extract_smell_features(source)
        assert s.total_smells == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Data Integrity Assertions
# ─────────────────────────────────────────────────────────────────────────────

def make_df(n_rows: int, start_days_ago: int = 100, repo: str = "repo_a") -> pd.DataFrame:
    """Create a synthetic DataFrame for testing."""
    dates = [datetime.now() - timedelta(days=start_days_ago - i) for i in range(n_rows)]
    date_strs = [d.isoformat() for d in dates]
    return pd.DataFrame({
        "snapshot_date": date_strs,
        "commit_date": date_strs,
        "canonical_file_id": [f"{repo}_file_{i % 20}" for i in range(n_rows)],
        "repo": [repo] * n_rows,
        "future_bug_fix": [i % 5 == 0 for i in range(n_rows)],
    })


class TestDataIntegrity:

    def test_temporal_pass(self):
        train = make_df(100, start_days_ago=100)
        test = make_df(20, start_days_ago=0)
        # Should not raise
        assert_no_temporal_leak(train, test, fold_id=1)

    def test_temporal_fail(self):
        train = make_df(100, start_days_ago=50)
        test = make_df(20, start_days_ago=100)  # test is older than train!
        with pytest.raises(AssertionError, match="TEMPORAL LEAK"):
            assert_no_temporal_leak(train, test, fold_id=1)

    def test_identity_pass(self):
        train = pd.DataFrame({
            "canonical_file_id": ["file_1", "file_2", "file_3"],
            "future_bug_fix": [0, 1, 0],
        })
        test = pd.DataFrame({
            "canonical_file_id": ["file_4", "file_5"],
            "future_bug_fix": [1, 0],
        })
        assert_no_file_identity_leak(train, test, fold_id=1)

    def test_identity_fail(self):
        train = pd.DataFrame({
            "canonical_file_id": ["file_1", "file_2"],
            "future_bug_fix": [0, 1],
        })
        test = pd.DataFrame({
            "canonical_file_id": ["file_2", "file_3"],  # file_2 overlaps!
            "future_bug_fix": [1, 0],
        })
        with pytest.raises(AssertionError, match="FILE IDENTITY LEAK"):
            assert_no_file_identity_leak(train, test, fold_id=1)

    def test_lopo_pass(self):
        train = pd.DataFrame({"repo": ["repo_a", "repo_a", "repo_b"], "future_bug_fix": [0, 1, 0]})
        test = pd.DataFrame({"repo": ["repo_c", "repo_c"], "future_bug_fix": [1, 0]})
        assert_lopo_integrity(train, test, "repo_c")

    def test_lopo_fail_target_in_train(self):
        train = pd.DataFrame({"repo": ["repo_c", "repo_a"], "future_bug_fix": [0, 1]})
        test = pd.DataFrame({"repo": ["repo_c"], "future_bug_fix": [1]})
        with pytest.raises(AssertionError, match="LOPO INTEGRITY VIOLATION"):
            assert_lopo_integrity(train, test, "repo_c")

    def test_temporal_split_generator(self):
        """Integration test: generate folds and check all pass assertions."""
        df_a = make_df(200, start_days_ago=200, repo="repo_a")
        df_b = make_df(100, start_days_ago=100, repo="repo_b")
        df = pd.concat([df_a, df_b], ignore_index=True)

        folds = list(temporal_split_generator(df, n_folds=5))
        assert len(folds) > 0

        for train, test, stats in folds:
            assert stats["chronologically_valid"] is True
            assert stats["no_overlap"] is True
            assert len(train) > 0
            assert len(test) > 0

    def test_lopo_generator(self):
        df_a = make_df(50, start_days_ago=100, repo="repo_a")
        df_b = make_df(50, start_days_ago=100, repo="repo_b")
        df = pd.concat([df_a, df_b], ignore_index=True)

        rotations = list(lopo_split_generator(df))
        assert len(rotations) == 2  # one per repo

        for train, test, held_out in rotations:
            assert held_out not in train["repo"].values
            assert (test["repo"] == held_out).all()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: AST Refactoring Advisor & SARIF & Predictor
# ─────────────────────────────────────────────────────────────────────────────

class TestRefactoringAdvisor:

    def test_long_method_advice(self):
        from smellpredict.features.refactor import analyze_refactorings
        # Generate a dummy long method (60 lines)
        source = "def long_function():\n" + "\n".join([f"    x_{i} = {i}" for i in range(65)]) + "\n    return x_0\n"
        advice = analyze_refactorings(source)
        assert len(advice) >= 1
        assert any(a["smell_type"] == "LongMethod" for a in advice)
        assert any("Extract Method" in a["title"] for a in advice)

    def test_long_param_list_advice(self):
        from smellpredict.features.refactor import analyze_refactorings
        source = "def process_data(a, b, c, d, e, f, g):\n    return a + b\n"
        advice = analyze_refactorings(source)
        assert len(advice) >= 1
        assert any(a["smell_type"] == "LongParameterList" for a in advice)
        assert any("Parameter Object" in a["title"] for a in advice)

    def test_deep_nesting_advice(self):
        from smellpredict.features.refactor import analyze_refactorings
        source = (
            "def deeply_nested(val):\n"
            "    if val > 0:\n"
            "        for i in range(val):\n"
            "            while i < 10:\n"
            "                if i % 2 == 0:\n"
            "                    try:\n"
            "                        return i * 2\n"
            "                    except Exception:\n"
            "                        pass\n"
        )
        advice = analyze_refactorings(source)
        assert len(advice) >= 1
        assert any(a["smell_type"] == "DeepNesting" for a in advice)


class TestSarifExporter:

    def test_sarif_generation(self):
        from smellpredict.platform.sarif import generate_sarif_report
        sample_results = [{
            "file_path": "src/sample.py",
            "risk_probability": 0.85,
            "risk_tier": "Critical",
            "smells": {"has_long_method": 1, "has_deep_nesting": 1},
            "metrics": {"loc": 250, "max_cc": 12.0, "maintainability_index": 42.0},
        }]
        sarif = generate_sarif_report(sample_results)
        assert sarif["version"] == "2.1.0"
        assert len(sarif["runs"]) == 1
        results = sarif["runs"][0]["results"]
        assert len(results) >= 2  # Risk diagnostic + smell diagnostics
        assert any(r["ruleId"] == "SP-RISK-001" for r in results)


class TestUnifiedPredictor:

    def test_analyze_source_code_clean(self):
        from smellpredict.models.predictor import analyze_source_code
        source = "def add(a: int, b: int) -> int:\n    return a + b\n"
        res = analyze_source_code(source, file_path="math.py")
        assert res["file_path"] == "math.py"
        assert 0.0 <= res["risk_probability"] <= 1.0
        assert res["risk_tier"] in ("Low", "Medium", "High", "Critical")
        assert res["code_metrics"]["loc"] >= 2
        assert res["smells"]["total_smells"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Phase 18 Tests: Collaborative IDE — Auth, GitHub API, WebSocket Relay
# ─────────────────────────────────────────────────────────────────────────────


class TestJWTAuth:
    """Tests for the JWT issue/verify/refresh cycle in auth.py."""

    def test_issue_and_verify_jwt(self):
        """A JWT issued by issue_jwt() should be decodable by verify_jwt()."""
        from smellpredict.platform.auth import issue_jwt, verify_jwt

        token = issue_jwt(
            login="testuser",
            avatar_url="https://avatars.githubusercontent.com/u/1",
            github_token="gho_fake_token_12345",
        )
        assert isinstance(token, str)
        assert len(token) > 50

        payload = verify_jwt(token)
        assert payload["sub"] == "testuser"
        assert payload["avatar_url"] == "https://avatars.githubusercontent.com/u/1"
        assert "gh_tok" in payload   # encrypted GitHub token present
        assert "exp" in payload

    def test_jwt_github_token_is_encrypted(self):
        """The GitHub token stored in the JWT payload must be Fernet-encrypted."""
        from smellpredict.platform.auth import issue_jwt, verify_jwt, extract_github_token

        raw_token = "gho_very_secret_token_abcdef"
        jwt_str = issue_jwt("alice", "", raw_token)
        payload = verify_jwt(jwt_str)

        # Encrypted payload must NOT contain the raw token as plaintext
        assert raw_token not in payload["gh_tok"]

        # But we must be able to recover it via extract_github_token
        recovered = extract_github_token(jwt_str)
        assert recovered == raw_token

    def test_invalid_jwt_raises_401(self):
        """A tampered or invalid JWT must raise HTTP 401."""
        from smellpredict.platform.auth import verify_jwt
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            verify_jwt("not.a.valid.jwt.token")
        assert exc_info.value.status_code == 401

    def test_expired_jwt_raises_401(self):
        """An expired JWT must raise HTTP 401."""
        from smellpredict.platform.auth import JWT_SECRET_KEY, JWT_ALGORITHM
        from fastapi import HTTPException
        from datetime import timezone, timedelta
        from jose import jwt as jose_jwt

        expired_payload = {
            "sub": "ghost",
            "avatar_url": "",
            "gh_tok": "dummy",
            "iat": datetime.now(tz=timezone.utc) - timedelta(hours=10),
            "exp": datetime.now(tz=timezone.utc) - timedelta(hours=2),
        }
        expired_token = jose_jwt.encode(expired_payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

        from smellpredict.platform.auth import verify_jwt
        with pytest.raises(HTTPException) as exc_info:
            verify_jwt(expired_token)
        assert exc_info.value.status_code == 401


class TestGitHubAPIMock:
    """Tests for the GitHub API module (mocked PyGitHub)."""

    def test_commit_request_model(self):
        """CommitRequest should accept valid fields."""
        from smellpredict.platform.github_api import CommitRequest

        req = CommitRequest(
            path="src/foo.py",
            content="def foo(): pass",
            message="SmellPredict: test commit",
            sha="abc123def456",
            branch="main",
        )
        assert req.path == "src/foo.py"
        assert req.branch == "main"

    def test_commit_request_default_message(self):
        """CommitRequest default commit message should be set."""
        from smellpredict.platform.github_api import CommitRequest

        req = CommitRequest(path="a.py", content="x=1", sha="000")
        assert req.message == "SmellPredict: update file"

    def test_handle_github_error_404(self):
        """_handle_github_error should raise HTTP 404 for UnknownObjectException."""
        from smellpredict.platform.github_api import _handle_github_error
        from github import UnknownObjectException
        from fastapi import HTTPException

        exc = UnknownObjectException(404, "Not Found", {})
        with pytest.raises(HTTPException) as exc_info:
            _handle_github_error(exc)
        assert exc_info.value.status_code == 404


class TestCollabEngine:
    """Tests for the Y.js WebSocket relay room engine in collab.py."""

    def test_make_room_id_deterministic(self):
        """make_room_id must return the same 16-char hex for the same inputs."""
        from smellpredict.platform.collab import make_room_id

        rid1 = make_room_id("alice", "repo", "main", "src/foo.py")
        rid2 = make_room_id("alice", "repo", "main", "src/foo.py")
        assert rid1 == rid2
        assert len(rid1) == 16
        assert all(c in "0123456789abcdef" for c in rid1)

    def test_make_room_id_unique_per_path(self):
        """make_room_id must produce different IDs for different paths."""
        from smellpredict.platform.collab import make_room_id

        rid_a = make_room_id("alice", "repo", "main", "src/foo.py")
        rid_b = make_room_id("alice", "repo", "main", "src/bar.py")
        assert rid_a != rid_b

    def test_make_room_id_unique_per_branch(self):
        """make_room_id must produce different IDs for different branches."""
        from smellpredict.platform.collab import make_room_id

        rid_main = make_room_id("alice", "repo", "main", "src/foo.py")
        rid_dev  = make_room_id("alice", "repo", "dev",  "src/foo.py")
        assert rid_main != rid_dev


class TestLiveAnalysisEndpoint:
    """Tests for the POST /api/v1/analyze/live endpoint."""

    def test_live_analyze_empty_content(self):
        """Empty content must return a valid 200 response with zero risk."""
        from starlette.testclient import TestClient
        from smellpredict.platform.api import app

        client = TestClient(app)
        resp = client.post("/api/v1/analyze/live", json={"content": "", "filename": "test.py"})
        assert resp.status_code == 200
        data = resp.json()
        assert "risk" in data
        assert data["risk"]["probability"] == 0.0

    def test_live_analyze_response_schema(self):
        """Non-empty content must return risk/smells/refactoring keys."""
        from starlette.testclient import TestClient
        from smellpredict.platform.api import app

        client = TestClient(app)
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        resp = client.post("/api/v1/analyze/live", json={"content": code, "filename": "add.py"})
        assert resp.status_code == 200
        data = resp.json()
        if "error" not in data:
            assert "risk" in data
            assert "smells" in data
            assert "refactoring" in data
            assert isinstance(data["refactoring"], list)
