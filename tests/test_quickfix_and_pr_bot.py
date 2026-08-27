"""
SmellPredict — Quick-Fix, GitHub Webhook & PR Bot Test Suite
=============================================================
Tests:
  - One-click AST & Polyglot quick-fix patch synthesis
  - Unified diff generation
  - Quick-fix REST API endpoint (/api/v1/refactor/quick-fix)
  - GitHub Webhook HMAC SHA256 signature verification
  - PR Review Bot engine and SARIF diagnostics
  - Real-time line annotations and team chat REST endpoints
"""

import hmac
import hashlib
import json
import pytest
from fastapi.testclient import TestClient

from smellpredict.features.refactor import (
    generate_diff,
    generate_quick_fix_patch,
)
from smellpredict.platform.webhook import verify_webhook_signature
from smellpredict.platform.pr_bot import analyze_pr_files, generate_pr_review_markdown
from smellpredict.platform.api import app

client = TestClient(app)


class TestQuickFixPatches:
    SAMPLE_PYTHON = """
def long_method_example():
    val = 1
    val += 2
    return val
"""

    SAMPLE_NESTED_PYTHON = """
def nested_check(x):
    if x > 0:
        return x * 2
    return 0
"""

    SAMPLE_JAVA = """
public class Handler {
    public void execute() {
        System.out.println("Processing");
    }
}
"""

    def test_extract_method_python(self):
        patch = generate_quick_fix_patch(
            self.SAMPLE_PYTHON,
            smell_type="LongMethod",
            line_number=2,
            language="python",
            target_name="long_method_example",
        )
        assert patch["applied"] is True
        assert "_helper_long_method_example" in patch["refactored_code"]
        assert len(patch["diff"]) > 0

    def test_guard_clause_python(self):
        patch = generate_quick_fix_patch(
            self.SAMPLE_NESTED_PYTHON,
            smell_type="DeepNesting",
            line_number=2,
            language="python",
        )
        assert patch["applied"] is True
        assert "guard clause" in patch["refactored_code"].lower()
        assert len(patch["diff"]) > 0

    def test_parameter_object_typescript(self):
        src = "export function setup(a: string, b: number, c: boolean, d: any, e: string) {}"
        patch = generate_quick_fix_patch(
            src,
            smell_type="LongParameterList",
            line_number=1,
            language="typescript",
            target_name="setup",
        )
        assert patch["applied"] is True
        assert "interface SetupOptions" in patch["refactored_code"]

    def test_quick_fix_api_endpoint(self):
        payload = {
            "content": "def do_work():\n    pass\n",
            "filename": "work.py",
            "smell_type": "LongMethod",
            "line_number": 1,
            "target_name": "do_work",
        }
        resp = client.post("/api/v1/refactor/quick-fix", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "diff" in data
        assert "refactored_code" in data
        assert data["applied"] is True


class TestWebhookHMACVerification:
    def test_hmac_valid_signature(self):
        secret = "secret_key_123"
        payload = b'{"action":"opened","pull_request":{"number":1}}'
        sig = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

        assert verify_webhook_signature(payload, sig, secret) is True

    def test_hmac_invalid_signature(self):
        secret = "secret_key_123"
        payload = b'{"action":"opened"}'
        sig = "sha256=invalid_hash_value_12345"

        assert verify_webhook_signature(payload, sig, secret) is False

    def test_webhook_ping_endpoint(self):
        resp = client.post("/github/webhook", json={"event": "ping"})
        assert resp.status_code == 200
        assert "Pong" in resp.json().get("message", "")


class TestPRReviewBotEngine:
    def test_analyze_pr_files_multi_lang(self):
        files_map = {
            "src/calculator.py": "def add(a, b):\n    return a + b\n",
            "src/Service.java": """
            public class Service {
                public void doAll(int a, int b, int c, int d, int e, int f) {
                    if (a > 0) { if (b > 0) { if (c > 0) { if (d > 0) { System.out.println("Deep"); } } } }
                }
            }
            """,
        }
        res = analyze_pr_files(files_map)
        assert res["analyzed_files_count"] == 2
        assert len(res["files"]) == 2
        assert "overall_risk_tier" in res
        assert "markdown_summary" in res
        assert "SmellPredict AI Code Review" in res["markdown_summary"]

    def test_generate_pr_review_markdown(self):
        files = [
            {
                "path": "app.py",
                "badge": "🐍 Python",
                "risk_icon": "🟢",
                "risk_tier": "Low",
                "risk_probability": 0.12,
                "total_smells": 0,
                "metrics": {"loc": 50},
            }
        ]
        md = generate_pr_review_markdown(
            avg_risk=0.12,
            overall_tier="Low",
            overall_icon="🟢",
            verdict="APPROVE",
            files=files,
        )
        assert "APPROVE" in md
        assert "app.py" in md
        assert "Low Risk" in md


class TestCollabCommentsAndChatEndpoints:
    def test_comments_crud_lifecycle(self):
        room_id = "test_room_123"

        # 1. Add comment
        add_resp = client.post(f"/rooms/{room_id}/comments", json={
            "line_number": 15,
            "text": "Refactor this method to use early return",
            "author": "Alice",
        })
        assert add_resp.status_code == 200
        comment = add_resp.json()
        assert comment["line_number"] == 15
        assert comment["author"] == "Alice"
        comment_id = comment["id"]

        # 2. Get comments
        get_resp = client.get(f"/rooms/{room_id}/comments")
        assert get_resp.status_code == 200
        comments = get_resp.json()["comments"]
        assert len(comments) >= 1

        # 3. Resolve comment
        res_resp = client.post(f"/rooms/{room_id}/comments/{comment_id}/resolve")
        assert res_resp.status_code == 200
        assert res_resp.json()["comment"]["resolved"] is True

        # 4. Delete comment
        del_resp = client.delete(f"/rooms/{room_id}/comments/{comment_id}")
        assert del_resp.status_code == 200

    def test_chat_messages(self):
        room_id = "chat_room_456"

        # 1. Post chat message
        post_resp = client.post(f"/rooms/{room_id}/chat", json={
            "text": "Hey team, check out the PR review on line 42",
            "author": "Bob",
        })
        assert post_resp.status_code == 200
        msg = post_resp.json()
        assert msg["text"] == "Hey team, check out the PR review on line 42"
        assert msg["author"] == "Bob"

        # 2. Get chat history
        get_resp = client.get(f"/rooms/{room_id}/chat")
        assert get_resp.status_code == 200
        messages = get_resp.json()["messages"]
        assert len(messages) >= 1
        assert any(m["author"] == "Bob" for m in messages)
