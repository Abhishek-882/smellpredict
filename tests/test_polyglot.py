"""
SmellPredict — Comprehensive Polyglot Multi-Language Tests
===========================================================
Tests AST and heuristic extraction across:
  - Java, Kotlin, JavaScript, TypeScript, Go, Rust, C++, Ruby
  - Live polyglot routing in FastAPI backend
"""

import pytest
from fastapi.testclient import TestClient

from smellpredict.features.polyglot import (
    polyglot_analyze,
    extract_polyglot_metrics,
    detect_polyglot_smells,
    EXTENSION_MAP,
    LANGUAGE_BADGES,
)
from smellpredict.platform.api import app

client = TestClient(app)


class TestPolyglotExtensionMapping:
    def test_extension_detection(self):
        assert EXTENSION_MAP[".java"] == "java"
        assert EXTENSION_MAP[".kt"] == "kotlin"
        assert EXTENSION_MAP[".ts"] == "typescript"
        assert EXTENSION_MAP[".js"] == "javascript"
        assert EXTENSION_MAP[".go"] == "go"
        assert EXTENSION_MAP[".rs"] == "rust"
        assert EXTENSION_MAP[".cpp"] == "cpp"
        assert EXTENSION_MAP[".rb"] == "ruby"

    def test_badges_defined(self):
        for lang in ("python", "java", "kotlin", "javascript", "typescript", "go", "rust"):
            assert lang in LANGUAGE_BADGES
            assert len(LANGUAGE_BADGES[lang]) > 2


class TestJavaSmellDetection:
    JAVA_LONG_METHOD = """
    package com.example;
    public class OrderProcessor {
        public void processOrder(String a, int b, double c, boolean d, String e, String f) {
            // Complex logic
            if (a != null) {
                if (b > 0) {
                    for (int i = 0; i < b; i++) {
                        if (c > 100.0) {
                            System.out.println("High value item: " + i);
                        }
                    }
                }
            }
        }
    }
    """

    def test_java_analysis(self):
        res = polyglot_analyze(self.JAVA_LONG_METHOD, file_path="OrderProcessor.java")
        assert res.language == "java"
        assert res.metrics.loc > 10
        assert res.metrics.max_nesting_depth >= 4
        assert res.metrics.max_param_count >= 5
        assert res.smells.has_deep_nesting == 1
        assert res.smells.has_long_param_list == 1
        assert res.is_ml_prediction is False
        assert res.risk_probability == 0.0


class TestKotlinSmellDetection:
    KOTLIN_CODE = """
    package com.example
    class UserService {
        suspend fun authenticate(user: String, pass: String, token: String, ip: String, deviceId: String, session: String) {
            if (user.isNotEmpty()) {
                if (pass.length > 8) {
                    while (true) {
                        if (token == "valid") {
                            println("Authenticated")
                        }
                    }
                }
            }
        }
    }
    """

    def test_kotlin_analysis(self):
        res = polyglot_analyze(self.KOTLIN_CODE, file_path="UserService.kt")
        assert res.language == "kotlin"
        assert res.metrics.max_nesting_depth >= 4
        assert res.metrics.max_param_count >= 5
        assert res.smells.has_deep_nesting == 1
        assert res.smells.has_long_param_list == 1


class TestTypeScriptJavaScriptSmellDetection:
    TS_CODE = """
    export class DataPipeline {
        public async transform(raw: any, schema: any, validate: boolean, mode: string, opt: any, callback: any) {
            if (raw) {
                if (schema) {
                    for (const key of Object.keys(raw)) {
                        if (key.startsWith("user_")) {
                            console.log("Transforming user key:", key);
                        }
                    }
                }
            }
        }
    }
    """

    def test_typescript_analysis(self):
        res = polyglot_analyze(self.TS_CODE, file_path="pipeline.ts")
        assert res.language == "typescript"
        assert res.metrics.max_nesting_depth >= 4
        assert res.metrics.max_param_count >= 5
        assert res.smells.total_smells >= 2


class TestGoSmellDetection:
    GO_CODE = """
    package engine
    import "fmt"

    func ComputeEngine(p1 string, p2 int, p3 bool, p4 float64, p5 []string, p6 map[string]string) error {
        if p1 != "" {
            if p2 > 0 {
                for _, s := range p5 {
                    if s == "run" {
                        fmt.Println("Running...")
                    }
                }
            }
        }
        return nil
    }
    """

    def test_go_analysis(self):
        res = polyglot_analyze(self.GO_CODE, file_path="engine.go")
        assert res.language == "go"
        assert res.metrics.max_nesting_depth >= 4
        assert res.metrics.max_param_count >= 5
        assert res.smells.has_deep_nesting == 1


class TestRustSmellDetection:
    RUST_CODE = """
    pub struct ComputeCluster;

    impl ComputeCluster {
        pub fn execute_job(&self, job_id: &str, priority: u32, timeout: u64, flags: u8, nodes: Vec<String>, dry_run: bool) {
            if !job_id.is_empty() {
                if priority > 0 {
                    for node in nodes {
                        if timeout > 1000 {
                            println!("Processing on node {}", node);
                        }
                    }
                }
            }
        }
    }
    """

    def test_rust_analysis(self):
        res = polyglot_analyze(self.RUST_CODE, file_path="cluster.rs")
        assert res.language == "rust"
        assert res.metrics.max_nesting_depth >= 4
        assert res.smells.has_deep_nesting == 1


class TestLiveApiPolyglotRouting:
    def test_live_analyze_java_file(self):
        payload = {
            "content": "public class Calculator { public int add(int a, int b) { return a + b; } }",
            "filename": "Calculator.java"
        }
        response = client.post("/api/v1/analyze/live", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "java"
        assert "☕ Java" in data.get("language_badge", "")
        assert "risk" in data
        assert "metrics" in data

    def test_live_analyze_typescript_file(self):
        payload = {
            "content": "export function greet(name: string): string { return `Hello ${name}`; }",
            "filename": "greeter.ts"
        }
        response = client.post("/api/v1/analyze/live", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "typescript"
        assert "🔷 TypeScript" in data.get("language_badge", "")
        assert data["is_ml_prediction"] is False
        assert data["risk"] is None

    def test_live_analyze_python_file(self):
        payload = {
            "content": "def add(a, b):\n    return a + b\n",
            "filename": "math_helper.py"
        }
        response = client.post("/api/v1/analyze/live", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "python"
        assert "🐍 Python" in data.get("language_badge", "")
