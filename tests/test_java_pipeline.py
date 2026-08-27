"""
SmellPredict — Java ML Training Pipeline & Extractor Test Suite
===============================================================
Comprehensive test cases for:
  - Java AST feature extraction (javalang + regex fallback)
  - Java code smell detectors (Long Method, Long Param List, Large Class, Deep Nesting, High Complexity)
  - Java mining configuration & test file exclusions
  - Java live API routing and health check
  - Java model training pipeline & inference predictor
  - Java live terminal monitor state parsing
"""

from pathlib import Path
import pytest
import yaml
from fastapi.testclient import TestClient

from smellpredict.features.java_extractor import (
    extract_java_metrics,
    extract_java_smells,
    _cyclomatic_complexity,
    _compute_max_nesting,
)
from smellpredict.models.java_predictor import (
    analyze_java_source_code,
    classify_java_risk_tier,
    compute_java_heuristic_risk,
)
from smellpredict.platform.api import app
from smellpredict.platform.java_monitor import parse_java_mining_state


JAVA_CLEAN_FIXTURE = """
package org.example;

import java.util.List;
import java.util.ArrayList;

public class OrderService {
    private final List<String> orders = new ArrayList<>();

    public void addOrder(String id) {
        if (id != null && !id.trim().isEmpty()) {
            orders.add(id);
        }
    }

    public int getOrderCount() {
        return orders.size();
    }
}
"""

JAVA_SMELLY_FIXTURE = """
package org.example;

import java.util.*;

public class GodOrderProcessor {
    // Large method with deep nesting and many parameters
    public void processComplexOrders(
        String customerId,
        String orderId,
        double amount,
        String currency,
        List<String> items,
        Map<String, Object> metadata,
        boolean isExpedited,
        String shippingAddress
    ) {
        if (customerId != null) {
            if (orderId != null) {
                if (amount > 0) {
                    if (items != null && !items.isEmpty()) {
                        if (currency != null && currency.equals("USD")) {
                            for (String item : items) {
                                if (item != null) {
                                    System.out.println("Processing item: " + item);
                                }
                            }
                        }
                    }
                }
            }
        }
        // Additional 55 lines to exceed 50 LOC method threshold
        int v01 = 1;
        int v02 = 2;
        int v03 = 3;
        int v04 = 4;
        int v05 = 5;
        int v06 = 6;
        int v07 = 7;
        int v08 = 8;
        int v09 = 9;
        int v10 = 10;
        int v11 = 11;
        int v12 = 12;
        int v13 = 13;
        int v14 = 14;
        int v15 = 15;
        int v16 = 16;
        int v17 = 17;
        int v18 = 18;
        int v19 = 19;
        int v20 = 20;
        int v21 = 21;
        int v22 = 22;
        int v23 = 23;
        int v24 = 24;
        int v25 = 25;
        int v26 = 26;
        int v27 = 27;
        int v28 = 28;
        int v29 = 29;
        int v30 = 30;
        int v31 = 31;
        int v32 = 32;
        int v33 = 33;
        int v34 = 34;
        int v35 = 35;
        int v36 = 36;
        int v37 = 37;
        int v38 = 38;
        int v39 = 39;
        int v40 = 40;
        int v41 = 41;
        int v42 = 42;
        int v43 = 43;
        int v44 = 44;
        int v45 = 45;
        int v46 = 46;
        int v47 = 47;
        int v48 = 48;
        int v49 = 49;
        int v50 = 50;
        int v51 = 51;
        int v52 = 52;
        int v53 = 53;
        int v54 = 54;
        int v55 = 55;
    }
}
"""


class TestJavaParserAndExtractor:
    def test_javalang_installed(self):
        import javalang
        assert javalang is not None

    def test_java_metrics_basic(self):
        metrics = extract_java_metrics(JAVA_CLEAN_FIXTURE)
        assert metrics["loc"] > 10
        assert metrics["sloc"] > 5
        assert metrics["class_count"] == 1
        assert metrics["function_count"] == 2
        assert metrics["import_count"] == 2
        assert metrics["parse_fallback"] is False

    def test_java_metrics_empty_file(self):
        metrics = extract_java_metrics("")
        assert metrics["loc"] == 0
        assert metrics["sloc"] == 0
        assert metrics["function_count"] == 0

    def test_java_metrics_syntax_error_fallback(self):
        invalid_java = "public class Broken { unclosed syntax ("
        metrics = extract_java_metrics(invalid_java)
        assert metrics["loc"] == 1
        assert metrics["parse_fallback"] is True

    def test_java_cyclomatic_complexity(self):
        cc = _cyclomatic_complexity("if (a && b) { for (int i=0; i<10; i++) { while(true) {} } }")
        assert cc >= 4

    def test_java_nesting_depth(self):
        depth = _compute_max_nesting(JAVA_SMELLY_FIXTURE)
        assert depth >= 5


class TestJavaSmellDetection:
    def test_java_smell_clean_code(self):
        metrics = extract_java_metrics(JAVA_CLEAN_FIXTURE)
        smells = extract_java_smells(JAVA_CLEAN_FIXTURE, metrics)
        assert smells["has_long_method"] == 0
        assert smells["has_long_param_list"] == 0
        assert smells["has_deep_nesting"] == 0
        assert smells["total_smells"] == 0

    def test_java_smell_long_method(self):
        metrics = extract_java_metrics(JAVA_SMELLY_FIXTURE)
        smells = extract_java_smells(JAVA_SMELLY_FIXTURE, metrics)
        assert smells["has_long_method"] == 1
        assert smells["long_method_count"] >= 1

    def test_java_smell_long_param_list(self):
        metrics = extract_java_metrics(JAVA_SMELLY_FIXTURE)
        smells = extract_java_smells(JAVA_SMELLY_FIXTURE, metrics)
        assert smells["has_long_param_list"] == 1
        assert smells["long_param_count"] >= 1

    def test_java_smell_deep_nesting(self):
        metrics = extract_java_metrics(JAVA_SMELLY_FIXTURE)
        smells = extract_java_smells(JAVA_SMELLY_FIXTURE, metrics)
        assert smells["has_deep_nesting"] == 1

    def test_java_smell_total_count(self):
        metrics = extract_java_metrics(JAVA_SMELLY_FIXTURE)
        smells = extract_java_smells(JAVA_SMELLY_FIXTURE, metrics)
        assert smells["total_smells"] >= 2


class TestJavaMiningConfigAndMiner:
    def test_java_mining_config_valid(self):
        cfg_path = Path("config/java_mining_config.yaml")
        assert cfg_path.exists()
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert "repositories" in cfg
        assert "tier1" in cfg["repositories"]
        assert "tier2" in cfg["repositories"]
        assert "tier3" in cfg["repositories"]

        total_repos = sum(len(r) for r in cfg["repositories"].values())
        assert total_repos >= 50

    def test_java_mining_exclude_tests(self):
        from smellpredict.mining.java_miner import _should_exclude_java
        assert _should_exclude_java("src/test/java/OrderTest.java") is True
        assert _should_exclude_java("src/test/java/TestOrder.java") is True
        assert _should_exclude_java("src/test/java/OrderIT.java") is True
        assert _should_exclude_java("target/generated-sources/Proto.java") is True
        assert _should_exclude_java("src/main/java/OrderService.java") is False


class TestJavaPredictorAndAPI:
    def test_java_predictor_clean(self):
        res = analyze_java_source_code(JAVA_CLEAN_FIXTURE, "OrderService.java")
        assert res["language"] == "java"
        assert res["language_badge"] == "☕ Java"
        assert res["risk_tier"] in ("Low", "Medium")
        assert res["risk_probability"] < 0.60

    def test_java_predictor_smelly(self):
        res = analyze_java_source_code(JAVA_SMELLY_FIXTURE, "GodOrderProcessor.java")
        assert res["language"] == "java"
        assert res["risk_tier"] in ("High", "Critical")
        assert res["risk_probability"] >= 0.50
        assert res["smells"]["total_smells"] >= 2

    def test_java_live_api_routing(self):
        client = TestClient(app)
        resp = client.post(
            "/api/v1/analyze/live",
            json={"content": JAVA_CLEAN_FIXTURE, "filename": "OrderService.java"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["language"] == "java"
        assert data["language_badge"] == "☕ Java"
        assert "risk" in data
        assert "metrics" in data
        assert "smells" in data

    def test_health_check_has_java_model_field(self):
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_loaded_java" in data

    def test_java_monitor_state_parsing(self):
        state = parse_java_mining_state()
        assert "total_targets" in state
        assert state["total_targets"] >= 50
        assert "tier_breakdown" in state
        assert "completed_count" in state


class TestJavaModelTraining:
    def test_synthetic_java_corpus_and_training(self, tmp_path):
        from smellpredict.models.java_trainer import run_java_training_pipeline
        out_dir = tmp_path / "models"
        res = run_java_training_pipeline(data_path=None, output_dir=out_dir, n_trials=5)
        assert res["pr_auc"] > 0.50
        assert (out_dir / "java_best_model.pkl").exists()
