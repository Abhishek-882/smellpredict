"""
SmellPredict — Streamlit Dashboard
=====================================
Interactive platform frontend providing:
  - 📁 File Upload Analyzer: upload any .py file, get risk score + explanations
  - 🔬 Commit Labeler: test the bug-fix heuristic interactively
  - 📊 Dataset Explorer: view mining statistics and smell prevalence
  - 🧪 Sensitivity Analysis: see how thresholds affect predictions
  - 📖 Research Explorer: key findings from the paper
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SmellPredict — Code Bug Risk Predictor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .risk-card {
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .risk-low    { background: #d4edda; border: 2px solid #28a745; }
    .risk-medium { background: #fff3cd; border: 2px solid #ffc107; }
    .risk-high   { background: #ffe5cc; border: 2px solid #fd7e14; }
    .risk-critical { background: #f8d7da; border: 2px solid #dc3545; }
    .metric-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .smell-detected { color: #dc3545; font-weight: 600; }
    .smell-clear    { color: #28a745; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# API Config
# ─────────────────────────────────────────────────────────────────────────────

API_URL = st.sidebar.text_input(
    "API URL",
    value="http://localhost:8000",
    help="URL of the SmellPredict FastAPI backend",
)


def api_post(endpoint: str, **kwargs):
    """Safe POST to the API."""
    try:
        r = requests.post(f"{API_URL}{endpoint}", **kwargs, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_get(endpoint: str, **kwargs):
    """Safe GET from the API."""
    try:
        r = requests.get(f"{API_URL}{endpoint}", **kwargs, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    [
        "⚡ Live Mining Progress",
        "🏠 Home",
        "📁 File Analyzer",
        "🔬 Commit Labeler",
        "📊 Dataset Explorer",
        "🧪 Sensitivity Analysis",
        "📖 Research Findings",
    ],
)

# API health check
health = api_get("/health")
if health:
    model_status = "✅ Real Model Loaded" if health.get("model_loaded") else "ℹ️ Awaiting Real Training (AST Analyzer Active)"
    st.sidebar.success(f"API: Online | {model_status}")
else:
    st.sidebar.warning("⚠️ API offline — Start the FastAPI server")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Live Mining Progress
# ─────────────────────────────────────────────────────────────────────────────

if page == "⚡ Live Mining Progress":
    st.header("⚡ Live Background Repository Mining Progress")
    st.markdown("Real-time telemetry showing snapshot extraction from GitHub repositories.")

    from smellpredict.platform.monitor import parse_mining_state
    state = parse_mining_state()

    total_target = state["total_targets"]
    completed_count = state["completed_count"]
    pct = completed_count / max(total_target, 1)

    # Top KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Repositories", f"{completed_count} / {total_target}", f"{pct * 100:.1f}%")
    with col2:
        st.metric("Snapshots Extracted", f"{state['total_snapshots']:,} rows")
    with col3:
        st.metric("Active Repository Target", f"#{completed_count + 1} {state['active_repo']}")
    with col4:
        st.metric("Storage Format", "Parquet + DuckDB")

    st.markdown("### 📊 Overall Pipeline Progress")
    st.progress(pct, text=f"{completed_count} of {total_target} Repositories Completed ({pct * 100:.1f}%)")

    # Completed Repositories Table & Chart
    if state["completed_repos"]:
        df_completed = pd.DataFrame(state["completed_repos"])
        df_completed.columns = ["Repository", "Snapshots", "Bug Fixes", "Bug-Fix %", "Size (KB)"]

        col_t, col_c = st.columns([3, 2])
        with col_t:
            st.markdown("#### 📁 Completed Repository Datasets (`data/raw/`)")
            st.dataframe(df_completed, width="stretch")

        with col_c:
            st.markdown("#### 📈 Snapshots per Repository")
            fig = px.bar(
                df_completed,
                x="Repository",
                y="Snapshots",
                color="Bug-Fix %",
                color_continuous_scale="Viridis",
                title="Extracted Rows by Repository",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Live Log viewer
    st.markdown("### 📜 Real-Time Log Stream (`logs/mining.log`)")
    log_text = "\n".join(state["latest_logs"])
    st.code(log_text if log_text else "Waiting for next event...", language="log")

    if st.button("🔄 Refresh Data Now"):
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Page: Home
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🏠 Home":
    st.markdown('<div class="main-header">🔍 SmellPredict</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Code Smell × Bug Prediction — '
        'Do code smells improve future bug-fix prediction?</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Target Repositories", "50+")
    with col2:
        st.metric("Target File Snapshots", "200K+")
    with col3:
        st.metric("Models Trained", "5+")
    with col4:
        st.metric("Smell Detectors", "8+")

    st.markdown("---")
    st.markdown("### 🎯 What is SmellPredict?")
    st.markdown("""
    SmellPredict is a research platform that:

    1. **Mines** Python repositories to extract code metrics, smells, and history features
    2. **Trains** defect prediction models using temporally-valid splits
    3. **Evaluates** whether code smells (Long Method, Deep Nesting, etc.) improve bug prediction
    4. **Explains** predictions using SHAP and LIME interpretability methods
    5. **Calibrates** risk scores into meaningful probabilities

    ### 🧠 Central Research Question
    > *"After controlling for code and history metrics, do code smells provide additional predictive power for future bug fixes?"*

    Based on [the paper](https://arxiv.org/), we find that code smells have a **small, direction-inconsistent effect** 
    — suggesting traditional metrics already capture most of the predictive signal.

    ### 🚀 Quick Start
    Use the sidebar to navigate to **📁 File Analyzer** to analyze any Python file,
    or **🔬 Commit Labeler** to test the bug-fix heuristic.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Page: File Analyzer
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📁 File Analyzer":
    st.header("📁 Python File Analyzer")
    st.markdown("Upload a `.py` file to get bug risk prediction, smell analysis, and code metrics.")

    col_upload, col_config = st.columns([2, 1])

    with col_config:
        model_choice = st.selectbox(
            "Prediction Model",
            ["xgboost", "random_forest", "logistic_regression"],
        )
        st.markdown("**Smell Thresholds**")
        t_long_method = st.slider("Long Method (lines)", 20, 100, 50, step=10)
        t_long_param = st.slider("Long Param List (params)", 2, 10, 5)
        t_nesting = st.slider("Deep Nesting (levels)", 2, 8, 4)

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Python file",
            type=["py"],
            help="Upload any Python .py file to analyze",
        )

    if uploaded_file is not None:
        with st.spinner("Analyzing file..."):
            result = api_post(
                "/api/v1/analyze/file",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/x-python")},
                params={"model_name": model_choice},
            )

        if result:
            st.markdown("---")
            risk = result.get("risk", {})
            tier = risk.get("tier", "Unknown")
            prob = risk.get("probability", 0.0)
            icon = risk.get("icon", "⚪")

            # Risk Display
            tier_class = {
                "Low": "risk-low", "Medium": "risk-medium",
                "High": "risk-high", "Critical": "risk-critical"
            }.get(tier, "metric-box")

            st.markdown(
                f'<div class="risk-card {tier_class}">'
                f'<h1>{icon} {tier} Risk</h1>'
                f'<h2>Bug-Fix Probability: {prob:.1%}</h2>'
                f'<p>{risk.get("recommendation", "")}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Metrics columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### 📐 Code Metrics")
                cm = result.get("code_metrics", {})
                st.metric("Lines of Code", cm.get("loc", "—"))
                st.metric("Functions", cm.get("function_count", "—"))
                st.metric("Max Cyclomatic Complexity", cm.get("max_cyclomatic_complexity", "—"))
                st.metric("Max Nesting Depth", cm.get("max_nesting_depth", "—"))
                st.metric("Cognitive Complexity", cm.get("cognitive_complexity", "—"))

            with col2:
                st.markdown("#### 👃 Code Smells")
                smells = result.get("smells", {})
                smell_items = [
                    ("Long Method", smells.get("has_long_method", 0)),
                    ("Long Param List", smells.get("has_long_param_list", 0)),
                    ("Large Class", smells.get("has_large_class", 0)),
                    ("Deep Nesting", smells.get("has_deep_nesting", 0)),
                    ("High Complexity", smells.get("has_high_complexity", 0)),
                ]
                for name, detected in smell_items:
                    badge = "🔴 Detected" if detected else "🟢 Clear"
                    st.markdown(f"**{name}**: {badge}")

                st.metric("Total Smells", smells.get("total_smells", 0))

            with col3:
                st.markdown("#### 📊 Quality Indicators")
                st.metric("Maintainability Index", f"{cm.get('maintainability_index', 0):.1f}/100")
                st.metric("Halstead Bug Estimate", f"{cm.get('halstead_bugs', 0):.4f}")
                st.metric("Classes", cm.get("class_count", "—"))

            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Bug-Fix Risk (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1f77b4"},
                    "steps": [
                        {"range": [0, 20], "color": "#d4edda"},
                        {"range": [20, 50], "color": "#fff3cd"},
                        {"range": [50, 75], "color": "#ffe5cc"},
                        {"range": [75, 100], "color": "#f8d7da"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": prob * 100,
                    },
                },
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("⚠️ API unavailable. Start the backend server with: `uvicorn smellpredict.platform.api:app --reload`")

            # Fallback: compute locally
            st.info("Running local analysis (without API)...")
            try:
                from smellpredict.features.extractor import extract_file_features
                source = uploaded_file.getvalue().decode("utf-8", errors="replace")
                code_m, smell_m = extract_file_features(source)
                st.markdown("**Local Analysis (API offline)**")
                st.json({
                    "loc": code_m.loc,
                    "function_count": code_m.function_count,
                    "max_cyclomatic_complexity": code_m.max_cyclomatic_complexity,
                    "total_smells": smell_m.total_smells,
                    "note": "No risk probability without trained model"
                })
            except Exception as e:
                st.error(f"Local analysis failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Commit Labeler
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔬 Commit Labeler":
    st.header("🔬 Bug-Fix Commit Labeler")
    st.markdown(
        "Test the bug-fix heuristic (Equation 1 from the paper) on any commit message. "
        "Adjust the score threshold to explore precision-recall tradeoffs."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        commit_msg = st.text_area(
            "Commit Message",
            value="Fix null pointer exception in login handler (#1234)",
            height=120,
        )
    with col2:
        threshold = st.slider("Score Threshold", 0.0, 1.0, 0.40, step=0.05)
        st.markdown("**Equation 1:**")
        st.markdown("score = 0.20·k + 0.15·i + 0.25·s − 0.15·n")

    if st.button("🔍 Label This Commit", type="primary"):
        result = api_post(
            "/api/v1/analyze/commit",
            json={"message": commit_msg, "threshold": threshold},
        )

        if result:
            score = result.get("score", 0)
            is_bug_fix = result.get("is_bug_fix", False)
            components = result.get("components", {})

            col_a, col_b = st.columns(2)
            with col_a:
                if is_bug_fix:
                    st.success(f"✅ **BUG-FIX** (score = {score:.3f})")
                else:
                    st.info(f"❌ **NOT a bug fix** (score = {score:.3f})")

            with col_b:
                st.markdown("**Score Components:**")
                st.markdown(f"- Keywords (k={components.get('k', 0)}, ×0.20): +{0.20 * components.get('k', 0):.2f}")
                st.markdown(f"- Issue refs (i={components.get('i', 0)}, ×0.15): +{0.15 * components.get('i', 0):.2f}")
                st.markdown(f"- Strong patterns (s={components.get('s', 0)}, ×0.25): +{0.25 * components.get('s', 0):.2f}")
                st.markdown(f"- Noise penalty (n={components.get('n', 0)}, ×0.15): −{0.15 * components.get('n', 0):.2f}")

        else:
            # Local fallback
            from smellpredict.labeling.heuristic import score_commit
            cs = score_commit(commit_msg, threshold)
            if cs.is_bug_fix:
                st.success(f"✅ **BUG-FIX** (score = {cs.score:.3f}) [local mode]")
            else:
                st.info(f"❌ **NOT a bug fix** (score = {cs.score:.3f}) [local mode]")

    # Batch testing
    st.markdown("---")
    st.markdown("### 🧪 Batch Test Examples")
    examples = [
        "Fix null pointer exception in core.py (#1234)",
        "Add new feature: user authentication",
        "Fix crash when input is empty",
        "Merge pull request #42 from dev/feature-branch",
        "Refactor database connection logic",
        "bug fix: resolved timeout regression in API handler",
        "Update documentation for endpoints",
        "hotfix: patch for broken login under load",
        "Improve test coverage for utils module",
        "resolve memory leak in file parser",
    ]

    from smellpredict.labeling.heuristic import score_commit
    rows = []
    for msg in examples:
        cs = score_commit(msg, threshold)
        rows.append({
            "Commit Message": msg[:65] + ("..." if len(msg) > 65 else ""),
            "Score": cs.score,
            "Label": "✅ Bug Fix" if cs.is_bug_fix else "❌ Not Bug Fix",
            "k": cs.k, "i": cs.i, "s": cs.s, "n": cs.n,
        })

    df_examples = pd.DataFrame(rows)
    st.dataframe(df_examples, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Dataset Explorer
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📊 Dataset Explorer":
    st.header("📊 Dataset Explorer")

    stats = api_get("/api/v1/dataset/stats")

    if stats and "error" not in str(stats).lower() and "message" not in stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{stats.get('total_rows', 0):,}")
        with col2:
            st.metric("Repositories", stats.get("repos", 0))
        with col3:
            st.metric("Unique Files", f"{stats.get('unique_files', 0):,}")
        with col4:
            st.metric("Positive Rate", f"{stats.get('positive_rate_pct', 0):.1f}%")
    else:
        st.warning("Dataset not yet mined. Run the mining pipeline first.")
        st.code("python -m smellpredict.mining.miner --config config/mining_config.yaml")

        # Show placeholder charts with paper data
        st.markdown("### Paper (v1) Dataset Preview")
        paper_data = {
            "Repository": ["colorama", "records", "gspread", "click"],
            "Files": [18, 14, 22, 119],
            "Snapshots": [342, 672, 1012, 2096],
            "Bug-Fix %": [6.4, 9.7, 7.8, 8.8],
        }
        df_paper = pd.DataFrame(paper_data)
        st.dataframe(df_paper, width="stretch")

        fig = px.bar(
            df_paper, x="Repository", y="Bug-Fix %",
            color="Bug-Fix %",
            color_continuous_scale="RdYlGn_r",
            title="Bug-Fix Prevalence by Repository (Paper v1)",
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Sensitivity Analysis
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🧪 Sensitivity Analysis":
    st.header("🧪 Smell Threshold Sensitivity Analysis")
    st.markdown("""
    This page shows how prediction performance changes when smell thresholds are varied.
    Each smell is tested at 5 threshold values to assess robustness.
    """)

    sens_path = Path("data/processed/sensitivity_results.json")
    if sens_path.exists():
        with open(sens_path, "r", encoding="utf-8") as f:
            sens_data = json.load(f)
        
        st.success("✅ Empirical sensitivity analysis results loaded (20,832 genuine snapshots)")
        
        smells = list(sens_data.keys())
        selected_smell = st.selectbox("Select Smell", smells)
        smell_records = sens_data[selected_smell]
        
        thresholds = [r["threshold"] for r in smell_records]
        bug_rates = [r["active_bug_rate"] for r in smell_records]
        correlations = [r["defect_correlation"] for r in smell_records]
        active_counts = [r["active_snapshots"] for r in smell_records]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[str(t) for t in thresholds],
            y=bug_rates,
            mode="lines+markers",
            name="Defect Rate in Smelly Code (%)",
            line=dict(color="#6366f1", width=3),
        ))
        fig.add_trace(go.Bar(
            x=[str(t) for t in thresholds],
            y=correlations,
            name="Defect Correlation (Point-Biserial r)",
            yaxis="y2",
            marker_color=["#10b981" if c > 0 else "#f43f5e" for c in correlations],
            opacity=0.7,
        ))
        fig.update_layout(
            title=f"Empirical Sensitivity to {selected_smell} Detection Threshold",
            xaxis_title=f"Threshold Metric Value for {selected_smell}",
            yaxis_title="Empirical Bug Rate (%)",
            yaxis2=dict(
                title="Correlation with Defects (r)",
                overlaying="y",
                side="right",
                range=[-0.2, 0.5],
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Detailed Threshold Breakdown")
        df_sens_display = pd.DataFrame(smell_records)
        df_sens_display.columns = ["Threshold", "Active Snapshots", "Bug Fix Rate (%)", "Defect Correlation (r)"]
        st.dataframe(df_sens_display, use_container_width=True)
    else:
        st.info("Sensitivity analysis dataset not found at data/processed/sensitivity_results.json.")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Research Findings
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📖 Research Findings":
    st.header("📖 Key Research Findings")

    st.markdown("""
    ### Paper: *Code Smells and Bug Prediction in Python Repositories — An Empirical Study*

    #### Central Finding (RQ6)
    > Code smells provide **small, direction-inconsistent, statistically non-significant** improvements 
    > over baseline code + history metrics in predicting future bug fixes.

    This suggests traditional metrics already capture most predictive signal from smells.
    """)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ✅ What We Found")
        st.markdown("""
        - Smell prevalence varies widely across repos (6.4%–9.8% bug-fix rate)
        - Development history is the strongest predictor
        - Smells add marginal information at best
        - Effect size is small (Cohen's d < 0.2)
        - Results inconsistent across projects (LOPO)
        """)

    with col2:
        st.markdown("#### 🔬 Validated Methodology")
        st.markdown("""
        - **Temporal validation**: strict date-cutoff splits
        - **Identity exclusion**: files never in both train and test
        - **5-fold temporal + LOPO cross-validation**
        - **3 feature groups**: code-only, +history, +smells
        - **VIF screening** for multicollinearity
        """)

    st.markdown("---")
    st.markdown("#### 📊 Model Performance Summary (Paper v1)")
    perf_data = {
        "Model": ["LogReg", "RF", "GBM", "LogReg", "RF", "GBM"],
        "Feature Group": ["Code+Hist", "Code+Hist", "Code+Hist",
                          "Code+Hist+Smells", "Code+Hist+Smells", "Code+Hist+Smells"],
        "PR-AUC": [0.289, 0.312, 0.334, 0.291, 0.315, 0.338],
        "ΔAUC (Smells)": [None, None, None, 0.002, 0.003, 0.004],
    }
    df_perf = pd.DataFrame(perf_data)
    st.dataframe(df_perf, width="stretch")
    st.caption("Based on 4 repos, 5-fold temporal CV (paper v1). v2 will expand to 50+ repos.")
