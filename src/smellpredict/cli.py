"""
SmellPredict — Command Line Interface (CLI)
=============================================
Unified CLI entry point for the SmellPredict platform.

Commands:
  smellpredict mine        — Mine repositories from config
  smellpredict train       — Train models and run evaluations
  smellpredict label       — Interactive/batch commit message labeler
  smellpredict analyze     — Analyze a single Python file
  smellpredict api         — Start the FastAPI backend
  smellpredict dashboard   — Start the Streamlit UI dashboard
  smellpredict version     — Show package version
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Reconfigure stdout for utf-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import typer
import yaml
from loguru import logger
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="smellpredict",
    help="SmellPredict: Code Smell × Bug Prediction Platform CLI",
    add_completion=False,
)
console = Console(highlight=False)



@app.command()
def version():
    """Display the SmellPredict version."""
    import smellpredict
    console.print(f"[bold green]SmellPredict[/bold green] version: {smellpredict.__version__}")


@app.command()
def mine(
    config_path: Path = typer.Option(
        Path("config/mining_config.yaml"),
        "--config", "-c",
        help="Path to mining configuration YAML",
    ),
    clone_dir: Path = typer.Option(
        Path("data/clones"),
        "--clones",
        help="Directory to store git repository clones",
    ),
    output_dir: Path = typer.Option(
        Path("data/raw"),
        "--output", "-o",
        help="Directory to write parquet snapshot datasets",
    ),
    db_path: Path = typer.Option(
        Path("data/smellpredict.duckdb"),
        "--db",
        help="DuckDB database path",
    ),
):
    """Mine target repositories and extract features into DuckDB and Parquet."""
    if not config_path.exists():
        console.print(f"[bold red]Error:[/bold red] Config file not found at {config_path}")
        raise typer.Exit(code=1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    console.print(f"[bold blue]Starting repository mining using config:[/bold blue] {config_path}")
    from smellpredict.mining.miner import mine_all_repositories

    df = mine_all_repositories(
        config=config,
        clone_base_dir=clone_dir,
        output_dir=output_dir,
        db_path=db_path,
    )
    console.print(f"[bold green]Mining complete![/bold green] Total records: {len(df):,}")


@app.command()
def train(
    dataset_path: Path = typer.Option(
        Path("data/raw/all_repos_merged.parquet"),
        "--data", "-d",
        help="Path to merged parquet dataset",
    ),
    experiment_name: str = typer.Option(
        "smellpredict_v2",
        "--exp", "-e",
        help="MLflow experiment name",
    ),
    output_results: Path = typer.Option(
        Path("data/processed/experiment_results.csv"),
        "--out", "-o",
        help="Path to save evaluation CSV results",
    ),
):
    """Train models across feature groups with temporal walk-forward & LOPO validation."""
    if not dataset_path.exists():
        console.print(f"[bold red]Error:[/bold red] Dataset not found at {dataset_path}")
        console.print("Run [bold yellow]smellpredict mine[/bold yellow] first or specify valid --data path.")
        raise typer.Exit(code=1)

    import pandas as pd
    from smellpredict.models.trainer import ExperimentRunner

    console.print(f"[bold blue]Loading dataset:[/bold blue] {dataset_path}")
    df = pd.read_parquet(dataset_path)

    console.print(f"[bold cyan]Dataset rows:[/bold cyan] {len(df):,} across {df['repo'].nunique()} repos")
    runner = ExperimentRunner(df=df, experiment_name=experiment_name)

    console.print("[bold blue]Executing training and statistical validation suite...[/bold blue]")
    results_df = runner.run_all()

    output_results.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_results, index=False)
    console.print(f"[bold green]Training complete![/bold green] Saved results to: {output_results}")


@app.command()
def label(
    message: Optional[str] = typer.Argument(None, help="Commit message to classify"),
    threshold: float = typer.Option(0.40, "--threshold", "-t", help="Bug-fix score threshold"),
    build_sample: bool = typer.Option(False, "--build-sample", help="Build 500-commit stratified annotation sample from repositories"),
    compute_kappa: bool = typer.Option(False, "--compute-kappa", help="Compute Cohen's kappa on annotated sample"),
    sample_output: Path = typer.Option(Path("data/annotation/sample_500.json"), "--sample-out", help="Output path for annotation sample JSON"),
    clones_dir: Path = typer.Option(Path("data/clones"), "--clones", help="Path to cloned repositories"),
):
    """Classify a commit message or build/evaluate human annotation benchmark samples."""
    from smellpredict.labeling.heuristic import (
        score_commit,
        build_annotation_sample,
        export_annotation_sample,
        heuristic_vs_human_metrics,
        cohens_kappa,
    )

    if build_sample:
        from pydriller import Repository
        console.print("[bold blue]Gathering commits from local repository clones for stratified sampling...[/bold blue]")
        clone_paths = sorted([d for d in clones_dir.iterdir() if d.is_dir() and (d / ".git").exists()])
        
        all_commits = []
        for clone_path in clone_paths:
            count = 0
            try:
                for commit in Repository(str(clone_path)).traverse_commits():
                    count += 1
                    msg = commit.msg.strip()
                    if len(msg) > 5:
                        all_commits.append({
                            "hash": commit.hash,
                            "repo": clone_path.name,
                            "message": msg,
                        })
                    if count >= 120:
                        break
            except Exception as e:
                logger.warning(f"Error reading {clone_path.name}: {e}")
        
        console.print(f"[bold cyan]Collected {len(all_commits)} total candidate commits across {len(clone_paths)} repositories.[/bold cyan]")
        samples = build_annotation_sample(all_commits, n_high_pos=125, n_boundary=125, n_high_neg=125, n_near_miss=125)
        sample_output.parent.mkdir(parents=True, exist_ok=True)
        export_annotation_sample(samples, sample_output)
        console.print(f"[bold green]Successfully built stratified sample ({len(samples)} commits) at {sample_output}[/bold green]")
        return

    if compute_kappa:
        if not sample_output.exists():
            console.print(f"[bold red]Annotation file not found at {sample_output}.[/bold red] Run with [bold yellow]--build-sample[/bold yellow] first.")
            raise typer.Exit(code=1)
        
        import json
        with open(sample_output, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        labeled = [d for d in data if d.get("human_label") is not None]
        if not labeled:
            console.print("[bold yellow]Sample loaded. Computing strata distribution...[/bold yellow]")
            table = Table(title=f"Sample Strata Distribution ({len(data)} items)")
            table.add_column("Stratum", style="cyan")
            table.add_column("Count", style="magenta")
            table.add_column("Heuristic Positive Rate", style="green")
            
            strata_counts = {}
            for d in data:
                s = d["stratum"]
                if s not in strata_counts:
                    strata_counts[s] = {"total": 0, "pos": 0}
                strata_counts[s]["total"] += 1
                if d["heuristic_label"]:
                    strata_counts[s]["pos"] += 1
            
            for s, counts in strata_counts.items():
                pos_rate = counts["pos"] / counts["total"] * 100
                table.add_row(s, str(counts["total"]), f"{pos_rate:.1f}%")
            console.print(table)
            return

        heuristic_labels = [d["heuristic_label"] for d in labeled]
        human_labels = [bool(d["human_label"]) for d in labeled]
        metrics = heuristic_vs_human_metrics(heuristic_labels, human_labels)

        table = Table(title="Inter-Rater Agreement & Accuracy (Heuristic vs. Human)")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold green")
        table.add_row("Cohen's Kappa (kappa)", f"{metrics['kappa']:.4f}")
        table.add_row("Interpretation", metrics["kappa_interpretation"])
        table.add_row("Precision", f"{metrics['precision']:.4f}")
        table.add_row("Recall", f"{metrics['recall']:.4f}")
        table.add_row("F1 Score", f"{metrics['f1']:.4f}")
        table.add_row("Annotated Sample Size", str(metrics["support"]))
        console.print(table)
        return

    if message is None:
        console.print("[bold yellow]Please provide a commit message or use --build-sample / --compute-kappa[/bold yellow]")
        raise typer.Exit(code=1)

    res = score_commit(message, threshold=threshold)
    table = Table(title="Commit Bug-Fix Classification")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Message", res.message)
    table.add_row("Score", f"{res.score:.3f}")
    table.add_row("Threshold", f"{threshold:.2f}")
    table.add_row("Is Bug-Fix", "[bold green]YES[/bold green]" if res.is_bug_fix else "[bold red]NO[/bold red]")
    table.add_row("Keywords (k)", str(res.k))
    table.add_row("Issue Refs (i)", str(res.i))
    table.add_row("Strong Fix Patterns (s)", str(res.s))
    table.add_row("Noise Patterns (n)", str(res.n))
    table.add_row("Noise Dominated", str(res.noise_dominated))

    console.print(table)


@app.command()
def analyze(
    file_path: Path = typer.Argument(..., help="Path to Python file to analyze"),
):
    """Analyze a single Python file for code metrics, defect risk, and refactoring advice."""
    if not file_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
        raise typer.Exit(code=1)

    from smellpredict.models.predictor import analyze_source_code

    source = file_path.read_text(encoding="utf-8", errors="replace")
    res = analyze_source_code(source, file_path=str(file_path))
    code = res["code_metrics"]
    smells = res["smells"]
    prob = res["risk_probability"]
    tier = res["risk_tier"]
    icon = res["risk_icon"]

    # Main Metrics Table
    table = Table(title=f"Analysis: {file_path.name}")
    table.add_column("Category", style="cyan")
    table.add_column("Metric / Indicator", style="bold white")
    table.add_column("Value / Status", style="green")

    # Defect Risk Tier
    risk_style = "bold red" if tier in ("Critical", "High") else "bold yellow" if tier == "Medium" else "bold green"
    table.add_row("Defect Risk", "ML Defect Probability", f"[{risk_style}]{prob*100:.1f}% ({icon} {tier})[/{risk_style}]")
    table.add_row("Defect Risk", "Recommendation", res["recommendation"])

    table.add_row("Code Metrics", "Lines of Code (LOC)", str(code["loc"]))
    table.add_row("Code Metrics", "Max Nesting Depth", str(code["max_nesting_depth"]))
    table.add_row("Code Metrics", "Max Cyclomatic Complexity", f"{code['max_cyclomatic_complexity']:.1f}")
    table.add_row("Code Metrics", "Maintainability Index", f"{code['maintainability_index']:.1f}/100")
    table.add_row("Code Metrics", "Cognitive Complexity", str(code["cognitive_complexity"]))

    table.add_row("Code Smells", "Long Method (>=50 LOC)", "[red]Detected[/red]" if smells["has_long_method"] else "[green]Clean[/green]")
    table.add_row("Code Smells", "Long Parameter List (>=5)", "[red]Detected[/red]" if smells["has_long_param_list"] else "[green]Clean[/green]")
    table.add_row("Code Smells", "Large Class (>=300 LOC)", "[red]Detected[/red]" if smells["has_large_class"] else "[green]Clean[/green]")
    table.add_row("Code Smells", "Deep Nesting (>=4 levels)", "[red]Detected[/red]" if smells["has_deep_nesting"] else "[green]Clean[/green]")
    table.add_row("Code Smells", "High Complexity (CC>=10)", "[red]Detected[/red]" if smells["has_high_complexity"] else "[green]Clean[/green]")
    table.add_row("Code Smells", "Total Smells Count", str(smells["total_smells"]))

    console.print(table)

    # Refactoring Advice
    advice = res.get("refactoring_advice", [])
    if advice:
        console.print(f"\n[bold yellow]🛠️  Refactoring Advice ({len(advice)} actionable recommendations):[/bold yellow]")
        for i, adv in enumerate(advice, 1):
            console.print(f"\n[bold cyan]{i}. {adv['title']}[/bold cyan] [dim](Line {adv['line_number']})[/dim]")
            console.print(f"   [white]{adv['description']}[/white]")
            console.print(f"   [green]Action:[/green] {adv['suggested_action']}")
            if adv.get("code_template"):
                console.print(f"   [dim]Proposed Template:[/dim]")
                for line in adv["code_template"].splitlines():
                    console.print(f"     [magenta]{line}[/magenta]")


@app.command()
def scan(
    target_path: Path = typer.Argument(Path("."), help="Directory or file to scan"),
    fail_under: Optional[float] = typer.Option(None, "--fail-under", "-f", help="Exit code 1 if any file risk >= threshold (e.g. 0.80)"),
    sarif_output: Optional[Path] = typer.Option(None, "--sarif-output", "-s", help="Output path for SARIF report"),
    json_output: Optional[Path] = typer.Option(None, "--json-output", "-j", help="Output path for JSON report"),
    max_files: int = typer.Option(200, "--max-files", "-m", help="Maximum files to scan"),
):
    """Scan a codebase for code smells, complexity, and ML defect risk."""
    import json
    from smellpredict.models.predictor import analyze_source_code
    from smellpredict.platform.sarif import export_sarif_file

    if not target_path.exists():
        console.print(f"[bold red]Error:[/bold red] Target path not found: {target_path}")
        raise typer.Exit(code=1)

    # Collect Python files
    if target_path.is_file():
        py_files = [target_path] if target_path.suffix == ".py" else []
    else:
        ignore_dirs = {".git", "__pycache__", "venv", ".venv", "build", "dist", "node_modules", "data", "models"}
        py_files = []
        for p in target_path.rglob("*.py"):
            if not any(part in ignore_dirs for part in p.parts):
                py_files.append(p)
                if len(py_files) >= max_files:
                    break

    if not py_files:
        console.print(f"[yellow]No Python files found in {target_path}[/yellow]")
        return

    console.print(f"[bold blue]🔍 Scanning {len(py_files)} Python files in [cyan]{target_path}[/cyan]...[/bold blue]\n")

    results = []
    failed_threshold = []

    for fpath in py_files:
        try:
            source = fpath.read_text(encoding="utf-8", errors="replace")
            res = analyze_source_code(source, file_path=str(fpath))
            results.append(res)
            if fail_under is not None and res["risk_probability"] >= fail_under:
                failed_threshold.append(res)
        except Exception as e:
            logger.debug(f"Failed to scan {fpath}: {e}")

    # Summary table
    table = Table(title=f"SmellPredict Codebase Scan Results ({len(results)} files)")
    table.add_column("File Path", style="cyan", no_wrap=True)
    table.add_column("Risk Score", justify="right")
    table.add_column("Tier", justify="center")
    table.add_column("LOC", justify="right", style="dim")
    table.add_column("MI", justify="right", style="dim")
    table.add_column("Smells", justify="center")

    results.sort(key=lambda x: x["risk_probability"], reverse=True)

    tier_counts = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}

    for r in results:
        prob = r["risk_probability"]
        tier = r["risk_tier"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        color = "red" if tier in ("Critical", "High") else "yellow" if tier == "Medium" else "green"

        smell_str = str(r["smells"]["total_smells"])
        if r["smells"]["total_smells"] > 0:
            smell_str = f"[red]{smell_str}[/red]"
        else:
            smell_str = "[green]0[/green]"

        table.add_row(
            r["file_path"],
            f"[{color}]{prob*100:.1f}%[/{color}]",
            f"[{color}]{r['risk_icon']} {tier}[/{color}]",
            str(r["code_metrics"]["loc"]),
            f"{r['code_metrics']['maintainability_index']:.1f}",
            smell_str,
        )

    console.print(table)

    # Health summary
    console.print("\n[bold]Codebase Health Overview:[/bold]")
    total = len(results)
    console.print(f"  🟢 Low Risk:      {tier_counts['Low']} ({tier_counts['Low']/total*100:.1f}%)")
    console.print(f"  🟡 Medium Risk:   {tier_counts['Medium']} ({tier_counts['Medium']/total*100:.1f}%)")
    console.print(f"  🟠 High Risk:     {tier_counts['High']} ({tier_counts['High']/total*100:.1f}%)")
    console.print(f"  🔴 Critical Risk: {tier_counts['Critical']} ({tier_counts['Critical']/total*100:.1f}%)")

    # Exports
    if sarif_output:
        sarif_path = export_sarif_file(results, sarif_output)
        console.print(f"\n[bold green]✓ SARIF report written to:[/bold green] {sarif_path}")

    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        console.print(f"[bold green]✓ JSON report written to:[/bold green] {json_output}")

    # Threshold gate check
    if fail_under is not None and failed_threshold:
        console.print(
            f"\n[bold red]❌ Quality Gate FAILED:[/bold red] {len(failed_threshold)} file(s) exceeded risk threshold of {fail_under*100:.1f}%"
        )
        raise typer.Exit(code=1)
    elif fail_under is not None:
        console.print(f"\n[bold green]✓ Quality Gate PASSED:[/bold green] All files below risk threshold of {fail_under*100:.1f}%")



@app.command()
def api(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
):
    """Start the FastAPI backend server."""
    import uvicorn
    console.print(f"[bold green]Starting SmellPredict API on http://{host}:{port}...[/bold green]")
    uvicorn.run("smellpredict.platform.api:app", host=host, port=port, reload=reload)


@app.command()
def dashboard(
    port: int = typer.Option(8501, "--port", "-p", help="Streamlit port"),
):
    """Launch the Streamlit interactive dashboard."""
    import subprocess
    dash_file = Path(__file__).parent / "platform" / "dashboard.py"
    console.print(f"[bold green]Launching Streamlit Dashboard on port {port}...[/bold green]")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dash_file), "--server.port", str(port)])


@app.command()
def start(
    api_port: int = typer.Option(8000, "--api-port", help="FastAPI port"),
    dash_port: int = typer.Option(8501, "--dash-port", help="Streamlit port"),
):
    """Start BOTH the FastAPI backend and Streamlit dashboard simultaneously."""
    import subprocess
    import threading
    import time

    console.print("[bold green]🚀 Launching SmellPredict Platform (API + Dashboard)...[/bold green]")
    
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "smellpredict.platform.api:app", "--host", "0.0.0.0", "--port", str(api_port)]
    )
    
    time.sleep(2)  # Give API 2 seconds to bind
    
    dash_file = Path(__file__).parent / "platform" / "dashboard.py"
    console.print(f"[bold cyan]API running on: http://localhost:{api_port}[/bold cyan]")
    console.print(f"[bold cyan]Dashboard running on: http://localhost:{dash_port}[/bold cyan]")
    console.print("[bold yellow]Press Ctrl+C at any time to stop both servers.[/bold yellow]\n")

    try:
        dash_proc = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(dash_file), "--server.port", str(dash_port)]
        )
    except KeyboardInterrupt:
        pass
    finally:
        console.print("\n[bold red]Shutting down API server...[/bold red]")
        api_proc.terminate()


@app.command()
def watch(
    refresh: float = typer.Option(2.0, "--refresh", "-r", help="Refresh rate in seconds"),
):
    """Launch the real-time live terminal progress monitor for background mining."""
    from smellpredict.platform.monitor import run_live_monitor
    run_live_monitor(refresh_rate=refresh)


@app.command()
def monitor(
    refresh: float = typer.Option(2.0, "--refresh", "-r", help="Refresh rate in seconds"),
):
    """Alias for 'watch': Launch live progress monitor."""
    from smellpredict.platform.monitor import run_live_monitor
    run_live_monitor(refresh_rate=refresh)


@app.command(name="mine-java")
def mine_java(
    config_path: Path = typer.Option(
        Path("config/java_mining_config.yaml"),
        "--config", "-c",
        help="Path to Java mining configuration YAML",
    ),
    clones_dir: Path = typer.Option(
        Path("data/java/clones"),
        "--clones",
        help="Directory to store Java git repository clones",
    ),
    output_dir: Path = typer.Option(
        Path("data/java/raw"),
        "--output", "-o",
        help="Directory to write Java parquet snapshot datasets",
    ),
    db_path: Path = typer.Option(
        Path("data/java_smellpredict.duckdb"),
        "--db",
        help="Java DuckDB database path",
    ),
):
    """Mine 50+ Java repositories and extract AST metrics into DuckDB and Parquet."""
    if not config_path.exists():
        console.print(f"[bold red]Error:[/bold red] Java config file not found at {config_path}")
        raise typer.Exit(code=1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    console.print(f"[bold green]☕ Starting Java repository mining using config:[/bold green] {config_path}")
    from smellpredict.mining.java_miner import mine_all_java_repositories

    df = mine_all_java_repositories(
        config=config,
        clone_base_dir=clones_dir,
        output_dir=output_dir,
        db_path=db_path,
    )
    console.print(f"[bold green]Java mining complete![/bold green] Total records: {len(df):,}")


@app.command(name="train-java")
def train_java(
    dataset_path: Optional[Path] = typer.Option(
        None,
        "--data", "-d",
        help="Path to all_java_merged.parquet (optional)",
    ),
    output_dir: Path = typer.Option(
        Path("models"),
        "--output", "-o",
        help="Output directory for java_best_model.pkl",
    ),
    trials: int = typer.Option(
        50,
        "--trials", "-t",
        help="Number of Optuna HPO optimization trials",
    ),
):
    """Train dedicated Java defect prediction models using CatBoost, RF, and Optuna HPO."""
    console.print("[bold green]☕ Launching Java Model Training Pipeline...[/bold green]")
    from scripts.train_java_model import run_java_training_pipeline

    results = run_java_training_pipeline(
        data_path=dataset_path,
        output_dir=output_dir,
        n_trials=trials,
    )
    console.print(f"[bold green]✓ Java model trained successfully![/bold green]")
    console.print(f"  Model Type:  [bold cyan]{results['model_type']}[/bold cyan]")
    console.print(f"  PR-AUC:      [bold green]{results['pr_auc']:.4f}[/bold green]")
    console.print(f"  ROC-AUC:     [bold green]{results['roc_auc']:.4f}[/bold green]")
    console.print(f"  Artifact:    [bold yellow]{results['model_path']}[/bold yellow]")


@app.command(name="watch-java")
def watch_java(
    refresh: float = typer.Option(2.0, "--refresh", "-r", help="Refresh rate in seconds"),
    log_path: Path = typer.Option(Path("logs/java_mining.log"), "--log", help="Java mining log file"),
    raw_dir: Path = typer.Option(Path("data/java/raw"), "--data", help="Java raw parquet directory"),
    config_path: Path = typer.Option(Path("config/java_mining_config.yaml"), "--config", help="Java mining config"),
):
    """Launch the real-time live terminal progress monitor for Java mining."""
    from smellpredict.platform.java_monitor import run_java_live_monitor
    run_java_live_monitor(refresh_rate=refresh, log_path=log_path, raw_dir=raw_dir, config_path=config_path)


def main():
    app()


if __name__ == "__main__":
    main()
