"""
SmellPredict — Live Terminal Progress Monitor
==============================================
Renders a live, real-time updating terminal dashboard showing:
  - Overall repository progress bar (e.g. 11/50 repos)
  - Current repository snapshot progress bar (e.g. 45/116 snapshots)
  - Completed repositories summary table with row counts
  - Real-time mining speed & memory stats
  - Latest log entries from logs/mining.log
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text


def parse_mining_state(
    log_path: Path = Path("logs/mining.log"),
    raw_dir: Path = Path("data/raw"),
    config_path: Path = Path("config/mining_config.yaml"),
) -> dict:
    """Parse the current mining state from logs, config, and raw parquet files."""
    total_target_repos = 50
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                all_repos = []
                for tier in cfg.get("repositories", {}).values():
                    all_repos.extend(tier)
                total_target_repos = len(all_repos)
        except Exception:
            pass

    # Find completed parquet files
    completed_repos = []
    total_snapshots = 0
    if raw_dir.exists():
        for p in raw_dir.glob("*.parquet"):
            if p.name == "all_repos_merged.parquet":
                continue
            try:
                # Fast parquet read
                df = pd.read_parquet(p, columns=["future_bug_fix"])
                n_rows = len(df)
                n_pos = int(df["future_bug_fix"].sum()) if "future_bug_fix" in df.columns else 0
                completed_repos.append({
                    "name": p.stem,
                    "rows": n_rows,
                    "positives": n_pos,
                    "pos_rate": (n_pos / max(n_rows, 1)) * 100,
                    "size_kb": p.stat().st_size / 1024,
                })
                total_snapshots += n_rows
            except Exception:
                pass

    # Parse active repo & snapshot progress from log files
    active_repo = "None"
    active_current = 0
    active_total = 100
    active_pct = 0.0
    latest_logs = []

    # Check all active task logs dynamically for live tqdm snapshot progress
    tasks_dir = Path("C:/Users/Asus/.gemini/antigravity-ide/brain/1f2b0c5f-16a4-4bd2-8676-d60f3306a254/.system_generated/tasks")
    if tasks_dir.exists():
        task_files = sorted(
            [f for f in tasks_dir.glob("task-*.log") if time.time() - f.stat().st_mtime < 120],
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        for t_file in task_files:
            try:
                # Read last 150KB of task log
                with open(t_file, "rb") as f:
                    f.seek(max(0, t_file.stat().st_size - 150000))
                    tail_text = f.read().decode("utf-8", errors="replace")
                    matches = re.findall(r"Snapshots \[([\w-]+)\]:\s+(\d+)%\|.*?\|\s+(\d+)/(\d+)", tail_text)
                    if matches:
                        last_m = matches[-1]
                        active_repo = last_m[0]
                        active_pct = float(last_m[1])
                        active_current = int(last_m[2])
                        active_total = int(last_m[3])
                        break
            except Exception:
                pass

    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            latest_logs = lines[-8:] if len(lines) >= 8 else lines
            completed_names = {r["name"] for r in completed_repos}
            if active_repo == "None" or active_repo in completed_names:
                for line in reversed(lines):
                    m_clone = re.search(r"Cloning .* → data\\clones\\([\w-]+)", line)
                    if m_clone and m_clone.group(1) not in completed_names:
                        active_repo = m_clone.group(1)
                        active_current = 0
                        active_total = 100
                        active_pct = 0.0
                        break
                    m_repo = re.search(r"Mining repository:\s+([\w-]+)", line)
                    if m_repo and m_repo.group(1) not in completed_names:
                        active_repo = m_repo.group(1)
                        break
        except Exception:
            pass

    total_bugs = sum(r.get("positives", 0) for r in completed_repos)
    overall_bug_rate = (total_bugs / max(total_snapshots, 1)) * 100 if total_snapshots > 0 else 0.0

    all_target_names = [
        "flask", "fastapi", "requests", "scrapy", "click", "rich", "typer", "httpx",
        "celery", "pytest", "black", "ruff", "jinja2", "sqlalchemy", "django",
        "scikit-learn", "xgboost", "lightgbm", "tox", "pydantic",
        "fabric", "paramiko", "pillow", "arrow", "poetry", "pip", "gunicorn",
        "uvicorn", "sphinx", "mypy", "pylint", "tornado", "aiohttp", "boto3", "werkzeug",
        "colorama", "records", "gspread", "python-fire", "tqdm", "attrs", "cattrs",
        "marshmallow", "python-dotenv", "loguru", "structlog", "tenacity", "backoff", "schedule", "boltons"
    ]

    return {
        "total_targets": total_target_repos,
        "completed_count": len(completed_repos),
        "completed_repos": completed_repos,
        "total_snapshots": total_snapshots,
        "total_bugs": total_bugs,
        "overall_bug_rate": overall_bug_rate,
        "all_target_names": all_target_names,
        "active_repo": active_repo,
        "active_current": active_current,
        "active_total": active_total,
        "active_pct": active_pct,
        "latest_logs": latest_logs,
    }


def render_dashboard(state: dict) -> Panel:
    """Generate the Rich visual panel dashboard."""
    # 1. Overall Progress
    pct_total = (state["completed_count"] / max(state["total_targets"], 1)) * 100
    bar_width = 40
    filled = int(bar_width * (state["completed_count"] / max(state["total_targets"], 1)))
    bar_str = f"[{'█' * filled}{'░' * (bar_width - filled)}]"

    overall_header = Text.from_markup(
        f"[bold cyan]Total Pipeline Progress:[/bold cyan] {bar_str} "
        f"[bold green]{state['completed_count']}/{state['total_targets']} Repositories[/bold green] "
        f"([bold yellow]{pct_total:.1f}%[/bold yellow])"
    )

    stats_text = Text.from_markup(
        f"[bold white]Extracted Snapshots:[/bold white] [bold green]{state['total_snapshots']:,} rows[/bold green] | "
        f"[bold white]Active Mining Target:[/bold white] [bold magenta]#{state['completed_count'] + 1} {state['active_repo']}[/bold magenta]"
    )

    # 2. Table of Completed Repos
    table = Table(
        title="Completed Repositories & Extracted Datasets",
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Repository", style="bold white")
    table.add_column("Snapshots", justify="right", style="green")
    table.add_column("Bug Fixes", justify="right", style="yellow")
    table.add_column("Bug-Fix %", justify="right", style="magenta")
    table.add_column("Disk Size", justify="right", style="cyan")

    for i, r in enumerate(state["completed_repos"], 1):
        table.add_row(
            str(i),
            r["name"],
            f"{r['rows']:,}",
            f"{r['positives']:,}",
            f"{r['pos_rate']:.1f}%",
            f"{r['size_kb']:.1f} KB",
        )

    if not state["completed_repos"]:
        table.add_row("—", "Initializing first repository...", "0", "0", "0.0%", "0.0 KB")

    # 3. Log view
    log_text = "\n".join(state["latest_logs"])
    log_panel = Panel(
        Text(log_text if log_text else "Waiting for log events...", style="dim white"),
        title="Live Log Stream (logs/mining.log)",
        border_style="blue",
    )

    content = Group(
        overall_header,
        stats_text,
        Text(""),
        table,
        Text(""),
        log_panel,
    )

    return Panel(
        content,
        title="[bold green]🔍 SmellPredict — Live Empirical Mining Monitor[/bold green]",
        subtitle="[dim]Auto-refreshing every 2s • Press Ctrl+C to exit monitor[/dim]",
        border_style="green",
    )


def run_live_monitor(refresh_rate: float = 2.0):
    """Run the live terminal dashboard in an auto-refreshing loop."""
    console = Console()
    with Live(console=console, screen=True, refresh_per_second=1) as live:
        while True:
            state = parse_mining_state()
            panel = render_dashboard(state)
            live.update(panel)
            time.sleep(refresh_rate)


if __name__ == "__main__":
    run_live_monitor()
