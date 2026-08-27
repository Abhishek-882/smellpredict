"""
SmellPredict — Java Mining Live Terminal Progress Monitor
=========================================================
Renders a live, real-time updating terminal dashboard showing:
  - Overall Java repository progress bar (e.g. 18/50 repos)
  - Active repository commit progress bar (e.g. 111/500 commits 22.2%)
  - Completed Java repositories summary table with snapshot counts, positive bug rate, and parse fallback %
  - Tier breakdown (Tier 1 Frameworks, Tier 2 Big Data, Tier 3 Tooling)
  - Real-time mining speed & memory stats
  - Live log stream from logs/java_mining.log
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def parse_java_mining_state(
    log_path: Path = Path("logs/java_mining.log"),
    raw_dir: Path = Path("data/java/raw"),
    config_path: Path = Path("config/java_mining_config.yaml"),
    status_path: Path = Path("data/java/mining_status.json"),
) -> dict:
    """Parse the current Java mining state from status JSON, task logs, parquet files, and config."""
    total_target_repos = 50
    tier_targets = {"tier1": 20, "tier2": 15, "tier3": 15}
    tier_repos_map = {}

    if Path(config_path).exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                all_repos = []
                for tier, repos in cfg.get("repositories", {}).items():
                    tier_targets[tier] = len(repos)
                    for r in repos:
                        tier_repos_map[r["name"]] = tier
                    all_repos.extend(repos)
                total_target_repos = len(all_repos)
        except Exception:
            pass

    # Find completed parquet files
    completed_repos = []
    total_snapshots = 0
    parse_fallback_total = 0
    tier_counts = {"tier1": 0, "tier2": 0, "tier3": 0}

    if Path(raw_dir).exists():
        for p in Path(raw_dir).glob("*.parquet"):
            if p.name == "all_java_merged.parquet":
                continue
            try:
                df_head = pd.read_parquet(p)
                n_rows = len(df_head)
                n_pos = int(df_head["future_bug_fix"].sum()) if "future_bug_fix" in df_head.columns else 0
                n_fallback = int(df_head["parse_fallback"].sum()) if "parse_fallback" in df_head.columns else 0

                tier = tier_repos_map.get(p.stem, "tier1")
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

                completed_repos.append({
                    "name": p.stem,
                    "tier": tier,
                    "rows": n_rows,
                    "positives": n_pos,
                    "pos_rate": (n_pos / max(n_rows, 1)) * 100,
                    "fallback_count": n_fallback,
                    "fallback_rate": (n_fallback / max(n_rows, 1)) * 100,
                    "size_kb": p.stat().st_size / 1024,
                })
                total_snapshots += n_rows
                parse_fallback_total += n_fallback
            except Exception:
                pass

    # Read live heartbeat status if available
    active_repo = "None"
    active_current = 0
    active_total = 500
    active_pct = 0.0
    active_snapshots = 0

    if Path(status_path).exists():
        try:
            with open(status_path, "r", encoding="utf-8") as sf:
                st_data = json.load(sf)
                if time.time() - st_data.get("timestamp", 0) < 300:
                    active_repo = st_data.get("active_repo", "None")
                    active_current = int(st_data.get("active_current", 0))
                    active_total = int(st_data.get("active_total", 500))
                    active_pct = float(st_data.get("active_pct", 0.0))
                    active_snapshots = int(st_data.get("snapshots_count", 0))
        except Exception:
            pass

    # If active_repo not determined yet, scan task logs in background system
    if active_current == 0:
        try:
            brain_dir = os.path.expanduser("~/.gemini/antigravity-ide/brain")
            if os.path.exists(brain_dir):
                for root, _, files in os.walk(brain_dir):
                    for f in files:
                        if f.startswith("task-") and f.endswith(".log"):
                            fp = os.path.join(root, f)
                            try:
                                if time.time() - os.path.getmtime(fp) < 300:
                                    with open(fp, "rb") as lf:
                                        lf.seek(max(0, os.path.getsize(fp) - 20000))
                                        txt = lf.read().decode("utf-8", errors="replace")
                                        matches = re.findall(r"Java-Snapshots \[([\w-]+)\]:\s+(\d+)%\|.*?\|\s+(\d+)/(\d+)", txt)
                                        if matches:
                                            last_m = matches[-1]
                                            active_repo = last_m[0]
                                            active_pct = float(last_m[1])
                                            active_current = int(last_m[2])
                                            active_total = int(last_m[3])
                                            break
                            except Exception:
                                pass
                    if active_current > 0:
                        break
        except Exception:
            pass

    # Read latest logs
    latest_logs = []
    if Path(log_path).exists():
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as lf:
                lines = [l.strip() for l in lf.readlines() if l.strip()]
                # Sanitize arrows and unicode for safe Windows rendering
                sanitized = [l.replace("→", "->").replace("✔", "[OK]").replace("☕", "[JAVA]") for l in lines]
                latest_logs = sanitized[-7:] if len(sanitized) >= 7 else sanitized
                
                if active_repo == "None":
                    for line in reversed(sanitized):
                        m_repo = re.search(r"Mining Java repository:\s+([\w-]+)", line)
                        if m_repo:
                            active_repo = m_repo.group(1)
                            break
                        m_clone = re.search(r"Cloning .* -> data[/\\]java[/\\]clones[/\\]([\w-]+)", line)
                        if m_clone:
                            active_repo = m_clone.group(1)
                            break
        except Exception:
            pass

    total_bugs = sum(r.get("positives", 0) for r in completed_repos)
    overall_bug_rate = (total_bugs / max(total_snapshots, 1)) * 100 if total_snapshots > 0 else 0.0
    overall_fallback_rate = (parse_fallback_total / max(total_snapshots, 1)) * 100 if total_snapshots > 0 else 0.0

    return {
        "total_targets": total_target_repos,
        "completed_count": len(completed_repos),
        "completed_repos": completed_repos,
        "total_snapshots": total_snapshots + active_snapshots,
        "total_bugs": total_bugs,
        "overall_bug_rate": overall_bug_rate,
        "parse_fallback_count": parse_fallback_total,
        "parse_fallback_pct": overall_fallback_rate,
        "tier_breakdown": {
            "tier1": f"{tier_counts.get('tier1', 0)}/{tier_targets.get('tier1', 20)}",
            "tier2": f"{tier_counts.get('tier2', 0)}/{tier_targets.get('tier2', 15)}",
            "tier3": f"{tier_counts.get('tier3', 0)}/{tier_targets.get('tier3', 15)}",
        },
        "active_repo": active_repo,
        "active_current": active_current,
        "active_total": active_total,
        "active_pct": active_pct,
        "active_snapshots": active_snapshots,
        "latest_logs": latest_logs,
    }


def render_java_dashboard(state: dict) -> Panel:
    """Generate the Rich visual panel dashboard for Java mining (CP1252 & UTF-8 safe)."""
    pct_total = (state["completed_count"] / max(state["total_targets"], 1)) * 100
    bar_width = 30
    filled = int(bar_width * (state["completed_count"] / max(state["total_targets"], 1)))
    bar_str = f"[{'#' * filled}{'-' * (bar_width - filled)}]"

    overall_header = Text.from_markup(
        f"[bold cyan]Total Pipeline Progress:[/bold cyan] {bar_str} "
        f"[bold green]{state['completed_count']}/{state['total_targets']} Repositories[/bold green] "
        f"([bold yellow]{pct_total:.1f}%[/bold yellow])"
    )

    tiers = state.get("tier_breakdown", {})
    tier_text = Text.from_markup(
        f"[dim]Tiers:[/dim] [bold blue]Tier 1 Frameworks:[/bold blue] {tiers.get('tier1', '0/20')} | "
        f"[bold magenta]Tier 2 Streaming & DB:[/bold magenta] {tiers.get('tier2', '0/15')} | "
        f"[bold cyan]Tier 3 Tools & Sec:[/bold cyan] {tiers.get('tier3', '0/15')}"
    )

    # Active Repo Commit Progress Bar
    act_name = state.get("active_repo", "None")
    act_cur = state.get("active_current", 0)
    act_tot = max(state.get("active_total", 500), 1)
    act_pct = state.get("active_pct", 0.0)
    act_filled = int(24 * (act_cur / act_tot))
    act_bar_str = f"[{'#' * act_filled}{'-' * (24 - act_filled)}]"

    active_header = Text.from_markup(
        f"[bold white]Active Mining:[/bold white] [bold magenta]#{state['completed_count'] + 1} {act_name}[/bold magenta] | "
        f"[bold cyan]Commit Progress:[/bold cyan] {act_bar_str} [bold yellow]{act_cur}/{act_tot} ({act_pct:.1f}%)[/bold yellow]"
    )

    stats_text = Text.from_markup(
        f"[bold white]Total Extracted Snapshots:[/bold white] [bold green]{state['total_snapshots']:,} rows[/bold green] | "
        f"[bold white]AST Fallback %:[/bold white] [bold yellow]{state['parse_fallback_pct']:.1f}%[/bold yellow] | "
        f"[bold white]Avg Bug Rate:[/bold white] [bold magenta]{state['overall_bug_rate']:.1f}%[/bold magenta]"
    )

    # Table of Completed Repos
    table = Table(
        title="Completed Java Repositories & Extracted AST Datasets",
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Repository", style="bold white")
    table.add_column("Tier", justify="center", style="blue")
    table.add_column("Snapshots", justify="right", style="green")
    table.add_column("Bug Fixes", justify="right", style="yellow")
    table.add_column("Bug %", justify="right", style="magenta")
    table.add_column("Fallback %", justify="right", style="yellow")
    table.add_column("Disk Size", justify="right", style="cyan")

    for i, r in enumerate(state["completed_repos"], 1):
        table.add_row(
            str(i),
            r["name"],
            r.get("tier", "tier1").upper(),
            f"{r['rows']:,}",
            f"{r['positives']:,}",
            f"{r['pos_rate']:.1f}%",
            f"{r.get('fallback_rate', 0.0):.1f}%",
            f"{r['size_kb']:.1f} KB",
        )

    if not state["completed_repos"]:
        table.add_row(
            "[*]",
            f"{act_name} (Analyzing {act_cur}/{act_tot} commits...)",
            "TIER1",
            f"{state.get('active_snapshots', 0)}",
            "--",
            "--",
            "0.0%",
            "--",
        )

    # Log view
    log_text = "\n".join(state["latest_logs"])
    log_panel = Panel(
        Text(log_text if log_text else "Waiting for Java mining log events...", style="dim white"),
        title="Live Log Stream (logs/java_mining.log)",
        border_style="blue",
    )

    content = Group(
        overall_header,
        tier_text,
        Text(""),
        active_header,
        stats_text,
        Text(""),
        table,
        Text(""),
        log_panel,
    )

    return Panel(
        content,
        title="[bold green][JAVA] SmellPredict -- Java Empirical Mining Live Monitor[/bold green]",
        subtitle="[dim]Auto-refreshing every 2s | Press Ctrl+C to exit monitor[/dim]",
        border_style="green",
    )


def run_java_live_monitor(
    refresh_rate: float = 2.0,
    log_path: Path = Path("logs/java_mining.log"),
    raw_dir: Path = Path("data/java/raw"),
    config_path: Path = Path("config/java_mining_config.yaml"),
    status_path: Path = Path("data/java/mining_status.json"),
):
    """Run the live terminal dashboard for Java mining in an auto-refreshing loop."""
    console = Console()
    with Live(console=console, screen=True, refresh_per_second=1) as live:
        while True:
            state = parse_java_mining_state(
                log_path=log_path,
                raw_dir=raw_dir,
                config_path=config_path,
                status_path=status_path,
            )
            panel = render_java_dashboard(state)
            live.update(panel)
            time.sleep(refresh_rate)


if __name__ == "__main__":
    run_java_live_monitor()
