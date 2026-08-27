"""
SmellPredict — Pull Request Defect Risk & Refactoring Bot
==========================================================
Analyzes modified Python files in a pull request diff, computes defect
risk using the trained Random Forest model, and generates a formatted
Markdown summary comment and SARIF diagnostics report.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


from smellpredict.models.predictor import analyze_source_code
from smellpredict.platform.sarif import export_sarif_file


def get_git_diff_files(base_ref: str = "origin/main") -> List[Path]:
    """Retrieve list of modified Python files compared to base branch."""
    try:
        cmd = ["git", "diff", "--name-only", "--diff-filter=d", f"{base_ref}...HEAD"]
        output = subprocess.check_output(cmd, text=True, errors="replace")
        files = [Path(line.strip()) for line in output.splitlines() if line.strip().endswith(".py")]
        return [f for f in files if f.exists()]
    except Exception:
        # Fallback: scan all local .py files in repo
        return [p for p in Path(".").rglob("*.py") if ".git" not in p.parts][:25]


def generate_pr_markdown_report(results: List[Dict[str, Any]], fail_under: float = 0.80) -> str:
    """Format PR defect analysis into clean GitHub Flavored Markdown."""
    if not results:
        return "### 🔍 SmellPredict PR Analysis\n\nNo Python files were modified in this pull request."

    total_files = len(results)
    high_risk_files = [r for r in results if r["risk_probability"] >= 0.65]
    critical_files = [r for r in results if r["risk_probability"] >= 0.80]

    # PR Health Badge
    if critical_files:
        status_badge = "🔴 **CRITICAL DEFECT RISK DETECTED**"
    elif high_risk_files:
        status_badge = "🟠 **HIGH DEFECT RISK — REVIEW ADVISED**"
    else:
        status_badge = "🟢 **LOW DEFECT RISK — CODEBASE HEALTHY**"

    lines = [
        "## 🔍 SmellPredict Quality Gate & Defect Risk Report",
        f"\n{status_badge}\n",
        f"Analyzed **{total_files}** modified Python file(s) against the trained ML defect prediction engine (*Random Forest FG-B, PR-AUC 0.8297*).\n",
        "### 📊 Modified Files Risk Breakdown",
        "| File | Defect Risk | Risk Tier | Maintainability | Smells | Status |",
        "|:-----|:-----------:|:---------:|:---------------:|:------:|:------:|",
    ]

    for r in results:
        prob = r["risk_probability"]
        tier = r["risk_tier"]
        icon = r["risk_icon"]
        cm = r["code_metrics"]
        smells = r["smells"]
        fpath = r["file_path"]

        status = "❌ Blocked" if prob >= fail_under else "⚠️ Warning" if prob >= 0.65 else "✅ Passed"

        lines.append(
            f"| `{fpath}` | **{prob*100:.1f}%** | {icon} {tier} | {cm['maintainability_index']:.1f}/100 | {smells['total_smells']} | {status} |"
        )

    # Refactoring Advice Section
    all_advice = []
    for r in results:
        for adv in r.get("refactoring_advice", []):
            all_advice.append((r["file_path"], adv))

    if all_advice:
        lines.append("\n### 🛠️ Actionable Refactoring Suggestions")
        for fpath, adv in all_advice[:6]:
            lines.append(f"\n<details><summary><b>{adv['title']}</b> in <code>{fpath}</code> (Line {adv['line_number']})</summary>\n")
            lines.append(f"> **Why:** {adv['description']}\n")
            lines.append(f"> **Action:** {adv['suggested_action']}\n")
            if adv.get("code_template"):
                lines.append(f"```python\n{adv['code_template']}\n```")
            lines.append("</details>")

    lines.append("\n---\n*Report generated automatically by [SmellPredict CI/CD](https://github.com/smellpredict/smellpredict)*")
    return "\n".join(lines)


def run_pr_reporter(
    base_ref: str = "origin/main",
    sarif_out: Path = Path("reports/smellpredict.sarif"),
    md_out: Path = Path("reports/pr_report.md"),
    fail_under: float = 0.80,
) -> int:
    """Execute PR scan, output reports, and return exit status."""
    files = get_git_diff_files(base_ref)
    print(f"SmellPredict PR Bot: Analyzing {len(files)} diff files against {base_ref}...")

    results = []
    for fpath in files:
        try:
            source = fpath.read_text(encoding="utf-8", errors="replace")
            res = analyze_source_code(source, file_path=str(fpath))
            results.append(res)
        except Exception as e:
            print(f"Error analyzing {fpath}: {e}", file=sys.stderr)

    # Write Markdown PR report
    md_content = generate_pr_markdown_report(results, fail_under=fail_under)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(md_content, encoding="utf-8")
    print(f"✓ PR Markdown report written to: {md_out}")

    # Write SARIF report
    export_sarif_file(results, sarif_out)
    print(f"✓ SARIF report written to: {sarif_out}")

    # Check failure gate
    failing = [r for r in results if r["risk_probability"] >= fail_under]
    if failing:
        print(f"❌ PR Quality Gate FAILED: {len(failing)} file(s) exceeded threshold of {fail_under*100:.1f}%")
        return 1

    print("✓ PR Quality Gate PASSED: All modified files are within acceptable defect risk bounds.")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SmellPredict PR Reporter Bot")
    parser.add_argument("--base-ref", default="origin/main", help="Git base ref for diff")
    parser.add_argument("--sarif-out", default="reports/smellpredict.sarif", type=Path)
    parser.add_argument("--md-out", default="reports/pr_report.md", type=Path)
    parser.add_argument("--fail-under", default=0.80, type=float, help="Risk threshold for non-zero exit")
    args = parser.parse_args()

    sys.exit(run_pr_reporter(
        base_ref=args.base_ref,
        sarif_out=args.sarif_out,
        md_out=args.md_out,
        fail_under=args.fail_under,
    ))
