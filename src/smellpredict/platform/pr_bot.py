"""
SmellPredict — Automated GitHub PR Review Bot Engine
=====================================================
Analyzes pull request diffs and changed files to synthesize comprehensive
pull request code reviews, SARIF inline diagnostics, and automated quick fixes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("smellpredict.pr_bot")

from smellpredict.features.polyglot import polyglot_analyze, EXTENSION_MAP, LANGUAGE_BADGES
from smellpredict.features.refactor import generate_quick_fix_patch
from smellpredict.models.predictor import analyze_source_code


def analyze_pr_files(files_map: Dict[str, str]) -> Dict[str, Any]:
    """
    Analyze a dictionary of {file_path: file_content} for a pull request.
    Returns composite PR risk score, per-file analysis, and inline comments list.
    """
    file_results: List[Dict[str, Any]] = []
    inline_comments: List[Dict[str, Any]] = []
    total_risk_score = 0.0
    analyzed_count = 0

    for fpath, content in files_map.items():
        if not content or not content.strip():
            continue

        ext = Path(fpath).suffix.lower()
        if ext not in EXTENSION_MAP and ext != ".py":
            continue

        analyzed_count += 1
        if ext == ".py":
            try:
                res = analyze_source_code(content, file_path=fpath)
                risk_prob = float(res.get("risk_probability", 0.1))
                risk_tier = res.get("risk_tier", "Low")
                risk_icon = res.get("risk_icon", "🟢")
                smells = res.get("smells", {})
                refactoring = res.get("refactoring_advice", [])
                metrics = res.get("code_metrics", {})
                language = "python"
                badge = "🐍 Python"
            except Exception as e:
                logger.warning(f"Python analysis failed for {fpath}: {e}")
                continue
        else:
            pres = polyglot_analyze(content, file_path=fpath)
            risk_prob = pres.risk_probability
            risk_tier = pres.risk_tier
            risk_icon = pres.risk_icon
            smells = {
                "has_long_method": pres.smells.has_long_method,
                "has_long_param_list": pres.smells.has_long_param_list,
                "has_large_class": pres.smells.has_large_class,
                "has_deep_nesting": pres.smells.has_deep_nesting,
                "total_smells": pres.smells.total_smells,
            }
            refactoring = pres.refactoring_advice
            metrics = {
                "loc": pres.metrics.loc,
                "sloc": pres.metrics.sloc,
                "cyclomatic_complexity": pres.metrics.cyclomatic_complexity,
                "max_nesting_depth": pres.metrics.max_nesting_depth,
            }
            language = pres.language
            badge = LANGUAGE_BADGES.get(pres.language, "📄 File")

        total_risk_score += risk_prob
        file_summary = {
            "path": fpath,
            "filename": Path(fpath).name,
            "language": language,
            "badge": badge,
            "risk_probability": round(risk_prob, 3),
            "risk_tier": risk_tier,
            "risk_icon": risk_icon,
            "total_smells": smells.get("total_smells", 0),
            "metrics": metrics,
            "refactoring_count": len(refactoring),
        }
        file_results.append(file_summary)

        # Generate inline review comments for high-risk smells
        for advice in refactoring[:3]:
            line_no = advice.get("line_number", 1) if isinstance(advice, dict) else getattr(advice, "line_number", 1)
            title = advice.get("title", "Refactoring Suggestion") if isinstance(advice, dict) else getattr(advice, "title", "Refactoring")
            smell_type = advice.get("smell_type", "CodeSmell") if isinstance(advice, dict) else getattr(advice, "smell_type", "CodeSmell")
            desc = advice.get("description", "") if isinstance(advice, dict) else getattr(advice, "description", "")
            action = advice.get("suggested_action", "") if isinstance(advice, dict) else getattr(advice, "suggested_action", "")

            # Generate quick-fix snippet
            patch = generate_quick_fix_patch(content, smell_type=smell_type, line_number=line_no, language=language)

            comment_body = (
                f"### ⚠️ SmellPredict SARIF Diagnostic: `{smell_type}`\n\n"
                f"**{title}**\n\n"
                f"{desc}\n\n"
                f"> **💡 Recommendation**: {action}\n\n"
            )
            if patch.get("diff"):
                comment_body += f"```diff\n{patch['diff']}\n```\n"

            inline_comments.append({
                "path": fpath,
                "line": max(1, line_no),
                "side": "RIGHT",
                "body": comment_body,
            })

    avg_risk = total_risk_score / max(1, analyzed_count)
    if avg_risk < 0.25:
        overall_tier = "Low"
        overall_icon = "🟢"
        verdict = "APPROVE"
    elif avg_risk < 0.60:
        overall_tier = "Medium"
        overall_icon = "🟡"
        verdict = "COMMENT"
    elif avg_risk < 0.80:
        overall_tier = "High"
        overall_icon = "🟠"
        verdict = "REQUEST_CHANGES"
    else:
        overall_tier = "Critical"
        overall_icon = "🔴"
        verdict = "REQUEST_CHANGES"

    markdown_summary = generate_pr_review_markdown(
        avg_risk=avg_risk,
        overall_tier=overall_tier,
        overall_icon=overall_icon,
        verdict=verdict,
        files=file_results,
    )

    return {
        "overall_risk_probability": round(avg_risk, 3),
        "overall_risk_tier": overall_tier,
        "overall_icon": overall_icon,
        "verdict": verdict,
        "analyzed_files_count": analyzed_count,
        "files": file_results,
        "inline_comments": inline_comments,
        "markdown_summary": markdown_summary,
    }


def generate_pr_review_markdown(
    avg_risk: float,
    overall_tier: str,
    overall_icon: str,
    verdict: str,
    files: List[Dict[str, Any]],
) -> str:
    """Format markdown report suitable for GitHub PR review comments."""
    pct = round(avg_risk * 100, 1)
    rows = []
    for f in files:
        rows.append(
            f"| `{f['path']}` | {f['badge']} | {f['risk_icon']} **{f['risk_tier']}** ({round(f['risk_probability']*100,1)}%) | {f['total_smells']} | {f.get('metrics', {}).get('loc', 0)} |"
        )
    table_rows = "\n".join(rows) if rows else "| *No analyzed code files* | - | - | - | - |"

    return f"""## {overall_icon} SmellPredict AI Code Review: **{overall_tier} Risk** ({pct}% Bug-Fix Probability)

### 📊 Executive Summary
The automated SmellPredict ML & Polyglot inspection completed analysis on **{len(files)} modified file(s)**.

| Metric | Assessment |
|---|---|
| **Composite PR Defect Probability** | **{pct}%** |
| **Risk Classification** | {overall_icon} **{overall_tier}** |
| **Recommended Review Action** | `{verdict}` |

---

### 📁 File-by-File Breakdown
| File | Language | Bug Risk | Code Smells | LOC |
|---|---|---|---|---|
{table_rows}

---
*Generated by [SmellPredict AI](https://github.com/marketplace/actions/smellpredict-ci-code-smell-bug-prediction)* 🚀
"""


async def post_github_pr_review(
    github_token: str,
    repo_full_name: str,
    pull_number: int,
    commit_sha: str,
    review_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Post review summary and inline comments to GitHub Pull Request API.
    POST /repos/{owner}/{repo}/pulls/{pull_number}/reviews
    """
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pull_number}/reviews"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "SmellPredict-PR-Bot",
    }

    event = review_result.get("verdict", "COMMENT")
    body = review_result.get("markdown_summary", "SmellPredict Review")
    comments = review_result.get("inline_comments", [])[:10]  # Cap at 10 inline comments to avoid rate limits

    payload = {
        "commit_id": commit_sha,
        "body": body,
        "event": event if event in ("APPROVE", "REQUEST_CHANGES", "COMMENT") else "COMMENT",
        "comments": comments,
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code in (200, 201):
                logger.info(f"Successfully posted PR review to {repo_full_name}#{pull_number}")
                return {"status": "success", "response": resp.json()}
            else:
                logger.warning(f"GitHub API review post failed ({resp.status_code}): {resp.text}")
                return {"status": "error", "status_code": resp.status_code, "error": resp.text}
        except Exception as exc:
            logger.error(f"Error posting review to GitHub: {exc}")
            return {"status": "error", "error": str(exc)}
