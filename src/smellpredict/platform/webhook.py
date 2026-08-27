"""
SmellPredict — GitHub Webhook Receiver & PR Review Ingestion
============================================================
Handles incoming GitHub webhook events:
  - Validates HMAC SHA256 signatures (X-Hub-Signature-256)
  - Processes `pull_request` events (opened, synchronize, reopened)
  - Extracts modified files and triggers automated Polyglot/ML reviews
  - Posts SARIF diagnostics and markdown review comments back to the PR
"""

from __future__ import annotations

import hmac
import hashlib
import json
import os
from typing import Any, Dict, Optional
from fastapi import APIRouter, Header, HTTPException, Request, BackgroundTasks
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("smellpredict.webhook")

from smellpredict.platform.pr_bot import analyze_pr_files, post_github_pr_review

router = APIRouter(prefix="/github", tags=["webhooks"])

GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")


def verify_webhook_signature(payload_bytes: bytes, signature_header: Optional[str], secret: str) -> bool:
    """
    Verify GitHub HMAC SHA256 signature from `X-Hub-Signature-256` header.
    Format: 'sha256=<hex_digest>'
    """
    if not secret:
        # If no secret configured in environment, allow for development/testing
        return True
    if not signature_header or not signature_header.startswith("sha256="):
        return False

    expected_sig = hmac.new(
        secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()

    received_sig = signature_header.replace("sha256=", "")
    return hmac.compare_digest(expected_sig, received_sig)


async def _process_pr_event(payload: Dict[str, Any], token: Optional[str] = None):
    """Background task to fetch PR files, run analysis, and post review."""
    action = payload.get("action")
    pull_request = payload.get("pull_request", {})
    repository = payload.get("repository", {})

    repo_full_name = repository.get("full_name")
    pull_number = pull_request.get("number")
    head_sha = pull_request.get("head", {}).get("sha")

    logger.info(f"Processing PR #{pull_number} ({action}) for {repo_full_name} @ {head_sha}")

    # If mock/inline files are passed in payload (e.g. testing)
    files_map = payload.get("files_map", {})

    # If no mock files and we have a token, fetch changed files from GitHub
    if not files_map and token and repo_full_name and pull_number:
        import httpx
        files_url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pull_number}/files"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(files_url, headers=headers)
                if resp.status_code == 200:
                    files_list = resp.json()
                    for f in files_list:
                        patch_content = f.get("patch", "")
                        filename = f.get("filename", "")
                        if patch_content and filename:
                            files_map[filename] = patch_content
        except Exception as e:
            logger.warning(f"Failed to fetch PR files from GitHub API: {e}")

    if not files_map:
        logger.info(f"No code files detected for PR #{pull_number}")
        return

    # Run analysis
    review_result = analyze_pr_files(files_map)
    logger.info(
        f"PR #{pull_number} analysis complete: {review_result['overall_risk_tier']} "
        f"({review_result['overall_risk_probability']}), {len(review_result['inline_comments'])} inline comments"
    )

    # Post review to GitHub if token is available
    if token and repo_full_name and pull_number and head_sha:
        await post_github_pr_review(
            github_token=token,
            repo_full_name=repo_full_name,
            pull_number=pull_number,
            commit_sha=head_sha,
            review_result=review_result,
        )


@router.post("/webhook", summary="GitHub Webhook Receiver for automated PR reviews")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: Optional[str] = Header(None, alias="X-GitHub-Event"),
    x_hub_signature_256: Optional[str] = Header(None, alias="X-Hub-Signature-256"),
):
    """
    Ingests GitHub webhook events.
    Verifies HMAC SHA256 signature and schedules PR inspection.
    """
    raw_body = await request.body()

    # Verify signature
    secret = os.getenv("GITHUB_WEBHOOK_SECRET", "")
    if secret and not verify_webhook_signature(raw_body, x_hub_signature_256, secret):
        logger.warning("GitHub webhook signature verification failed")
        raise HTTPException(status_code=401, detail="Invalid HMAC signature")

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}")

    event = x_github_event or payload.get("event", "ping")

    if event == "ping":
        return {"status": "ok", "message": "Pong! SmellPredict webhook receiver is active."}

    if event == "pull_request":
        action = payload.get("action")
        if action in ("opened", "synchronize", "reopened"):
            token = os.getenv("GITHUB_TOKEN", "")
            background_tasks.add_task(_process_pr_event, payload, token)
            return {
                "status": "queued",
                "event": "pull_request",
                "action": action,
                "pr_number": payload.get("pull_request", {}).get("number"),
            }
        return {"status": "ignored", "reason": f"Action '{action}' does not trigger PR review"}

    return {"status": "ignored", "event": event}
