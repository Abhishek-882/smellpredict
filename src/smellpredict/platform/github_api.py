"""
SmellPredict — GitHub Repository API
======================================
Provides FastAPI endpoints for browsing GitHub repositories,
reading file contents, and committing changes back — all authenticated
via the GitHub OAuth token embedded in the SmellPredict JWT.

Endpoints:
  GET  /github/repos                            → list user repos
  GET  /github/repos/{owner}/{repo}/branches    → list branches
  GET  /github/repos/{owner}/{repo}/tree        → recursive file tree
  GET  /github/repos/{owner}/{repo}/file        → file content + SHA
  POST /github/repos/{owner}/{repo}/commit      → commit file to GitHub

All endpoints require:  Authorization: Bearer <smellpredict_jwt>
The JWT contains a Fernet-encrypted GitHub access token (see auth.py).
"""

from __future__ import annotations

import base64
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from github import Github, GithubException, UnknownObjectException
from loguru import logger
from pydantic import BaseModel, Field

from smellpredict.platform.auth import _bearer_token, extract_github_token

# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/github", tags=["github"])


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────


class CommitRequest(BaseModel):
    """Request body for committing a file change to GitHub."""
    path: str = Field(..., description="File path within the repo (e.g. src/foo.py)")
    content: str = Field(..., description="New file content (plain text, not base64)")
    message: str = Field(
        default="SmellPredict: update file",
        description="Commit message",
    )
    sha: str = Field(
        ...,
        description=(
            "The blob SHA of the file being replaced. "
            "Must be retrieved from GET /github/repos/{o}/{r}/file first."
        ),
    )
    branch: str = Field(default="main", description="Target branch for the commit")


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────


def _gh(request: Request) -> Github:
    """Construct an authenticated PyGitHub client from the request's JWT."""
    token = _bearer_token(request)
    github_token = extract_github_token(token)
    return Github(github_token)


def _handle_github_error(exc: GithubException) -> None:
    """Convert PyGitHub exceptions into FastAPI HTTPExceptions."""
    if isinstance(exc, UnknownObjectException):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    raise HTTPException(
        status_code=exc.status if exc.status else 502,
        detail=f"GitHub API error: {exc.data}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/repos", summary="List authenticated user's GitHub repositories")
async def list_repos(request: Request):
    """
    Returns the list of repositories the authenticated user has access to,
    including name, description, visibility, primary language, and default branch.
    """
    try:
        gh = _gh(request)
        user = gh.get_user()
        repos = []
        for repo in user.get_repos(sort="updated", type="all"):
            repos.append({
                "id": repo.id,
                "name": repo.name,
                "full_name": repo.full_name,
                "description": repo.description or "",
                "private": repo.private,
                "language": repo.language or "",
                "default_branch": repo.default_branch,
                "updated_at": repo.updated_at.isoformat() if repo.updated_at else "",
                "url": repo.html_url,
            })
        logger.info(f"Listed {len(repos)} repos for user: {user.login}")
        return JSONResponse({"repos": repos, "count": len(repos)})
    except GithubException as exc:
        _handle_github_error(exc)


@router.get(
    "/repos/{owner}/{repo}/branches",
    summary="List branches for a repository",
)
async def list_branches(request: Request, owner: str, repo: str):
    """Returns all branches for the specified repository."""
    try:
        gh = _gh(request)
        repository = gh.get_repo(f"{owner}/{repo}")
        branches = [
            {
                "name": b.name,
                "sha": b.commit.sha,
                "protected": b.protected,
            }
            for b in repository.get_branches()
        ]
        return JSONResponse({"branches": branches})
    except GithubException as exc:
        _handle_github_error(exc)


@router.get(
    "/repos/{owner}/{repo}/tree",
    summary="Get recursive file tree for a branch",
)
async def get_file_tree(
    request: Request,
    owner: str,
    repo: str,
    ref: str = Query(default="", description="Branch, tag, or commit SHA (default: repo default branch)"),
):
    """
    Returns the full recursive file tree of the repository at the given ref.
    Each node contains: path, type (blob/tree), sha, size.
    """
    try:
        gh = _gh(request)
        repository = gh.get_repo(f"{owner}/{repo}")
        branch = ref or repository.default_branch
        git_tree = repository.get_git_tree(sha=branch, recursive=True)
        tree = []
        for item in git_tree.tree:
            tree.append({
                "path": item.path,
                "type": item.type,   # "blob" = file, "tree" = directory
                "sha": item.sha,
                "size": item.size or 0,
            })
        logger.info(f"Fetched tree for {owner}/{repo}@{branch}: {len(tree)} items")
        return JSONResponse({
            "owner": owner,
            "repo": repo,
            "ref": branch,
            "tree": tree,
        })
    except GithubException as exc:
        _handle_github_error(exc)


@router.get(
    "/repos/{owner}/{repo}/file",
    summary="Get file content and SHA from a repository",
)
async def get_file(
    request: Request,
    owner: str,
    repo: str,
    path: str = Query(..., description="File path within the repo, e.g. src/foo.py"),
    ref: str = Query(default="", description="Branch, tag, or commit SHA"),
):
    """
    Returns the decoded file content (UTF-8) and the current blob SHA.
    The SHA must be sent back when committing changes to prevent conflicts.
    """
    try:
        gh = _gh(request)
        repository = gh.get_repo(f"{owner}/{repo}")
        branch = ref or repository.default_branch
        file_content = repository.get_contents(path, ref=branch)

        # PyGitHub returns a list if the path is a directory — reject that
        if isinstance(file_content, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Path '{path}' is a directory, not a file.",
            )

        # Decode content (GitHub returns base64-encoded content)
        try:
            decoded = base64.b64decode(file_content.content).decode("utf-8")
        except (UnicodeDecodeError, Exception):
            # Binary file — return base64 as-is with a flag
            return JSONResponse({
                "path": path,
                "sha": file_content.sha,
                "encoding": "base64",
                "content": file_content.content,
                "binary": True,
            })

        return JSONResponse({
            "path": path,
            "sha": file_content.sha,
            "encoding": "utf-8",
            "content": decoded,
            "binary": False,
            "size": file_content.size,
        })
    except GithubException as exc:
        _handle_github_error(exc)


@router.post(
    "/repos/{owner}/{repo}/commit",
    summary="Commit a file change to GitHub",
)
async def commit_file(
    request: Request,
    owner: str,
    repo: str,
    body: CommitRequest,
):
    """
    Commits a single file change to the specified GitHub repository and branch.

    IMPORTANT: The `sha` field must contain the current blob SHA retrieved
    from GET /github/repos/{owner}/{repo}/file. GitHub requires this to
    prevent accidental overwrites (optimistic concurrency control).

    On success, returns the new commit SHA and URL.
    """
    try:
        gh = _gh(request)
        repository = gh.get_repo(f"{owner}/{repo}")

        # GitHub Contents API requires base64-encoded content
        encoded_content = base64.b64encode(body.content.encode("utf-8")).decode()

        result = repository.update_file(
            path=body.path,
            message=body.message,
            content=encoded_content,
            sha=body.sha,
            branch=body.branch,
        )

        commit = result["commit"]
        sha_short = (commit.sha or "")[:7]
        logger.info(
            f"Committed {body.path} to {owner}/{repo}@{body.branch}: "
            f"{sha_short} — {body.message}"
        )
        # commit.commit (inner GitCommit) may be None in some PyGitHub versions
        inner = getattr(commit, "commit", None)
        return JSONResponse({
            "success": True,
            "commit": {
                "sha": commit.sha or "",
                "message": (inner.message if inner else body.message),
                "url": commit.html_url or "",
                "author": (inner.author.name if inner and inner.author else ""),
                "date": (inner.author.date.isoformat() if inner and inner.author else ""),
            },
            "path": body.path,
            "branch": body.branch,
        })
    except GithubException as exc:
        _handle_github_error(exc)
