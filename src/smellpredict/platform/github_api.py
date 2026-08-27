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
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("smellpredict.github_api")
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


# ─────────────────────────────────────────────────────────────────────────────
# Public Repository Explorer (Zero Authentication Required)
# ─────────────────────────────────────────────────────────────────────────────

import httpx

@router.get("/public/presets", summary="Get curated open-source repositories for instant exploration")
async def get_public_presets():
    return JSONResponse({
        "presets": [
            {"name": "pallets/click", "desc": "Composable command line interface toolkit", "lang": "Python"},
            {"name": "pallets/flask", "desc": "Lightweight WSGI web application framework", "lang": "Python"},
            {"name": "psf/requests", "desc": "HTTP for Humans — Python library", "lang": "Python"},
            {"name": "tiangolo/fastapi", "desc": "High performance modern Python API framework", "lang": "Python"},
            {"name": "Abhishek-882/smellpredict", "desc": "SmellPredict Defect Intelligence platform", "lang": "Python"},
            {"name": "iluwatar/java-design-patterns", "desc": "Design patterns implemented in Java", "lang": "Java"},
        ]
    })


# Built-in cached trees for common presets to guarantee instant exploration even under cloud IP rate limits
PRESET_FALLBACK_TREES = {
    "pallets/click": [
        {"path": "src/click/core.py", "size": 95000, "sha": "tree_click_1"},
        {"path": "src/click/decorators.py", "size": 18000, "sha": "tree_click_2"},
        {"path": "src/click/parser.py", "size": 24000, "sha": "tree_click_3"},
        {"path": "src/click/types.py", "size": 32000, "sha": "tree_click_4"},
        {"path": "src/click/utils.py", "size": 15000, "sha": "tree_click_5"},
        {"path": "tests/test_basic.py", "size": 12000, "sha": "tree_click_6"},
        {"path": "README.md", "size": 4000, "sha": "tree_click_7"},
    ],
    "psf/requests": [
        {"path": "src/requests/api.py", "size": 6000, "sha": "tree_req_1"},
        {"path": "src/requests/sessions.py", "size": 32000, "sha": "tree_req_2"},
        {"path": "src/requests/models.py", "size": 41000, "sha": "tree_req_3"},
        {"path": "src/requests/adapters.py", "size": 26000, "sha": "tree_req_4"},
        {"path": "src/requests/utils.py", "size": 29000, "sha": "tree_req_5"},
        {"path": "tests/test_requests.py", "size": 28000, "sha": "tree_req_6"},
        {"path": "README.md", "size": 5000, "sha": "tree_req_7"},
    ],
    "tiangolo/fastapi": [
        {"path": "fastapi/applications.py", "size": 25000, "sha": "tree_fa_1"},
        {"path": "fastapi/routing.py", "size": 85000, "sha": "tree_fa_2"},
        {"path": "fastapi/params.py", "size": 12000, "sha": "tree_fa_3"},
        {"path": "fastapi/dependencies/utils.py", "size": 45000, "sha": "tree_fa_4"},
        {"path": "fastapi/encoders.py", "size": 16000, "sha": "tree_fa_5"},
        {"path": "README.md", "size": 12000, "sha": "tree_fa_6"},
    ],
    "iluwatar/java-design-patterns": [
        {"path": "factory/src/main/java/com/iluwatar/factory/App.java", "size": 3000, "sha": "tree_j_1"},
        {"path": "singleton/src/main/java/com/iluwatar/singleton/IvoryTower.java", "size": 2000, "sha": "tree_j_2"},
        {"path": "observer/src/main/java/com/iluwatar/observer/Weather.java", "size": 4000, "sha": "tree_j_3"},
        {"path": "builder/src/main/java/com/iluwatar/builder/Hero.java", "size": 5000, "sha": "tree_j_4"},
        {"path": "README.md", "size": 8000, "sha": "tree_j_5"},
    ],
}


@router.get("/public/tree", summary="Fetch file tree for any public GitHub repository")
async def get_public_tree(
    request: Request,
    repo: str = Query(..., description="Repository in owner/name format or GitHub URL"),
):
    """
    Fetches the public git tree for any GitHub repository.
    Automatically attaches user's token if logged in to avoid rate limits.
    Falls back gracefully for presets if rate limited.
    """
    clean_repo = repo.strip().replace("https://github.com/", "").rstrip("/")
    if "/" not in clean_repo:
        return JSONResponse({"detail": "Repository must be in owner/repo format (e.g. pallets/click)."}, status_code=400)

    headers = {
        "User-Agent": "SmellPredict-Explorer",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # If user is authenticated, attach their token for 5,000 req/hr rate limit
    auth_hdr = request.headers.get("Authorization", "")
    if auth_hdr.startswith("Bearer "):
        try:
            gh_token = extract_github_token(auth_hdr[7:])
            if gh_token and gh_token != "guest_token":
                auth_val = f"Bearer {gh_token}" if gh_token.startswith("github_pat_") or gh_token.startswith("ghp_") else f"token {gh_token}"
                headers["Authorization"] = auth_val
        except Exception:
            pass

    async with httpx.AsyncClient(timeout=15) as client:
        # 1. Get default branch
        repo_resp = await client.get(f"https://api.github.com/repos/{clean_repo}", headers=headers)
        
        if repo_resp.status_code == 403 or repo_resp.status_code == 429:
            # Check if we have preset fallback tree
            if clean_repo in PRESET_FALLBACK_TREES:
                return JSONResponse({
                    "repo": clean_repo,
                    "branch": "main",
                    "tree": PRESET_FALLBACK_TREES[clean_repo],
                    "count": len(PRESET_FALLBACK_TREES[clean_repo]),
                    "stars": 25000,
                    "description": f"Public repository {clean_repo} (preset view)",
                })
            return JSONResponse({
                "detail": "GitHub API public rate limit reached on shared cloud IP. Please sign in with your GitHub Personal Access Token in the top-right to browse unlimited repositories.",
                "rate_limited": True,
            }, status_code=403)

        if repo_resp.status_code != 200:
            return JSONResponse({"detail": f"Could not load repository '{clean_repo}'. HTTP {repo_resp.status_code}"}, status_code=repo_resp.status_code)

        repo_data = repo_resp.json()
        default_branch = repo_data.get("default_branch", "main")

        # 2. Get recursive tree
        tree_resp = await client.get(
            f"https://api.github.com/repos/{clean_repo}/git/trees/{default_branch}?recursive=1",
            headers=headers,
        )

        tree_items = []
        if tree_resp.status_code == 200:
            raw_tree = tree_resp.json().get("tree", [])
            for item in raw_tree:
                if item.get("type") == "blob":
                    p = item.get("path", "")
                    ext = p.split(".")[-1].lower() if "." in p else ""
                    if ext in ["py", "java", "js", "ts", "cpp", "c", "go", "rs", "html", "json", "md", "yaml", "yml"]:
                        tree_items.append({
                            "path": p,
                            "size": item.get("size", 0),
                            "sha": item.get("sha", ""),
                            "url": item.get("url", ""),
                        })
        elif clean_repo in PRESET_FALLBACK_TREES:
            tree_items = PRESET_FALLBACK_TREES[clean_repo]

    return JSONResponse({
        "repo": clean_repo,
        "branch": default_branch,
        "tree": tree_items,
        "count": len(tree_items),
        "stars": repo_data.get("stargazers_count", 0),
        "description": repo_data.get("description", ""),
    })


@router.get("/public/file", summary="Fetch raw content of a file from any public GitHub repository")
async def get_public_file(
    request: Request,
    repo: str = Query(..., description="Repository in owner/name format"),
    path: str = Query(..., description="File path within the repository"),
    branch: str = Query("main", description="Branch name"),
):
    clean_repo = repo.strip().replace("https://github.com/", "").rstrip("/")
    raw_url = f"https://raw.githubusercontent.com/{clean_repo}/{branch}/{path}"
    
    headers = {"User-Agent": "SmellPredict-Explorer"}
    auth_hdr = request.headers.get("Authorization", "")
    if auth_hdr.startswith("Bearer "):
        try:
            gh_token = extract_github_token(auth_hdr[7:])
            if gh_token and gh_token != "guest_token":
                auth_val = f"Bearer {gh_token}" if gh_token.startswith("github_pat_") or gh_token.startswith("ghp_") else f"token {gh_token}"
                headers["Authorization"] = auth_val
        except Exception:
            pass

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(raw_url, headers=headers)
        if resp.status_code != 200:
            # Try raw master branch if main failed
            alt_url = f"https://raw.githubusercontent.com/{clean_repo}/master/{path}"
            resp = await client.get(alt_url, headers=headers)
            if resp.status_code != 200:
                return JSONResponse({"detail": f"File not found: {path} on repository {clean_repo}"}, status_code=404)
        content = resp.text

    return JSONResponse({
        "repo": clean_repo,
        "branch": branch,
        "path": path,
        "content": content,
    })
