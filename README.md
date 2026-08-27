# SmellPredict 🔬

**Code Smell × Bug Prediction Platform — Collaborative IDE + Research Engine**

> Predict the probability of future bug-fix commits using code smells, complexity metrics,
> and repository history — with a real-time collaborative IDE backed by GitHub and
> powered by a CatBoost ML model trained on 20,832 real code snapshots.

[![PR-AUC](https://img.shields.io/badge/PR--AUC-0.8297-brightgreen)](#)
[![Repos](https://img.shields.io/badge/Training%20Repos-50-blue)](#)
[![Snapshots](https://img.shields.io/badge/Snapshots-20%2C832-blue)](#)
[![Cohen's κ](https://img.shields.io/badge/Cohen%27s%20κ-0.816-green)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688)](#)
[![License](https://img.shields.io/badge/License-MIT-yellow)](#)

---

## What is SmellPredict?

SmellPredict answers the research question:

> **"Do code smells improve future bug-fix prediction beyond complexity metrics alone?"**

It combines:
- **Bug-fix labeling heuristic** (keyword regex + pattern scoring, κ=0.816 vs human)
- **Code smell detection** (long method, large class, deep nesting, long param list)
- **Complexity metrics** (cyclomatic complexity, Halstead, maintainability index)
- **ML model** (CatBoost, trained on 50 OSS repos, PR-AUC 0.8297)
- **Collaborative IDE** (Monaco editor + Y.js CRDT sync + GitHub storage)

---

## Features

| Feature | Description |
|---|---|
| 🔬 **ML Prediction** | CatBoost model — predicts bug-fix probability per file |
| 🧪 **Code Smell Detection** | Long Method, Large Class, Deep Nesting, Long Param List |
| 📊 **Research Dashboard** | Streamlit + SHAP explainability, temporal CV results |
| 🖥️ **Collaborative IDE** | Monaco editor with Y.js real-time multi-user sync |
| 🐙 **GitHub Integration** | OAuth login, repo browser, file read/write via GitHub API |
| ⚡ **Live Analysis** | Debounced smell + risk scoring as you type in the IDE |
| 🔒 **JWT Auth** | GitHub tokens Fernet-encrypted inside JWT payloads |
| 🐳 **Docker Ready** | One-command `docker compose up` with Redis + API |
| 🧾 **Pre-commit Hook** | Blocks high-risk commits with configurable threshold |
| 📄 **GitHub Action** | CI/CD integration for PR-level smell reporting |

---

## Quick Start (No Docker)

```powershell
# 1. Clone / enter project
cd smellpredict

# 2. Install
pip install -e .

# 3. Copy and fill environment variables
copy .env.example .env
# Edit .env with your GitHub OAuth credentials (see Setup below)

# 4. Start the server
python -m uvicorn smellpredict.platform.api:app --host 127.0.0.1 --port 8000 --reload
```

Open **http://localhost:8000/ui/ide.html** → Sign in with GitHub → Start coding collaboratively.

---

## Quick Start (Docker)

```bash
cp .env.example .env   # fill in credentials
docker compose up --build
```

| Service | URL |
|---|---|
| 🖥️ Collaborative IDE | http://localhost:8000/ui/ide.html |
| 📖 API Docs | http://localhost:8000/docs |
| 🏠 Platform Dashboard | http://localhost:8000/ui/ |
| 💚 Health | http://localhost:8000/health |
| 📊 Streamlit Research UI | http://localhost:8501 |

---

## GitHub OAuth App Setup

1. Go to [github.com/settings/developers](https://github.com/settings/developers) → **New OAuth App**
2. Fill in:
   - **Application name**: `SmellPredict IDE`
   - **Homepage URL**: `http://localhost:8000`
   - **Redirect URI**: `http://localhost:8000/auth/github/callback`
3. Copy **Client ID** and generate **Client Secret**
4. Edit `.env`:

```env
GITHUB_CLIENT_ID=your_client_id
GITHUB_CLIENT_SECRET=your_client_secret
GITHUB_REDIRECT_URI=http://localhost:8000/auth/github/callback
JWT_SECRET_KEY=<run: python -c "import secrets; print(secrets.token_hex(32))">
REDIS_URL=redis://redis:6379
```

---

## Project Structure

```
smellpredict/
├── src/smellpredict/
│   ├── platform/
│   │   ├── api.py          # FastAPI app — all endpoints
│   │   ├── auth.py         # GitHub OAuth + JWT + Fernet encryption
│   │   ├── github_api.py   # Repo browser + file read/write endpoints
│   │   ├── collab.py       # Y.js WebSocket relay + Redis room persistence
│   │   └── monitor.py      # Health monitoring
│   ├── features/
│   │   ├── extractor.py    # Code metrics + smell feature extraction
│   │   └── refactor.py     # AST-based refactoring suggestions
│   ├── models/
│   │   ├── predictor.py    # analyze_source_code() — full pipeline
│   │   └── trainer.py      # Model training + temporal CV
│   ├── labeling/
│   │   └── heuristic.py    # Bug-fix commit labeling (κ=0.816)
│   ├── evaluation/
│   │   └── assertions.py   # No-leak integrity assertions
│   └── explainability/
│       └── explain.py      # SHAP + risk tier classification
├── platform_ui/
│   ├── ide.html            # Collaborative Monaco IDE (Y.js + GitHub)
│   └── index.html          # Platform dashboard
├── tests/
│   └── test_all.py         # 369+ line test suite (18 test classes)
├── scripts/
│   ├── collect_data.py     # PyDriller repo mining
│   └── generate_paper_artifacts.py
├── reports/                # Research tables, figures, paper
├── docker-compose.yml      # FastAPI + Redis services
├── Dockerfile
├── pyproject.toml
├── .env.example
└── action.yml              # GitHub Action for CI/CD smell reporting
```

---

## API Reference

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/auth/github` | Start GitHub OAuth flow |
| GET | `/auth/github/callback` | OAuth callback → JWT redirect |
| GET | `/auth/me` | Get current user info |
| GET | `/auth/refresh` | Refresh JWT |

### GitHub Repository API
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/github/repos` | List user's repositories |
| GET | `/github/repos/{owner}/{repo}/branches` | List branches |
| GET | `/github/repos/{owner}/{repo}/tree` | Recursive file tree |
| GET | `/github/repos/{owner}/{repo}/file` | File content + SHA |
| POST | `/github/repos/{owner}/{repo}/commit` | Commit file to GitHub |

### Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/analyze` | Analyze a Python file |
| POST | `/api/v1/analyze/live` | Live IDE analysis (debounced) |
| GET | `/api/v1/predict` | Batch prediction |
| GET | `/health` | Service health + model status |

### Collaboration
| Method | Endpoint | Description |
|--------|----------|-------------|
| WS | `/ws/room/{room_id}` | Y.js WebSocket room (JWT-authenticated) |

---

## Research Results

| Metric | Value |
|--------|-------|
| PR-AUC (temporal CV) | **0.8297** |
| ROC-AUC | 0.847 |
| Training repos | 50 OSS Python repos |
| Snapshots | 20,832 |
| Labeling agreement κ | 0.816 |
| Model | CatBoost + calibrated isotonic regression |
| Feature set | Smell flags + complexity metrics + Halstead + MI |

**Key finding**: Code smell features improve PR-AUC by +0.041 over complexity-only baseline.

---

## Development

### Run tests
```powershell
pytest tests/test_all.py -v
```

### Pre-commit hook
```powershell
pip install pre-commit
pre-commit install
```

### Train model from scratch
```powershell
python -m smellpredict.models.trainer
```

### Mine repository data
```powershell
python scripts/collect_data.py --repos repos.txt --out data/commits.parquet
```

---

## Completed Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Dataset design & research question | ✅ |
| 2 | Bug-fix labeling heuristic (κ=0.816) | ✅ |
| 3 | Code smell feature extraction | ✅ |
| 4 | Complexity metric extraction (Radon + Halstead) | ✅ |
| 5 | Temporal CV + LOPO integrity assertions | ✅ |
| 6 | CatBoost model training + calibration | ✅ |
| 7 | SHAP explainability + risk tiers | ✅ |
| 8 | FastAPI REST platform | ✅ |
| 9 | Platform web UI (dashboard) | ✅ |
| 10 | Streamlit research dashboard | ✅ |
| 11 | Pre-commit hook integration | ✅ |
| 12 | GitHub Action for CI/CD | ✅ |
| 13 | Refactor advisor (AST-based suggestions) | ✅ |
| 14 | Research paper generation | ✅ |
| 15 | Docker containerization | ✅ |
| 16 | Dataset integrity & leak prevention | ✅ |
| 17 | CLI tooling | ✅ |
| **18** | **Collaborative IDE + GitHub storage** | ✅ |
| 19 | Multi-language live analysis | 🔜 |

---

## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Make changes (pre-commit hooks will auto-check smell risk)
4. Push and open a PR — the GitHub Action will report smell analysis

---

## License

MIT © SmellPredict Team — Abhishek-882
