# RepoLens AI — GitHub Repository Intelligence Platform

RepoLens AI is a production-ready, AI-powered codebase intelligence platform that enables developers to understand public GitHub repositories through natural-language questions grounded in verified code citations. It combines Retrieval-Augmented Generation (RAG), Abstract Syntax Tree (AST) parsing, hybrid semantic and keyword search, and flexible LLM integrations (local Ollama models and cloud providers) to generate accurate answers with verifiable file paths and exact line ranges.

---

## 📸 Application Screenshots

### Dashboard
![RepoLens AI Dashboard](screenshots/dashboardpage.png)

### Repository Analysis
![Repository Analysis](screenshots/repository-analysis.png)

### AI Chat with Verified Code References
![AI Chat Response](screenshots/ai-chat.png)

---

## 🚀 Key Features

- **GitHub Repository Ingestion & Analysis** — Clones, parses, and indexes public GitHub repositories for code navigation and Q&A.
- **AST-Aware Code Parsing** — Deconstructs Python source code into classes, functions, and modules, preserving symbol names, docstrings, and exact line boundaries.
- **Hybrid RAG Pipeline** — Combines vector similarity search (ChromaDB) with keyword/symbol matching to retrieve the most relevant source snippets.
- **Strict Citation Verification** — Validates every generated citation against current repository source lines to guarantee zero fabricated references or phantom lines.
- **Multi-Provider LLM Engine**:
  - **Local Ollama** (default development): zero-cost local inference with `qwen2.5-coder:1.5b` or `qwen2.5-coder:7b`.
  - **Google Gemini**: cloud-ready inference with `gemini-2.5-flash` or `gemini-1.5-pro`.
  - **OpenAI**: cloud-ready inference with `gpt-4o-mini` or `gpt-4o`.
- **Dimension-Consistent Embedding Architecture** — Standardized 384-dimensional vector space supporting both local Sentence Transformers (`all-MiniLM-L6-v2`) and deterministic Mock embeddings.
- **Repository Isolation & Persistence** — Independent SQLite and ChromaDB collections per repository, preventing cross-repo data leakage.
- **User Authentication & Session Management** — Secure user registration and login with bcrypt password hashing and JWT access tokens.
- **Production-Ready Web Deployment** — Pre-configured for deployment on **Vercel** (frontend) and **Render** (backend) with persistent volume support.

---

## 🏗️ Architecture & Data Flow

```text
                                  +---------------------------------------+
                                  |         User Browser / Client         |
                                  |   (React 18 + Vite + Tailwind CSS)    |
                                  +---------------------------------------+
                                                      |
                                                      | HTTP / SSE (EventSource)
                                                      v
                                  +---------------------------------------+
                                  |          FastAPI Backend API          |
                                  |   (Uvicorn + Pydantic + SQLAlchemy)   |
                                  +---------------------------------------+
                                           /          |          \
                 Ingestion / Sync         /           |           \ Query / RAG
                                         v            |            v
+------------------------------------------+          |   +----------------------------------+
|            Ingestion Service             |          |   |            RAG Engine            |
| - Git Clone / Fetch                      |          |   | - Question Classifier            |
| - Language Detection                     |          |   | - Query Formulation              |
| - AST Code Parser (Chunking)             |          |   | - Hybrid Retrieval               |
+------------------------------------------+          |   +----------------------------------+
                     |                                |            /                \
                     v                                |           /                  \
+------------------------------------------+          |          v                    v
|           Embedding Pipeline             |          |   +---------------+   +------------------+
| - Sentence Transformers (384-d)          |          |   | Vector Search |   | Keyword / SQLite |
|   or Fast Mock Embeddings (384-d)        |          |   |  (ChromaDB)   |   |   (Source Files) |
+------------------------------------------+          |   +---------------+   +------------------+
                     |                                |           \                  /
                     v                                |            \                /
+------------------------------------------+          |             v              v
|              Storage Layer               |          |   +----------------------------------+
| - SQLite: repos, files, users, chats     |<---------+   |       Context Construction       |
| - ChromaDB: 384-d vector embeddings      |              +----------------------------------+
+------------------------------------------+                               |
                                                                           v
                                                          +----------------------------------+
                                                          |           LLM Provider           |
                                                          | - Ollama (host.docker.internal)  |
                                                          | - Gemini (gemini-2.5-flash)      |
                                                          | - OpenAI (gpt-4o-mini)           |
                                                          +----------------------------------+
                                                                           |
                                                                           v
                                                          +----------------------------------+
                                                          |  Citation Validator & Formatter  |
                                                          | - AST / Source Line Validation   |
                                                          | - Groundedness Verification      |
                                                          +----------------------------------+
                                                                           |
                                                                           v
                                                              Verified Stream to Frontend
```

---

## 💻 Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend** | React 18, Vite, Tailwind CSS, React Router v6, Axios, Lucide React |
| **Backend** | Python 3.11+, FastAPI, Uvicorn, SQLAlchemy, SQLite, Pydantic, python-jose (JWT), passlib (bcrypt) |
| **AI / RAG** | ChromaDB, Sentence Transformers (`all-MiniLM-L6-v2`), Ollama (`qwen2.5-coder:1.5b`), Google Generative AI, OpenAI SDK, Python AST |
| **DevOps / Containers** | Docker, Docker Compose, Nginx, Vercel (Frontend), Render (Backend) |

---

## ⚙️ Environment Variables

### Backend Configuration (`backend/.env`)

| Variable | Default Value | Description |
|---|---|---|
| `APP_ENV` | `development` | Environment mode (`development` or `production`) |
| `DATABASE_URL` | `sqlite:///./repolens.db` | SQLAlchemy database URL (use persistent disk path in cloud) |
| `CHROMA_PERSIST_DIRECTORY` | `./chroma_data` | Directory where ChromaDB indexes are stored |
| `SECRET_KEY` | *(dev fallback in dev)* | JWT token signing key. **Required in production** (min 16 chars). Application fails fast on startup if missing, default, or placeholder in production. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `10080` | JWT token lifetime (7 days) |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Allowed frontend origins. In production, **must match exact deployed frontend origin** (not `*`) when credentials are enabled. |
| `LLM_PROVIDER` | `ollama` | Active LLM (`ollama`, `gemini`, `openai`, or `mock`) |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama API endpoint (`http://localhost:11434` for bare-metal) |
| `OLLAMA_MODEL` | `qwen2.5-coder:1.5b` | Ollama model identifier |
| `OLLAMA_TIMEOUT` | `120.0` | Ollama HTTP request timeout in seconds |
| `GEMINI_API_KEY` | *(optional)* | Google Gemini API key for hosted deployment |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name |
| `OPENAI_API_KEY` | *(optional)* | OpenAI API key for hosted deployment |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `EMBEDDING_PROVIDER` | `mock` | Embedding engine (`mock`, `local`, `openai`, `gemini`) |
| `EMBEDDING_DIMENSION` | `384` | Embedding dimensionality (must match stored vectors) |

### Frontend Configuration (`frontend/.env` or Vercel Environment)

| Variable | Default Value | Description |
|---|---|---|
| `VITE_API_BASE_URL` | `/api` | Base URL of the backend API (set to Render URL in production) |

---

## 🛠️ Local Development Setup

### Prerequisites
- **Python 3.11+** installed
- **Node.js 18+** and **npm** installed
- **Git** installed
- **Ollama** installed on your host machine ([Download Ollama](https://ollama.com/download))

### 1. Configure and Start Ollama

Start the Ollama daemon and pull the recommended coding model:
```bash
ollama serve
ollama pull qwen2.5-coder:1.5b
```

Verify that the model is ready:
```bash
ollama list
```

### 2. Set Up the Backend (Native / Local Virtualenv)

```bash
cd backend
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate

# Install dependencies:
pip install -r requirements.txt

# Start backend server:
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

### 3. Set Up the Frontend

In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```
- Frontend UI: [http://localhost:5173](http://localhost:5173)

---

## 🐳 Running with Docker Compose

Docker Compose runs both frontend and backend in isolated containers while connecting seamlessly to Ollama running on your Windows host.

### Windows Host Configuration
In `docker-compose.yml`, the backend communicates with Ollama via `host.docker.internal`:
```yaml
environment:
  - LLM_PROVIDER=ollama
  - OLLAMA_BASE_URL=http://host.docker.internal:11434
  - OLLAMA_MODEL=qwen2.5-coder:1.5b
  - EMBEDDING_PROVIDER=mock
  - EMBEDDING_DIMENSION=384
```

### Start Containers
```bash
docker compose up -d --build
```

### Check Logs & Health
```bash
docker compose logs -f backend
```
- Frontend: [http://localhost:3000](http://localhost:3000)
- Backend: [http://localhost:8000](http://localhost:8000)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

To stop services without deleting persistent volumes:
```bash
docker compose down
```
*(Never pass `-v` to preserve repository databases and indexed vector data).*

---

## 🧪 Automated Testing

RepoLens AI includes a comprehensive automated test suite covering unit, integration, and RAG retrieval functionality:

```bash
cd backend

# Run full backend test suite:
.\venv\Scripts\python -m pytest tests/

# Run LLM provider integration tests specifically:
.\venv\Scripts\python -m pytest tests/test_llm_provider_integration.py

# Verify Chroma vector index dimension compatibility:
.\venv\Scripts\python reindex_repo.py --verify-dimensions

# Verify frontend production build:
cd ../frontend
npm run build
```

---

## 🚀 Production Deployment

RepoLens AI is architected for seamless cloud deployment:
- **Frontend**: [Vercel](https://vercel.com) (Edge hosting with Vite SPA routing via `vercel.json`)
- **Backend**: [Render](https://render.com) (FastAPI web service with persistent disk via `render.yaml`)

For a complete step-by-step walkthrough, see [DEPLOYMENT_CHECKLIST.md](file:///DEPLOYMENT_CHECKLIST.md).

### Using Hosted LLMs in Production
Cloud instances (like Render or Vercel) cannot access local desktop Ollama instances. For production, switch to Google Gemini or OpenAI:

1. **Google Gemini (Recommended for cost & speed)**:
   ```env
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   ```
2. **OpenAI**:
   ```env
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4o-mini
   ```
3. **Zero-cost Portfolio Demonstration**:
   ```env
   LLM_PROVIDER=mock
   ```

---

## 📋 Common Repository Questions Supported

RepoLens AI is specifically optimized to answer:
1. **Repository Overview**: *"What is this repository and what is its main purpose?"*
2. **Tech Stack & Dependencies**: *"Which programming languages, frameworks, and technologies does it use?"*
3. **Project Structure**: *"Explain the folder and file structure."*
4. **File Purpose**: *"What is the purpose of `backend/app/ai/rag.py`? Explain its main responsibilities."*
5. **Function & Class Anatomy**: *"Explain the purpose of `RAGEngine` and its execution flow."*
6. **Feature Implementation**: *"Where is JWT authentication implemented?"*
7. **API Endpoints**: *"Which file contains the repository analysis API endpoints?"*
8. **Client-Server Communication**: *"How does the frontend communicate with the backend?"*

Every response includes exact file paths and line number ranges checked against actual source files.

---

## ⚠️ Known Limitations

- **Language Support**: Deep AST structural parsing is currently optimized for Python. Other languages are indexed using line-aware text and symbol chunking.
- **Dynamic Reflection**: Dynamically imported or runtime-evaluated dependencies cannot be parsed statically.
- **Resource Constraints**: Local LLM speed is bounded by your machine's CPU/GPU capabilities. For 8GB RAM systems, `qwen2.5-coder:1.5b` is strongly recommended.
