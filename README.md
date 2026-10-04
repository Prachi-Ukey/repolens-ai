# RepoLens AI — AI Codebase Assistant & RAG Analyzer

**RepoLens AI** is a production-quality, full-stack AI application that connects to public GitHub repositories, understands their structure and source code using an AST-aware Retrieval-Augmented Generation (RAG) pipeline, and provides grounded, line-level answers with source citations.

---

## Key Features

- **GitHub Repository Ingestion**: Submit any public GitHub URL to validate, clone, parse, and chunk code. Real-time timeline status reporting (`validating`, `downloading`, `detecting_structure`, `parsing_files`, `chunking`, `embedding`, `completed`).
- **AST & Scope-Aware Code Chunker**: Preserves function, class, and method boundaries with start and end line metadata while ignoring `.git`, `node_modules`, `dist`, binaries, and secrets.
- **Repository-Isolated ChromaDB Vector Storage**: Vector embeddings are isolated per repository collection (`repo_{repo_id}`).
- **Grounded RAG Engine & Line-Level Citations**: Answers questions using repository code context only. Every response includes clickable source badges (`[server/middleware/auth.js:12-38]`) opening the exact lines in an interactive Code Viewer.
- **Prompt Injection Protection**: Protects against malicious code/README instructions by wrapping repository context in untrusted data tags `<repository_code_context>`.
- **3-Panel Developer Workspace**:
  - **Left Panel**: Interactive file tree explorer with search filtering.
  - **Center Panel**: AI Chat assistant with quick action buttons ("Explain Architecture", "Find Auth", "Find Security Risks", etc.).
  - **Right Panel**: Syntax-highlighted code viewer with line numbers, highlighted cited line ranges, and search in file.
- **Architecture Visualization**: Generates interactive component node graphs (Frontend, API, Auth, Database, External Services) with zoom/pan capabilities.
- **Code Quality & Dependency Audit**: Scans for potential security concerns (hardcoded secrets, bare exceptions, SQL string formatting) and parses package manifests (`package.json`, `requirements.txt`).
- **AI Documentation Generator**: Export custom READMEs, API docs, Architecture overviews, and Onboarding guides.
- **Demo Mode**: Includes pre-packaged sample chat application dataset for instant zero-config testing.

---

## Tech Stack

### Frontend
- **Framework**: React 18 + Vite
- **Styling**: Tailwind CSS (Dark-first VS Code / GitHub developer UI)
- **Icons**: Lucide React
- **HTTP**: Axios with JWT interceptors
- **Routing**: React Router v6

### Backend
- **Framework**: Python 3.11+ / FastAPI
- **Web Server**: Uvicorn
- **ORM & DB**: SQLAlchemy 2.0 + SQLite (default) / PostgreSQL
- **Security**: JWT tokens, bcrypt password hashing

### AI & Vector Storage
- **RAG Framework**: AST/Line-aware chunker + ChromaDB Persistent Vector Store
- **Providers**: Abstraction layer supporting OpenAI (`gpt-4o-mini`), Google Gemini (`gemini-1.5-flash`), and Mock/Local Fallback.

---

## How RAG Pipeline Works

```
User Question
    │
    ▼
Embed Question (EmbeddingProvider: OpenAI / Gemini / Mock)
    │
    ▼
Vector Similarity Search (ChromaDB Collection repo_{repo_id})
    │
    ▼
Retrieve Relevant Code Chunks (Line numbers, File path, Symbol name)
    │
    ▼
Prompt Injection Protection Wrapper (<repository_code_context> + System Prompts)
    │
    ▼
LLM Generation (OpenAI / Gemini / Mock)
    │
    ▼
Grounded Answer + Clickable Line Citations [file_path:start_line-end_line]
```

---

## Installation & Local Execution

### 1. Backend Setup
```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
python app/main.py
```
Backend API will run at `http://localhost:8000`. OpenAPI docs available at `http://localhost:8000/docs`.

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Frontend web UI will run at `http://localhost:5173`.

---

## Environment Variables

Edit `backend/.env`:
```env
APP_ENV=development
DEBUG=True
SECRET_KEY=repolens-super-secret-jwt-key-2026
DATABASE_URL=sqlite:///./repolens.db

# Provider options: mock, openai, gemini
LLM_PROVIDER=mock
EMBEDDING_PROVIDER=mock
OPENAI_API_KEY=
GEMINI_API_KEY=
CHROMA_PERSIST_DIRECTORY=./chroma_data
```

---

## Running with Docker Compose

```bash
docker-compose up --build
```
- Web Application: `http://localhost:3000`
- Backend API: `http://localhost:8000`

---

## Running Automated Test Suite

```bash
cd backend
python tests/run_tests.py
```

---

## License

MIT License
